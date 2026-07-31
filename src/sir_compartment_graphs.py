"""
sir_compartment_graphs.py
=========================
Visualizzazione del grafo spaziale per ogni singolo compartimento.

Per ogni compartimento (es. "PT_boundary", "vascular_core", ...):
  - I nodi sono i singoli spot Visium appartenenti al compartimento
  - Gli archi sono le connessioni kNN interne (Wsp filtrato per nodi del comp.)
  - I pesi degli archi sono i pesi gaussiani di Wsp
  - Il layout usa le coordinate spaziali Visium reali

Due figure per compartimento
-----------------------------
Fig A — Statico R₀:
    Tutti i nodi colorati con R₀ = β/γ del compartimento (costante per nodo,
    since R₀ is defined at compartment level). Shows the internal topological
    structure and the connectivity degree of the spots.

Fig B — Snapshot I(t):
    Pannello multi-istante: i nodi sono colorati con I(t) espanso tramite inv2.
    Mostra come il segnale infiammatorio si distribuisce spazialmente
    all'interno del compartimento nel tempo.

Gerarchia coerente con il modello
-----------------------------------
    Wsp (kNN spot)  →  Wc2 (coarse-grained)  →  dinamica SIR
    Fig A/B show the Wsp level, consistent with how Wc2 was constructed.

Funzioni esportate
-------------------
plot_compartment_graph_static    : Fig A per un singolo compartimento
plot_compartment_graph_snapshots : Fig B per un singolo compartimento
plot_all_compartments            : itera su tutti i compartimenti
"""

import numpy as np
import scipy.sparse as sp
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import networkx as nx
import warnings


# ═══════════════════════════════════════════════════════════════════════════
# INTERNAL UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def _subgraph_for_compartment(
    Wsp:       sp.spmatrix,
    node_mask: np.ndarray,
    threshold_w: float = 1e-4,
) -> tuple:
    """
    Estrae il sottografo di Wsp per i nodi del compartimento.

    Restituisce solo gli archi tra nodi entrambi nel compartimento
    (archi interni), filtrando quelli con peso < threshold_w.

    Parameters
    ----------
    Wsp       : sparse (N, N) — grafo kNN originale tra spot
    node_mask : ndarray bool (N,) — nodi del compartimento
    threshold_w : float — soglia minima peso arco

    Returns
    -------
    G         : nx.Graph con nodi = indici originali degli spot
    pos       : dict {node_idx: (x, y)} con coordinate Visium normalizzate
    local_idx : ndarray — indici originali dei nodi nel compartimento
    """
    local_idx = np.where(node_mask)[0]
    if len(local_idx) == 0:
        return nx.Graph(), {}, local_idx

    # Sottomatrice quadrata per i nodi del compartimento
    Wsub = Wsp[np.ix_(local_idx, local_idx)].tocsr()

    G = nx.Graph()
    G.add_nodes_from(local_idx.tolist())

    cx = Wsub.tocoo()
    for i_loc, j_loc, w in zip(cx.row, cx.col, cx.data):
        if i_loc < j_loc and w > threshold_w:
            G.add_edge(
                int(local_idx[i_loc]),
                int(local_idx[j_loc]),
                weight=float(w),
            )

    return G, local_idx


def _visium_pos(coords: np.ndarray, node_idx: np.ndarray) -> dict:
    """
    Costruisce il dict pos per networkx dalle coordinate Visium.
    Normalizza in [0,1] sugli assi del sottografo, inverte y.
    """
    xy = coords[node_idx].astype(float)
    x_min, x_max = xy[:, 0].min(), xy[:, 0].max()
    y_min, y_max = xy[:, 1].min(), xy[:, 1].max()

    x_range = x_max - x_min if x_max > x_min else 1.0
    y_range = y_max - y_min if y_max > y_min else 1.0

    pos = {}
    for i, idx in enumerate(node_idx):
        xn = (xy[i, 0] - x_min) / x_range
        yn = 1.0 - (xy[i, 1] - y_min) / y_range   # y invertito
        pos[int(idx)] = (xn, yn)
    return pos


# ═══════════════════════════════════════════════════════════════════════════
# FIGURA A — GRAFO STATICO R₀
# ═══════════════════════════════════════════════════════════════════════════

def plot_compartment_graph_static(
    comp_name:    str,
    comp_idx:     int,
    Wsp:          sp.spmatrix,
    inv2:         np.ndarray,
    coords:       np.ndarray,
    sir_params:   dict,
    macro_of:     np.ndarray,
    region_of:    np.ndarray,
    boundary_mask: np.ndarray,
    figsize:      tuple = (7, 6),
    node_size:    float = 40.0,
    edge_scale:   float = 3.0,
    threshold_w:  float = 1e-4,
    save_path:    str   = None,
):
    """
    Grafo statico per un singolo compartimento — R₀ sui nodi.

    Since R₀ is uniform per compartment (β/γ is defined at
    compartmental level), the colour is uniform. The figure therefore shows:
    - Internal topological structure (spot connectivity)
    - Il grado di ogni nodo (hub vs nodi periferici nel compartimento)
    - Il titolo riporta R₀, macro_type e region

    Parameters
    ----------
    comp_name  : str  — nome del compartimento (es. "PT_boundary")
    comp_idx   : int  — indice del compartimento in comps2
    Wsp        : sparse (N,N) — grafo kNN spaziale originale
    inv2       : ndarray int (N,) — indice compartimento per spot
    coords     : ndarray (N,2) — coordinate spaziali Visium
    sir_params : dict — output di build_default_params_SIR
    save_path  : str o None — se fornito, salva la figura
    """
    beta_v  = np.asarray(sir_params["beta"],  dtype=float)
    gamma_v = np.asarray(sir_params["gamma"], dtype=float)
    R0_comp = beta_v[comp_idx] / max(gamma_v[comp_idx], 1e-12)

    node_mask = inv2 == comp_idx
    n_nodes   = node_mask.sum()
    if n_nodes == 0:
        warnings.warn(f"Compartimento '{comp_name}' vuoto — skip.")
        return

    G, local_idx = _subgraph_for_compartment(Wsp, node_mask, threshold_w)
    pos          = _visium_pos(coords, local_idx)

    # Node degrees (internal connectivity)
    degrees    = dict(G.degree())
    deg_vals   = np.array([degrees.get(int(i), 0) for i in local_idx])
    deg_max    = deg_vals.max() if deg_vals.max() > 0 else 1.0

    # Node colour: internal connectivity degree (blue = isolated, yellow = hub)
    norm_deg   = mcolors.Normalize(vmin=0, vmax=deg_max)
    cmap_deg   = cm.YlGnBu
    node_colors = [cmap_deg(norm_deg(degrees.get(int(i), 0))) for i in local_idx]
    node_sizes  = [node_size * (0.8 + 0.5 * degrees.get(int(i), 0) / deg_max)
                   for i in local_idx]

    # Archi
    edges    = list(G.edges(data=True))
    edge_ws  = np.array([d["weight"] for _, _, d in edges]) if edges else np.array([])
    if len(edge_ws) > 0:
        ew_n      = edge_ws / (edge_ws.max() + 1e-12)
        edge_widths = edge_scale * ew_n
        edge_alphas = 0.3 + 0.5 * ew_n
    else:
        edge_widths = []
        edge_alphas = []

    is_boundary = boundary_mask[comp_idx]
    border_col  = "#cc0000" if is_boundary else "#336699"
    border_lw   = 2.5       if is_boundary else 1.0

    fig, ax = plt.subplots(figsize=figsize)

    # Archi
    for (u, v, d), lw, al in zip(edges, edge_widths, edge_alphas):
        nx.draw_networkx_edges(
            G, pos, edgelist=[(u, v)],
            width=float(lw), alpha=float(al),
            edge_color="#888888", ax=ax,
        )

    # Nodi
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=[int(i) for i in local_idx],
        node_color=node_colors,
        node_size=node_sizes,
        linewidths=border_lw,
        edgecolors=border_col,
        ax=ax,
    )

    # Colorbar grado
    sm = cm.ScalarMappable(cmap=cmap_deg, norm=norm_deg)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("Internal degree (connectivity)", fontsize=9)

    seed_label = "  [SEED]" if is_boundary else ""
    ax.set_title(
        f"{comp_name}{seed_label}\n"
        f"R₀ = {R0_comp:.2f}  |  β={beta_v[comp_idx]:.3f}  "
        f"γ={gamma_v[comp_idx]:.3f}  |  {n_nodes} spot  "
        f"|  {G.number_of_edges()} archi interni",
        fontsize=10, pad=10,
    )
    ax.axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"    Salvato: {save_path}")

    plt.show()
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# FIGURA B — SNAPSHOT I(t)
# ═══════════════════════════════════════════════════════════════════════════

def plot_compartment_graph_snapshots(
    comp_name:    str,
    comp_idx:     int,
    sol,
    Wsp:          sp.spmatrix,
    inv2:         np.ndarray,
    coords:       np.ndarray,
    boundary_mask: np.ndarray,
    macro_of:     np.ndarray,
    times:        list  = None,
    n_panels:     int   = 6,
    figsize:      tuple = (16, 9),
    node_size:    float = 50.0,
    edge_scale:   float = 2.5,
    threshold_w:  float = 1e-4,
    cmap_I:       str   = "YlOrRd",
    save_path:    str   = None,
    I_max_global: float = None,   # Fix 2: colorbar globale condivisa
):
    """
    Snapshot temporali di I(t) per un singolo compartimento.

    The I value assigned to each spot is that of its compartment,
    appartenenza (costante dentro il compartimento per ogni t), espanso
    via inv2. This is consistent with the granularity of the SIR model,
    che opera a livello compartimentale.

    Parameters
    ----------
    sol       : OdeResult o dict — output di run_SIR
    times     : lista di istanti da visualizzare (None = automatico)
    n_panels  : numero di pannelli se times=None
    save_path : str o None
    """
    # Estrai traiettoria I
    if hasattr(sol, "t"):
        t_arr  = sol.t
        nC_tot = sol.y.shape[0] // 3
        I_traj = sol.y[nC_tot:2*nC_tot, :]   # (C, T)
    else:
        t_arr  = sol["t"]
        I_traj = sol["I"]

    # Scegli istanti
    if times is None:
        idx_list = np.linspace(1, len(t_arr) - 1, n_panels, dtype=int)
        times_plot = t_arr[idx_list].tolist()
    else:
        idx_list   = [int(np.argmin(np.abs(t_arr - tt))) for tt in times]
        times_plot = times

    node_mask = inv2 == comp_idx
    n_nodes   = node_mask.sum()
    if n_nodes == 0:
        warnings.warn(f"Compartimento '{comp_name}' vuoto — skip.")
        return

    G, local_idx = _subgraph_for_compartment(Wsp, node_mask, threshold_w)
    pos          = _visium_pos(coords, local_idx)

    # Fix 2: usa colorbar globale se fornita, altrimenti massimo locale
    I_vals_comp = I_traj[comp_idx, :]
    if I_max_global is not None:
        I_max = float(I_max_global)
    else:
        I_max = float(I_vals_comp.max()) if I_vals_comp.max() > 0 else 0.01
    norm_I      = mcolors.Normalize(vmin=0.0, vmax=I_max)
    cmap        = matplotlib.colormaps[cmap_I]

    # Archi (fissi)
    edges    = list(G.edges(data=True))
    edge_ws  = np.array([d["weight"] for _, _, d in edges]) if edges else np.array([])
    if len(edge_ws) > 0:
        ew_n        = edge_ws / (edge_ws.max() + 1e-12)
        edge_widths = edge_scale * ew_n
        edge_alphas = 0.2 + 0.4 * ew_n
    else:
        edge_widths = []
        edge_alphas = []

    is_boundary = boundary_mask[comp_idx]
    border_col  = "#cc0000" if is_boundary else "#336699"

    n_cols = min(3, len(idx_list))
    n_rows = int(np.ceil(len(idx_list) / n_cols))
    # Fix: figsize dinamico + constrained_layout evita tight_layout warning
    dyn_w = max(figsize[0], n_cols * 5.5)
    dyn_h = max(figsize[1] * n_rows / 2, n_rows * 4.5)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(dyn_w, dyn_h),
                             constrained_layout=True,
                             squeeze=False)

    for panel, (k, tt) in enumerate(zip(idx_list, times_plot)):
        row = panel // n_cols
        col = panel  % n_cols
        ax  = axes[row][col]

        # I is uniform for all spots within the compartment
        I_now   = float(I_traj[comp_idx, k])
        n_color = cmap(norm_I(I_now))

        # Archi
        for (u, v, d), lw, al in zip(edges, edge_widths, edge_alphas):
            nx.draw_networkx_edges(
                G, pos, edgelist=[(u, v)],
                width=float(lw), alpha=float(al),
                edge_color="#aaaaaa", ax=ax,
            )

        # Nodi — tutti con lo stesso colore I(t) del compartimento
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=[int(i) for i in local_idx],
            node_color=[n_color] * len(local_idx),
            node_size=node_size,
            linewidths=1.2,
            edgecolors=border_col,
            ax=ax,
        )

        ax.set_title(f"t = {tt:.1f}   I = {I_now:.3f}", fontsize=10)
        ax.axis("off")

    # Pannelli vuoti
    for panel in range(len(idx_list), n_rows * n_cols):
        axes[panel // n_cols][panel % n_cols].axis("off")

    # Colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm_I)
    sm.set_array([])
    fig.colorbar(sm, ax=axes, shrink=0.6, pad=0.02, label="I(t)")

    seed_label = "  [SEED — infezione inizia qui]" if is_boundary else ""
    fig.suptitle(
        f"Propagazione I(t) — {comp_name}{seed_label}\n"
        f"{n_nodes} spot  |  {G.number_of_edges()} archi interni  "
        f"|  macro_type: {macro_of[comp_idx]}",
        fontsize=12, y=1.02,
    )
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"    Salvato: {save_path}")

    plt.show()
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# WRAPPER — ITERA SU TUTTI I COMPARTIMENTI
# ═══════════════════════════════════════════════════════════════════════════

def plot_all_compartments(
    sol,
    Wsp:          sp.spmatrix,
    sir_params:   dict,
    comps2:       np.ndarray,
    macro_of:     np.ndarray,
    region_of:    np.ndarray,
    boundary_mask: np.ndarray,
    inv2:         np.ndarray,
    adata_st,
    times:        list  = None,
    n_panels:     int   = 6,
    only:         list  = None,
    threshold_w:  float = 1e-4,
    save_dir:     str   = None,
    min_spots:    int   = 20,     # Fix 1: soglia minima spot per visualizzazione
    global_cbar:  bool  = True,   # Fix 2: colorbar globale condivisa
):
    """
    Genera Fig A + Fig B per ogni compartimento.

    Parameters
    ----------
    only        : lista di nomi compartimento da visualizzare.
                  Se None, genera tutte. Esempio: ["PT_boundary", "PT_core"]
    save_dir    : cartella dove salvare le figure (None = non salva)
    adata_st    : AnnData con obsm['spatial']
    min_spots   : int — compartimenti con meno di min_spots spot vengono
                  saltati. Default 20. Difendibile: sotto questa soglia
                  the internal graph is not structurally informative.
    global_cbar : bool — se True usa una colorbar I(t) globale uguale per
                  tutti i compartimenti, rendendo le figure confrontabili.

    Esempio
    -------
    >>> plot_all_compartments(
    ...     sol_sir, comp["Wsp"], sir_params,
    ...     comp["comps2"], comp["macro_of"], comp["region_of"],
    ...     comp["boundary_mask"], comp["inv2"], adata_st2,
    ...     only=["PT_boundary", "PT_core", "vascular_boundary"],
    ...     n_panels=6,
    ... )
    """
    import os
    coords = adata_st.obsm["spatial"].astype(np.float32)

    # Fix 2: calcola I_max_global una volta sola su tutti i compartimenti
    # da includere (rispettando il filtro min_spots e only)
    I_max_global = None
    if global_cbar:
        if hasattr(sol, "t"):
            nC_tot = sol.y.shape[0] // 3
            I_traj_all = sol.y[nC_tot:2*nC_tot, :]
        else:
            I_traj_all = sol["I"]

        candidate_max = 0.0
        for ci, cname in enumerate(comps2):
            if only is not None and cname not in only:
                continue
            if int((inv2 == ci).sum()) < min_spots:
                continue
            candidate_max = max(candidate_max, float(I_traj_all[ci, :].max()))
        I_max_global = candidate_max if candidate_max > 0 else 0.01
        print(f"  Colorbar globale I_max = {I_max_global:.4f}")

    for comp_idx, comp_name in enumerate(comps2):

        # Filtro opzionale per nome
        if only is not None and comp_name not in only:
            continue

        n_spots = int((inv2 == comp_idx).sum())

        # Fix 1: salta compartimenti con troppi pochi spot
        if n_spots < min_spots:
            print(f"  [{comp_name}] — {n_spots} spot < {min_spots} → skip "
                  f"(soglia min_spots)")
            continue

        print(f"\n{'─'*55}")
        print(f"  Compartimento: {comp_name}  ({n_spots} spot)")
        print(f"{'─'*55}")

        # Percorsi di salvataggio opzionali
        save_static = None
        save_snap   = None
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            safe = comp_name.replace("/", "_")
            save_static = os.path.join(save_dir, f"{safe}_static_R0.png")
            save_snap   = os.path.join(save_dir, f"{safe}_snapshots_I.png")

        # Fig A — statico R₀
        plot_compartment_graph_static(
            comp_name, comp_idx,
            Wsp, inv2, coords,
            sir_params, macro_of, region_of, boundary_mask,
            threshold_w=threshold_w,
            save_path=save_static,
        )

        # Fig B — snapshot I(t)
        plot_compartment_graph_snapshots(
            comp_name, comp_idx,
            sol, Wsp, inv2, coords, boundary_mask, macro_of,
            times=times, n_panels=n_panels,
            threshold_w=threshold_w,
            save_path=save_snap,
            I_max_global=I_max_global,   # Fix 2
        )