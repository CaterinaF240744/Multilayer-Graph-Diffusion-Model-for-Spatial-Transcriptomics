"""
sir_drug_graphs.py
==================
Visualizzazioni a grafo della dinamica SIRD+farmaco.

Livello 2 — Grafo compartimentale con nodi colorati
-----------------------------------------------------
I nodi sono i compartimenti spaziali (PT_boundary, PT_mid, ...).
Each node is a circle with:
  - Colour  = I(t) on YlOrRd scale  (inflammatory intensity)
  - Raggio  ∝ D(t)                  (danno accumulato)
  - Bordo   = spessore ∝ R(t)       (recovery)
  - Bordo rosso spesso = nodo boundary (seed)
Gli archi sono pesati dal flusso inter-compartimentale (da Wc2).
Tre snapshot: t=t_dose, t=t_peak_I, t=t_end.

Livello 3 — Rete multilayer con dinamica farmaco
-------------------------------------------------
Estende la figura multilayer esistente (PT/DCT/TAL affiancati con
coordinate Visium reali). I nodi hanno:
  - Colour      = I(t) from the SIRD model (inflammatory intensity)
  - Raggio      = proporzionale a D(t) (danno accumulato)
  - Bordo rosso = spot seed (boundary)
Affiancamento farmaco vs no-farmaco sulle righe.
Tre colonne = tre istanti temporali (t_dose, t_peak, t_end).

Livello 4 — Confronto SIRD a t_dose diversi (figure separate)
--------------------------------------------------------------
Tre figure indipendenti per t_dose = {pre-picco, picco, post-picco}.
Ogni figura mostra il grafo compartimentale colorato a tre snapshot
temporali (t_dose, t_peak_I, t_end).

Funzioni esportate
------------------
plot_drug_state_graph        : Livello 2 — nodi colorati, 3 snapshot
plot_drug_multilayer         : Livello 3 — multilayer con I e D, drug vs nodrug
plot_drug_tdose_comparison   : Livello 4 — figure separate per t_dose diversi
plot_all_drug_graphs         : runner integrato Livello 2+3+4
"""

import numpy as np
import scipy.sparse as sp
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
import warnings

try:
    import networkx as nx
    _HAS_NX = True
except ImportError:
    _HAS_NX = False
    warnings.warn("networkx non trovato — plot_drug_state_graph disabilitato",
                  ImportWarning)


# ═══════════════════════════════════════════════════════════════════════════
# COSTANTI
# ═══════════════════════════════════════════════════════════════════════════

LAYER_COLORS = {"PT": "#c0392b", "DCT": "#2471a3", "TAL": "#1e8449"}
MULTILAYER_MACROS = ["PT", "DCT", "TAL"]

# Colormap condivisa per I(t)
_CMAP_I = matplotlib.colormaps["YlOrRd"]
_NORM_I = mcolors.Normalize(vmin=0.0, vmax=0.35)


# ═══════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def _extract_SIRD_snapshot(sol, nC: int, t_idx: int) -> dict:
    """Estrae S, I, I₂, R, D, F a un dato indice temporale."""
    y = sol.y[:, t_idx]
    return {
        "S":  np.clip(y[0*nC:1*nC], 0.0, 1.0),
        "I":  np.clip(y[1*nC:2*nC], 0.0, 1.0),
        "I2": np.clip(y[2*nC:3*nC], 0.0, 1.0),
        "R":  np.clip(y[3*nC:4*nC], 0.0, 1.0),
        "D":  np.clip(y[4*nC:5*nC], 0.0, 1.0),
        "F":  np.clip(y[5*nC:6*nC], 0.0, None),
    }


def _find_t_peak_I(sol, nC: int) -> int:
    """Indice temporale at peak di I medio."""
    I_mean = sol.y[1*nC:2*nC, :].mean(axis=0)
    return int(np.argmax(I_mean))


def _draw_circle_node(ax, center, radius, I_val, D_val, R_val,
                      macro_type="other", is_boundary=False):
    """
    Disegna un nodo circolare stile network:
      - Fill color  = I_val mapped to YlOrRd (inflammatory intensity)
      - Radius      ∝ D_val (danno accumulato)
      - Bordo color = macro_type (PT=rosso, DCT=blu, TAL=verde, altri=grigio)
      - Bordo lw    = spesso se boundary seed, medio altrimenti
      - Alone esterno sottile = R_val (recovery, colore verde pallido)
    """
    face_color  = _CMAP_I(_NORM_I(I_val))

    # Colore bordo = macro_type
    macro_border = {
        "PT":       "#c0392b",
        "DCT":      "#2471a3",
        "TAL":      "#1e8449",
        "vascular": "#8e44ad",
        "immune":   "#d35400",
        "other":    "#7f8c8d",
    }
    edge_color = macro_border.get(str(macro_type), "#555555")
    edge_lw    = 3.5 if is_boundary else 1.8

    # Alone R: cerchio esterno trasparente verde proporzionale a R
    if R_val > 0.05:
        halo = Circle(center, radius * (1.0 + 0.35 * R_val),
                      facecolor="none",
                      edgecolor="#27ae60",
                      linewidth=0.8,
                      alpha=min(0.15 + 0.5 * R_val, 0.55),
                      zorder=3)
        ax.add_patch(halo)

    circ = Circle(center, radius,
                  facecolor=face_color,
                  edgecolor=edge_color,
                  linewidth=edge_lw,
                  alpha=0.92,
                  zorder=4)
    ax.add_patch(circ)


def _build_network_layout(nC, macro_of, region_of, Wc2, edge_thresh=0.01):
    """
    Layout Kamada-Kawai pesato da Wc2.

    The most connected nodes (hubs) converge to the centre, peripheral nodes
    ai bordi — riflette naturalmente la gerarchia biologica.
    The layout is deterministic and stable across different snapshots.

    Returns
    -------
    G   : nx.Graph con pesi Wc2
    pos : dict {node_id: (x, y)} normalizzato in [0,1]
    """
    Wc2_csr = sp.csr_matrix(Wc2)
    G = nx.Graph()
    G.add_nodes_from(range(nC))
    cx = Wc2_csr.tocoo()
    for i, j, w in zip(cx.row, cx.col, cx.data):
        if i < j and w > edge_thresh:
            G.add_edge(i, j, weight=float(w))

    # Kamada-Kawai con distanza = 1/peso (nodi molto connessi si avvicinano)
    try:
        raw_pos = nx.kamada_kawai_layout(G, weight="weight")
    except Exception:
        # Fallback a spring_layout se KK fallisce (grafo disconnesso)
        raw_pos = nx.spring_layout(G, weight="weight", seed=42)

    # Normalizza in [0.05, 0.95] per evitare nodi ai bordi della figura
    xs = np.array([raw_pos[i][0] for i in range(nC)])
    ys = np.array([raw_pos[i][1] for i in range(nC)])
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    x_range = max(x_max - x_min, 1e-6)
    y_range = max(y_max - y_min, 1e-6)

    pos = {
        i: (
            0.05 + 0.90 * (raw_pos[i][0] - x_min) / x_range,
            0.05 + 0.90 * (raw_pos[i][1] - y_min) / y_range,
        )
        for i in range(nC)
    }
    return G, pos


def _build_snapshots(sol, nC, t_dose):
    """Restituisce lista di (t_idx, label) per i 3 snapshot standard."""
    t = sol.t
    t_dose_idx = int(np.argmin(np.abs(t - t_dose)))
    t_peak_idx = _find_t_peak_I(sol, nC)
    t_end_idx  = len(t) - 1
    return [
        (t_dose_idx, f"t = {t[t_dose_idx]:.1f}  [somministrazione]"),
        (t_peak_idx, f"t = {t[t_peak_idx]:.1f}  [picco I]"),
        (t_end_idx,  f"t = {t[t_end_idx]:.1f}   [final]"),
    ]


def _add_colorbar(fig, axes):
    """Aggiunge colorbar I(t) condivisa alla figura."""
    sm = cm.ScalarMappable(cmap=_CMAP_I, norm=_NORM_I)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.45, pad=0.02, aspect=25)
    cbar.set_label("I(t) — infetti", fontsize=9)
    return cbar


# ═══════════════════════════════════════════════════════════════════════════
# LIVELLO 2 — GRAFO COMPARTIMENTALE CON NODI COLORATI
# ═══════════════════════════════════════════════════════════════════════════

def plot_drug_state_graph(
    sol_drug:      object,
    sol_nodrug:    object,
    comps2:        np.ndarray,
    macro_of:      np.ndarray,
    region_of:     np.ndarray,
    boundary_mask: np.ndarray,
    Wc2:           sp.spmatrix,
    t_dose:        float,
    figsize:       tuple = (20, 12),
    node_radius:   float = 0.05,
    edge_thresh:   float = 0.01,
    save_path:     str   = None,
) -> None:
    """
    Livello 2 — Grafo compartimentale con stato SIRD sovrapposto.

    Layout
    ------
    - 2 righe: farmaco (sopra) e no-farmaco (sotto)
    - 3 colonne: t=t_dose, t=t_peak_I, t=t_end
    - Ogni nodo = cerchio con:
        colore ∝ I(t)   (YlOrRd)
        raggio ∝ D(t)   (danno)
        bordo  ∝ R(t)   (recovery)
        bordo rosso     = boundary seed
    - Edges = Wc2 connectivity

    Parameters
    ----------
    sol_drug, sol_nodrug : OdeResult — simulazioni con e senza farmaco
    comps2               : ndarray (C,) — nomi compartimenti
    macro_of, region_of  : ndarray (C,) — tipo e regione
    boundary_mask        : ndarray bool (C,)
    Wc2                  : sparse (C,C) — pesi compartimentali
    t_dose               : float — primo snapshot temporale
    """
    if not _HAS_NX:
        print("  [SKIP] plot_drug_state_graph richiede networkx")
        return

    nC        = sol_drug.y.shape[0] // 6
    snapshots = _build_snapshots(sol_drug, nC, t_dose)

    # Layout Kamada-Kawai — calcolato una sola volta, stabile tra snapshot
    G, pos = _build_network_layout(nC, macro_of, region_of, Wc2, edge_thresh)

    # Pesi archi per rendering
    edge_weights = {(i, j): d["weight"]
                    for i, j, d in G.edges(data=True)}
    w_max = max(edge_weights.values(), default=1.0)

    fig, axes = plt.subplots(
        2, 3, figsize=figsize,
        gridspec_kw={"hspace": 0.35, "wspace": 0.15},
        facecolor="#f8f8f8",
    )

    row_labels = ["+farmaco", "−farmaco"]
    sols       = [sol_drug, sol_nodrug]

    for row, (sol, rlabel) in enumerate(zip(sols, row_labels)):
        for col, (t_idx, t_label) in enumerate(snapshots):
            ax   = axes[row][col]
            ax.set_facecolor("#f8f8f8")
            snap = _extract_SIRD_snapshot(sol, nC, t_idx)

            # ── Archi ────────────────────────────────────────────────────
            for (i, j), w in edge_weights.items():
                xi, yi = pos[i]
                xj, yj = pos[j]
                # Colore arco = blend tra i colori macro dei due nodi
                lw    = 0.8 + 3.5 * (w / w_max)
                alpha = 0.25 + 0.50 * (w / w_max)
                ax.plot([xi, xj], [yi, yj],
                        color="#888888",
                        lw=lw, alpha=alpha,
                        solid_capstyle="round",
                        zorder=1)

            # ── Nodi ─────────────────────────────────────────────────────
            d_max = snap["D"].max() + 1e-6
            for i in range(nC):
                xc, yc = pos[i]
                # Raggio: base fissa + incremento per D
                r = node_radius * (0.55 + 1.8 * snap["D"][i] / d_max)
                _draw_circle_node(
                    ax, (xc, yc), r,
                    I_val=snap["I"][i],
                    D_val=snap["D"][i],
                    R_val=snap["R"][i],
                    macro_type=str(macro_of[i]),
                    is_boundary=bool(boundary_mask[i]),
                )
                # Etichetta: solo primo snapshot, prima riga
                if row == 0 and col == 0:
                    # Nome corto: rimuove prefisso macro_type
                    label = str(comps2[i])
                    ax.text(xc, yc - r - 0.028, label,
                            ha="center", va="top", fontsize=5.2,
                            color="#222222", zorder=6,
                            bbox=dict(boxstyle="round,pad=0.1",
                                      fc="white", ec="none", alpha=0.6))

            ax.set_xlim(-0.08, 1.08)
            ax.set_ylim(-0.08, 1.08)
            ax.set_aspect("equal")
            ax.axis("off")

            col_title = t_label if row == 0 else t_label.split("[")[0].strip()
            ax.set_title(col_title, fontsize=9, pad=5,
                         fontweight="bold" if row == 0 else "normal")

            if col == 0:
                ax.text(-0.10, 0.5, rlabel, transform=ax.transAxes,
                        ha="center", va="center", fontsize=10,
                        fontweight="bold", rotation=90,
                        color="#c0392b" if row == 0 else "#555555")

    # ── Legenda ──────────────────────────────────────────────────────────
    macro_border = {
        "PT": "#c0392b", "DCT": "#2471a3",
        "TAL": "#1e8449", "vascular": "#8e44ad", "other": "#7f8c8d",
    }
    legend_els = (
        # Macro-type (colore bordo)
        [Line2D([0], [0], marker="o", color="w",
                markerfacecolor="#dddddd",
                markeredgecolor=c, markeredgewidth=2.0,
                markersize=9, label=m)
         for m, c in macro_border.items()] +
        # I
        [Line2D([0], [0], marker="o", color="w",
                markerfacecolor=_CMAP_I(_NORM_I(0.30)),
                markersize=9, label="I alto"),
         Line2D([0], [0], marker="o", color="w",
                markerfacecolor=_CMAP_I(_NORM_I(0.04)),
                markersize=9, label="I basso")] +
        # D e boundary
        [Line2D([0], [0], marker="o", color="w",
                markerfacecolor="#aaaaaa", markersize=13,
                label="nodo grande = D alto"),
         Line2D([0], [0], marker="o", color="w",
                markerfacecolor="none", markeredgecolor="#c0392b",
                markeredgewidth=3.0, markersize=9,
                label="boundary (seed)"),
         Line2D([0], [0], marker="o", color="w",
                markerfacecolor="none", markeredgecolor="#27ae60",
                markeredgewidth=1.0, markersize=11,
                label="alone verde = R alto")]
    )
    fig.legend(handles=legend_els, loc="lower center",
               ncol=len(legend_els) // 2 + 1,
               fontsize=7.5, framealpha=0.95,
               bbox_to_anchor=(0.5, -0.03))

    _add_colorbar(fig, axes)

    fig.suptitle(
        "Grafo compartimentale SIRD+Farmaco  —  layout Kamada-Kawai\n"
        "[fill = I(t) | raggio ∝ D(t) | bordo = macro-type | "
        "alone = R(t) | archi = Wc2]",
        fontsize=11, y=1.01,
    )

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"    Salvato: {save_path}")
    plt.tight_layout()
    plt.show()
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# LIVELLO 3 — RETE MULTILAYER (invariato rispetto all'originale)
# ═══════════════════════════════════════════════════════════════════════════

def plot_drug_multilayer(
    sol_drug:      object,
    sol_nodrug:    object,
    net:           dict,
    inv2:          np.ndarray,
    coords:        np.ndarray,
    Wsp:           sp.spmatrix,
    interlayer_w:  dict,
    t_dose:        float,
    figsize:       tuple = (22, 12),
    n_inter_edges: int   = 80,
    save_path:     str   = None,
) -> None:
    """
    Livello 3 — Rete multilayer PT/DCT/TAL con stato SIRD sovrapposto.

    Layout
    ------
    - 2 righe: +farmaco (sopra) / −farmaco (sotto)
    - 3 colonne: t=t_dose, t=t_peak_I, t=t_end
    - Colore nodo  = I(t) su scala YlOrRd
    - Raggio nodo  ∝ D(t)
    - Bordo rosso  = spot boundary (seed)
    - Archi viola  = coupling inter-layer
    """
    layers = net["layers"]
    macros = [m for m in MULTILAYER_MACROS if m in layers]

    global_index = net["global_index"]
    nC           = sol_drug.y.shape[0] // 6
    t            = sol_drug.t

    x_range  = float(np.ptp(coords[:, 0]))
    x_pad    = x_range * 0.10
    x_offset = {m: i * (x_range + x_pad) for i, m in enumerate(macros)}

    snapshots = _build_snapshots(sol_drug, nC, t_dose)

    cmap_I = _CMAP_I
    norm_I = _NORM_I

    # Raggio base da distanza mediana tra spot vicini
    dists = []
    Wsp_coo = sp.csr_matrix(Wsp).tocoo()
    for i, j in zip(Wsp_coo.row[:500], Wsp_coo.col[:500]):
        if i < j:
            dists.append(np.linalg.norm(coords[i] - coords[j]))
    r_base = float(np.median(dists)) * 0.45 if dists else 15.0

    # Macro-type per spot
    macro_of_spot = np.full(len(inv2), "other", dtype=object)
    for macro in macros:
        node_idx = layers[macro]["nodes"]
        macro_of_spot[np.isin(inv2, node_idx)] = macro

    # Boundary spots
    boundary_spots = np.zeros(len(inv2), dtype=bool)
    for macro in macros:
        info  = layers[macro]
        nodes = info["nodes"]
        regs  = info.get("regions", [])
        for ci, rg in zip(nodes, regs):
            if "boundary" in str(rg):
                boundary_spots[inv2 == ci] = True

    # Archi inter-layer (campionamento)
    rng         = np.random.default_rng(0)
    inter_edges = {}
    for (ma, mb), w_il in interlayer_w.items():
        if ma not in layers or mb not in layers:
            continue
        spots_a = np.where(macro_of_spot == ma)[0]
        spots_b = np.where(macro_of_spot == mb)[0]
        if len(spots_a) == 0 or len(spots_b) == 0:
            continue
        n_draw = min(n_inter_edges, len(spots_a) * len(spots_b))
        sa_idx = rng.choice(len(spots_a),
                            size=min(n_draw, len(spots_a)), replace=False)
        sb_idx = rng.choice(len(spots_b),
                            size=min(n_draw, len(spots_b)), replace=False)
        inter_edges[(ma, mb)] = [
            (spots_a[a], spots_b[b], w_il)
            for a, b in zip(sa_idx, sb_idx)
        ]

    fig, axes = plt.subplots(
        2, 3, figsize=figsize,
        gridspec_kw={"hspace": 0.20, "wspace": 0.05},
    )

    for row, (sol, rlabel) in enumerate(
            zip([sol_drug, sol_nodrug], ["+farmaco", "−farmaco"])):
        for col, (t_idx, t_label) in enumerate(snapshots):
            ax   = axes[row][col]
            snap = _extract_SIRD_snapshot(sol, nC, t_idx)

            I_spot = np.zeros(len(inv2))
            D_spot = np.zeros(len(inv2))
            for ci in range(nC):
                mask = inv2 == ci
                I_spot[mask] = snap["I"][ci]
                D_spot[mask] = snap["D"][ci]
            D_max = D_spot.max() + 1e-6

            for macro in macros:
                info      = layers[macro]
                node_idx  = info["nodes"]
                spot_mask = np.isin(inv2, node_idx)
                if spot_mask.sum() == 0:
                    continue

                cx_ = coords[spot_mask, 0] + x_offset[macro]
                cy_ = coords[spot_mask, 1]
                I_m = I_spot[spot_mask]
                D_m = D_spot[spot_mask]
                bnd = boundary_spots[spot_mask]

                # Archi intra-layer
                Wsp_csr  = sp.csr_matrix(Wsp)
                spot_idx = np.where(spot_mask)[0]
                drawn    = 0
                for si in spot_idx:
                    if drawn >= 600:
                        break
                    for sj in Wsp_csr.indices[
                            Wsp_csr.indptr[si]:Wsp_csr.indptr[si+1]]:
                        if sj <= si or not spot_mask[sj]:
                            continue
                        ax.plot(
                            [coords[si, 0] + x_offset[macro],
                             coords[sj, 0] + x_offset[macro]],
                            [coords[si, 1], coords[sj, 1]],
                            color="#cccccc", lw=0.3, alpha=0.25, zorder=1)
                        drawn += 1

                # Nodi
                node_colors = cmap_I(norm_I(I_m))
                node_sizes  = r_base * (0.4 + 1.8 * D_m / D_max)

                ax.scatter(cx_[~bnd], cy_[~bnd],
                           s=(node_sizes[~bnd])**2 * 0.003,
                           c=node_colors[~bnd],
                           linewidths=0.3, edgecolors="#888888",
                           zorder=3, alpha=0.85)
                if bnd.any():
                    ax.scatter(cx_[bnd], cy_[bnd],
                               s=(node_sizes[bnd])**2 * 0.004,
                               c=node_colors[bnd],
                               linewidths=1.2, edgecolors="#cc0000",
                               zorder=4, alpha=0.90)

                if row == 0 and col == 0:
                    ax.text(cx_.mean(), cy_.max() + r_base * 2,
                            macro, ha="center", va="bottom",
                            fontsize=11, fontweight="bold",
                            color=LAYER_COLORS.get(macro, "k"))

            # Archi inter-layer
            for (ma, mb), edges in inter_edges.items():
                w_il  = interlayer_w.get((ma, mb),
                         interlayer_w.get((mb, ma), 0.1))
                lw    = 0.4 + 1.2 * w_il
                alpha = min(0.12 + 0.25 * w_il, 0.38)
                for sa, sb, _ in edges:
                    ax.plot(
                        [coords[sa, 0] + x_offset[macro_of_spot[sa]],
                         coords[sb, 0] + x_offset[macro_of_spot[sb]]],
                        [coords[sa, 1], coords[sb, 1]],
                        color="#8e44ad", lw=lw, alpha=alpha, zorder=2)

            ax.set_aspect("equal")
            ax.axis("off")
            if row == 0:
                ax.set_title(t_label, fontsize=9, pad=5)
            if col == 0:
                ax.text(-0.04, 0.5, rlabel, transform=ax.transAxes,
                        ha="right", va="center", fontsize=10,
                        fontweight="bold",
                        color="#c0392b" if row == 0 else "#555555",
                        rotation=90)

    _add_colorbar(fig, axes)

    legend_els = (
        [mpatches.Patch(color=LAYER_COLORS[m], label=f"Layer {m}")
         for m in macros] +
        [Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
                markeredgecolor="#cc0000", markersize=8,
                markeredgewidth=1.5, label="Boundary (seed)"),
         Line2D([0], [0], color="#8e44ad", lw=1.5,
                label="Archi inter-layer"),
         Line2D([0], [0], marker="o", color="w",
                markerfacecolor="#cccccc", markersize=5,
                label="Nodo grande = D alto")]
    )
    fig.legend(handles=legend_els, loc="lower center",
               ncol=len(legend_els), fontsize=8,
               framealpha=0.9, bbox_to_anchor=(0.5, -0.03))

    fig.suptitle(
        "Rete multilayer SIRD+Farmaco\n"
        "[colore = I(t) | dimensione ∝ D(t) | viola = archi inter-layer]",
        fontsize=12, y=1.01,
    )

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"    Salvato: {save_path}")
    plt.tight_layout()
    plt.show()
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# LIVELLO 4 — CONFRONTO SIRD A t_dose DIVERSI (figure separate)
# ═══════════════════════════════════════════════════════════════════════════

def plot_drug_tdose_comparison(
    comp:          dict,
    sir_params:    dict,
    t_end:         float       = 80.0,
    n_steps:       int         = 800,
    seed_I:        float       = 0.10,
    dose_amount:   float       = 1.0,
    dose_mode:     str         = "bolus",
    node_radius:   float       = 0.05,
    edge_thresh:   float       = 0.01,
    figsize:       tuple       = (22, 18),
    save_prefix:   str         = None,
) -> dict:
    """
    Livello 4 — Figura unica 3×3 per confronto t_dose diversi.

    Layout
    ------
    - 3 righe = 3 scenari di somministrazione:
        riga 0 = pre-picco  (t_dose = t_peak_I / 2)
        riga 1 = at peak   (t_dose = t_peak_I)
        riga 2 = post-picco (t_dose = t_peak_I * 1.5)
    - 3 colonne = 3 snapshot temporali per ogni scenario:
        col 0 = t_dose      (momento somministrazione)
        col 1 = t_peak_I    (picco degli infetti)
        col 2 = t_end       (stato final)
    - Layout Kamada-Kawai calcolato una sola volta e condiviso tra
      tutte le 9 celle → i nodi restano nella stessa posizione,
      cambia solo colore (I) e dimensione (D).
    - Etichette compartimento visibili solo nella cella (0,0).
    - Label di riga a sinistra con t_dose e scenario.
    - Label di colonna in alto con snapshot temporale.

    Parameters
    ----------
    comp        : output di build_compartment2
    sir_params  : output di build_default_params_SIR
    t_end       : durata simulazione
    n_steps     : passi integrazione ODE
    seed_I      : frazione iniziale infetti ai boundary
    dose_amount : drug amount per dose
    dose_mode   : "bolus" | "infusion" | "repeated"
    node_radius : raggio base dei nodi
    edge_thresh : soglia minima peso arco per visualizzazione
    figsize     : dimensioni figura (default 22×18 per 3×3)
    save_prefix : str o None — prefisso per salvare la figura

    Returns
    -------
    dict con chiavi "pre_peak", "at_peak", "post_peak",
    ciascuna contenente il risultato di run_SIRD_drug
    """
    from sir_drug import build_drug_params, run_SIRD_drug

    if not _HAS_NX:
        print("  [SKIP] plot_drug_tdose_comparison richiede networkx")
        return {}

    comps2        = comp["comps2"]
    Lc2           = comp["Lc2"]
    macro_of      = comp["macro_of"]
    region_of     = comp["region_of"]
    boundary_mask = comp["boundary_mask"]
    nC            = len(comps2)

    drug_params = build_drug_params(comps2, macro_of, region_of, sir_params)

    # ── Stima t_peak_I dalla simulazione di riferimento senza farmaco ─────
    print("  [L4] Simulazione riferimento per stima t_peak_I...")
    ref = run_SIRD_drug(
        comps2, Lc2, boundary_mask, macro_of, region_of,
        sir_params  = sir_params,
        drug_params = drug_params,
        seed_I      = seed_I,
        t_end       = t_end,
        n_steps     = n_steps,
        t_dose      = t_end * 2,   # farmaco mai somministrato
        dose_amount = dose_amount,
        dose_mode   = dose_mode,
        run_nodrug  = False,
    )
    t_peak_I = float(ref["sol_drug"].t[
        _find_t_peak_I(ref["sol_drug"], nC)
    ])
    print(f"  [L4] t_peak_I = {t_peak_I:.2f}")

    # ── Scenari ───────────────────────────────────────────────────────────
    t_pre  = max(1.0, t_peak_I / 2.0)
    t_at   = t_peak_I
    t_post = min(t_end * 0.75, t_peak_I * 1.5)

    scenarios = [
        ("pre_peak",  t_pre,
         f"pre-picco\nt_dose = {t_pre:.1f}"),
        ("at_peak",   t_at,
         f"at peak\nt_dose = {t_at:.1f}"),
        ("post_peak", t_post,
         f"post-picco\nt_dose = {t_post:.1f}"),
    ]

    # ── Layout KK — calcolato una sola volta per tutte le 9 celle ─────────
    G, pos = _build_network_layout(
        nC, macro_of, region_of, comp["Wc2"], edge_thresh
    )
    edge_weights = {(i, j): d["weight"]
                    for i, j, d in G.edges(data=True)}
    w_max = max(edge_weights.values(), default=1.0)

    # ── Simulazioni ───────────────────────────────────────────────────────
    results   = {}
    sols      = {}
    snap_sets = {}   # {scenario_key: [(t_idx, label), ...]}

    for scenario_key, t_dose_val, _ in scenarios:
        print(f"  [L4] Simulazione {scenario_key} (t_dose={t_dose_val:.1f})...")
        result = run_SIRD_drug(
            comps2, Lc2, boundary_mask, macro_of, region_of,
            sir_params  = sir_params,
            drug_params = drug_params,
            seed_I      = seed_I,
            t_end       = t_end,
            n_steps     = n_steps,
            t_dose      = t_dose_val,
            dose_amount = dose_amount,
            dose_mode   = dose_mode,
            run_nodrug  = False,
        )
        results[scenario_key]   = result
        sols[scenario_key]      = result["sol_drug"]
        snap_sets[scenario_key] = _build_snapshots(
            result["sol_drug"], nC, t_dose_val
        )

    # ── Intestazioni colonne (dal primo scenario — gli snapshot temporali
    #    sono simili tra scenari ma usiamo quelli del pre-picco come ref) ──
    col_labels = [
        snap_sets["pre_peak"][0][1],   # t_dose snapshot
        snap_sets["pre_peak"][1][1],   # t_peak snapshot
        snap_sets["pre_peak"][2][1],   # t_end snapshot
    ]
    # Semplifica le label di colonna: solo t e tag
    col_labels_short = ["somministrazione", "picco I", "final"]

    # ── Figura 3 × 3 ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        3, 3, figsize=figsize,
        gridspec_kw={"hspace": 0.10, "wspace": 0.08},
        facecolor="#f5f5f5",
    )

    for row, (scenario_key, t_dose_val, row_label) in enumerate(scenarios):
        sol       = sols[scenario_key]
        snapshots = snap_sets[scenario_key]

        for col, (t_idx, t_label) in enumerate(snapshots):
            ax   = axes[row][col]
            ax.set_facecolor("#f5f5f5")
            snap = _extract_SIRD_snapshot(sol, nC, t_idx)

            # ── Archi ─────────────────────────────────────────────────
            for (i, j), w in edge_weights.items():
                xi, yi = pos[i]
                xj, yj = pos[j]
                ax.plot([xi, xj], [yi, yj],
                        color="#888888",
                        lw=0.8 + 3.5 * (w / w_max),
                        alpha=0.22 + 0.45 * (w / w_max),
                        solid_capstyle="round",
                        zorder=1)

            # ── Nodi ──────────────────────────────────────────────────
            d_max = snap["D"].max() + 1e-6
            for i in range(nC):
                xc, yc = pos[i]
                r = node_radius * (0.55 + 1.8 * snap["D"][i] / d_max)
                _draw_circle_node(
                    ax, (xc, yc), r,
                    I_val=snap["I"][i],
                    D_val=snap["D"][i],
                    R_val=snap["R"][i],
                    macro_type=str(macro_of[i]),
                    is_boundary=bool(boundary_mask[i]),
                )
                # Etichette nodo: solo cella (0,0)
                if row == 0 and col == 0:
                    label = str(comps2[i])
                    ax.text(xc, yc - r - 0.025, label,
                            ha="center", va="top", fontsize=4.8,
                            color="#222222", zorder=6,
                            bbox=dict(boxstyle="round,pad=0.08",
                                      fc="white", ec="none", alpha=0.6))

            ax.set_xlim(-0.10, 1.10)
            ax.set_ylim(-0.10, 1.10)
            ax.set_aspect("equal")
            ax.axis("off")

            # ── Titolo colonna (solo prima riga) ──────────────────────
            if row == 0:
                t_val = sol.t[t_idx]
                ax.set_title(
                    f"{col_labels_short[col]}\nt = {t_val:.1f}",
                    fontsize=9, pad=6, fontweight="bold",
                )

            # ── Label riga (solo prima colonna) ───────────────────────
            if col == 0:
                ax.text(-0.08, 0.5, row_label,
                        transform=ax.transAxes,
                        ha="right", va="center",
                        fontsize=9, fontweight="bold",
                        rotation=90,
                        color=["#c0392b", "#e67e22", "#7f8c8d"][row])

            # ── Linea verticale tratteggiata: segna t_dose nella cella ─
            # (solo col 0 dove t = t_dose: aggiunge piccola freccia)
            if col == 0:
                ax.annotate(
                    "↓ dose",
                    xy=(0.5, 1.02), xycoords="axes fraction",
                    ha="center", va="bottom",
                    fontsize=7, color="#2980b9",
                    fontweight="bold",
                )

    # ── Legenda condivisa ─────────────────────────────────────────────────
    macro_border = {
        "PT":       "#c0392b",
        "DCT":      "#2471a3",
        "TAL":      "#1e8449",
        "vascular": "#8e44ad",
        "other":    "#7f8c8d",
    }
    legend_els = (
        [Line2D([0], [0], marker="o", color="w",
                markerfacecolor="#e8e8e8",
                markeredgecolor=c, markeredgewidth=2.2,
                markersize=9, label=m)
         for m, c in macro_border.items()] +
        [Line2D([0], [0], marker="o", color="w",
                markerfacecolor=_CMAP_I(_NORM_I(0.32)),
                markersize=10, label="I alto"),
         Line2D([0], [0], marker="o", color="w",
                markerfacecolor=_CMAP_I(_NORM_I(0.04)),
                markersize=10, label="I basso"),
         Line2D([0], [0], marker="o", color="w",
                markerfacecolor="#bbbbbb", markersize=14,
                label="grande = D alto"),
         Line2D([0], [0], marker="o", color="w",
                markerfacecolor="none",
                markeredgecolor="#c0392b",
                markeredgewidth=3.2, markersize=10,
                label="boundary (seed)"),
         Line2D([0], [0], marker="o", color="w",
                markerfacecolor="none",
                markeredgecolor="#27ae60",
                markeredgewidth=1.0, markersize=13,
                label="alone = R alto")]
    )
    fig.legend(
        handles=legend_els,
        loc="lower center",
        ncol=len(legend_els),
        fontsize=8,
        framealpha=0.95,
        bbox_to_anchor=(0.5, -0.02),
    )

    _add_colorbar(fig, axes)

    fig.suptitle(
        "Confronto SIRD+Farmaco — effetto del timing di somministrazione\n"
        "[layout Kamada-Kawai  |  fill = I(t)  |  raggio ∝ D(t)  |"
        "  bordo = macro-type  |  alone = R(t)]",
        fontsize=11, y=1.01,
    )

    if save_prefix:
        sp_ = f"{save_prefix}_L4_tdose_comparison.png"
        plt.savefig(sp_, dpi=150, bbox_inches="tight")
        print(f"    Salvato: {sp_}")

    plt.tight_layout()
    plt.show()
    plt.close()

    return results


# ═══════════════════════════════════════════════════════════════════════════
# LIVELLO 5 — CONFRONTO CURVE TEMPORALI I/D/R PER t_dose DIVERSI
# ═══════════════════════════════════════════════════════════════════════════

def plot_drug_tdose_curves(
    comp:          dict,
    sir_params:    dict,
    t_end:         float = 80.0,
    n_steps:       int   = 800,
    seed_I:        float = 0.10,
    dose_amount:   float = 1.0,
    dose_mode:     str   = "bolus",
    figsize:       tuple = (20, 16),
    save_prefix:   str   = None,
) -> dict:
    """
    Livello 5 — Figura unica 3×3 con curve temporali I(t), D(t), R(t)
    per tre scenari di somministrazione del farmaco a t_dose diversi.

    Layout
    ------
    - 3 righe = 3 scenari:
        riga 0 = pre-picco  (t_dose = t_peak_I / 2)
        riga 1 = at peak   (t_dose = t_peak_I)
        riga 2 = post-picco (t_dose = t_peak_I * 1.5)
    - 3 colonne = variabili di stato:
        col 0 = I(t) — infetti/infiammati
        col 1 = D(t) — danno irreversibile
        col 2 = R(t) — recovered
    - Per ogni cella: una curva per macro_type
        linea continua  = +farmaco
        linea tratteggiata = −farmaco
    - Linea verticale tratteggiata blu = t_dose dello scenario
    - Linea verticale grigia = t_peak_I di riferimento (no-drug)

    Parameters
    ----------
    comp        : output di build_compartment2
    sir_params  : output di build_default_params_SIR
    t_end       : durata simulazione
    n_steps     : passi integrazione ODE
    seed_I      : frazione iniziale infetti ai boundary
    dose_amount : drug amount per dose
    dose_mode   : "bolus" | "infusion" | "repeated"
    figsize     : dimensioni figura
    save_prefix : str o None — prefisso per salvare la figura

    Returns
    -------
    dict con chiavi "pre_peak", "at_peak", "post_peak",
    ciascuna con "sol_drug" e "sol_nodrug"
    """
    from sir_drug import build_drug_params, run_SIRD_drug

    comps2        = comp["comps2"]
    Lc2           = comp["Lc2"]
    macro_of      = comp["macro_of"]
    region_of     = comp["region_of"]
    boundary_mask = comp["boundary_mask"]
    nC            = len(comps2)

    drug_params = build_drug_params(comps2, macro_of, region_of, sir_params)

    # Colori per macro_type — coerenti con il resto del codice
    MACRO_COLORS = {
        "PT":       "#27ae60",
        "DCT":      "#2471a3",
        "TAL":      "#922b21",
        "vascular": "#00bcd4",
        "other":    "#888888",
        "immune":   "#e67e22",
    }

    # ── Simulazione di riferimento senza farmaco per t_peak_I ────────────
    print("  [L5] Simulazione riferimento per t_peak_I...")
    ref = run_SIRD_drug(
        comps2, Lc2, boundary_mask, macro_of, region_of,
        sir_params  = sir_params,
        drug_params = drug_params,
        seed_I      = seed_I,
        t_end       = t_end,
        n_steps     = n_steps,
        t_dose      = t_end * 2,
        dose_amount = dose_amount,
        dose_mode   = dose_mode,
        run_nodrug  = False,
    )
    t_peak_I = float(ref["sol_drug"].t[
        _find_t_peak_I(ref["sol_drug"], nC)
    ])
    print(f"  [L5] t_peak_I = {t_peak_I:.2f}")

    # ── Scenari ───────────────────────────────────────────────────────────
    t_pre  = max(1.0, t_peak_I / 2.0)
    t_at   = t_peak_I
    t_post = min(t_end * 0.75, t_peak_I * 1.5)

    scenarios = [
        ("pre_peak",  t_pre,
         f"pre-picco  (t_dose = {t_pre:.1f})",  "#2980b9"),
        ("at_peak",   t_at,
         f"at peak   (t_dose = {t_at:.1f})",   "#e67e22"),
        ("post_peak", t_post,
         f"post-picco (t_dose = {t_post:.1f})", "#7f8c8d"),
    ]

    # ── Simulazioni ───────────────────────────────────────────────────────
    results = {}
    for scenario_key, t_dose_val, scenario_label, _ in scenarios:
        print(f"  [L5] Simulazione {scenario_key} (t_dose={t_dose_val:.1f})...")
        result = run_SIRD_drug(
            comps2, Lc2, boundary_mask, macro_of, region_of,
            sir_params  = sir_params,
            drug_params = drug_params,
            seed_I      = seed_I,
            t_end       = t_end,
            n_steps     = n_steps,
            t_dose      = t_dose_val,
            dose_amount = dose_amount,
            dose_mode   = dose_mode,
            run_nodrug  = True,
        )
        results[scenario_key] = result

    # ── Macro-type unici presenti ─────────────────────────────────────────
    macros_present = list(dict.fromkeys(
        m for m in macro_of if m in MACRO_COLORS
    ))

    # Helper: media per macro_type di una variabile dal vettore sol.y
    def _macro_mean(sol, block_idx):
        """
        Estrae il blocco [block_idx*nC : (block_idx+1)*nC] e
        restituisce dict {macro: array(T,)} con la media per macro_type.
        """
        traj = sol.y[block_idx*nC:(block_idx+1)*nC, :]  # (nC, T)
        out  = {}
        for m in macros_present:
            mask = macro_of == m
            if mask.sum() > 0:
                out[m] = traj[mask, :].mean(axis=0)
        return out

    # ── Figura 3 × 3 ─────────────────────────────────────────────────────
    col_vars   = [
        (1, "I (infetti)",       (0.0, None)),
        (4, "D (danno/morti)",   (0.0, None)),
        (3, "R (recovered)",     (0.0, 1.05)),
    ]

    fig, axes = plt.subplots(
        3, 3, figsize=figsize,
        gridspec_kw={"hspace": 0.38, "wspace": 0.28},
    )

    for row, (scenario_key, t_dose_val, scenario_label, dose_color) \
            in enumerate(scenarios):

        sol_drug   = results[scenario_key]["sol_drug"]
        sol_nodrug = results[scenario_key]["sol_nodrug"]
        t_vec      = sol_drug.t

        for col, (block_idx, col_title, ylim) in enumerate(col_vars):
            ax = axes[row][col]

            drug_means   = _macro_mean(sol_drug,   block_idx)
            nodrug_means = _macro_mean(sol_nodrug, block_idx)

            for m in macros_present:
                color = MACRO_COLORS.get(m, "#555555")
                if m in drug_means:
                    ax.plot(t_vec, drug_means[m],
                            color=color, lw=1.8,
                            linestyle="-", alpha=0.90,
                            label=m if row == 0 and col == 0 else "_")
                if m in nodrug_means:
                    ax.plot(t_vec, nodrug_means[m],
                            color=color, lw=1.4,
                            linestyle="--", alpha=0.55,
                            label="_")

            # Linea verticale t_dose
            ax.axvline(t_dose_val, color=dose_color,
                       lw=1.5, ls=":", alpha=0.85,
                       label=f"t_dose={t_dose_val:.1f}" if col == 0 else "_")

            # Linea verticale t_peak_I riferimento
            ax.axvline(t_peak_I, color="#aaaaaa",
                       lw=1.0, ls="--", alpha=0.6,
                       label=f"t_peak={t_peak_I:.1f}" if col == 0 else "_")

            ax.set_xlim(0, t_end)
            if ylim[1] is not None:
                ax.set_ylim(ylim[0], ylim[1])
            else:
                ax.set_ylim(bottom=ylim[0])

            ax.set_xlabel("t", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.25, lw=0.5)
            ax.spines[["top", "right"]].set_visible(False)

            # Titolo colonna (solo prima riga)
            if row == 0:
                ax.set_title(col_title, fontsize=10, fontweight="bold", pad=6)

            # Label riga (solo prima colonna)
            if col == 0:
                ax.set_ylabel(scenario_label,
                              fontsize=8, labelpad=8,
                              color=dose_color, fontweight="bold")

            # Annotazione t_dose sulla linea verticale
            ymax = ax.get_ylim()[1]
            ax.annotate(
                f"dose\nt={t_dose_val:.0f}",
                xy=(t_dose_val, ymax * 0.92),
                fontsize=6.5, color=dose_color,
                ha="center", va="top",
                bbox=dict(boxstyle="round,pad=0.2",
                          fc="white", ec=dose_color,
                          alpha=0.75, lw=0.8),
            )

    # ── Legenda macro_type (prima cella) ─────────────────────────────────
    drug_line   = Line2D([0], [0], color="black", lw=2.0,
                         linestyle="-",  label="+farmaco")
    nodrug_line = Line2D([0], [0], color="black", lw=1.4,
                         linestyle="--", label="−farmaco",
                         alpha=0.55)
    macro_lines = [
        Line2D([0], [0], color=MACRO_COLORS.get(m, "#555"),
               lw=2.0, label=m)
        for m in macros_present
    ]
    peak_line = Line2D([0], [0], color="#aaaaaa", lw=1.0,
                       linestyle="--", label=f"t_peak ref ({t_peak_I:.1f})")

    fig.legend(
        handles=macro_lines + [drug_line, nodrug_line, peak_line],
        loc="lower center",
        ncol=len(macro_lines) + 3,
        fontsize=8,
        framealpha=0.95,
        bbox_to_anchor=(0.5, -0.03),
    )

    fig.suptitle(
        "Confronto SIRD+Farmaco — curve temporali per timing di somministrazione\n"
        "[continua = +farmaco  |  tratteggiata = −farmaco  |  "
        "media per macro_type  |  : = t_dose  |  -- = t_peak ref]",
        fontsize=11, y=1.01,
    )

    if save_prefix:
        sp_ = f"{save_prefix}_L5_tdose_curves.png"
        plt.savefig(sp_, dpi=150, bbox_inches="tight")
        print(f"    Salvato: {sp_}")

    plt.tight_layout()
    plt.show()
    plt.close()

    return results


# ═══════════════════════════════════════════════════════════════════════════
# RUNNER INTEGRATO
# ═══════════════════════════════════════════════════════════════════════════

def plot_all_drug_graphs(
    drug_result:   dict,
    ml_result:     dict,
    comp:          dict,
    coords_visium: np.ndarray,
    Wsp:           sp.spmatrix,
    sir_params:    dict        = None,
    t_dose:        float       = 20.0,
    t_end:         float       = 80.0,
    seed_I:        float       = 0.10,
    save_prefix:   str         = None,
) -> None:
    """
    Genera Livello 2, 3 e 4 in sequenza.

    Parameters
    ----------
    drug_result   : output di run_full_drug_analysis
    ml_result     : output di run_SIR_multilayer
    comp          : output di build_compartment2
    coords_visium : ndarray (N,2) — coordinate spaziali spot Visium
    Wsp           : sparse (N,N) — grafo kNN Visium
    sir_params    : dict parametri SIR (necessario per Livello 4)
    t_dose        : float — tempo somministrazione default
    t_end         : float — durata simulazione
    seed_I        : float — frazione iniziale infetti ai boundary
    save_prefix   : str o None — prefisso per salvare le figure
    """
    result     = drug_result["result"]
    sol_drug   = result["sol_drug"]
    sol_nodrug = result["sol_nodrug"]

    print("\n  [DRUG GRAPHS] Livello 2 — grafo compartimentale con nodi colorati...")
    plot_drug_state_graph(
        sol_drug      = sol_drug,
        sol_nodrug    = sol_nodrug,
        comps2        = comp["comps2"],
        macro_of      = comp["macro_of"],
        region_of     = comp["region_of"],
        boundary_mask = comp["boundary_mask"],
        Wc2           = comp["Wc2"],
        t_dose        = t_dose,
        save_path     = f"{save_prefix}_L2_state_graph.png" if save_prefix else None,
    )

    print("  [DRUG GRAPHS] Livello 3 — rete multilayer con dinamica farmaco...")
    plot_drug_multilayer(
        sol_drug      = sol_drug,
        sol_nodrug    = sol_nodrug,
        net           = ml_result["net"],
        inv2          = comp["inv2"],
        coords        = coords_visium,
        Wsp           = Wsp,
        interlayer_w  = ml_result["interlayer_w"],
        t_dose        = t_dose,
        save_path     = f"{save_prefix}_L3_multilayer.png" if save_prefix else None,
    )

    if sir_params is not None:
        print("  [DRUG GRAPHS] Livello 4 — confronto t_dose diversi (grafi KK)...")
        plot_drug_tdose_comparison(
            comp        = comp,
            sir_params  = sir_params,
            t_end       = t_end,
            seed_I      = seed_I,
            save_prefix = save_prefix,
        )
        print("  [DRUG GRAPHS] Livello 5 — confronto curve temporali t_dose diversi...")
        plot_drug_tdose_curves(
            comp        = comp,
            sir_params  = sir_params,
            t_end       = t_end,
            seed_I      = seed_I,
            save_prefix = save_prefix,
        )
    else:
        print("  [DRUG GRAPHS] Livelli 4 e 5 saltati — sir_params non fornito.")