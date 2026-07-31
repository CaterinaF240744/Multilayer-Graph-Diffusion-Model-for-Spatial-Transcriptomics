"""
sir_multilayer.py
=================
Modello SIR su rete multilayer per il rene murino.

Struttura della rete
---------------------
Tre layer biologici: PT, DCT, TAL
  - Ogni layer contiene i compartimenti del corrispondente macro_type
    (es. PT_cortex, PT_outer_medulla, PT_inner_medulla)
  - Archi INTRA-layer: sottomatrice di Wc2 per i nodi dello stesso layer,
    Laplaciano normalizzato separato per layer
  - Archi INTER-layer: coupling biologico PT↔DCT, DCT↔TAL, PT↔TAL,
    costruito dalla sottomatrice OFF-DIAGONALE di Wc2 pesata per D_I

Coupling inter-layer (versione corretta)
-----------------------------------------
The weight w_{ij} between node i (layer A) and node j (layer B) is extracted
DIRECTLY from Wc2[i,j] — i.e. the actual number of kNN edges
che connettono gli spot dei due compartimenti nel tessuto.
Questo viene poi moltiplicato per √(D_I_i · D_I_j) per incorporare
the specific diffusion capacity of each compartment.

Confronto con la versione precedente
--------------------------------------
Versione precedente:  w(A,B) = 1 / (1 + dist_centroide(A,B))
  — ignored the actual kNN edge connectivity
  — sensibile a rotazioni/traslazioni del tessuto
  — non incorporava D_I dei compartimenti

Versione corretta:    W_inter[i,j] = Wc2[i,j] · √(D_I_i · D_I_j)
  — deriva direttamente dalla struttura del grafo spot
  — nodo-a-nodo invece di layer-a-layer (campo medio)
  — invariante a geometria globale del tessuto
  — PT_boundary connesso a DCT_boundary, non alla media dell'intero DCT

Equazioni (per ogni nodo i nel layer ℓ) — exposure-driven (Eq. 1-2-4)
---------------------------------------------------------------------
    λ_i(t) = Σ_{j∈layer_ℓ} w̃_ij · D_j(t)            (Eq. 1, intra-layer)
           + Σ_{j∈altri_layer} W_inter_ij · D_j(t)    (Eq. 4, inter-layer)

    dS_i/dt = -β_i · S_i · λ_i(t)                     (Eq. 2)
              - D_S_i · Σ_{j∈layer_ℓ} L_ij^intra · S_j

    dI_i/dt = +β_i · S_i · λ_i(t) - γ_i · I_i         (Eq. 2-3)
              - D_I_i · Σ_{j∈layer_ℓ} L_ij^intra · I_j

    dR_i/dt =  γ_i · I_i                              (Eq. 3)

Il termine di infezione è exposure-driven (β·S·λ), NON bilineare (β·S·I).
Questo allinea il multilayer alle Eq. 1-2-4 dell'articolo e al modello
single-layer di sir_compartments.py. L'esposizione λ combina:
  - intra-layer: W̃_l @ D_l  (matrice di adiacenza normalizzata del layer)
  - inter-layer: W_inter @ I (coupling nodo-a-nodo tra layer)

The inter-layer term is now node-to-node: each node i of PT is coupled
specificamente ai nodi j di DCT/TAL con cui condivide archi kNN nel
grafo spot-level (filtrati dal coarse-graining in Wc2).

Topologia biologica inter-layer
---------------------------------
PT↔DCT: continuo anatomico del nefrone (tubulo prossimale → distale)
DCT↔TAL: ansa di Henle → tubulo distale (confine ISOM/cortex)
PT↔TAL: microambiente midollare condiviso (OSOM — entrambi presenti)

Funzioni esportate
-------------------
build_multilayer_network     : struttura rete multilayer
build_interlayer_coupling    : matrice W_inter da Wc2 off-diagonal + D_I
simulate_SIR_multilayer      : integrazione ODE con coupling nodo-a-nodo
multilayer_metrics           : effective R₀, most vulnerable layer, etc.
plot_multilayer_trajectories : traiettorie S/I/R per tutti i layer
plot_interlayer_coupling     : heatmap matrice di coupling
plot_multilayer_spatial      : mappe spaziali S/I/R su tessuto Visium
plot_multilayer_graph        : grafo rete completa (spot reali Visium)
run_SIR_multilayer           : runner completo con metriche e figure
"""

from __future__ import annotations

import warnings
import numpy as np
import scipy.sparse as sp

# ── numpy compat shim ────────────────────────────────────────────────────────
try:
    _trapz = np.trapezoid
except AttributeError:
    _trapz = np.trapezoid
from scipy.integrate import solve_ivp
from scipy.sparse.csgraph import laplacian as sp_laplacian
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# ═══════════════════════════════════════════════════════════════════════════
# COSTANTI
# ═══════════════════════════════════════════════════════════════════════════

MULTILAYER_MACROS = ("PT", "DCT", "TAL")

# Topologia biologica inter-layer
INTERLAYER_TOPOLOGY = [
    ("PT",  "DCT"),
    ("DCT", "TAL"),
    ("PT",  "TAL"),
]


# ═══════════════════════════════════════════════════════════════════════════
# COSTRUZIONE RETE MULTILAYER
# ═══════════════════════════════════════════════════════════════════════════

def build_multilayer_network(
    Wc2:       sp.spmatrix,
    comps2:    np.ndarray,
    macro_of:  np.ndarray,
    region_of: np.ndarray,
    macros:    tuple = MULTILAYER_MACROS,
) -> dict:
    """
    Costruisce la struttura della rete multilayer.

    Estrae nodi e archi intra-layer da Wc2 per ciascun macro_type.
    Costruisce l'indice globale per il sistema ODE.

    Parameters
    ----------
    Wc2       : sparse (C, C) — grafo compartimentale globale
    comps2    : ndarray (C,)
    macro_of  : ndarray (C,)
    region_of : ndarray (C,)
    macros    : tuple — layer da includere

    Returns
    -------
    dict con chiavi:
        layers          : {macro: {nodes, L_intra, n, regions, names}}
        global_index    : ndarray — posizione_globale → indice in comps2
        layer_of_global : ndarray — posizione_globale → layer (str)
        n_total         : int
        node_ranges     : {macro: (start, end)}
    """
    Wc2 = sp.csr_matrix(Wc2)
    layers = {}

    for macro in macros:
        node_idx = np.where(macro_of == macro)[0]
        if len(node_idx) == 0:
            warnings.warn(f"Layer '{macro}': nessun nodo trovato — skip.",
                          UserWarning)
            continue

        W_sub   = Wc2[np.ix_(node_idx, node_idx)]
        W_sub   = (W_sub + W_sub.T) * 0.5
        L_intra = sp_laplacian(W_sub, normed=True).tocsr()

        # Row-normalized exposure matrix W̃_l (Eq. 1 of the paper)
        # W̃_l[i,j] = W_raw[i,j] / Σ_k W_raw[i,k],  diagonal = 0
        # Used for the exposure-driven infection term: λ_l = W̃_l @ D_l
        W_raw_l  = np.diag(np.diag(L_intra.toarray())) - L_intra.toarray()
        np.fill_diagonal(W_raw_l, 0.0)
        row_sum_l = W_raw_l.sum(axis=1, keepdims=True)
        W_tilde_l = np.where(row_sum_l > 1e-12, W_raw_l / row_sum_l, 0.0)

        layers[macro] = dict(
            nodes    = node_idx,
            L_intra  = L_intra,
            W_tilde  = W_tilde_l,        # exposure matrix (Eq. 1)
            n        = len(node_idx),
            regions  = region_of[node_idx],
            names    = comps2[node_idx],
        )

    global_index    = []
    layer_of_global = []
    node_ranges     = {}
    pos = 0
    for macro in macros:
        if macro not in layers:
            continue
        n = layers[macro]["n"]
        node_ranges[macro] = (pos, pos + n)
        global_index.extend(layers[macro]["nodes"].tolist())
        layer_of_global.extend([macro] * n)
        pos += n

    return dict(
        layers          = layers,
        global_index    = np.array(global_index),
        layer_of_global = np.array(layer_of_global),
        n_total         = pos,
        node_ranges     = node_ranges,
    )


def build_interlayer_coupling(
    net:        dict,
    Wsp:        sp.spmatrix,
    inv2:       np.ndarray,
    sir_params: dict,
    topology:   list = INTERLAYER_TOPOLOGY,
    D_inter:    float = 1.0,
) -> sp.csr_matrix:
    """
    Costruisce la matrice di coupling inter-layer W_inter (n_total × n_total)
    aggregando gli archi kNN spot-level (Wsp) tra compartimenti di layer diversi.

    Why Wsp and not Wc2
    ---------------------
    Wc2 is the compartment-compartment graph obtained by coarse-graining.
    Per costruzione, due compartimenti di tipo cellulare diverso (es. PT_cortex
    e DCT_cortex) hanno archi in Wc2 solo se i loro spot sono fisicamente
    adiacenti nel kNN. In pratica PT e DCT tendono a formare zone separate nel
    tessuto, quindi Wc2[PT_*, DCT_*] ≈ 0 → coupling zero.

    Wsp invece contiene tutti gli archi spot-spot a prescindere dal tipo.
    Aggregando gli archi Wsp che attraversano il confine tra macro_type A e
    macro_type B otteniamo il flusso reale di contatto tra i due tipi:

        flux(A→B) = Σ_{i∈spots(A)} Σ_{j∈spots(B)} Wsp[i,j]

    Questo viene poi normalizzato per sqrt(N_A · N_B) e moltiplicato per
    sqrt(D_I_A · D_I_B) per ottenere il coupling effettivo tra compartimenti.

    Formula per ogni coppia compartimento (ca, cb) con ca∈layerA, cb∈layerB:

        W_inter[ca_pos, cb_pos] =
            D_inter
            · (Σ_{i∈spots(ca)} Σ_{j∈spots(cb)} Wsp[i,j]) / sqrt(N_ca · N_cb)
            · sqrt(D_I_ca · D_I_cb)

    Parameters
    ----------
    net        : output di build_multilayer_network
    Wsp        : sparse (N_spots, N_spots) — grafo kNN spot-level
    inv2       : ndarray (N_spots,) — mappa spot → indice compartimento
    sir_params : dict — contiene D_I (C,) per ogni compartimento in comps2
    topology   : lista di coppie (macro_a, macro_b)
    D_inter    : scalare moltiplicativo globale

    Returns
    -------
    W_inter : csr_matrix (n_total, n_total)
    """
    Wsp = sp.csr_matrix(Wsp)
    n_total      = net["n_total"]
    global_index = net["global_index"]
    layers       = net["layers"]
    node_ranges  = net["node_ranges"]

    D_I_global = np.asarray(sir_params["D_I"], dtype=float)

    # Pre-calcola: per ogni compartimento globale → lista spot
    comp_spots: dict[int, np.ndarray] = {}
    for gpos, comp_idx in enumerate(global_index):
        comp_spots[int(comp_idx)] = np.where(inv2 == comp_idx)[0]

    rows, cols, vals = [], [], []

    for macro_a, macro_b in topology:
        if macro_a not in layers or macro_b not in layers:
            continue

        nodes_a = layers[macro_a]["nodes"]   # indici in comps2
        nodes_b = layers[macro_b]["nodes"]

        sa, ea = node_ranges[macro_a]
        sb, eb = node_ranges[macro_b]

        # Mappa comp_idx → posizione nel vettore globale
        comp_to_gpos_a = {int(global_index[sa + k]): sa + k
                          for k in range(ea - sa)}
        comp_to_gpos_b = {int(global_index[sb + k]): sb + k
                          for k in range(eb - sb)}

        DI_a = D_I_global[nodes_a]
        DI_b = D_I_global[nodes_b]

        for ka, ci_a in enumerate(nodes_a):
            spots_a = comp_spots.get(int(ci_a))
            if spots_a is None or len(spots_a) == 0:
                continue
            gpa = comp_to_gpos_a.get(int(ci_a))
            if gpa is None:
                continue

            # Riga aggregata: somma pesi Wsp da tutti gli spot di ci_a
            # verso tutti gli spot di ciascun compartimento di layer B
            # shape: (N_spots,) — poi indicizziamo per compartimento B
            row_sum = np.asarray(Wsp[spots_a, :].sum(axis=0)).reshape(-1)

            for kb, ci_b in enumerate(nodes_b):
                spots_b = comp_spots.get(int(ci_b))
                if spots_b is None or len(spots_b) == 0:
                    continue
                gpb = comp_to_gpos_b.get(int(ci_b))
                if gpb is None:
                    continue

                # Flusso aggregato tra i due compartimenti
                flux = float(row_sum[spots_b].sum())
                if flux <= 0:
                    continue

                # Normalizza per sqrt(N_ca · N_cb)
                n_ca = len(spots_a)
                n_cb = len(spots_b)
                w_norm = flux / max(np.sqrt(n_ca * n_cb), 1.0)

                # Moltiplica per sqrt(D_I_ca · D_I_cb)
                w = D_inter * w_norm * np.sqrt(DI_a[ka] * DI_b[kb])

                # Arco bidirezionale
                rows.append(gpa); cols.append(gpb); vals.append(w)
                rows.append(gpb); cols.append(gpa); vals.append(w)

    if not vals:
        warnings.warn(
            "[build_interlayer_coupling] Nessun arco inter-layer trovato. "
            "Verificare che Wsp e inv2 siano corretti e che i macro_type "
            "abbiano spot adiacenti nel grafo kNN.",
            UserWarning, stacklevel=2,
        )
        return sp.csr_matrix((n_total, n_total))

    W_inter = sp.csr_matrix(
        (np.array(vals), (np.array(rows), np.array(cols))),
        shape=(n_total, n_total),
    )
    W_inter.setdiag(0)
    W_inter.eliminate_zeros()

    nnz = W_inter.nnz
    vmax = float(W_inter.data.max()) if nnz > 0 else 0.0
    print(f"    [build_interlayer_coupling] archi non-zero: {nnz}  "
          f"w_max={vmax:.5f}")
    return W_inter


def _interlayer_summary(W_inter: sp.csr_matrix, net: dict) -> dict:
    """
    Calcola un riepilogo scalare del coupling inter-layer per le figure
    e le metriche (mantiene la stessa interfaccia del vecchio interlayer_w).

    Returns
    -------
    dict {(macro_a, macro_b): float} — somma dei pesi off-diagonal tra i
    due layer, normalizzata per sqrt(n_a * n_b).
    """
    layers      = net["layers"]
    node_ranges = net["node_ranges"]
    W_arr       = W_inter.toarray()
    summary = {}
    macros  = list(layers.keys())
    for i, ma in enumerate(macros):
        for mb in macros[i+1:]:
            if ma not in node_ranges or mb not in node_ranges:
                continue
            sa, ea = node_ranges[ma]
            sb, eb = node_ranges[mb]
            block  = W_arr[sa:ea, sb:eb]
            na, nb = ea - sa, eb - sb
            denom  = max(np.sqrt(na * nb), 1.0)
            w_avg  = float(block.sum() / denom)
            summary[(ma, mb)] = w_avg
            summary[(mb, ma)] = w_avg
    return summary


# ═══════════════════════════════════════════════════════════════════════════
# INTEGRATORE SIR MULTILAYER  (coupling nodo-a-nodo)
# ═══════════════════════════════════════════════════════════════════════════

def simulate_SIR_multilayer(
    net:            dict,
    W_inter:        sp.csr_matrix,
    sir_params:     dict,
    boundary_mask:  np.ndarray,
    seed_I:         float = 0.10,
    t_end:          float = 50.0,
    n_steps:        int   = 500,
    method:         str   = "BDF",
    rtol:           float = 1e-6,
    atol:           float = 1e-9,
) -> tuple:
    """
    Integra il modello SIR multilayer exposure-driven (Eq. 1-2-4).

    Stato: [S_0..S_{n-1} | I_0..I_{n-1} | R_0..R_{n-1}]
    Dimensione: 3 · n_total

    Modello
    --------
    L'infezione è exposure-driven (Eq. 2): infect_i = β_i · S_i · λ_i,
    dove l'esposizione λ_i combina intra- e inter-layer:

        λ_i = (W̃_l @ D_l)_i  +  (W_inter @ I)_i     (Eq. 1 + Eq. 4)

    W̃_l è la matrice di adiacenza normalizzata intra-layer (Eq. 1);
    W_inter è la matrice di coupling inter-layer nodo-a-nodo (Eq. 4).
    Il termine diffusivo (Lapliciano intra-layer) resta sui soli stati
    S e I, come nel modello single-layer.

    Parameters
    ----------
    net           : output di build_multilayer_network
    W_inter       : sparse (n_total, n_total) — output di build_interlayer_coupling
    sir_params    : dict — parametri per tutti i compartimenti in comps2
    boundary_mask : ndarray bool (C,) in comps2 — nodi seed
    seed_I        : valore iniziale di I sui nodi boundary

    Returns
    -------
    S_fin, I_fin, R_fin : dict {macro: ndarray}
    sol                  : OdeResult
    y0_info              : dict diagnostico
    """
    layers       = net["layers"]
    global_index = net["global_index"]
    node_ranges  = net["node_ranges"]
    n_total      = net["n_total"]
    macros       = [m for m in MULTILAYER_MACROS if m in layers]

    beta_vec  = np.asarray(sir_params["beta"],  dtype=float)[global_index]
    gamma_vec = np.asarray(sir_params["gamma"], dtype=float)[global_index]

    # W_inter: matrice di coupling inter-layer (esposizione cross-layer, Eq. 4)
    # Nel modello exposure-driven, W_inter @ I dà l'esposizione di ciascun
    # nodo ai compartimenti malati degli altri layer. Non serve deg_inter
    # (il Laplaciano inter-layer è sostituito dal termine di esposizione).
    W_inter = sp.csr_matrix(W_inter)

    # Condizioni iniziali
    S0 = np.ones(n_total)
    I0 = np.zeros(n_total)
    R0 = np.zeros(n_total)
    for i_glob, comp_idx in enumerate(global_index):
        if boundary_mask[comp_idx]:
            I0[i_glob] = seed_I
            S0[i_glob] = 1.0 - seed_I

    y0_info = dict(n_seed=int((I0 > 0).sum()), seed_I=seed_I)
    y0      = np.concatenate([S0, I0, R0])
    t_eval  = np.linspace(0.0, float(t_end), int(n_steps))

    # Laplaciani intra-layer, matrici di esposizione W̃_l e vettori D_S, D_I
    L_list     = [layers[m]["L_intra"]  for m in macros]
    Wt_list    = [layers[m]["W_tilde"]  for m in macros]
    DS_list = [
        np.asarray(sir_params["D_S"], dtype=float)[layers[m]["nodes"]]
        for m in macros
    ]
    DI_list = [
        np.asarray(sir_params["D_I"], dtype=float)[layers[m]["nodes"]]
        for m in macros
    ]

    def rhs(t, y):
        S = np.clip(y[:n_total], 0.0, 1.0)
        I = np.clip(y[n_total:2*n_total], 0.0, 1.0)
        # Forza S+I ≤ 1 per evitare R < 0
        excess = np.maximum(S + I - 1.0, 0.0)
        denom  = S + I + 1e-15
        S = S - excess * S / denom
        I = I - excess * I / denom

        dS = np.zeros(n_total)
        dI = np.zeros(n_total)
        dR = np.zeros(n_total)

        # ── Esposizione inter-layer (Eq. 4): λ_inter = Σ_β κ_αβ · W̃_inter · D ──
        # W_inter è già la matrice di coupling normalizzata nodo-a-nodo;
        # il termine W_inter @ I dà l'esposizione cross-layer al nodo i.
        lam_inter = W_inter @ I                       # (n_total,)  Eq. 4

        # ── Dinamica intra-layer (exposure-driven, Eq. 1-2) ──────────────
        for macro, L, Wt, D_S, D_I in zip(macros, L_list, Wt_list, DS_list, DI_list):
            s, e = node_ranges[macro]
            S_l  = S[s:e]
            I_l  = I[s:e]
            b    = beta_vec[s:e]
            g    = gamma_vec[s:e]

            # Esposizione locale: λ_i = Σ_j w̃_ij · D_j  (Eq. 1, intra-layer)
            #   + esposizione cross-layer da lam_inter (Eq. 4)
            lam_l = Wt @ I_l + lam_inter[s:e]          # (n_l,)  Eq. 1 + Eq. 4

            # Infezione exposure-driven (Eq. 2): infect = β · H · λ
            inf_l = b * S_l * lam_l                    # (n_l,)  Eq. 2

            dS[s:e] += -inf_l - D_S * (L @ S_l)
            dI[s:e] +=  inf_l - g * I_l - D_I * (L @ I_l)
            dR[s:e] +=  g * I_l

        return np.concatenate([dS, dI, dR])

    sol = solve_ivp(
        rhs, (0.0, float(t_end)), y0,
        t_eval=t_eval, method=method, rtol=rtol, atol=atol,
    )

    if not sol.success:
        warnings.warn(
            f"simulate_SIR_multilayer: solver failed — {sol.message}",
            RuntimeWarning, stacklevel=2,
        )

    S_fin = {}; I_fin = {}; R_fin = {}
    for macro in macros:
        s, e = node_ranges[macro]
        S_r = np.clip(sol.y[s:e,                 -1], 0.0, 1.0)
        I_r = np.clip(sol.y[n_total+s:n_total+e, -1], 0.0, 1.0)
        R_r = np.clip(sol.y[2*n_total+s:2*n_total+e, -1], 0.0, 1.0)
        tot = np.where(S_r + I_r + R_r > 0, S_r + I_r + R_r, 1.0)
        S_fin[macro] = S_r / tot
        I_fin[macro] = I_r / tot
        R_fin[macro] = R_r / tot

    return S_fin, I_fin, R_fin, sol, y0_info


# ═══════════════════════════════════════════════════════════════════════════
# METRICHE MULTILAYER
# ═══════════════════════════════════════════════════════════════════════════

def multilayer_metrics(
    S_fin:        dict,
    I_fin:        dict,
    R_fin:        dict,
    sol,
    net:          dict,
    sir_params:   dict,
    interlayer_w: dict,   # scalar summary for output compatibility
    threshold_I:  float = 0.10,
) -> dict:
    """
    Metriche del modello SIR multilayer.

    Parameters
    ----------
    interlayer_w : dict {(ma,mb): float} — riepilogo scalare del coupling
                   (output di _interlayer_summary). Usato solo per
                   interlayer_flow, non per la dinamica.
    """
    layers      = net["layers"]
    node_ranges = net["node_ranges"]
    n_total     = net["n_total"]
    macros      = list(layers.keys())

    beta_g  = np.asarray(sir_params["beta"],  dtype=float)
    gamma_g = np.asarray(sir_params["gamma"], dtype=float)
    t       = sol.t

    by_layer = {}
    for macro in macros:
        s, e     = node_ranges[macro]
        node_idx = layers[macro]["nodes"]
        I_traj   = sol.y[n_total+s:n_total+e, :]
        I_mean_t = I_traj.mean(axis=0)

        peak_I     = float(I_mean_t.max())
        t_peak     = float(t[np.argmax(I_mean_t)])
        try:
            auc_I = float(_trapz(I_mean_t, t))
        except AttributeError:
            auc_I = float(np.trapezoid(I_mean_t, t))
        above    = I_mean_t > threshold_I
        inv_time = float(t[above][0]) if above.any() else None

        R0_layer  = beta_g[node_idx] / np.where(gamma_g[node_idx] > 0,
                                                  gamma_g[node_idx], 1e-12)
        I_peak_nd = I_traj.max(axis=1)
        frac_inv  = float((I_peak_nd > threshold_I).mean())

        by_layer[macro] = dict(
            mean_I        = float(I_fin[macro].mean()),
            mean_R        = float(R_fin[macro].mean()),
            mean_R0       = float(R0_layer.mean()),
            frac_invaded  = frac_inv,
            peak_I        = peak_I,
            t_peak        = t_peak,
            auc_I         = auc_I,
            invasion_time = inv_time,
            n_nodes       = int(e - s),
        )

    times_valid     = {m: v["invasion_time"] for m, v in by_layer.items()
                       if v["invasion_time"] is not None}
    most_vulnerable = (min(times_valid, key=times_valid.get)
                       if times_valid else None)

    all_I    = np.concatenate([I_fin[m] for m in macros])
    all_R    = np.concatenate([R_fin[m] for m in macros])
    global_R = float(all_R.mean())
    global_I = float(all_I.mean())

    total_nodes = sum(by_layer[m]["n_nodes"] for m in macros)
    eff_R0 = sum(
        by_layer[m]["mean_R0"] * by_layer[m]["n_nodes"] / total_nodes
        for m in macros
    )

    # ── Global peak_I, t_peak, invasion_time — media pesata per n_nodes ──
    # Consistent with global_R, global_I and effective_R0 already computed above.
    # Ogni layer contribuisce proporzionalmente al numero di compartimenti
    # che contiene, non in parti uguali tra i layer.
    global_peak_I = sum(
        by_layer[m]["peak_I"] * by_layer[m]["n_nodes"] / total_nodes
        for m in macros
    )
    global_t_peak = sum(
        by_layer[m]["t_peak"] * by_layer[m]["n_nodes"] / total_nodes
        for m in macros
    )
    inv_valid = {m: by_layer[m]["invasion_time"] for m in macros
                 if by_layer[m]["invasion_time"] is not None}
    if inv_valid:
        n_inv_total = sum(by_layer[m]["n_nodes"] for m in inv_valid)
        global_inv_time = sum(
            v * by_layer[m]["n_nodes"] / n_inv_total
            for m, v in inv_valid.items()
        )
    else:
        global_inv_time = None

    interlayer_flow = {}
    for (a, b), w in interlayer_w.items():
        if a < b:
            flow = w * abs(
                by_layer.get(a, {}).get("mean_I", 0) -
                by_layer.get(b, {}).get("mean_I", 0)
            )
            interlayer_flow[(a, b)] = float(flow)

    return dict(
        by_layer         = by_layer,
        most_vulnerable  = most_vulnerable,
        global_R         = global_R,
        global_I         = global_I,
        global_peak_I    = float(global_peak_I),
        global_t_peak    = float(global_t_peak),
        global_inv_time  = (float(global_inv_time)
                            if global_inv_time is not None else None),
        effective_R0     = float(eff_R0),
        interlayer_flow  = interlayer_flow,
    )


# ═══════════════════════════════════════════════════════════════════════════
# VISUALIZZAZIONI
# ═══════════════════════════════════════════════════════════════════════════

def plot_multilayer_trajectories(
    sol,
    net:          dict,
    sir_params:   dict,
    threshold_I:  float = 0.10,
    figsize:      tuple = (16, 5),
    title:        str   = "",
) -> None:
    """Traiettorie S/I/R per tutti i layer insieme."""
    layers      = net["layers"]
    node_ranges = net["node_ranges"]
    n_total     = net["n_total"]
    macros      = list(layers.keys())
    t           = sol.t

    layer_colors = {"PT": "#e74c3c", "DCT": "#2980b9", "TAL": "#27ae60"}

    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=False)
    state_labels = ["S (Suscettibili)", "I (Infetti)", "R (Recovered)"]

    for ax, slbl in zip(axes, state_labels):
        ax.set_xlabel("Time", fontsize=11)
        ax.set_ylabel(slbl, fontsize=11)
        ax.set_title(f"{slbl}(t) per layer", fontsize=11)
        ax.set_ylim(-0.02, 1.05)
        ax.axhline(0, color="k", lw=0.5, ls=":")

    for macro in macros:
        s, e  = node_ranges[macro]
        color = layer_colors.get(macro, "gray")

        S_traj = sol.y[s:e,                :]
        I_traj = sol.y[n_total+s:n_total+e,    :]
        R_traj = sol.y[2*n_total+s:2*n_total+e, :]

        S_mean = S_traj.mean(axis=0); S_std = S_traj.std(axis=0)
        I_mean = I_traj.mean(axis=0); I_std = I_traj.std(axis=0)
        R_mean = R_traj.mean(axis=0); R_std = R_traj.std(axis=0)

        label = f"{macro} ({e-s} comp.)"
        for ax, mean, std in zip(axes,
                                  [S_mean, I_mean, R_mean],
                                  [S_std,  I_std,  R_std]):
            ax.plot(t, mean, color=color, lw=2.2, label=label)
            ax.fill_between(t, mean - std, mean + std,
                            color=color, alpha=0.15)

        pk_idx = int(np.argmax(I_mean))
        axes[1].scatter([t[pk_idx]], [I_mean[pk_idx]],
                        color=color, zorder=5, s=60)
        axes[1].annotate(
            f"  {macro}\n  pk={I_mean[pk_idx]:.2f}\n  t={t[pk_idx]:.1f}",
            (t[pk_idx], I_mean[pk_idx]),
            fontsize=7.5, color=color,
            xytext=(8, 5), textcoords="offset points",
        )

    axes[1].axhline(threshold_I, color="k", ls="--", lw=1.2,
                    label=f"soglia={threshold_I}", alpha=0.6)
    for ax in axes:
        ax.legend(fontsize=9, loc="upper right")

    fig.suptitle(
        (title or "SIR Multilayer — traiettorie per layer") +
        "\n[media ± std intra-layer | coupling inter-layer nodo-a-nodo]",
        fontsize=12,
    )
    plt.tight_layout()
    plt.show()


def plot_interlayer_coupling(
    interlayer_w: dict,
    metrics:      dict,
    figsize:      tuple = (7, 5),
) -> None:
    """Heatmap della matrice di coupling inter-layer (summary scalare)."""
    macros = list({m for pair in interlayer_w for m in pair})
    macros = sorted(set(macros))
    C = len(macros)
    idx = {m: i for i, m in enumerate(macros)}

    mat = np.zeros((C, C))
    for (a, b), w in interlayer_w.items():
        mat[idx[a], idx[b]] = w

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(mat, cmap="YlOrRd", vmin=0)
    ax.set_xticks(range(C)); ax.set_yticks(range(C))
    ax.set_xticklabels(macros, fontsize=12)
    ax.set_yticklabels(macros, fontsize=12)
    ax.set_title("Coupling inter-layer\n(W_off-diag · √D_I, normalizzato)",
                 fontsize=12)

    thr = mat.mean()
    for i in range(C):
        for j in range(C):
            ax.text(j, i, f"{mat[i,j]:.3f}", ha="center", va="center",
                    fontsize=10,
                    color="white" if mat[i, j] > thr else "black")

    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.show()


def plot_multilayer_spatial(
    S_fin: dict,
    I_fin: dict,
    R_fin: dict,
    net:   dict,
    adata_st,
    inv2:  np.ndarray,
    comps2: np.ndarray,
) -> None:
    """Mappe spaziali S/I/R su tessuto Visium per ogni layer."""
    import scanpy as sc

    layers      = net["layers"]
    node_ranges = net["node_ranges"]
    global_index = net["global_index"]
    n_total      = net["n_total"]

    for macro in layers:
        s, e     = node_ranges[macro]
        node_idx = global_index[s:e]   # indici in comps2

        S_arr = np.zeros(len(comps2)); S_arr[node_idx] = S_fin[macro]
        I_arr = np.zeros(len(comps2)); I_arr[node_idx] = I_fin[macro]
        R_arr = np.zeros(len(comps2)); R_arr[node_idx] = R_fin[macro]

        adata_st.obs[f"ML_S_{macro}"] = S_arr[inv2].astype(np.float32)
        adata_st.obs[f"ML_I_{macro}"] = I_arr[inv2].astype(np.float32)
        adata_st.obs[f"ML_R_{macro}"] = R_arr[inv2].astype(np.float32)

    keys = []
    for macro in layers:
        keys += [f"ML_S_{macro}", f"ML_I_{macro}", f"ML_R_{macro}"]
    titles = [f"{k.replace('ML_','').replace('_',' ')}" for k in keys]

    sc.pl.spatial(adata_st, color=keys, cmap="viridis", title=titles)


def plot_multilayer_graph(
    net:          dict,
    interlayer_w: dict,
    I_fin:        dict,
    sir_params:   dict,
    inv2:         np.ndarray,
    coords:       np.ndarray,
    comps2:       np.ndarray,
    Wsp=None,
    D_inter:      float = 0.05,
    save_path:    str   = None,
) -> None:
    """
    Grafo della rete multilayer — layout a tre pannelli affiancati.

    I tre layer (PT, DCT, TAL) sono disposti orizzontalmente con un gap
    netto tra di loro (offset = 1.15 × larghezza del tessuto per layer).
    Gli archi inter-layer attraversano il gap come ponti.

    Node colour  = internal degree (intra-layer connectivity, plasma)
    Dimensione   = I final del compartimento
    Bordo rosso  = spot boundary/outer_medulla (seed)
    Archi grigi  = archi kNN intra-layer (alpha basso)
    Archi azzurri= diffusione inter-layer (W_Wc2 · √D_I)
    """
    from matplotlib.lines import Line2D

    layers       = net["layers"]
    node_ranges  = net["node_ranges"]
    global_index = net["global_index"]
    n_total      = net["n_total"]
    macros       = list(layers.keys())

    layer_colors = {"PT": "#e74c3c", "DCT": "#2980b9", "TAL": "#27ae60"}

    # ── Valori I per ogni spot ───────────────────────────────────────────
    C_full = len(comps2)
    I_comp = np.zeros(C_full)
    for macro in macros:
        s, e = node_ranges[macro]
        I_comp[global_index[s:e]] = I_fin[macro]
    I_spot = I_comp[inv2]
    I_max  = max(float(I_spot.max()), 1e-6)

    # ── Grado interno ────────────────────────────────────────────────────
    if Wsp is not None:
        Wsp_csr = sp.csr_matrix(Wsp)
        degree_internal = np.asarray(Wsp_csr.sum(axis=1)).reshape(-1)
    else:
        degree_internal = np.ones(len(coords))

    # ── Maschere spot per layer ──────────────────────────────────────────
    spot_mask_layer: dict[str, np.ndarray] = {}
    for macro in macros:
        s, e     = node_ranges[macro]
        node_idx = global_index[s:e]
        spot_mask_layer[macro] = np.isin(inv2, node_idx)

    # ── Offset X: ogni layer occupa la sua colonna, gap = 15% del range ──
    # Each layer is translated by (i * (x_range + gap)) along the X axis
    # in modo che i tre pannelli siano chiaramente separati
    x_range = float(np.ptp(coords[:, 0]))
    gap     = x_range * 0.15          # spazio bianco tra layer adiacenti
    step    = x_range + gap            # passo tra origini dei layer

    x_offsets = {macro: i * step for i, macro in enumerate(macros)}

    # ── Figura ───────────────────────────────────────────────────────────
    total_width  = step * len(macros) - gap
    y_range      = float(np.ptp(coords[:, 1]))
    aspect       = total_width / max(y_range, 1.0)
    fig_w        = max(20, 7 * len(macros))
    fig_h        = max(8,  fig_w / aspect * 0.55)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor="#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    # ── Archi intra-layer (kNN) ──────────────────────────────────────────
    if Wsp is not None:
        Wsp_coo = Wsp_csr.tocoo()
        for macro in macros:
            mask     = spot_mask_layer[macro]
            x_off    = x_offsets[macro]
            col      = layer_colors.get(macro, "gray")
            spot_set = set(np.where(mask)[0])
            for u, v in zip(Wsp_coo.row, Wsp_coo.col):
                if u >= v:
                    continue
                if u in spot_set and v in spot_set:
                    ax.plot(
                        [coords[u, 0] + x_off, coords[v, 0] + x_off],
                        [coords[u, 1],          coords[v, 1]],
                        color=col, lw=0.25, alpha=0.07, zorder=1,
                    )

    # ── Archi inter-layer ────────────────────────────────────────────────
    # Disegna linee campionate spot_A → spot_B (non centroide)
    # per mostrare la distribuzione spaziale reale del coupling
    max_w = max((v for v in interlayer_w.values()), default=1e-9)
    rng   = np.random.default_rng(42)

    for (ma, mb), w_scalar in interlayer_w.items():
        if ma >= mb or w_scalar <= 0:
            continue
        if ma not in macros or mb not in macros:
            continue
        mask_a = spot_mask_layer[ma]
        mask_b = spot_mask_layer[mb]
        if mask_a.sum() == 0 or mask_b.sum() == 0:
            continue

        idxs_a  = np.where(mask_a)[0]
        idxs_b  = np.where(mask_b)[0]
        x_off_a = x_offsets[ma]
        x_off_b = x_offsets[mb]

        # Campiona coppie (spot_A, spot_B) — max 60 linee per coppia di layer
        n_lines = min(60, len(idxs_a), len(idxs_b))
        samp_a  = rng.choice(idxs_a, size=n_lines, replace=len(idxs_a) < n_lines)
        samp_b  = rng.choice(idxs_b, size=n_lines, replace=len(idxs_b) < n_lines)

        alpha_line = min(0.30, max(0.06, float(w_scalar / max_w) * 0.35))
        lw_line    = max(0.3, float(w_scalar / max_w) * 1.2)

        for ia, ib in zip(samp_a, samp_b):
            x0 = coords[ia, 0] + x_off_a
            y0 = coords[ia, 1]
            x1 = coords[ib, 0] + x_off_b
            y1 = coords[ib, 1]
            ax.plot([x0, x1], [y0, y1],
                    color="#8899cc", lw=lw_line, alpha=alpha_line, zorder=2)

    # ── Nodi ────────────────────────────────────────────────────────────
    cmap_deg       = plt.cm.plasma
    node_size_min  = 6
    node_size_max  = 80
    d_max          = float(degree_internal.max()) + 1e-9
    norm_deg       = plt.Normalize(0, d_max)

    for macro in macros:
        mask   = spot_mask_layer[macro]
        idxs   = np.where(mask)[0]
        x_off  = x_offsets[macro]
        xs     = coords[idxs, 0] + x_off
        ys     = coords[idxs, 1]
        I_s    = I_spot[idxs]
        degs   = degree_internal[idxs]

        colors_node = [cmap_deg(norm_deg(d)) for d in degs]
        sizes = node_size_min + (I_s / I_max) * (node_size_max - node_size_min)

        boundary_comps = set(
            layers[macro]["nodes"][
                layers[macro]["regions"] == "outer_medulla"
            ].tolist()
            + layers[macro]["nodes"][
                layers[macro]["regions"] == "boundary"
            ].tolist()
        )
        edge_colors = [
            "#ff4444" if inv2[i] in boundary_comps else "#22223b"
            for i in idxs
        ]

        ax.scatter(xs, ys, s=sizes, c=colors_node,
                   edgecolors=edge_colors, linewidths=0.6,
                   zorder=3, alpha=0.92)

        # Etichetta layer centrata sopra il cloud di spot
        xc = float(xs.mean())
        yc = float(ys.max()) + y_range * 0.05
        ax.text(xc, yc, macro,
                ha="center", va="bottom",
                fontsize=20, fontweight="bold",
                color=layer_colors.get(macro, "white"), zorder=10)

    # ── Colorbar grado ───────────────────────────────────────────────────
    sm_deg = plt.cm.ScalarMappable(cmap=cmap_deg, norm=norm_deg)
    sm_deg.set_array([])
    cbar = fig.colorbar(sm_deg, ax=ax, shrink=0.30, pad=0.01,
                        location="right", anchor=(0, 0.7))
    cbar.set_label("Grado interno (connettivita')", fontsize=9, color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    # ── Legenda ──────────────────────────────────────────────────────────
    legend_els = [
        Line2D([0],[0], color="#e74c3c", lw=1.5, label="PT archi kNN"),
        Line2D([0],[0], color="#2980b9", lw=1.5, label="DCT archi kNN"),
        Line2D([0],[0], color="#27ae60", lw=1.5, label="TAL archi kNN"),
        Line2D([0],[0], color="#8899cc", lw=1.5,
               label="Diffusione inter-layer"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor="gray",
               markersize=6, markeredgecolor="#ff4444", markeredgewidth=1.5,
               label="Spot boundary / outer_medulla (seed)"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor="gray",
               markersize=3, label="I basso (dim. piccola)"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor="gray",
               markersize=8, label="I alto (dim. grande)"),
    ]
    ax.legend(handles=legend_els, fontsize=9, loc="lower center",
              facecolor="#1a1a2e", labelcolor="white",
              edgecolor="#334", framealpha=0.9,
              ncol=4, bbox_to_anchor=(0.5, -0.07))

    ax.axis("off")
    ax.set_title(
        "Rete Multilayer SIR — topologia spot reali Visium\n"
        "Colore = connettivita'  |  Dimensione = I final  |  "
        "Archi azzurri = diffusione transcompartimentale",
        fontsize=13, color="white", pad=14,
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    plt.show()
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# RUNNER COMPLETO
# ═══════════════════════════════════════════════════════════════════════════

def run_SIR_multilayer(
    comp:        dict,
    sir_params:  dict,
    adata_st,
    seed_I:      float = 0.10,
    D_inter:     float = 0.05,
    t_end:       float = 50.0,
    n_steps:     int   = 500,
    macros:      tuple = MULTILAYER_MACROS,
    threshold_I: float = 0.10,
    plot:        bool= True,
) -> dict:
    """
    Runner completo per il modello SIR multilayer.

    Sequenza
    ---------
    1. build_multilayer_network  — struttura rete
    2. build_interlayer_coupling — matrice W_inter da Wc2 off-diagonal + D_I
    3. simulate_SIR_multilayer   — integrazione ODE nodo-a-nodo
    4. multilayer_metrics        — metriche aggregate
    5. Figure (traiettorie, coupling, mappe Visium, grafo)

    Parameters
    ----------
    D_inter : scalare moltiplicativo applicato a W_inter.
              Valori suggeriti:
              0.01 = barriera tubulare forte
              0.05 = coupling moderato (default)
              0.15 = infiammazione diffusa

    Returns
    -------
    dict: net, W_inter, interlayer_w, S_fin, I_fin, R_fin, sol, metrics
    """
    comps2        = comp["comps2"]
    Wc2           = comp["Wc2"]
    macro_of      = comp["macro_of"]
    region_of     = comp["region_of"]
    boundary_mask = comp["boundary_mask"]
    inv2          = comp["inv2"]
    Wsp           = comp["Wsp"]
    coords        = adata_st.obsm["spatial"].astype(np.float32)

    print("\n" + "="*60)
    print("SIR MULTILAYER — rete PT / DCT / TAL")
    print("="*60)

    # 1. Struttura rete
    print("\n  [1] Costruzione rete multilayer...")
    net = build_multilayer_network(Wc2, comps2, macro_of, region_of,
                                   macros=macros)
    for macro, info in net["layers"].items():
        s, e = net["node_ranges"][macro]
        print(f"    Layer {macro}: {info['n']} nodi  "
              f"(idx globale {s}–{e-1})  "
              f"zone: {list(np.unique(info['regions']))}")

    # 2. Coupling inter-layer (da Wsp spot-level + D_I)
    print(f"\n  [2] Costruzione coupling inter-layer "
          f"(Wsp spot-level · √D_I · D_inter={D_inter})...")
    W_inter = build_interlayer_coupling(
        net, Wsp, inv2, sir_params,
        topology=INTERLAYER_TOPOLOGY,
        D_inter=D_inter,
    )
    interlayer_w = _interlayer_summary(W_inter, net)
    print(f"    Archi inter-layer non zero: {W_inter.nnz}")
    for (a, b), w in interlayer_w.items():
        if a < b:
            print(f"    w_avg({a}↔{b}) = {w:.5f}")

    # 3. Simulazione
    print(f"\n  [3] Simulazione SIR multilayer "
          f"(seed_I={seed_I}  t_end={t_end})...")
    S_fin, I_fin, R_fin, sol, y0_info = simulate_SIR_multilayer(
        net, W_inter, sir_params, boundary_mask,
        seed_I=seed_I, t_end=t_end, n_steps=n_steps,
    )
    print(f"    Nodi seed (boundary): {y0_info['n_seed']}")
    print(f"    Solver: {'OK' if sol.success else 'FAILED — ' + sol.message}")

    # 4. Metriche
    print("\n  [4] Metriche multilayer...")
    metrics = multilayer_metrics(
        S_fin, I_fin, R_fin, sol, net, sir_params, interlayer_w, threshold_I
    )
    print(f"\n  Attack rate globale  : {metrics['global_R']:.3f}")
    print(f"  Infetti residui (I)  : {metrics['global_I']:.3f}")
    print(f"  R₀ effettivo         : {metrics['effective_R0']:.2f}")
    print(f"  Most vulnerable layer: {metrics['most_vulnerable']}")
    print(f"\n  {'Layer':8s} {'mean_I':8s} {'mean_R':8s} {'R0':6s} "
          f"{'peak_I':8s} {'t_peak':8s} {'inv_time':10s} {'invaded':8s}")
    print("  " + "-"*65)
    for macro, v in metrics["by_layer"].items():
        it = f"{v['invasion_time']:.1f}" if v["invasion_time"] is not None else "None"
        print(f"  {macro:8s} {v['mean_I']:8.3f} {v['mean_R']:8.3f} "
              f"{v['mean_R0']:6.2f} {v['peak_I']:8.3f} "
              f"{v['t_peak']:8.1f} {it:10s} {v['frac_invaded']:8.0%}")
    # Global row — media pesata per n_nodes (coerente con Tabella 4 del paper)
    git = (f"{metrics['global_inv_time']:.1f}"
           if metrics["global_inv_time"] is not None else "None")
    print("  " + "-"*65)
    print(f"  {'Global':8s} {metrics['global_I']:8.3f} {metrics['global_R']:8.3f} "
          f"{metrics['effective_R0']:6.2f} {metrics['global_peak_I']:8.3f} "
          f"{metrics['global_t_peak']:8.1f} {git:10s}")
    print("\n  Flusso inter-layer stimato:")
    for (a, b), f in metrics["interlayer_flow"].items():
        print(f"    {a}↔{b}: {f:.5f}")

    # 5. Figure
    print("\n  [5] Figure...")
    plot_multilayer_trajectories(
        sol, net, sir_params, threshold_I=threshold_I,
        title=f"SIR Multilayer  D_inter={D_inter}  seed_I={seed_I}",
    )
    plot_interlayer_coupling(interlayer_w, metrics)
    plot_multilayer_spatial(S_fin, I_fin, R_fin, net, adata_st,
                            inv2, comps2)
    plot_multilayer_graph(
        net, interlayer_w, I_fin, sir_params,
        inv2, coords, comps2,
        Wsp=Wsp, D_inter=D_inter,
    )

    return dict(
        net          = net,
        W_inter      = W_inter,
        interlayer_w = interlayer_w,
        S_fin        = S_fin,
        I_fin        = I_fin,
        R_fin        = R_fin,
        sol          = sol,
        metrics      = metrics,
    )