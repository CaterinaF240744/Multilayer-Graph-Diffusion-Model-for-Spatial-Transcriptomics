"""
sir_section7.py
================
Mechanistic interpretability of spatial diffusion (manuscript Section 7).

Refactored from DiffComp_Inter.ipynb into a clean, importable Python module.

Three self-contained analyses:
  7.1  Edge importance maps      — intra vs inter-layer flux ranking
  7.2  Edge influence & perturbation — counterfactual ablation vs baseline
  7.3  Scenario-dependent mechanisms — seeding scenario comparison
       (spatial boundary / central / random boundary x 5 reps)

**IMPORTANT — initialisation requirement.**
This module operates on the *real* Visium-derived compartment graph, not a
hardcoded schematic.  You MUST call ``init_from_compartments(comp)`` with the
output of ``compartments.build_compartment2()`` before invoking any analysis
function.  A ``_check_initialised()`` guard raises ``RuntimeError`` if this
step is skipped, preventing silent runs on empty/stale state.

Functions exported
-------------------
init_from_compartments    : load real compartment graph (REQUIRED before use)
run_edge_importance       : Section 7.1 — directed flux ranking + Fig 7
run_edge_ablation         : Section 7.2 — class + single-edge ablation + Fig 8
run_seeding_scenarios     : Section 7.3 — 3 scenarios + Figs 9-11
run_section7_analysis     : runner that produces all figures + tables
"""

from __future__ import annotations

import os
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.stats import spearmanr

import matplotlib
matplotlib.rcParams['font.family'] = ['Liberation Sans', 'Arimo', 'DejaVu Sans']
matplotlib.rcParams['svg.fonttype'] = 'none'
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════
# SHARED MODEL DEFINITIONS  (populated by init_from_compartments)
# ═══════════════════════════════════════════════════════════════════════════

# --- Globals: set to empty/None until init_from_compartments() is called ---
COMPARTMENTS: list = []
N: int = 0
C_IDX: dict = {}
MACRO_MAP: dict = {}

W_raw: np.ndarray = np.zeros((0, 0))
W_norm: np.ndarray = np.zeros((0, 0))
BASE_EDGES: list = []

ZONE_BOUNDARY: list = []
ZONE_OUTER_MED: list = []
ZONE_INNER_MED: list = []
ZONE_CORTEX: list = []
ZONE_MID: list = []
ZONE_CORE: list = []

SPATIAL_SEED: list = []
CENTRAL_SEED: list = []

_INITIALISED = False


def _check_initialised():
    """Raise RuntimeError if init_from_compartments() has not been called."""
    if not _INITIALISED or N == 0 or len(COMPARTMENTS) == 0:
        raise RuntimeError(
            "sir_section7 is not initialised. Call "
            "init_from_compartments(comp) with the output of "
            "compartments.build_compartment2() before using any "
            "Section 7 function."
        )


def init_from_compartments(comp):
    """Populate all module-level globals from the real Visium compartment graph.

    Parameters
    ----------
    comp : dict
        Output of ``compartments.build_compartment2(adata_st2)``.
        Must contain keys: ``Wc2`` (C×C adjacency matrix),
        ``comps2`` (list of compartment names), ``macro_of`` (macro-type
        per compartment), ``region_of`` (anatomical zone per compartment),
        ``boundary_mask`` (boolean mask for boundary/outer-medulla compartments).

    This function MUST be called before any Section 7 analysis function.
    It replaces the previous hardcoded schematic graph with the real
    data-driven compartment graph used throughout Section 6.
    """
    global COMPARTMENTS, N, C_IDX, MACRO_MAP
    global W_raw, W_norm, BASE_EDGES
    global ZONE_BOUNDARY, ZONE_OUTER_MED, ZONE_INNER_MED
    global ZONE_CORTEX, ZONE_MID, ZONE_CORE
    global SPATIAL_SEED, CENTRAL_SEED
    global _INITIALISED

    comps2 = list(comp["comps2"])
    Wc2 = np.asarray(comp["Wc2"])
    macro_of = comp.get("macro_of", np.array([c.split("_")[0] for c in comps2]))
    region_of = comp.get("region_of", np.array([c.split("_", 1)[1] if "_" in c else "other" for c in comps2]))
    boundary_mask = comp.get("boundary_mask", np.zeros(len(comps2), dtype=bool))

    COMPARTMENTS = comps2
    N = len(COMPARTMENTS)
    C_IDX = {c: i for i, c in enumerate(COMPARTMENTS)}
    MACRO_MAP = {c: str(macro_of[i]) for i, c in enumerate(COMPARTMENTS)}

    # Build adjacency from the real compartment graph
    W_raw = Wc2.copy()
    rs = W_raw.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    W_norm = W_raw / rs

    # Build edge list from non-zero upper-triangle entries
    BASE_EDGES = []
    for i in range(N):
        for j in range(i + 1, N):
            if W_raw[i, j] > 0:
                BASE_EDGES.append((COMPARTMENTS[i], COMPARTMENTS[j], float(W_raw[i, j])))

    # Zone classification from region_of
    ZONE_BOUNDARY = [c for c in COMPARTMENTS if "boundary" in c]
    ZONE_OUTER_MED = [c for c in COMPARTMENTS if "outer_medulla" in c]
    ZONE_INNER_MED = [c for c in COMPARTMENTS if "inner_medulla" in c]
    ZONE_CORTEX = [c for c in COMPARTMENTS if "cortex" in c and "boundary" not in c]
    ZONE_MID = [c for c in COMPARTMENTS if "_mid" in c]
    ZONE_CORE = [c for c in COMPARTMENTS if "_core" in c]

    # Seed regions: boundary + outer_medulla (matches the IRI origin zone)
    SPATIAL_SEED = ZONE_BOUNDARY + ZONE_OUTER_MED
    CENTRAL_SEED = ZONE_MID + ZONE_CORE

    _INITIALISED = True

    # Diagnostics
    n_intra_vascular = sum(1 for a, b, _ in BASE_EDGES
                           if MACRO_MAP.get(a) == "vascular" and MACRO_MAP.get(b) == "vascular")
    print(f"[sir_section7] init_from_compartments: {N} compartments loaded, "
          f"{len(BASE_EDGES)} edges loaded, "
          f"intra_vascular edges present: {n_intra_vascular}")
    print(f"[sir_section7] SPATIAL_SEED ({len(SPATIAL_SEED)}): {SPATIAL_SEED}")
    print(f"[sir_section7] CENTRAL_SEED ({len(CENTRAL_SEED)}): {CENTRAL_SEED}")


# Parameters (Table 1, paper) — aligned to Table 1: vascular β=0.30, ρ=0.10
BETA_BASE = {"PT": 0.40, "DCT": 0.28, "TAL": 0.25,
             "vascular": 0.30, "immune": 0.20, "other": 0.22}
RHO_BASE = {"PT": 0.08, "DCT": 0.10, "TAL": 0.12,
            "vascular": 0.10, "immune": 0.25, "other": 0.10}
DD_BASE = {"PT": 0.20, "DCT": 0.20, "TAL": 0.20,
           "vascular": 0.35, "immune": 0.20, "other": 0.20}
OUTER_MULT = 1.25       # β multiplier for outer-medulla / boundary compartments
GAMMA_OUTER_MULT = 1.10  # ρ multiplier for outer-medulla / boundary compartments
                         # (aligned with sir_compartments.py _REGION_SIR)

INTRA_COLORS = {"PT": "#2166ac", "DCT": "#4dac26", "TAL": "#d01c8b",
                "vascular": "#f1a340", "other": "#878787"}
INTER_COLOR = "#d73027"
SEED_COLOR = "#ff1493"
ACTIVE_COLOR = "#ffcc00"


# ═══════════════════════════════════════════════════════════════════════════
# CORE SIMULATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

# D_H (D_S): spatial diffusion of the healthy state — uniform 0.05,
# matching SIR_DEFAULTS["D_S"] in sir_compartments.py (Eq. 18-20).
# This is deliberately NOT per-macro-type: the healthy state diffuses
# slowly and uniformly (structural inertia), unlike the damage signal
# which propagates faster and is cell-type-specific (DD_BASE).
DS_DEFAULT = 0.05


def build_params():
    """Build parameter vectors aligned with Table 1 and sir_compartments.py.

    Returns four vectors: beta, rho, DS, DD.

    Zone multipliers (outer_medulla/boundary):
      β  *= OUTER_MULT       (1.25)
      ρ  *= GAMMA_OUTER_MULT  (1.10)
    This matches sir_compartments.py _REGION_SIR and the paper's Table 2 R_diff.

    DS (D_H) is uniform 0.05 for all compartments — the healthy state
    diffuses slowly and uniformly (Eq. 18-20, SIR_DEFAULTS["D_S"]).
    DD (D_D) is per-macro-type from DD_BASE — the damage signal
    propagates faster and is cell-type-specific.
    """
    _check_initialised()
    beta = np.zeros(N); rho = np.zeros(N); DD = np.zeros(N)
    DS = np.full(N, DS_DEFAULT)  # uniform, NOT per-macro-type
    for i, c in enumerate(COMPARTMENTS):
        m = MACRO_MAP[c]
        b, r, d = BETA_BASE[m], RHO_BASE[m], DD_BASE[m]
        if "outer_medulla" in c or "boundary" in c:
            b *= OUTER_MULT
            r *= GAMMA_OUTER_MULT
        beta[i] = b; rho[i] = r; DD[i] = d
    return beta, rho, DS, DD


def build_adjacency_from_edges(edge_list):
    _check_initialised()
    W = np.zeros((N, N))
    for a, b, w in edge_list:
        if a in C_IDX and b in C_IDX:
            i, j = C_IDX[a], C_IDX[b]
            W[i, j] = w; W[j, i] = w
    rs = W.sum(axis=1, keepdims=True); rs[rs == 0] = 1.0
    return W, W / rs


# NOTE: W_raw and W_norm are populated by init_from_compartments().
# Do NOT call build_adjacency_from_edges() at module level — it requires
# init_from_compartments() to have been called first.


def make_ode(W_norm, beta, rho, DS, DD):
    """Build the ODE RHS for the exposure-driven H/D/R model (Eq. 18-20).

    DS (D_H) is used ONLY in the dpH equation (healthy-state spatial
    diffusion, Eq. 18).  DD (D_D) is used ONLY in the dpD equation
    (damage-signal spatial diffusion, Eq. 19).  This asymmetry — D_H <<
    D_D — encodes the biological assumption that damage propagates
    actively and faster than the healthy state.
    """
    L = np.diag(W_norm.sum(axis=1)) - W_norm

    def f(t, y):
        pH = np.clip(y[:N], 0, 1)
        pD = np.clip(y[N:2 * N], 0, 1)
        lam = W_norm @ pD
        dpH = -beta * lam * pH - DS * (L @ pH)    # Eq. 18: D_H for healthy
        dpD = beta * lam * pH - rho * pD - DD * (L @ pD)  # Eq. 19: D_D for damage
        dpR = -dpH - dpD
        return np.concatenate([dpH, dpD, dpR])
    return f


def run_sim(W_norm, beta, rho, DS, DD, y0, t_end=60.0, n_pts=601):
    t_eval = np.linspace(0, t_end, n_pts)
    sol = solve_ivp(make_ode(W_norm, beta, rho, DS, DD), [0, t_end], y0,
                    t_eval=t_eval, method="RK45", rtol=1e-7, atol=1e-9)
    pH_t = np.clip(sol.y[:N, :], 0, 1)
    pD_t = np.clip(sol.y[N:2 * N, :], 0, 1)
    pR_t = np.clip(1.0 - pH_t - pD_t, 0, 1)
    return sol.t, pH_t, pD_t, pR_t


def compute_metrics(t, pD_t, pR_t, threshold=0.10):
    metrics = {}
    for i, c in enumerate(COMPARTMENTS):
        pD = pD_t[i]; pR = pR_t[i]
        peak_D = pD.max()
        t_peak = t[np.argmax(pD)]
        auc_D = np.trapezoid(pD, t)
        cross = np.where(pD > threshold)[0]
        t_cross = t[cross[0]] if len(cross) > 0 else np.nan
        pR_fin = pR[-1]
        metrics[c] = dict(peakD=peak_D, tpeak=t_peak,
                          AUCD=auc_D, tcross=t_cross, pR_fin=pR_fin)
    return pd.DataFrame(metrics).T


def global_metrics(t, pD_t, threshold=0.10):
    tinv = []
    for i in range(N):
        c = np.where(pD_t[i] > threshold)[0]
        tinv.append(t[c[0]] if len(c) > 0 else np.nan)
    return np.nanmean(tinv), pD_t.mean(axis=0).max(), np.mean(~np.isnan(tinv))


def make_ic(seed_comps, pD0=0.05):
    pH = np.ones(N); pD = np.zeros(N); pR = np.zeros(N)
    for c in seed_comps:
        if c in C_IDX:
            i = C_IDX[c]; pD[i] = pD0; pH[i] = 1.0 - pD0
    return np.concatenate([pH, pD, pR])


def short(s):
    return (s.replace("_boundary", "_bnd")
             .replace("_outer_medulla", "_OM")
             .replace("_inner_medulla", "_IM")
             .replace("_cortex", "_ctx")
             .replace("_core", "_core"))


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 7.1 — EDGE IMPORTANCE MAPS
# ═══════════════════════════════════════════════════════════════════════════

def compute_edge_flux(t, pH_t, pD_t, W_raw, beta):
    """Directed cumulative flux: Φ_{i→j} = β_i · w_{ij} · ∫ pD_i(t) · pH_j(t) dt"""
    records = []
    for i, src in enumerate(COMPARTMENTS):
        for j, tgt in enumerate(COMPARTMENTS):
            w = W_raw[i, j]
            if w == 0:
                continue
            flux_t = beta[i] * w * pD_t[i, :] * pH_t[j, :]
            flux = np.trapezoid(flux_t, t)
            etype = "intra" if MACRO_MAP[src] == MACRO_MAP[tgt] else "inter"
            records.append(dict(
                source=src, target=tgt,
                macro_src=MACRO_MAP[src], macro_tgt=MACRO_MAP[tgt],
                weight=w, flux=flux, edge_type=etype,
            ))
    return pd.DataFrame(records).sort_values("flux", ascending=False).reset_index(drop=True)


def run_edge_importance(save_dir="./results/figures", dpi=200):
    """Section 7.1 — edge importance maps (Fig 7)."""
    _check_initialised()
    os.makedirs(save_dir, exist_ok=True)

    beta, rho, DS, DD = build_params()
    y0_base = make_ic(SPATIAL_SEED)
    t_base, pH_base, pD_base, pR_base = run_sim(W_norm, beta, rho, DS, DD, y0_base)

    df_flux = compute_edge_flux(t_base, pH_base, pD_base, W_raw, beta)
    intra_sum = df_flux[df_flux.edge_type == "intra"]["flux"].sum()
    inter_sum = df_flux[df_flux.edge_type == "inter"]["flux"].sum()
    frac_inter = inter_sum / (intra_sum + inter_sum)

    df_flux.to_csv(os.path.join(save_dir, "..", "edge_flux_table.csv"), index=False)

    # Figure 7 — ranked edge flux bar chart
    top_n = 25
    df_top = df_flux.head(top_n).copy().reset_index(drop=True)
    df_plot = df_top.iloc[::-1].reset_index(drop=True)

    def bar_color(row):
        return INTER_COLOR if row.edge_type == "inter" else INTRA_COLORS.get(row.macro_src, "#878787")

    colors = [bar_color(r) for _, r in df_plot.iterrows()]
    labels = [f"{short(r.source)}  →  {short(r.target)}" for _, r in df_plot.iterrows()]

    fig, ax = plt.subplots(figsize=(11, 12))
    y_pos = np.arange(top_n)
    bars = ax.barh(y_pos, df_plot["flux"].values, color=colors,
                   edgecolor="white", linewidth=0.5, height=0.65)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel(r"Cumulative directed flux  $\Phi_{i\to j}$", fontsize=10)
    ax.set_title("Edge importance map: top-25 directed edges by cumulative flux\n"
                 "(intra-layer = macro-type colour; inter-layer = red)",
                 fontsize=10, fontweight="bold", pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, df_plot["flux"].max() * 1.18)
    for bar, val in zip(bars, df_plot["flux"].values):
        ax.text(val + 0.015, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", ha="left", fontsize=7.5, color="#333333")

    legend_handles = [
        mpatches.Patch(color=INTRA_COLORS["PT"], label="Intra-layer: PT"),
        mpatches.Patch(color=INTRA_COLORS["DCT"], label="Intra-layer: DCT"),
        mpatches.Patch(color=INTRA_COLORS["TAL"], label="Intra-layer: TAL"),
        mpatches.Patch(color=INTRA_COLORS["vascular"], label="Intra-layer: Vascular"),
        mpatches.Patch(color=INTRA_COLORS["other"], label="Intra-layer: Other"),
        mpatches.Patch(color=INTER_COLOR, label="Inter-layer (cross-type)"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8.5,
              framealpha=0.92, edgecolor="#cccccc")
    ax.text(0.98, 0.01,
            f"Intra: {intra_sum:.1f}   Inter: {inter_sum:.1f}   ({frac_inter * 100:.0f}% inter)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            color="#555555", style="italic")

    plt.tight_layout()
    fig_path = os.path.join(save_dir, "fig7_edge_flux_ranked")
    plt.savefig(fig_path + ".png", dpi=dpi, bbox_inches="tight")
    plt.savefig(fig_path + ".svg", bbox_inches="tight")
    plt.close()

    print(f"Section 7.1 complete. Intra flux={intra_sum:.3f}, Inter flux={inter_sum:.3f} ({frac_inter * 100:.1f}%)")
    return dict(df_flux=df_flux, intra_sum=intra_sum, inter_sum=inter_sum,
                frac_inter=frac_inter, t_base=t_base, pD_base=pD_base, pH_base=pH_base,
                tinv_base=np.nanmean([t_base[np.where(pD_base[i] > 0.10)[0][0]]
                                      if len(np.where(pD_base[i] > 0.10)[0]) > 0 else np.nan
                                      for i in range(N)]))


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 7.2 — EDGE ABLATION
# ═══════════════════════════════════════════════════════════════════════════

def edge_class(a, b):
    ma, mb = MACRO_MAP.get(a, "other"), MACRO_MAP.get(b, "other")
    if ma == mb:
        return f"intra_{ma}"
    pair = tuple(sorted([ma, mb]))
    return "inter_" + "_".join(pair)


def run_edge_ablation(save_dir="./results/figures", dpi=200,
                      baseline_result=None):
    """Section 7.2 — edge ablation study (Table 12, Fig 8)."""
    _check_initialised()
    os.makedirs(save_dir, exist_ok=True)

    beta, rho, DS, DD = build_params()
    y0_base = make_ic(SPATIAL_SEED)

    if baseline_result is not None and "t_base" in baseline_result:
        t_base = baseline_result["t_base"]
        pD_base = baseline_result["pD_base"]
    else:
        t_base, _, pD_base, _ = run_sim(W_norm, beta, rho, DS, DD, y0_base)

    tinv_base, peakD_base, frac_base = global_metrics(t_base, pD_base)

    edge_classes = defaultdict(list)
    for idx, (a, b, w) in enumerate(BASE_EDGES):
        edge_classes[edge_class(a, b)].append(idx)

    # Need flux ranking for top-10 single-edge ablation
    if baseline_result is not None and "df_flux" in baseline_result:
        df_flux = baseline_result["df_flux"]
    else:
        _, pH_base, _, _ = run_sim(W_norm, beta, rho, DS, DD, y0_base)
        df_flux = compute_edge_flux(t_base, pH_base if 'pH_base' in dir() else
                                    run_sim(W_norm, beta, rho, DS, DD, y0_base)[1],
                                    pD_base, W_raw, beta)

    top10_edges = df_flux.head(10)[["source", "target"]].values.tolist()

    ablation_results = []

    # Class-level ablation
    for ec, idxs in sorted(edge_classes.items()):
        ablated = [e for i, e in enumerate(BASE_EDGES) if i not in idxs]
        _, W_norm_abl = build_adjacency_from_edges(ablated)
        t_a, _, pD_a, _ = run_sim(W_norm_abl, beta, rho, DS, DD, y0_base)
        tinv_a, peakD_a, frac_a = global_metrics(t_a, pD_a)
        ablation_results.append(dict(
            ablation_type="class", label=ec, n_edges_removed=len(idxs),
            mean_tinv=tinv_a, delta_tinv=tinv_a - tinv_base,
            peak_pD=peakD_a, delta_peakD=peakD_a - peakD_base,
            frac_invaded=frac_a))

    # Single-edge ablation
    for src, tgt in top10_edges:
        ablated = [e for e in BASE_EDGES
                   if not ((e[0] == src and e[1] == tgt) or (e[0] == tgt and e[1] == src))]
        _, W_norm_abl = build_adjacency_from_edges(ablated)
        t_a, _, pD_a, _ = run_sim(W_norm_abl, beta, rho, DS, DD, y0_base)
        tinv_a, peakD_a, frac_a = global_metrics(t_a, pD_a)
        ablation_results.append(dict(
            ablation_type="single_edge", label=f"{src}→{tgt}", n_edges_removed=1,
            mean_tinv=tinv_a, delta_tinv=tinv_a - tinv_base,
            peak_pD=peakD_a, delta_peakD=peakD_a - peakD_base,
            frac_invaded=frac_a))

    df_ablation = pd.DataFrame(ablation_results)
    df_ablation.to_csv(os.path.join(save_dir, "..", "edge_ablation_results.csv"), index=False)

    # Figure 8 — two-panel ablation
    df_class_abl = df_ablation[df_ablation.ablation_type == "class"].copy()
    df_single_abl = df_ablation[df_ablation.ablation_type == "single_edge"].copy()
    df_class_abl = df_class_abl.sort_values("delta_tinv", ascending=False).reset_index(drop=True)
    df_single_abl = df_single_abl.sort_values("delta_tinv", ascending=False).reset_index(drop=True)

    def abl_color(label):
        if label.startswith("intra_PT"): return INTRA_COLORS["PT"]
        if label.startswith("intra_DCT"): return INTRA_COLORS["DCT"]
        if label.startswith("intra_TAL"): return INTRA_COLORS["TAL"]
        if label.startswith("intra_vascular"): return INTRA_COLORS["vascular"]
        return INTER_COLOR

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    ax = axes[0]
    y_a = np.arange(len(df_class_abl))
    col_a = [abl_color(r.label) for _, r in df_class_abl.iterrows()]
    bars_a = ax.barh(y_a, df_class_abl["delta_tinv"].values, color=col_a,
                     edgecolor="white", linewidth=0.5, height=0.65)
    ax.axvline(0, color="#333333", linewidth=0.8, linestyle="--")
    ax.set_yticks(y_a)
    ax.set_yticklabels(df_class_abl["label"].str.replace("_", " "), fontsize=8.5)
    ax.set_xlabel(r"$\Delta t_{\mathrm{inv}}$ (ablated − baseline)", fontsize=9.5)
    ax.set_title("(A)  Edge-class ablation\n" + r"$\Delta t_{\mathrm{inv}}$ per removed class",
                 fontsize=9.5, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    xlim_a = max(abs(df_class_abl["delta_tinv"].min()), df_class_abl["delta_tinv"].max()) + 0.15
    ax.set_xlim(-xlim_a, xlim_a)
    for bar, val, row in zip(bars_a, df_class_abl["delta_tinv"].values, df_class_abl.itertuples()):
        xpos = val + 0.01 if val >= 0 else val - 0.01
        ax.text(xpos, bar.get_y() + bar.get_height() / 2,
                f"{val:+.3f}", va="center", ha="left" if val >= 0 else "right",
                fontsize=7.5, color="#333333")
        ax.text(ax.get_xlim()[1] * 0.97, bar.get_y() + bar.get_height() / 2,
                f"n={row.n_edges_removed}", va="center", ha="right", fontsize=7, color="#777777")

    ax = axes[1]
    def se_color(label):
        parts = label.split("→")
        if len(parts) == 2:
            ma = MACRO_MAP.get(parts[0].strip(), "other")
            mb = MACRO_MAP.get(parts[1].strip(), "other")
            if ma == mb: return INTRA_COLORS.get(ma, "#878787")
        return INTER_COLOR

    y_b = np.arange(len(df_single_abl))
    col_b = [se_color(r.label) for _, r in df_single_abl.iterrows()]
    labels_b = [short(r.label.replace("→", "  →  ")) for _, r in df_single_abl.iterrows()]
    bars_b = ax.barh(y_b, df_single_abl["delta_tinv"].values, color=col_b,
                     edgecolor="white", linewidth=0.5, height=0.65)
    ax.axvline(0, color="#333333", linewidth=0.8, linestyle="--")
    ax.set_yticks(y_b)
    ax.set_yticklabels(labels_b, fontsize=8.5)
    ax.set_xlabel(r"$\Delta t_{\mathrm{inv}}$ (ablated − baseline)", fontsize=9.5)
    ax.set_title("(B)  Single-edge ablation\n" + r"$\Delta t_{\mathrm{inv}}$ (top-10 flux edges)",
                 fontsize=9.5, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    xlim_b = max(abs(df_single_abl["delta_tinv"].min()), df_single_abl["delta_tinv"].max()) + 0.05
    ax.set_xlim(-xlim_b, xlim_b)
    for bar, val in zip(bars_b, df_single_abl["delta_tinv"].values):
        xpos = val + 0.005 if val >= 0 else val - 0.005
        ax.text(xpos, bar.get_y() + bar.get_height() / 2,
                f"{val:+.3f}", va="center", ha="left" if val >= 0 else "right",
                fontsize=7.5, color="#333333")

    legend_handles = [
        mpatches.Patch(color=INTRA_COLORS["PT"], label="Intra: PT"),
        mpatches.Patch(color=INTRA_COLORS["DCT"], label="Intra: DCT"),
        mpatches.Patch(color=INTRA_COLORS["TAL"], label="Intra: TAL"),
        mpatches.Patch(color=INTRA_COLORS["vascular"], label="Intra: Vascular"),
        mpatches.Patch(color=INTER_COLOR, label="Inter-layer"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=5,
               fontsize=8.5, framealpha=0.92, edgecolor="#cccccc", bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Edge ablation: counterfactual invasion time " +
                 r"$\Delta t_{\mathrm{inv}}$ vs. corrected baseline",
                 fontsize=11, fontweight="bold", y=1.01)
    plt.tight_layout(pad=1.5)
    fig_path = os.path.join(save_dir, "fig8_edge_ablation")
    plt.savefig(fig_path + ".png", dpi=dpi, bbox_inches="tight")
    plt.savefig(fig_path + ".svg", bbox_inches="tight")
    plt.close()

    print(f"Section 7.2 complete. Baseline t_inv={tinv_base:.3f}, {len(df_ablation)} ablation rows.")
    return dict(df_ablation=df_ablation, tinv_base=tinv_base, peakD_base=peakD_base)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 7.3 — SEEDING SCENARIOS
# ═══════════════════════════════════════════════════════════════════════════

def run_seeding_scenarios(save_dir="./results/figures", dpi=200):
    """Section 7.3 — seeding scenario comparison (Figs 9-11)."""
    _check_initialised()
    os.makedirs(save_dir, exist_ok=True)

    beta, rho, DS, DD = build_params()

    rng = np.random.default_rng(42)
    ALL_BOUNDARY = ZONE_BOUNDARY + ZONE_OUTER_MED
    N_RANDOM = max(1, len(ALL_BOUNDARY) // 2)

    RANDOM_SEEDS = []
    for rep in range(5):
        chosen = rng.choice(ALL_BOUNDARY, size=N_RANDOM, replace=False).tolist()
        RANDOM_SEEDS.append(chosen)

    t_sp, _, pD_sp, pR_sp = run_sim(W_norm, beta, rho, DS, DD, make_ic(SPATIAL_SEED))
    t_ce, _, pD_ce, pR_ce = run_sim(W_norm, beta, rho, DS, DD, make_ic(CENTRAL_SEED))

    rand_runs = []
    for seeds in RANDOM_SEEDS:
        t_r, _, pD_r, pR_r = run_sim(W_norm, beta, rho, DS, DD, make_ic(seeds))
        rand_runs.append((t_r, pD_r, pR_r))

    df_sp = compute_metrics(t_sp, pD_sp, pR_sp)
    df_ce = compute_metrics(t_ce, pD_ce, pR_ce)
    df_rand_list = [compute_metrics(t_r, pD_r, pR_r) for t_r, pD_r, pR_r in rand_runs]

    rank_sp = df_sp["tcross"].rank(method="min", na_option="bottom")
    rank_ce = df_ce["tcross"].rank(method="min", na_option="bottom")
    rs_central = spearmanr(rank_sp, rank_ce, nan_policy="omit").statistic

    rs_rand_list = []
    for df_r in df_rand_list:
        rk_r = df_r["tcross"].rank(method="min", na_option="bottom")
        rs_rand_list.append(spearmanr(rank_sp, rk_r, nan_policy="omit").statistic)
    rs_rand_mean = np.mean(rs_rand_list)
    rs_rand_std = np.std(rs_rand_list)

    def scenario_summary(df, label):
        valid = df["tcross"].dropna()
        return dict(scenario=label, mean_tinv=valid.mean(), std_tinv=valid.std(),
                    frac_invaded=(~df["tcross"].isna()).mean(),
                    global_peakD=df["peakD"].mean(), mean_pR_fin=df["pR_fin"].mean(),
                    first_invaded=df["tcross"].idxmin(),
                    last_invaded=df["tcross"].dropna().idxmax())

    summary_rows = [scenario_summary(df_sp, "Spatial boundary (paper default)"),
                    scenario_summary(df_ce, "Central seed")]
    for i, df_r in enumerate(df_rand_list):
        summary_rows.append(scenario_summary(df_r, f"Random boundary rep {i+1}"))
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(os.path.join(save_dir, "..", "seeding_scenario_summary.csv"), index=False)

    # ── Layout helpers ──────────────────────────────────────────────────────
    ZONE_ORDER = ["boundary", "outer_medulla", "cortex", "inner_medulla", "mid", "core"]
    MACRO_ORDER = ["PT", "DCT", "TAL", "vascular", "other"]

    def get_zone(c):
        for z in ["boundary", "outer_medulla", "inner_medulla", "cortex", "mid", "core"]:
            if z in c: return z
        return "cortex"

    def node_pos(c):
        z = get_zone(c)
        m = MACRO_MAP.get(c, "other")
        x = MACRO_ORDER.index(m)
        y = -(ZONE_ORDER.index(z))
        return x, y

    def draw_graph_with_seeds(ax, seed_comps, pD_t_snapshot, title):
        for a, b, w in BASE_EDGES:
            if a in C_IDX and b in C_IDX:
                xa, ya = node_pos(a)
                xb, yb = node_pos(b)
                ax.plot([xa, xb], [ya, yb], color="#cccccc", linewidth=w * 0.8, zorder=1)
        for c in COMPARTMENTS:
            x, y = node_pos(c)
            is_seed = c in seed_comps
            is_active = pD_t_snapshot[C_IDX[c]] > 0.05
            facecolor = SEED_COLOR if is_seed else (ACTIVE_COLOR if is_active else "#e0e0e0")
            edgecolor = "#cc0000" if is_seed else ("#ccaa00" if is_active else "#999999")
            zorder = 4 if is_seed else (3 if is_active else 2)
            size = 200 if is_seed else (150 if is_active else 80)
            ax.scatter(x, y, s=size, facecolors=facecolor, edgecolors=edgecolor,
                       linewidths=1.8, zorder=zorder)
            ax.text(x, y - 0.30, short(c), ha="center", va="top", fontsize=5.5, color="#444444", zorder=5)
        ax.set_xticks(range(len(MACRO_ORDER)))
        ax.set_xticklabels(MACRO_ORDER, fontsize=8)
        ax.set_yticks([-i for i in range(len(ZONE_ORDER))])
        ax.set_yticklabels(ZONE_ORDER, fontsize=8)
        ax.set_xlim(-0.6, len(MACRO_ORDER) - 0.4)
        ax.set_ylim(-len(ZONE_ORDER) + 0.4, 0.6)
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)

    # ── Fig 9: spatial boundary vs central seed ─────────────────────────────
    snap_idx = np.argmin(np.abs(t_sp - 10.0))
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    draw_graph_with_seeds(axes[0], SPATIAL_SEED, pD_sp[:, snap_idx],
                          "(A)  Spatial boundary seed\n(paper default) — t = 10")
    draw_graph_with_seeds(axes[1], CENTRAL_SEED, pD_ce[:, snap_idx],
                          "(B)  Central seed — t = 10")
    legend_handles_g = [
        mpatches.Patch(facecolor=SEED_COLOR, edgecolor="#cc0000", label="Seed node"),
        mpatches.Patch(facecolor=ACTIVE_COLOR, edgecolor="#ccaa00", label="Activated node (pD > 0.05)"),
        mpatches.Patch(facecolor="#e0e0e0", edgecolor="#999999", label="Inactive node"),
    ]
    fig.legend(handles=legend_handles_g, loc="lower center", ncol=3,
               fontsize=9, framealpha=0.92, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Seed-node visualisation: spatial boundary vs. central seed (t = 10)",
                 fontsize=11, fontweight="bold", y=1.01)
    plt.tight_layout(pad=1.5)
    fig_path = os.path.join(save_dir, "fig9_seed_vis_sp_central")
    plt.savefig(fig_path + ".png", dpi=dpi, bbox_inches="tight")
    plt.savefig(fig_path + ".svg", bbox_inches="tight")
    plt.close()

    # ── Fig 10: random boundary replicates ──────────────────────────────────
    fig, axes = plt.subplots(1, 5, figsize=(28, 6))
    for rep, (seeds, (t_r, pD_r, _)) in enumerate(zip(RANDOM_SEEDS, rand_runs)):
        snap_r = np.argmin(np.abs(t_r - 10.0))
        draw_graph_with_seeds(axes[rep], seeds, pD_r[:, snap_r],
                              f"Random rep {rep+1}\n({len(seeds)} seeds)")
    fig.legend(handles=legend_handles_g, loc="lower center", ncol=3,
               fontsize=9, framealpha=0.92, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Random boundary seed replicates: active seed nodes and propagation at t = 10",
                 fontsize=11, fontweight="bold", y=1.02)
    plt.tight_layout(pad=1.5)
    fig_path = os.path.join(save_dir, "fig10_seed_vis_random5")
    plt.savefig(fig_path + ".png", dpi=dpi, bbox_inches="tight")
    plt.savefig(fig_path + ".svg", bbox_inches="tight")
    plt.close()

    # ── Fig 11: four-panel scenario comparison ───────────────────────────────
    mean_pD_sp = pD_sp.mean(axis=0)
    mean_pD_ce = pD_ce.mean(axis=0)
    rand_pD_mean = np.mean([pD_r.mean(axis=0) for _, pD_r, _ in rand_runs], axis=0)
    rand_pD_std = np.std([pD_r.mean(axis=0) for _, pD_r, _ in rand_runs], axis=0)

    def get_tcross_arr(pD_t, t, thr=0.10):
        tc = []
        for i in range(N):
            c = np.where(pD_t[i] > thr)[0]
            tc.append(t[c[0]] if len(c) > 0 else np.nan)
        return np.array(tc)

    tcross_sp = get_tcross_arr(pD_sp, t_sp)
    tcross_ce = get_tcross_arr(pD_ce, t_ce)
    tcross_rand_list = [get_tcross_arr(pD_r, t_r) for t_r, pD_r, _ in rand_runs]
    tcross_rand_mean = np.nanmean(tcross_rand_list, axis=0)
    sort_idx = np.argsort(tcross_sp)
    comp_labels = [short(COMPARTMENTS[i]) for i in sort_idx]
    rand_tinv_means = [np.nanmean(tc) for tc in tcross_rand_list]

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.35)
    axA = fig.add_subplot(gs[0, 0]); axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0]); axD = fig.add_subplot(gs[1, 1])

    axA.plot(t_sp, mean_pD_sp, color="#2166ac", lw=2.0, label="Spatial boundary (paper default)")
    axA.plot(t_ce, mean_pD_ce, color="#d73027", lw=2.0, linestyle="--", label="Central seed")
    axA.plot(t_sp, rand_pD_mean, color="#4dac26", lw=2.0, linestyle="-.", label="Random boundary (mean, n=5)")
    axA.fill_between(t_sp, rand_pD_mean - rand_pD_std, rand_pD_mean + rand_pD_std, color="#4dac26", alpha=0.15)
    axA.set_xlabel("Simulation time", fontsize=9.5)
    axA.set_ylabel(r"Mean $p^D(t)$", fontsize=9.5)
    axA.set_title("(A)  Disease pressure dynamics\nper seeding scenario", fontsize=9.5, fontweight="bold")
    axA.legend(fontsize=8, framealpha=0.9)
    axA.spines[["top", "right"]].set_visible(False)

    hmap = np.vstack([tcross_sp[sort_idx], tcross_rand_mean[sort_idx], tcross_ce[sort_idx]])
    hmap_disp = np.where(np.isnan(hmap), np.nanmax(hmap) + 5, hmap)
    im = axB.imshow(hmap_disp, aspect="auto", cmap="YlOrRd", vmin=0, vmax=np.nanmax(hmap) + 2)
    axB.set_xticks(range(N))
    axB.set_xticklabels(comp_labels, rotation=55, ha="right", fontsize=6.5)
    axB.set_yticks([0, 1, 2])
    axB.set_yticklabels(["Spatial\nboundary", "Random\nboundary\n(mean)", "Central\nseed"], fontsize=8.5)
    axB.set_title(r"(B)  Activation time $t_{\mathrm{cross}}$ per compartment" +
                  "\n(sorted by spatial boundary order)", fontsize=9.5, fontweight="bold")
    cbar = fig.colorbar(im, ax=axB, fraction=0.03, pad=0.04)
    cbar.set_label(r"$t_{\mathrm{cross}}$", fontsize=8)

    sc_labels = ["Spatial\nboundary\nvs. itself", "Random\nboundary\n(mean±sd)", "Central\nseed"]
    rs_vals = [1.0, rs_rand_mean, rs_central]
    rs_errs = [0.0, rs_rand_std, 0.0]
    col_c = ["#2166ac", "#4dac26", "#d73027"]
    bars_c = axC.bar(sc_labels, rs_vals, yerr=rs_errs, color=col_c,
                     edgecolor="white", linewidth=0.5, capsize=5,
                     error_kw=dict(elinewidth=1.2, ecolor="#333333"))
    axC.axhline(0, color="#333333", linewidth=0.8, linestyle="--")
    axC.set_ylabel(r"Spearman $r_s$ (vs. spatial boundary)", fontsize=9.5)
    axC.set_title("(C)  Ranking stability\n(activation-time ordering)", fontsize=9.5, fontweight="bold")
    axC.set_ylim(-0.7, 1.15)
    axC.spines[["top", "right"]].set_visible(False)
    for bar, val in zip(bars_c, rs_vals):
        axC.text(bar.get_x() + bar.get_width() / 2, val + 0.04,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    sc_labels_d = ["Spatial\nboundary", "Random\nboundary\n(rep 1–5)", "Central\nseed"]
    tinv_means = [np.nanmean(tcross_sp), np.mean(rand_tinv_means), np.nanmean(tcross_ce)]
    tinv_errs = [0.0, np.std(rand_tinv_means), 0.0]
    bars_d = axD.bar(sc_labels_d, tinv_means, yerr=tinv_errs, color=col_c,
                     edgecolor="white", linewidth=0.5, capsize=5,
                     error_kw=dict(elinewidth=1.2, ecolor="#333333"))
    axD.set_ylabel(r"Mean invasion time $\bar{t}_{\mathrm{inv}}$", fontsize=9.5)
    axD.set_title("(D)  Mean invasion time\nper seeding scenario", fontsize=9.5, fontweight="bold")
    axD.spines[["top", "right"]].set_visible(False)
    for bar, val in zip(bars_d, tinv_means):
        axD.text(bar.get_x() + bar.get_width() / 2, val + 0.15,
                 f"{val:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    fig.suptitle("Seeding scenario comparison: spatial boundary, random boundary, central seed",
                 fontsize=11, fontweight="bold", y=1.01)
    fig_path = os.path.join(save_dir, "fig11_seeding_comparison")
    plt.savefig(fig_path + ".png", dpi=dpi, bbox_inches="tight")
    plt.savefig(fig_path + ".svg", bbox_inches="tight")
    plt.close()

    print(f"Section 7.3 complete. r_s random={rs_rand_mean:.3f}±{rs_rand_std:.3f}, r_s central={rs_central:.3f}")
    return dict(df_summary=df_summary, rs_rand_mean=rs_rand_mean, rs_rand_std=rs_rand_std,
                rs_central=rs_central, rs_rand_list=rs_rand_list)


# ═══════════════════════════════════════════════════════════════════════════
# RUNNER
# ═══════════════════════════════════════════════════════════════════════════

def run_section7_analysis(save_dir="./results/figures", dpi=200):
    """Run all three Section 7 analyses and return combined results."""
    print("=" * 60)
    print("SECTION 7 — Mechanistic interpretability of spatial diffusion")
    print("=" * 60)

    print("\n[7.1] Edge importance maps...")
    res1 = run_edge_importance(save_dir=save_dir, dpi=dpi)

    print("\n[7.2] Edge ablation...")
    res2 = run_edge_ablation(save_dir=save_dir, dpi=dpi, baseline_result=res1)

    print("\n[7.3] Seeding scenarios...")
    res3 = run_seeding_scenarios(save_dir=save_dir, dpi=dpi)

    # Print key numbers
    print("\n" + "=" * 60)
    print("KEY NUMBERS FOR MANUSCRIPT")
    print("=" * 60)
    print(f"7.1  intra total flux     : {res1['intra_sum']:.3f}")
    print(f"7.1  inter total flux     : {res1['inter_sum']:.3f}  ({res1['frac_inter'] * 100:.1f}%)")
    print(f"7.2  baseline t_inv       : {res2['tinv_base']:.3f}")
    print(f"7.3  r_s random vs spatial: {res3['rs_rand_mean']:.3f} ± {res3['rs_rand_std']:.3f}")
    print(f"7.3  r_s central vs spatial: {res3['rs_central']:.3f}")

    return dict(edge_importance=res1, edge_ablation=res2, seeding_scenarios=res3)


if __name__ == "__main__":
    run_section7_analysis()
