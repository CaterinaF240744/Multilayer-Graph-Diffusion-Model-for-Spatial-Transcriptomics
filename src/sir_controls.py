"""
sir_controls.py
===============
Control experiments for the compartmental diffusion model.

Addresses Reviewer 2, Point 6: "The case study would benefit from internal
controls. Rather than requiring extensive method benchmarking, the authors
should include simple controls such as alternative seed locations, randomized
graph edges, uniform parameters, removal of vascular edges, etc."

Also addresses Reviewer 1, Point 5b: R-state sensitivity analysis.

Two versions of each control are provided:
  1. **Visium pipeline** (primary): runs on the compartment graph derived from
     the V1 Mouse Kidney Visium dataset via build_compartment2 + run_SIR.
     These are directly comparable to the paper's main results.
  2. **Section 7 simplified model** (supplementary): runs on the hardcoded
     13-compartment model from sir_section7.py. Useful for rapid testing
     and as a simplified-model cross-check.

Controls implemented
---------------------
1. randomized_edge_null_model : Maslov-Sneppen degree-preserving rewiring
2. uniform_parameters         : all β, ρ, D_I set to global mean
3. vascular_edge_removal      : remove all vascular compartment edges
4. alternative_seeds          : cortex-only, inner-medulla-only, single-compartment
5. R_state_sensitivity        : vary recovery rate ρ, measure R prediction sensitivity

Functions exported
-------------------
run_randomized_edge_null_model_visium
run_uniform_parameters_control_visium
run_vascular_edge_removal_visium
run_alternative_seeds_visium
run_R_state_sensitivity
run_all_controls_visium
run_all_controls_section7   (supplementary, on simplified model)
"""

from __future__ import annotations

import os
import warnings
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import spearmanr, wilcoxon

import matplotlib
matplotlib.rcParams['font.family'] = ['Liberation Sans', 'Arimo', 'DejaVu Sans']
matplotlib.rcParams['svg.fonttype'] = 'none'
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

# ── Visium pipeline imports ─────────────────────────────────────────────────
from sir_compartments import (
    build_default_params_SIR, run_SIR, sir_metrics,
    build_default_params_SIR as _build_params,
)


# ═══════════════════════════════════════════════════════════════════════════
# HELPER: run a single simulation on the Visium compartment graph
# ═══════════════════════════════════════════════════════════════════════════

def _run_and_extract_visium(
    comp:        dict,
    sir_params:  dict,
    seed_I:      float = 0.05,
    t_end:       float = 60.0,
    n_steps:     int   = 400,
    threshold_I: float = 0.10,
    label:       str   = "",
    boundary_mask_override: np.ndarray = None,
    Lc2_override: sp.spmatrix = None,
) -> dict:
    """
    Run a single SIR simulation on the Visium compartment graph and
    extract summary metrics.

    Parameters
    ----------
    comp                : output of build_compartment2
    sir_params          : output of build_default_params_SIR
    seed_I              : initial diseased fraction in seed compartments
    boundary_mask_override : if provided, use this instead of comp["boundary_mask"]
                             (for alternative seed experiments)
    Lc2_override        : if provided, use this Laplacian instead of comp["Lc2"]
                         (for graph rewiring / edge removal experiments)

    Returns
    -------
    dict with: label, mean_tinv, peak_D, frac_invaded, macro_tinv, macro_Rdiff,
               df_metrics, t, D_t, R_t
    """
    comps2        = comp["comps2"]
    Lc2           = Lc2_override if Lc2_override is not None else comp["Lc2"]
    boundary_mask = (boundary_mask_override if boundary_mask_override is not None
                     else comp["boundary_mask"])
    macro_of      = comp["macro_of"]
    region_of     = comp["region_of"]

    S_fin, I_fin, R_fin, sol, _ = run_SIR(
        comps2, Lc2, boundary_mask, macro_of, region_of,
        sir_params=sir_params,
        seed_I=seed_I,
        seed_boundary_only=True,
        use_ivp=True,
        t_end=t_end,
        n_steps=n_steps,
    )

    # Extract trajectories
    nC = len(comps2)
    t_arr = sol.t
    D_traj = sol.y[nC:2*nC, :]   # D(t) = diseased
    R_traj = sol.y[2*nC:3*nC, :]  # R(t) = recovered

    # Global metrics
    D_mean = D_traj.mean(axis=0)
    peak_D = float(D_mean.max())
    t_peak = float(t_arr[np.argmax(D_mean)])

    # Invasion time per compartment
    inv_times = []
    for ci in range(nC):
        above = np.where(D_traj[ci, :] > threshold_I)[0]
        inv_times.append(float(t_arr[above[0]]) if len(above) > 0 else np.nan)
    inv_times = np.array(inv_times)
    mean_tinv = float(np.nanmean(inv_times))
    frac_invaded = float(np.mean(~np.isnan(inv_times)))

    # Per macro-type invasion order
    macro_tinv = {}
    for macro in np.unique(macro_of):
        mask = macro_of == macro
        vals = inv_times[mask]
        valid = vals[~np.isnan(vals)]
        macro_tinv[macro] = float(valid.mean()) if len(valid) > 0 else np.nan

    # R_diff per macro-type
    beta_v = np.asarray(sir_params["beta"], dtype=float)
    gamma_v = np.asarray(sir_params["gamma"], dtype=float)
    R_diff = beta_v / np.where(gamma_v > 0, gamma_v, 1e-12)
    macro_Rdiff = {}
    for macro in np.unique(macro_of):
        mask = macro_of == macro
        macro_Rdiff[macro] = float(R_diff[mask].mean())

    return dict(
        label=label, mean_tinv=mean_tinv, peak_D=peak_D,
        frac_invaded=frac_invaded, t_peak=t_peak,
        macro_tinv=macro_tinv, macro_Rdiff=macro_Rdiff,
        inv_times=inv_times,
        t=t_arr, D_t=D_traj, R_t=R_traj,
        S_fin=S_fin, I_fin=I_fin, R_fin=R_fin,
    )


# ═══════════════════════════════════════════════════════════════════════════
# CONTROL 1: RANDOMIZED EDGE NULL MODEL (Maslov-Sneppen rewiring)
# ═══════════════════════════════════════════════════════════════════════════

def maslov_sneppen_rewire_sparse(W: sp.spmatrix, n_swaps: int, rng) -> sp.csr_matrix:
    """
    Degree-preserving random rewiring (Maslov-Sneppen) for sparse matrices.

    Repeatedly pick two random edges (a,b) and (c,d), swap to (a,d) and (c,b)
    if the new edges don't already exist and don't create self-loops.
    """
    W = sp.lil_matrix(W).copy()
    edges = list(zip(*W.nonzero()))
    # Keep only upper triangle edges to avoid duplicates
    edges = [(i, j) for i, j in edges if i < j]
    n_edges = len(edges)
    if n_edges < 2:
        return W.tocsr()

    swaps_done = 0
    attempts = 0
    max_attempts = n_swaps * 20

    while swaps_done < n_swaps and attempts < max_attempts:
        attempts += 1
        idx1, idx2 = rng.integers(n_edges, size=2)
        a, b = edges[idx1]
        c, d = edges[idx2]
        if a == c or a == d or b == c or b == d:
            continue
        if W[a, d] > 0 or W[c, b] > 0:
            continue
        # Perform swap
        w_ab = W[a, b]
        w_cd = W[c, d]
        W[a, b] = W[b, a] = 0
        W[c, d] = W[d, c] = 0
        W[a, d] = W[d, a] = w_ab
        W[c, b] = W[b, c] = w_cd
        edges[idx1] = (a, d) if a < d else (d, a)
        edges[idx2] = (c, b) if c < b else (b, c)
        swaps_done += 1

    return W.tocsr()


def run_randomized_edge_null_model_visium(
    comp:        dict,
    sir_params:  dict,
    seed_I:      float = 0.05,
    t_end:       float = 60.0,
    n_steps:     int   = 400,
    n_rewires:   int   = 20,
    n_swaps_per: int   = 50,
    threshold_I: float = 0.10,
    save_dir:    str   = "./results/figures",
    dpi:         int   = 200,
) -> dict:
    """
    Control 1: Randomized edge null model on the Visium compartment graph.

    Rewires comp["Wc2"] preserving degree distribution, then compares
    invasion dynamics. If tissue structure drives the activation hierarchy,
    rewired graphs should show different/no hierarchy.
    """
    os.makedirs(save_dir, exist_ok=True)
    print("\n[Control 1] Randomized edge null model (Maslov-Sneppen, Visium graph)")

    # Baseline
    baseline = _run_and_extract_visium(
        comp, sir_params, seed_I=seed_I, t_end=t_end, n_steps=n_steps,
        threshold_I=threshold_I, label="Baseline",
    )
    print(f"  Baseline: t_inv={baseline['mean_tinv']:.3f}, peak_D={baseline['peak_D']:.3f}")

    # Rewired replicates
    rng = np.random.default_rng(42)
    Wc2 = comp["Wc2"]
    from scipy.sparse.csgraph import laplacian as sp_laplacian

    rewired_results = []
    for i in range(n_rewires):
        W_rewired = maslov_sneppen_rewire_sparse(Wc2, n_swaps_per, rng)
        L_rewired = sp_laplacian(W_rewired, normed=True).tocsr()
        res = _run_and_extract_visium(
            comp, sir_params, seed_I=seed_I, t_end=t_end, n_steps=n_steps,
            threshold_I=threshold_I, label=f"Rewired_{i+1}",
            Lc2_override=L_rewired,
        )
        rewired_results.append(res)
        if (i + 1) % 5 == 0:
            print(f"  Rewire {i+1}/{n_rewires}: t_inv={res['mean_tinv']:.3f}")

    # Aggregate
    tinv_rewired = [r["mean_tinv"] for r in rewired_results]
    peakD_rewired = [r["peak_D"] for r in rewired_results]

    # Spearman: compare macro-type invasion order (baseline vs each rewired)
    macros = list(baseline["macro_tinv"].keys())
    baseline_order = np.array([baseline["macro_tinv"][m] for m in macros])
    spearman_rewired = []
    for r in rewired_results:
        rewired_order = np.array([r["macro_tinv"].get(m, np.nan) for m in macros])
        valid = ~(np.isnan(baseline_order) | np.isnan(rewired_order))
        if valid.sum() >= 3:
            rs = spearmanr(baseline_order[valid], rewired_order[valid]).statistic
            spearman_rewired.append(rs)

    # Wilcoxon test: baseline vs rewired distributions
    try:
        w_stat, w_pval = wilcoxon(tinv_rewired, [baseline["mean_tinv"]] * len(tinv_rewired)).statistic, \
                         wilcoxon(tinv_rewired, [baseline["mean_tinv"]] * len(tinv_rewired)).pvalue
    except Exception:
        w_stat, w_pval = np.nan, np.nan

    # Summary table
    rows = [dict(scenario="Baseline", mean_tinv=baseline["mean_tinv"],
                 peak_D=baseline["peak_D"], frac_invaded=baseline["frac_invaded"],
                 spearman_vs_baseline=1.0)]
    for i, r in enumerate(rewired_results):
        rs = spearman_rewired[i] if i < len(spearman_rewired) else np.nan
        rows.append(dict(scenario=f"Rewired_{i+1}", mean_tinv=r["mean_tinv"],
                         peak_D=r["peak_D"], frac_invaded=r["frac_invaded"],
                         spearman_vs_baseline=rs))
    df_summary = pd.DataFrame(rows)
    df_summary.to_csv(os.path.join(save_dir, "..", "tables",
                                   "control1_null_model_visium.csv"), index=False)

    # Figure: comparison
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    ax.hist(tinv_rewired, bins=min(10, n_rewires), color="#4dac26", alpha=0.7,
            edgecolor="white", label="Rewired")
    ax.axvline(baseline["mean_tinv"], color="#d73027", lw=2.5, linestyle="--",
               label="Baseline")
    ax.set_xlabel(r"Mean invasion time $\bar{t}_{\mathrm{inv}}$", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title(f"(A) Invasion time\n(rewired vs baseline)\nWilcoxon p={w_pval:.3f}",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    ax.hist(peakD_rewired, bins=min(10, n_rewires), color="#2166ac", alpha=0.7,
            edgecolor="white", label="Rewired")
    ax.axvline(baseline["peak_D"], color="#d73027", lw=2.5, linestyle="--",
               label="Baseline")
    ax.set_xlabel(r"Peak $D(t)$", fontsize=10)
    ax.set_title("(B) Peak diseased fraction\n(rewired vs baseline)",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[2]
    if spearman_rewired:
        ax.hist(spearman_rewired, bins=min(10, n_rewires), color="#f1a340",
                alpha=0.7, edgecolor="white")
        ax.axvline(np.mean(spearman_rewired), color="#d73027", lw=2.5, linestyle="--",
                   label=f"mean={np.mean(spearman_rewired):.3f}")
    ax.set_xlabel(r"Spearman $r_s$ (vs baseline order)", fontsize=10)
    ax.set_title("(C) Activation order stability\n(rewired vs baseline)",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"Control 1: Randomized edge null model (Maslov-Sneppen, n={n_rewires}, Visium graph)",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig_path = os.path.join(save_dir, "control1_null_model_visium")
    plt.savefig(fig_path + ".png", dpi=dpi, bbox_inches="tight")
    plt.savefig(fig_path + ".svg", bbox_inches="tight")
    plt.close()

    print(f"  Rewired t_inv: {np.mean(tinv_rewired):.3f} ± {np.std(tinv_rewired):.3f} "
          f"(baseline: {baseline['mean_tinv']:.3f})")
    print(f"  Spearman r_s: {np.mean(spearman_rewired):.3f} ± {np.std(spearman_rewired):.3f}")
    print(f"  Wilcoxon p-value: {w_pval:.4f}")

    return dict(baseline=baseline, rewired=rewired_results, df_summary=df_summary,
                tinv_rewired=tinv_rewired, spearman_rewired=spearman_rewired,
                wilcoxon_pval=w_pval)


# ═══════════════════════════════════════════════════════════════════════════
# CONTROL 2: UNIFORM PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════

def run_uniform_parameters_control_visium(
    comp:        dict,
    sir_params:  dict,
    seed_I:      float = 0.05,
    t_end:       float = 60.0,
    n_steps:     int   = 400,
    threshold_I: float = 0.10,
    save_dir:    str   = "./results/figures",
    dpi:         int   = 200,
) -> dict:
    """
    Control 2: Uniform parameters on the Visium compartment graph.

    Sets all β, ρ, D_I to the global mean of the heterogeneous parameters.
    If cell-type heterogeneity drives the amplifier/absorber roles, uniform
    params should flatten the hierarchy.
    """
    os.makedirs(save_dir, exist_ok=True)
    print("\n[Control 2] Uniform parameters control (Visium graph)")

    baseline = _run_and_extract_visium(
        comp, sir_params, seed_I=seed_I, t_end=t_end, n_steps=n_steps,
        threshold_I=threshold_I, label="Heterogeneous (baseline)",
    )

    # Uniform: global mean
    uniform_params = {
        "beta":  np.full_like(sir_params["beta"],  sir_params["beta"].mean()),
        "gamma": np.full_like(sir_params["gamma"], sir_params["gamma"].mean()),
        "D_S":   np.full_like(sir_params["D_S"],   sir_params["D_S"].mean()),
        "D_I":   np.full_like(sir_params["D_I"],   sir_params["D_I"].mean()),
    }
    uniform = _run_and_extract_visium(
        comp, uniform_params, seed_I=seed_I, t_end=t_end, n_steps=n_steps,
        threshold_I=threshold_I, label="Uniform (global mean)",
    )

    print(f"  Heterogeneous: t_inv={baseline['mean_tinv']:.3f}, peak_D={baseline['peak_D']:.3f}")
    print(f"  Uniform:       t_inv={uniform['mean_tinv']:.3f}, peak_D={uniform['peak_D']:.3f}")

    # Compare macro-type R_diff and invasion order
    macros = list(baseline["macro_tinv"].keys())
    comparison_rows = []
    for m in macros:
        comparison_rows.append(dict(
            macro_type=m,
            R_diff_het=baseline["macro_Rdiff"].get(m, np.nan),
            R_diff_uni=uniform["macro_Rdiff"].get(m, np.nan),
            tinv_het=baseline["macro_tinv"].get(m, np.nan),
            tinv_uni=uniform["macro_tinv"].get(m, np.nan),
        ))
    df_comp = pd.DataFrame(comparison_rows)
    df_comp.to_csv(os.path.join(save_dir, "..", "tables",
                                "control2_uniform_params_visium.csv"), index=False)

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(macros))
    width = 0.35

    ax = axes[0]
    ax.bar(x - width/2, [baseline["macro_Rdiff"].get(m, 0) for m in macros], width,
           color="#2166ac", label="Heterogeneous", edgecolor="white")
    ax.bar(x + width/2, [uniform["macro_Rdiff"].get(m, 0) for m in macros], width,
           color="#f1a340", label="Uniform", edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(macros, fontsize=10, rotation=30)
    ax.set_ylabel(r"$R_{\mathrm{diff}} = \beta/\rho$", fontsize=10)
    ax.set_title("(A) Diffusion reproduction number\nper macro-type",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=9); ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    ax.bar(x - width/2, [baseline["macro_tinv"].get(m, 0) for m in macros], width,
           color="#2166ac", label="Heterogeneous", edgecolor="white")
    ax.bar(x + width/2, [uniform["macro_tinv"].get(m, 0) for m in macros], width,
           color="#f1a340", label="Uniform", edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(macros, fontsize=10, rotation=30)
    ax.set_ylabel(r"Mean invasion time $t_{\mathrm{inv}}$", fontsize=10)
    ax.set_title("(B) Invasion time\nper macro-type",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=9); ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Control 2: Uniform parameters — does heterogeneity drive the hierarchy? (Visium graph)",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig_path = os.path.join(save_dir, "control2_uniform_params_visium")
    plt.savefig(fig_path + ".png", dpi=dpi, bbox_inches="tight")
    plt.savefig(fig_path + ".svg", bbox_inches="tight")
    plt.close()

    return dict(baseline=baseline, uniform=uniform, df_comp=df_comp)


# ═══════════════════════════════════════════════════════════════════════════
# CONTROL 3: VASCULAR EDGE REMOVAL
# ═══════════════════════════════════════════════════════════════════════════

def run_vascular_edge_removal_visium(
    comp:        dict,
    sir_params:  dict,
    seed_I:      float = 0.05,
    t_end:       float = 60.0,
    n_steps:     int   = 400,
    threshold_I: float = 0.10,
    save_dir:    str   = "./results/figures",
    dpi:         int   = 200,
) -> dict:
    """
    Control 3: Vascular edge removal on the Visium compartment graph.

    Removes all edges connected to vascular compartments from Wc2.
    If vascular compartments are propagation corridors, removal should
    slow/delay invasion.
    """
    os.makedirs(save_dir, exist_ok=True)
    print("\n[Control 3] Vascular edge removal (Visium graph)")

    baseline = _run_and_extract_visium(
        comp, sir_params, seed_I=seed_I, t_end=t_end, n_steps=n_steps,
        threshold_I=threshold_I, label="Baseline (all edges)",
    )

    # Remove all edges touching vascular compartments
    macro_of = comp["macro_of"]
    vascular_mask = macro_of == "vascular"
    vascular_idx = np.where(vascular_mask)[0]

    Wc2 = sp.lil_matrix(comp["Wc2"]).copy()
    n_removed = 0
    for vi in vascular_idx:
        for j in range(Wc2.shape[1]):
            if Wc2[vi, j] != 0:
                Wc2[vi, j] = 0
                Wc2[j, vi] = 0
                n_removed += 1
    Wc2 = Wc2.tocsr()
    Wc2.eliminate_zeros()

    from scipy.sparse.csgraph import laplacian as sp_laplacian
    L_abl = sp_laplacian(Wc2, normed=True).tocsr()

    ablated = _run_and_extract_visium(
        comp, sir_params, seed_I=seed_I, t_end=t_end, n_steps=n_steps,
        threshold_I=threshold_I, label="Vascular edges removed",
        Lc2_override=L_abl,
    )

    print(f"  Baseline:          t_inv={baseline['mean_tinv']:.3f}, peak_D={baseline['peak_D']:.3f}")
    print(f"  Vascular removed:  t_inv={ablated['mean_tinv']:.3f}, peak_D={ablated['peak_D']:.3f}")
    print(f"  Edges removed: {n_removed}")

    # Summary
    rows = [
        dict(scenario="Baseline", mean_tinv=baseline["mean_tinv"],
             peak_D=baseline["peak_D"], frac_invaded=baseline["frac_invaded"],
             delta_tinv=0.0),
        dict(scenario="Vascular edges removed", mean_tinv=ablated["mean_tinv"],
             peak_D=ablated["peak_D"], frac_invaded=ablated["frac_invaded"],
             delta_tinv=ablated["mean_tinv"] - baseline["mean_tinv"]),
    ]
    df_summary = pd.DataFrame(rows)
    df_summary.to_csv(os.path.join(save_dir, "..", "tables",
                                   "control3_vascular_removal_visium.csv"), index=False)

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot(baseline["t"], baseline["D_t"].mean(axis=0), color="#2166ac",
            lw=2.0, label="Baseline")
    ax.plot(ablated["t"], ablated["D_t"].mean(axis=0), color="#d73027",
            lw=2.0, linestyle="--", label="Vascular edges removed")
    ax.set_xlabel("Simulation time", fontsize=10)
    ax.set_ylabel(r"Mean $D(t)$", fontsize=10)
    ax.set_title("(A) Diseased fraction dynamics", fontsize=10, fontweight="bold")
    ax.legend(fontsize=9); ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    macros = list(baseline["macro_tinv"].keys())
    x = np.arange(len(macros)); width = 0.35
    ax.bar(x - width/2, [baseline["macro_tinv"].get(m, 0) for m in macros], width,
           color="#2166ac", label="Baseline", edgecolor="white")
    ax.bar(x + width/2, [ablated["macro_tinv"].get(m, 0) for m in macros], width,
           color="#d73027", label="Vascular removed", edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(macros, fontsize=10, rotation=30)
    ax.set_ylabel(r"Mean invasion time", fontsize=10)
    ax.set_title("(B) Invasion time per macro-type", fontsize=10, fontweight="bold")
    ax.legend(fontsize=9); ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"Control 3: Vascular edge removal ({n_removed} edges removed, Visium graph)",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig_path = os.path.join(save_dir, "control3_vascular_removal_visium")
    plt.savefig(fig_path + ".png", dpi=dpi, bbox_inches="tight")
    plt.savefig(fig_path + ".svg", bbox_inches="tight")
    plt.close()

    return dict(baseline=baseline, ablated=ablated, df_summary=df_summary,
                n_removed=n_removed)


# ═══════════════════════════════════════════════════════════════════════════
# CONTROL 4: ALTERNATIVE SEED LOCATIONS
# ═══════════════════════════════════════════════════════════════════════════

def run_alternative_seeds_visium(
    comp:        dict,
    sir_params:  dict,
    seed_I:      float = 0.05,
    t_end:       float = 60.0,
    n_steps:     int   = 400,
    threshold_I: float = 0.10,
    save_dir:    str   = "./results/figures",
    dpi:         int   = 200,
) -> dict:
    """
    Control 4: Alternative seed locations on the Visium compartment graph.

    Tests cortex-only, inner-medulla-only, and single-compartment seeds
    in addition to the default outer-medulla seeding.
    """
    os.makedirs(save_dir, exist_ok=True)
    print("\n[Control 4] Alternative seed locations (Visium graph)")

    comps2   = comp["comps2"]
    macro_of = comp["macro_of"]
    region_of = comp["region_of"]

    # Define seed scenarios as boundary masks
    def _mask_by_region(regions):
        return np.array([r in regions for r in region_of])

    def _mask_by_comp(name):
        return np.array([c == name for c in comps2])

    def _mask_by_macro_region(macro, regions):
        return np.array([(m == macro and r in regions)
                         for m, r in zip(macro_of, region_of)])

    seed_scenarios = {
        "Outer medulla (default)": comp["boundary_mask"].copy(),
        "Cortex only":             _mask_by_region(["cortex"]),
        "Inner medulla only":      _mask_by_region(["inner_medulla"]),
        "PT outer medulla only":   _mask_by_macro_region("PT", ["outer_medulla"]),
        "Vascular only":           _mask_by_macro_region("vascular", ["outer_medulla", "cortex", "inner_medulla"]),
        "PT cortex only":          _mask_by_macro_region("PT", ["cortex"]),
    }

    results = {}
    for label, mask in seed_scenarios.items():
        if not mask.any():
            print(f"  {label:30s}: NO compartments match — skip")
            continue
        res = _run_and_extract_visium(
            comp, sir_params, seed_I=seed_I, t_end=t_end, n_steps=n_steps,
            threshold_I=threshold_I, label=label,
            boundary_mask_override=mask,
        )
        results[label] = res
        print(f"  {label:30s}: t_inv={res['mean_tinv']:.3f}, peak_D={res['peak_D']:.3f}")

    # Spearman vs baseline
    baseline_key = "Outer medulla (default)"
    if baseline_key not in results:
        baseline_key = list(results.keys())[0]
    baseline_order = np.array([results[baseline_key]["macro_tinv"].get(m, np.nan)
                               for m in results[baseline_key]["macro_tinv"]])
    macros = list(results[baseline_key]["macro_tinv"].keys())

    spearman_vals = {}
    for label, res in results.items():
        if label == baseline_key:
            spearman_vals[label] = 1.0
        else:
            alt_order = np.array([res["macro_tinv"].get(m, np.nan) for m in macros])
            valid = ~(np.isnan(baseline_order) | np.isnan(alt_order))
            if valid.sum() >= 3:
                spearman_vals[label] = spearmanr(baseline_order[valid],
                                                  alt_order[valid]).statistic
            else:
                spearman_vals[label] = np.nan

    # Summary table
    rows = []
    for label, res in results.items():
        rows.append(dict(
            scenario=label, mean_tinv=res["mean_tinv"], peak_D=res["peak_D"],
            frac_invaded=res["frac_invaded"],
            spearman_vs_baseline=spearman_vals.get(label, np.nan),
            n_seeds=int(seed_scenarios[label].sum()),
        ))
    df_summary = pd.DataFrame(rows)
    df_summary.to_csv(os.path.join(save_dir, "..", "tables",
                                   "control4_alternative_seeds_visium.csv"), index=False)

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    labels = list(results.keys())
    tinvs = [results[l]["mean_tinv"] for l in labels]
    spearmans = [spearman_vals.get(l, np.nan) for l in labels]
    colors = ["#2166ac"] + ["#4dac26"] * (len(labels) - 1)

    ax = axes[0]
    ax.barh(range(len(labels)), tinvs, color=colors, edgecolor="white", height=0.6)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel(r"Mean invasion time $\bar{t}_{\mathrm{inv}}$", fontsize=10)
    ax.set_title("(A) Invasion time per seed scenario", fontsize=10, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    valid_s = [(l, s) for l, s in zip(labels, spearmans) if not np.isnan(s)]
    if valid_s:
        ax.barh(range(len(valid_s)), [s for _, s in valid_s],
                color=["#2166ac"] + ["#f1a340"] * (len(valid_s) - 1),
                edgecolor="white", height=0.6)
        ax.set_yticks(range(len(valid_s)))
        ax.set_yticklabels([l for l, _ in valid_s], fontsize=9)
        ax.axvline(0.8, color="#d73027", lw=1.5, linestyle="--", label="r_s=0.8")
    ax.set_xlabel(r"Spearman $r_s$ vs baseline", fontsize=10)
    ax.set_title("(B) Activation order stability", fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Control 4: Alternative seed locations (Visium graph)",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig_path = os.path.join(save_dir, "control4_alternative_seeds_visium")
    plt.savefig(fig_path + ".png", dpi=dpi, bbox_inches="tight")
    plt.savefig(fig_path + ".svg", bbox_inches="tight")
    plt.close()

    return dict(results=results, df_summary=df_summary, spearman_vals=spearman_vals)


# ═══════════════════════════════════════════════════════════════════════════
# CONTROL 5: R-STATE SENSITIVITY ANALYSIS (R1 point 5b)
# ═══════════════════════════════════════════════════════════════════════════

def run_R_state_sensitivity(
    comp:        dict,
    sir_params:  dict,
    seed_I:      float = 0.05,
    t_end:       float = 60.0,
    n_steps:     int   = 400,
    threshold_I: float = 0.10,
    rho_multipliers: list = None,
    save_dir:    str   = "./results/figures",
    dpi:         int   = 200,
) -> dict:
    """
    Control 5: R-state sensitivity analysis (Reviewer 1, Point 5b).

    The reviewer notes that the recovered (R) state is not directly
    observable in real data, yet the manuscript reports precise numerical
    values (e.g., p_fin^R = 0.946). This control varies the recovery rate
    ρ by ±50% and measures how p_fin^R changes.

    If R predictions are highly sensitive to ρ, the reported values should
    be interpreted as model predictions conditional on recovery assumptions,
    not as measurements.
    """
    os.makedirs(save_dir, exist_ok=True)
    print("\n[Control 5] R-state sensitivity analysis (R1 point 5b)")

    if rho_multipliers is None:
        rho_multipliers = [0.50, 0.75, 1.00, 1.25, 1.50]

    baseline_R_fin = None
    results = []

    for mult in rho_multipliers:
        # Modify recovery rate
        mod_params = {k: v.copy() if isinstance(v, np.ndarray) else v
                      for k, v in sir_params.items()}
        mod_params["gamma"] = sir_params["gamma"] * mult

        res = _run_and_extract_visium(
            comp, mod_params, seed_I=seed_I, t_end=t_end, n_steps=n_steps,
            threshold_I=threshold_I, label=f"ρ × {mult:.2f}",
        )

        # Final R per macro-type
        R_fin = res["R_fin"]
        macro_of = comp["macro_of"]
        R_by_macro = {}
        for m in np.unique(macro_of):
            mask = macro_of == m
            R_by_macro[m] = float(R_fin[mask].mean())

        results.append(dict(
            rho_mult=mult, mean_tinv=res["mean_tinv"], peak_D=res["peak_D"],
            global_R=float(R_fin.mean()), R_by_macro=R_by_macro,
        ))

        if mult == 1.0:
            baseline_R_fin = float(R_fin.mean())

        print(f"  ρ × {mult:.2f}: R_fin={R_fin.mean():.4f}, "
              f"t_inv={res['mean_tinv']:.3f}, peak_D={res['peak_D']:.3f}")

    # Sensitivity: how much does R_fin change with ρ?
    R_vals = [r["global_R"] for r in results]
    R_range = max(R_vals) - min(R_vals)
    R_mean = np.mean(R_vals)

    print(f"\n  R_fin range across ρ sweep: [{min(R_vals):.4f}, {max(R_vals):.4f}]")
    print(f"  R_fin sensitivity (range/mean): {R_range/R_mean:.3f}")
    print(f"  → R predictions are {'HIGHLY' if R_range/R_mean > 0.2 else 'moderately' if R_range/R_mean > 0.1 else 'weakly'}"
          f" sensitive to ρ assumptions")

    # Summary table
    rows = []
    for r in results:
        row = dict(rho_mult=r["rho_mult"], global_R=r["global_R"],
                    mean_tinv=r["mean_tinv"], peak_D=r["peak_D"])
        for m, v in r["R_by_macro"].items():
            row[f"R_{m}"] = v
        rows.append(row)
    df_summary = pd.DataFrame(rows)
    df_summary.to_csv(os.path.join(save_dir, "..", "tables",
                                   "control5_R_sensitivity.csv"), index=False)

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    mults = [r["rho_mult"] for r in results]
    ax.plot(mults, R_vals, "o-", color="#2166ac", lw=2, markersize=8)
    if baseline_R_fin is not None:
        ax.axhline(baseline_R_fin, color="#d73027", ls="--", lw=1.5,
                   label=f"Baseline R = {baseline_R_fin:.3f}")
    ax.set_xlabel(r"Recovery rate multiplier ($\rho / \rho_0$)", fontsize=10)
    ax.set_ylabel(r"Final recovered fraction $p^R_{\mathrm{fin}}$", fontsize=10)
    ax.set_title("(A) R-state sensitivity to recovery rate\n"
                 "(R1 point 5b: R is not directly observable)",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=9); ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    macros_to_plot = ["PT", "DCT", "TAL", "vascular", "immune"]
    colors = ["#c0392b", "#2471a3", "#1e8449", "#7d3c98", "#d35400"]
    for m, c in zip(macros_to_plot, colors):
        vals = [r["R_by_macro"].get(m, np.nan) for r in results]
        if not all(np.isnan(v) for v in vals):
            ax.plot(mults, vals, "o-", color=c, lw=1.5, markersize=6, label=m)
    ax.set_xlabel(r"Recovery rate multiplier ($\rho / \rho_0$)", fontsize=10)
    ax.set_ylabel(r"$p^R_{\mathrm{fin}}$ per macro-type", fontsize=10)
    ax.set_title("(B) R-state sensitivity per macro-type", fontsize=10, fontweight="bold")
    ax.legend(fontsize=9); ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Control 5: R-state sensitivity (R1 point 5b) — Visium graph",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig_path = os.path.join(save_dir, "control5_R_sensitivity")
    plt.savefig(fig_path + ".png", dpi=dpi, bbox_inches="tight")
    plt.savefig(fig_path + ".svg", bbox_inches="tight")
    plt.close()

    return dict(results=results, df_summary=df_summary,
                R_range=R_range, R_sensitivity=R_range/R_mean)


# ═══════════════════════════════════════════════════════════════════════════
# MASTER RUNNER — VISIUM PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def run_all_controls_visium(
    comp:        dict,
    sir_params:  dict,
    seed_I:      float = 0.05,
    t_end:       float = 60.0,
    n_steps:     int   = 400,
    threshold_I: float = 0.10,
    n_rewires:   int   = 20,
    save_dir:    str   = "./results/figures",
    dpi:         int   = 200,
) -> dict:
    """
    Run all 5 control experiments on the Visium-derived compartment graph.

    Parameters
    ----------
    comp        : output of build_compartment2 (from main.py Step 4)
    sir_params  : output of build_default_params_SIR (from main.py Step 5)
    seed_I      : initial diseased fraction (default 0.05, matching paper)
    t_end       : simulation horizon
    n_steps     : ODE solver steps
    threshold_I : invasion threshold

    Returns
    -------
    dict with keys: null_model, uniform_params, vascular_removal,
                    alternative_seeds, R_sensitivity
    """
    print("=" * 60)
    print("CONTROL EXPERIMENTS — Visium pipeline (R2 point 6, R1 point 5b)")
    print("=" * 60)

    res1 = run_randomized_edge_null_model_visium(
        comp, sir_params, seed_I=seed_I, t_end=t_end, n_steps=n_steps,
        threshold_I=threshold_I, n_rewires=n_rewires,
        save_dir=save_dir, dpi=dpi)

    res2 = run_uniform_parameters_control_visium(
        comp, sir_params, seed_I=seed_I, t_end=t_end, n_steps=n_steps,
        threshold_I=threshold_I, save_dir=save_dir, dpi=dpi)

    res3 = run_vascular_edge_removal_visium(
        comp, sir_params, seed_I=seed_I, t_end=t_end, n_steps=n_steps,
        threshold_I=threshold_I, save_dir=save_dir, dpi=dpi)

    res4 = run_alternative_seeds_visium(
        comp, sir_params, seed_I=seed_I, t_end=t_end, n_steps=n_steps,
        threshold_I=threshold_I, save_dir=save_dir, dpi=dpi)

    res5 = run_R_state_sensitivity(
        comp, sir_params, seed_I=seed_I, t_end=t_end, n_steps=n_steps,
        threshold_I=threshold_I, save_dir=save_dir, dpi=dpi)

    print("\n" + "=" * 60)
    print("ALL CONTROLS COMPLETE (Visium pipeline)")
    print("=" * 60)

    return dict(null_model=res1, uniform_params=res2,
                vascular_removal=res3, alternative_seeds=res4,
                R_sensitivity=res5)


# ═══════════════════════════════════════════════════════════════════════════
# SUPPLEMENTARY: SECTION 7 SIMPLIFIED MODEL CONTROLS
# ═══════════════════════════════════════════════════════════════════════════

def run_all_controls_section7(
    save_dir: str = "./results/figures",
    dpi:      int = 200,
) -> dict:
    """
    Run controls on the simplified 13-compartment model (sir_section7.py).

    This is a supplementary analysis for cross-checking. The primary
    controls are run on the Visium pipeline (run_all_controls_visium).
    """
    from sir_section7 import (
        COMPARTMENTS, N, C_IDX, MACRO_MAP,
        BETA_BASE, RHO_BASE, DD_BASE, OUTER_MULT, GAMMA_OUTER_MULT,
        build_params, build_adjacency_from_edges,
        W_raw, W_norm, make_ode, run_sim, compute_metrics, global_metrics,
        make_ic, SPATIAL_SEED, CENTRAL_SEED,
        ZONE_BOUNDARY, ZONE_OUTER_MED, ZONE_INNER_MED, ZONE_CORTEX,
        BASE_EDGES,
    )
    from scipy.stats import spearmanr
    import scipy.sparse as sp

    os.makedirs(save_dir, exist_ok=True)
    tables_dir = os.path.join(save_dir, "..", "tables")
    os.makedirs(tables_dir, exist_ok=True)
    print("=" * 60)
    print("CONTROL EXPERIMENTS \u2014 Section 7 simplified model (supplementary)")
    print("=" * 60)

    beta, rho, DS, DD = build_params()
    y0 = make_ic(SPATIAL_SEED)

    # Baseline
    t_base, pH_base, pD_base, pR_base = run_sim(W_norm, beta, rho, DS, DD, y0, t_end=60.0)
    tinv_base, peakD_base, frac_base = global_metrics(t_base, pD_base)
    print(f"\n  Baseline: t_inv={tinv_base:.3f}, peak_pD={peakD_base:.3f}")

    # Helper: macro-type invasion order from a simulation
    macros_list = ["PT", "DCT", "TAL", "vascular", "other"]
    def _macro_tinv(t, pD):
        df_m = compute_metrics(t, pD, np.clip(1 - pD - 0, 0, 1))  # pR not needed for tcross
        result = {}
        for macro in macros_list:
            mask = [MACRO_MAP[c] == macro for c in COMPARTMENTS]
            vals = df_m["tcross"].values[mask]
            valid = vals[~np.isnan(vals)]
            result[macro] = float(valid.mean()) if len(valid) > 0 else np.nan
        return result

    baseline_macro_tinv = _macro_tinv(t_base, pD_base)

    # --- Control 1: Maslov-Sneppen rewiring ---
    print("\n  [C1] Maslov-Sneppen rewiring (n=20)...")
    rng = np.random.default_rng(42)
    tinv_rewired = []
    spearman_rewired = []
    peakD_rewired = []

    # W_raw is a dense numpy array; convert to sparse for rewiring
    W_raw_sparse = sp.csr_matrix(W_raw)

    for i in range(20):
        W_rew_sparse = maslov_sneppen_rewire_sparse(W_raw_sparse, 50, rng)
        W_rew_dense = np.asarray(W_rew_sparse.todense())
        # Row-normalize (dense)
        rs = W_rew_dense.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0
        W_rew_norm = W_rew_dense / rs

        t_r, _, pD_r, _ = run_sim(W_rew_norm, beta, rho, DS, DD, y0, t_end=60.0)
        tinv_r, peakD_r, _ = global_metrics(t_r, pD_r)
        tinv_rewired.append(tinv_r)
        peakD_rewired.append(peakD_r)

        rewired_macro = _macro_tinv(t_r, pD_r)
        baseline_order = []
        rewired_order = []
        for macro in macros_list:
            b_val = baseline_macro_tinv[macro]
            r_val = rewired_macro[macro]
            if not (np.isnan(b_val) or np.isnan(r_val)):
                baseline_order.append(b_val)
                rewired_order.append(r_val)
        if len(rewired_order) >= 3:
            spearman_rewired.append(spearmanr(baseline_order, rewired_order).statistic)

    print(f"    Rewired t_inv: {np.mean(tinv_rewired):.3f} \u00b1 {np.std(tinv_rewired):.3f}")
    print(f"    Rewired peak_pD: {np.mean(peakD_rewired):.3f} \u00b1 {np.std(peakD_rewired):.3f}")
    print(f"    Spearman r_s: {np.mean(spearman_rewired):.3f} \u00b1 {np.std(spearman_rewired):.3f}")

    # --- Control 2: Uniform parameters ---
    print("\n  [C2] Uniform parameters...")
    beta_uni = np.full(N, beta.mean())
    rho_uni = np.full(N, rho.mean())
    DD_uni = np.full(N, DD.mean())
    t_u, _, pD_u, _ = run_sim(W_norm, beta_uni, rho_uni, DS, DD_uni, y0, t_end=60.0)
    tinv_u, peakD_u, _ = global_metrics(t_u, pD_u)
    print(f"    Uniform: t_inv={tinv_u:.3f}, peak_pD={peakD_u:.3f} (baseline: {tinv_base:.3f}, {peakD_base:.3f})")

    # --- Control 3: Vascular edge removal ---
    print("\n  [C3] Vascular edge removal...")
    vascular_comps = [c for c in COMPARTMENTS if MACRO_MAP[c] == "vascular"]
    vascular_idx = set(C_IDX[c] for c in vascular_comps)
    ablated_edges = [(a, b, w) for a, b, w in BASE_EDGES
                     if C_IDX[a] not in vascular_idx and C_IDX[b] not in vascular_idx]
    _, W_abl_norm = build_adjacency_from_edges(ablated_edges)
    t_a, _, pD_a, _ = run_sim(W_abl_norm, beta, rho, DS, DD, y0, t_end=60.0)
    tinv_a, peakD_a, _ = global_metrics(t_a, pD_a)
    n_removed = len(BASE_EDGES) - len(ablated_edges)
    print(f"    Vascular removed: t_inv={tinv_a:.3f}, peak_pD={peakD_a:.3f} ({n_removed} edges removed)")

    # --- Control 4: Alternative seeds ---
    print("\n  [C4] Alternative seeds...")
    seed_scenarios = {
        "Spatial boundary (default)": SPATIAL_SEED,
        "Cortex only": ZONE_CORTEX,
        "Inner medulla only": ZONE_INNER_MED,
        "PT_boundary only": ["PT_boundary"],
        "Vascular boundary only": ["vascular_boundary"],
    }
    seed_results = {}
    for label, seeds in seed_scenarios.items():
        y0_s = make_ic(seeds)
        t_s, _, pD_s, _ = run_sim(W_norm, beta, rho, DS, DD, y0_s, t_end=60.0)
        tinv_s, peakD_s, _ = global_metrics(t_s, pD_s)
        seed_results[label] = dict(tinv=tinv_s, peakD=peakD_s)
        print(f"    {label:30s}: t_inv={tinv_s:.3f}, peak_pD={peakD_s:.3f}")

    # --- Control 5: R-state sensitivity ---
    print("\n  [C5] R-state sensitivity...")
    R_sens = []
    for mult in [0.50, 0.75, 1.00, 1.25, 1.50]:
        rho_mod = rho * mult
        t_m, _, pD_m, pR_m = run_sim(W_norm, beta, rho_mod, DS, DD, y0, t_end=60.0)
        R_fin = float(pR_m[:, -1].mean())
        R_sens.append(dict(rho_mult=mult, R_fin=R_fin))
        print(f"    \u03c1 \u00d7 {mult:.2f}: R_fin={R_fin:.4f}")

    # ── Save summary tables ──────────────────────────────────────────────
    # Control 1 summary
    df_c1 = pd.DataFrame([
        dict(scenario="Baseline", mean_tinv=tinv_base, peak_D=peakD_base,
             spearman_vs_baseline=1.0),
        *[dict(scenario=f"Rewired_{i+1}", mean_tinv=t, peak_D=p,
               spearman_vs_baseline=spearman_rewired[i] if i < len(spearman_rewired) else np.nan)
          for i, (t, p) in enumerate(zip(tinv_rewired, peakD_rewired))]
    ])
    df_c1.to_csv(os.path.join(tables_dir, "control1_null_model_section7.csv"), index=False)

    # Control 2 summary
    df_c2 = pd.DataFrame([
        dict(scenario="Heterogeneous", mean_tinv=tinv_base, peak_D=peakD_base),
        dict(scenario="Uniform", mean_tinv=tinv_u, peak_D=peakD_u),
    ])
    df_c2.to_csv(os.path.join(tables_dir, "control2_uniform_params_section7.csv"), index=False)

    # Control 3 summary
    df_c3 = pd.DataFrame([
        dict(scenario="Baseline", mean_tinv=tinv_base, peak_D=peakD_base, delta_tinv=0.0),
        dict(scenario="Vascular edges removed", mean_tinv=tinv_a, peak_D=peakD_a,
             delta_tinv=tinv_a - tinv_base),
    ])
    df_c3.to_csv(os.path.join(tables_dir, "control3_vascular_removal_section7.csv"), index=False)

    # Control 4 summary
    df_c4 = pd.DataFrame([
        dict(scenario=label, mean_tinv=r["tinv"], peak_D=r["peakD"])
        for label, r in seed_results.items()
    ])
    df_c4.to_csv(os.path.join(tables_dir, "control4_alternative_seeds_section7.csv"), index=False)

    # Control 5 summary
    df_c5 = pd.DataFrame(R_sens)
    df_c5.to_csv(os.path.join(tables_dir, "control5_R_sensitivity_section7.csv"), index=False)

    print("\n" + "=" * 60)
    print("ALL CONTROLS COMPLETE (Section 7 simplified model)")
    print("=" * 60)

    return dict(
        baseline=dict(tinv=tinv_base, peakD=peakD_base),
        null_model=dict(tinv_rewired=tinv_rewired, peakD_rewired=peakD_rewired,
                        spearman=spearman_rewired),
        uniform=dict(tinv=tinv_u, peakD=peakD_u),
        vascular_removal=dict(tinv=tinv_a, peakD=peakD_a, n_removed=n_removed),
        alternative_seeds=seed_results,
        R_sensitivity=R_sens,
    )

# Backward-compatible alias
run_all_controls = run_all_controls_visium
