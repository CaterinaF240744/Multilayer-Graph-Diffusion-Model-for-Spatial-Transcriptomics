"""
sir_transcompartment_metrics.py
================================
Trans-compartment invasion metrics for a spatial SIR model defined on a
weighted graph. Quantifies how inflammatory signals propagate *between*
biologically defined tissue compartments.

All functions work at compartment level (C compartments). Spatial maps
are obtained downstream via broadcasting through inv2.

Metrics implemented
-------------------
1. next_generation_matrix_dynamic   — K_AB^dyn  (integral over time)
2. structural_invasion_matrix       — K_AB  and  K_AB^norm  (static)
                                      NOW supports per-compartment β, γ
3. source_sink_scores               — source(A), sink(B)
4. retention_dispersal              — retention(A), dispersal(A)
5. invasion_entropy                 — H(t), mean H, peak H
6. spatial_propagation_distance     — d(t) weighted by I_i(t)
7. cross_compartment_flux           — F_{A→B}(t) + integrated matrix

Changes vs previous version
-----------------------------
- next_generation_matrix_dynamic: fully vectorised (no Python loop over T)
  using einsum — O(T·C²) instead of O(T·C²·|A|·|B|) with inner loop
- structural_invasion_matrix: accepts beta/gamma as ndarray (C,) per node
  in addition to scalar; uses per-node R0 = beta_i / gamma_i in K_AB
- cross_compartment_flux: vectorised over T with einsum
- np.trapezoid / np.trapezoid compatibility shim for numpy < 2.0

Output utilities
----------------
matrix_to_dataframe    — wraps numpy matrix with compartment labels
plot_invasion_matrix   — annotated heatmap of any (C,C) matrix
plot_entropy           — entropy time series
plot_source_sink       — scatter source vs sink
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# ── numpy compat shim ────────────────────────────────────────────────────────
try:
    _trapz = np.trapezoid
except AttributeError:
    _trapz = np.trapezoid   # numpy < 2.0


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _extract_SIR(sol):
    """
    Returns S (T,N), I (T,N), t (T,) from either:
      - scipy OdeResult  (sol.y shape (3C, T), sol.t shape (T,))
      - dict             with keys 't', 'S', 'I', 'R'  (each (C,T))
    """
    if hasattr(sol, "t"):
        t  = sol.t
        nC = sol.y.shape[0] // 3
        S  = sol.y[:nC,     :].T   # (T, C)
        I  = sol.y[nC:2*nC, :].T
    else:
        t = sol["t"]
        S = sol["S"].T
        I = sol["I"].T
    return S, I, t


def _to_dense(W) -> np.ndarray:
    if sp.issparse(W):
        return W.toarray().astype(np.float64)
    return np.asarray(W, dtype=np.float64)


# ═══════════════════════════════════════════════════════════════════════════
# 1. DYNAMIC NEXT-GENERATION INVASION MATRIX  (vectorised)
# ═══════════════════════════════════════════════════════════════════════════

def _receiver_flux_blocks(S, I, t, E, labels, beta):
    """Yield (ai, bi, integrand(t)) for the model's own transmission flux.

    In the model (Eqs. 2-4) node j becomes diseased at rate
        beta_j * p^H_j * lambda_j,   lambda_j = sum_i E_ji p^D_i,
    so the flux from source i to receiver j is
        F_{i->j}(t) = beta_j * E_ji * p^D_i(t) * p^H_j(t).
    R3: before R3 the code used beta_i (source) and the raw adjacency W_ij,
    i.e. a quantity that does not appear in the dynamics.
    """
    E = _to_dense(E)
    N = E.shape[0]
    beta_vec = (np.full(N, float(beta)) if np.isscalar(beta)
                else np.asarray(beta, dtype=np.float64))
    comps = np.unique(labels)
    comp_idx = {c: np.where(labels == c)[0] for c in comps}
    for ai, cA in enumerate(comps):
        idx_A = comp_idx[cA]
        I_A = I[:, idx_A]                                   # (T, |A|)
        for bi, cB in enumerate(comps):
            idx_B = comp_idx[cB]
            E_BA = E[np.ix_(idx_B, idx_A)]                  # receiver rows
            bS_B = S[:, idx_B] * beta_vec[idx_B][None, :]   # (T, |B|)
            yield ai, bi, ((I_A @ E_BA.T) * bS_B).sum(axis=1)


def next_generation_matrix_dynamic(
    S:      np.ndarray,
    I:      np.ndarray,
    t:      np.ndarray,
    W,
    labels: np.ndarray,
    beta,
    check:  bool = True,
) -> pd.DataFrame:
    """
    Compartment-to-compartment transmission matrix integrated over time.

        K_AB = int_0^T  sum_{i in A} sum_{j in B}  beta_j E_ji p^D_i(t) p^H_j(t) dt

    W must be the EXPOSURE matrix E used in the simulation (row j = receiver),
    e.g. blockdiag(W~[alpha]) + W_inter for the multilayer model, and beta the
    per-node susceptibility. K_AB is then an exact attribution of the infection
    that occurred in B to its sources: sum_A K_AB = sum_{j in B}
    (p^H_j(0) - p^H_j(T)). This identity is checked when check=True.

    Returns pd.DataFrame (C, C), rows = source, columns = receiver.
    """
    comps = np.unique(labels)
    C = len(comps)
    K = np.zeros((C, C), dtype=np.float64)
    for ai, bi, integrand in _receiver_flux_blocks(S, I, t, W, labels, beta):
        K[ai, bi] = _trapz(integrand, t)
    if check:
        lost_H = np.array([(S[0, labels == c] - S[-1, labels == c]).sum()
                           for c in comps])
        err = np.abs(K.sum(axis=0) - lost_H) / np.maximum(lost_H, 1e-12)
        if np.max(err) > 0.02:   # trapezoid error on the output grid
            warnings.warn(
                "K_dyn column sums differ from the infected mass by up to "
                f"{100*np.max(err):.1f}%: is W the exposure matrix used in "
                "the simulation? Increase n_steps if it is.", RuntimeWarning)
    return pd.DataFrame(K, index=comps, columns=comps)


def next_generation_matrix_dynamic_from_sol(
    sol,
    W,
    labels: np.ndarray,
    beta,
) -> pd.DataFrame:
    """Convenience wrapper that accepts a sol object directly."""
    S, I, t = _extract_SIR(sol)
    return next_generation_matrix_dynamic(S, I, t, W, labels, beta)


# ═══════════════════════════════════════════════════════════════════════════
# 2. STRUCTURAL INVASION POTENTIAL MATRIX  (per-node β/γ support)
# ═══════════════════════════════════════════════════════════════════════════

def structural_invasion_matrix(
    W,
    labels: np.ndarray,
    beta,
    gamma,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Static next-generation matrix with per-node or scalar β, γ.

    For each pair (A, B) (R3: receiver susceptibility, source recovery):

        K_AB = Σ_{i∈A} Σ_{j∈B}  β_j · E_ji / γ_i

    i.e. new infections caused in B by one unit of disease in A over its
    mean infectious period 1/γ_i. W must be the exposure matrix E.

    When beta/gamma are scalars the formula reduces to the original
        K_AB = (β/γ) · Σ_{i∈A} Σ_{j∈B} W_ij

    K_AB^norm = K_AB / Σ_j K_Aj   (fraction of A's output going to B)

    Parameters
    ----------
    W      : (N, N) weight matrix
    labels : (N,) compartment label
    beta   : float scalar OR ndarray (N,) per-node
    gamma  : float scalar OR ndarray (N,) per-node

    Returns
    -------
    (K_df, K_norm_df) — both pd.DataFrame (C, C)
    """
    W = _to_dense(W)
    N = W.shape[0]

    if np.isscalar(beta):
        beta_vec  = np.full(N, float(beta))
    else:
        beta_vec  = np.asarray(beta,  dtype=np.float64)
    if np.isscalar(gamma):
        gamma_vec = np.full(N, float(gamma))
    else:
        gamma_vec = np.asarray(gamma, dtype=np.float64)

    inv_gamma = 1.0 / np.where(gamma_vec > 0, gamma_vec, 1e-12)   # (N,)

    comps    = np.unique(labels)
    C        = len(comps)
    comp_idx = {c: np.where(labels == c)[0] for c in comps}

    K = np.zeros((C, C), dtype=np.float64)
    for ai, cA in enumerate(comps):
        idx_A  = comp_idx[cA]
        for bi, cB in enumerate(comps):
            idx_B  = comp_idx[cB]
            E_BA   = W[np.ix_(idx_B, idx_A)]   # receiver rows (|B|, |A|)
            K[ai, bi] = float(beta_vec[idx_B] @ E_BA @ inv_gamma[idx_A])

    row_totals = K.sum(axis=1, keepdims=True)
    K_norm     = np.where(row_totals > 0, K / row_totals, 0.0)

    K_df      = pd.DataFrame(K,      index=comps, columns=comps)
    K_norm_df = pd.DataFrame(K_norm, index=comps, columns=comps)
    return K_df, K_norm_df


# ═══════════════════════════════════════════════════════════════════════════
# 3. SOURCE AND SINK SCORES
# ═══════════════════════════════════════════════════════════════════════════

def source_sink_scores(K_df: pd.DataFrame, labels=None) -> pd.DataFrame:
    """
    Per-compartment source and receiver strengths from a transmission matrix.

        Source(A) = Σ_B K_AB   (row sum: infection A causes, incl. itself)
        Sink(B)   = Σ_A K_AB   (column sum: infection B receives)

    R3 additions (labels = node labels, to normalise by layer size):
        *_per_node   : the same divided by the number of nodes in the layer
        inter_out    : Σ_{B≠A} K_AB   (exported to other layers)
        inter_in     : Σ_{B≠A} K_BA   (imported from other layers)
        net_export   : inter_out - inter_in  (> 0 net exporter, < 0 net
                       importer). A net importer is a RECEIVER; it is an
                       absorber only if it also attenuates onward spread
                       (local R_diff < 1), which no tubular layer does.
    """
    K = K_df.values
    off = K - np.diag(np.diag(K))
    out = pd.DataFrame({
        "compartment":     K_df.index,
        "source_strength": K.sum(axis=1),
        "sink_strength":   K.sum(axis=0),
        "inter_out":       off.sum(axis=1),
        "inter_in":        off.sum(axis=0),
    })
    out["net_export"] = out["inter_out"] - out["inter_in"]
    if labels is not None:
        n = pd.Series(np.asarray(labels)).value_counts()
        out["n_nodes"] = out["compartment"].map(n).values
        out["source_per_node"] = out["source_strength"] / out["n_nodes"]
        out["sink_per_node"] = out["sink_strength"] / out["n_nodes"]
    return out.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════
# 4. SELF-RETENTION AND DISPERSAL INDICES
# ═══════════════════════════════════════════════════════════════════════════

def retention_dispersal(K_df: pd.DataFrame) -> pd.DataFrame:
    """
    Self-retention and dispersal for each compartment.

        Retention(A) = K_AA / Σ_B K_AB
        Dispersal(A) = 1 − Retention(A)

    Returns
    -------
    pd.DataFrame with columns: compartment, retention, dispersal
    """
    row_sums  = K_df.sum(axis=1)
    diag      = pd.Series(np.diag(K_df.values), index=K_df.index)
    retention = np.where(row_sums > 0, diag / row_sums, 0.0)
    dispersal = 1.0 - retention
    return pd.DataFrame({
        "compartment": K_df.index,
        "retention":   retention,
        "dispersal":   dispersal,
    }).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════
# 5. INVASION ENTROPY
# ═══════════════════════════════════════════════════════════════════════════

def invasion_entropy(
    I:      np.ndarray,
    t:      np.ndarray,
    labels: np.ndarray,
) -> dict:
    """
    Shannon entropy of the infection distribution across compartments.

        p_c(t)  = Σ_{i∈c} I_i(t)  /  Σ_i I_i(t)
        H(t)    = -Σ_c  p_c(t) log(p_c(t))

    Returns
    -------
    dict: H (T,), t (T,), mean_H, peak_H, t_peak_H, compartments
    """
    comps = np.unique(labels)
    T     = len(t)
    H     = np.zeros(T, dtype=np.float64)

    for k in range(T):
        I_total = I[k].sum()
        if I_total <= 0:
            continue
        p      = np.array([I[k, labels == c].sum() / I_total for c in comps])
        p_safe = np.clip(p, 1e-15, None)
        H[k]   = -np.sum(p * np.log(p_safe))

    return {
        "H":            H,
        "t":            t,
        "mean_H":       float(H.mean()),
        "peak_H":       float(H.max()),
        "t_peak_H":     float(t[np.argmax(H)]),
        "compartments": list(comps),
    }


def invasion_entropy_from_sol(sol, labels: np.ndarray) -> dict:
    S, I, t = _extract_SIR(sol)
    return invasion_entropy(I, t, labels)


# ═══════════════════════════════════════════════════════════════════════════
# 6. SPATIAL PROPAGATION DISTANCE
# ═══════════════════════════════════════════════════════════════════════════

def spatial_propagation_distance(
    I:        np.ndarray,
    t:        np.ndarray,
    coords:   np.ndarray,
    seed_idx: int | None = None,
) -> dict:
    """
    Infection-weighted mean distance from the seed.

        d(t) = Σ_i I_i(t) · d(i, seed)  /  Σ_i I_i(t)

    Returns
    -------
    dict: d (T,), t (T,), seed_idx, max_d, t_max_d
    """
    coords = np.asarray(coords, dtype=np.float64)
    if seed_idx is None:
        seed_idx = int(np.argmax(I[0]))

    dist_to_seed = np.linalg.norm(coords - coords[seed_idx], axis=1)
    T = len(t)
    d = np.zeros(T, dtype=np.float64)
    for k in range(T):
        I_total = I[k].sum()
        if I_total > 0:
            d[k] = (I[k] * dist_to_seed).sum() / I_total

    return {
        "d":        d,
        "t":        t,
        "seed_idx": seed_idx,
        "max_d":    float(d.max()),
        "t_max_d":  float(t[np.argmax(d)]),
    }


def spatial_propagation_distance_from_sol(sol, coords, seed_idx=None):
    S, I, t = _extract_SIR(sol)
    return spatial_propagation_distance(I, t, coords, seed_idx=seed_idx)


# ═══════════════════════════════════════════════════════════════════════════
# 7. CROSS-COMPARTMENT INFECTION FLUX  (vectorised)
# ═══════════════════════════════════════════════════════════════════════════

def cross_compartment_flux(
    S:      np.ndarray,
    I:      np.ndarray,
    t:      np.ndarray,
    W,
    labels: np.ndarray,
    beta,
) -> dict:
    """
    Instantaneous and integrated flux between compartments.

        F_{A→B}(t) = Σ_{i∈A} Σ_{j∈B} β_j · E_ji · S_j(t) · I_i(t)
    (R3: receiver β and exposure matrix E, as in the dynamics.)

    Vectorised over T using einsum.

    Parameters
    ----------
    beta : float scalar OR ndarray (N,)

    Returns
    -------
    dict: flux (T,C,C), integrated pd.DataFrame (C,C), t, compartments
    """
    comps = np.unique(labels)
    C, T  = len(comps), len(t)
    flux  = np.zeros((T, C, C), dtype=np.float64)
    for ai, bi, integrand in _receiver_flux_blocks(S, I, t, W, labels, beta):
        flux[:, ai, bi] = integrand
    integrated    = _trapz(flux, t, axis=0)
    integrated_df = pd.DataFrame(integrated, index=comps, columns=comps)

    return {
        "flux":         flux,
        "integrated":   integrated_df,
        "t":            t,
        "compartments": list(comps),
    }


def cross_compartment_flux_from_sol(sol, W, labels, beta):
    S, I, t = _extract_SIR(sol)
    return cross_compartment_flux(S, I, t, W, labels, beta)


# ═══════════════════════════════════════════════════════════════════════════
# OUTPUT UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def matrix_to_dataframe(mat: np.ndarray, compartments) -> pd.DataFrame:
    comps = list(compartments)
    return pd.DataFrame(mat, index=comps, columns=comps)


def plot_invasion_matrix(
    df:      pd.DataFrame,
    title:   str   = "Invasion matrix",
    cmap:    str   = "YlOrRd",
    figsize: tuple = (8, 6),
    fmt:     str   = ".3f",
    ax=None,
):
    """Annotated heatmap of a (C, C) invasion / flux matrix."""
    mat   = df.values.astype(float)
    comps = list(df.index)
    C     = len(comps)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    vmax = np.nanmax(mat) if np.nanmax(mat) > 0 else 1.0
    im   = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=0, vmax=vmax)

    ax.set_xticks(range(C)); ax.set_yticks(range(C))
    ax.set_xticklabels(list(df.columns), rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(comps, fontsize=9)
    ax.set_xlabel("Target compartment (B)", fontsize=10)
    ax.set_ylabel("Source compartment (A)", fontsize=10)
    ax.set_title(title, fontsize=12)

    valid = mat[~np.isnan(mat)]
    threshold = valid.mean() if valid.size > 0 else 0.0
    for i in range(C):
        for j in range(C):
            val = mat[i, j]
            txt = f"{val:{fmt}}" if not np.isnan(val) else "—"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                    color="white" if val > threshold else "black")

    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.show()
    return fig, ax


def plot_entropy(
    entropy_result: dict,
    figsize: tuple = (8, 4),
    title:   str   = "Invasion entropy H(t)",
):
    t  = entropy_result["t"]
    H  = entropy_result["H"]
    mH = entropy_result["mean_H"]
    pH = entropy_result["peak_H"]
    tP = entropy_result["t_peak_H"]

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(t, H, color="#2166ac", lw=2.0, label="H(t)")
    ax.axhline(mH, color="#d6604d", ls="--", lw=1.2,
               label=f"mean H = {mH:.3f}")
    ax.axvline(tP, color="#4dac26", ls=":", lw=1.2,
               label=f"peak H = {pH:.3f}  @  t={tP:.1f}")
    ax.fill_between(t, H, alpha=0.15, color="#2166ac")
    ax.set_xlabel("Time", fontsize=11)
    ax.set_ylabel("Shannon entropy H(t)", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.show()
    return fig, ax


def plot_source_sink(
    ss_df:   pd.DataFrame,
    figsize: tuple = (9, 7),
    title:   str   = "Source vs Sink strength",
):
    """
    Scatter plot source_strength (x) vs sink_strength (y).
    Tutti i compartimenti sono mostrati, non solo i 3 layer.
    I punti sono colorati per source-sink e dimensionati per
    source+sink (forza totale). Quadranti annotati.
    """
    fig, ax = plt.subplots(figsize=figsize)
    x    = ss_df["source_strength"].values
    y    = ss_df["sink_strength"].values
    lbls = ss_df["compartment"].values

    net_flow = x - y
    total    = x + y + 1e-12
    sizes    = 40 + 300 * (total / total.max())

    scatter = ax.scatter(x, y, s=sizes, c=net_flow, cmap="RdBu_r",
                         edgecolors="k", linewidths=0.4, zorder=3,
                         vmin=-np.abs(net_flow).max(),
                         vmax= np.abs(net_flow).max())
    fig.colorbar(scatter, ax=ax, label="source − sink (net emitter vs receiver)")

    # Etichette solo sui punti con forza totale > 5° percentile
    thresh = np.percentile(total, 20)
    for xi, yi, lbl, tot in zip(x, y, lbls, total):
        if tot >= thresh:
            ax.annotate(lbl, (xi, yi), textcoords="offset points",
                        xytext=(4, 3), fontsize=6.5, color="#333333")

    # Diagonale source = sink
    lim = max(x.max(), y.max()) * 1.12
    ax.plot([0, lim], [0, lim], "k--", lw=0.8, alpha=0.4,
            label="source = sink")

    # Annotazione quadranti
    ax.text(lim * 0.75, lim * 0.12, "emettitore netto\n(source > sink)",
            fontsize=7, color="#c0392b", alpha=0.7, ha="center")
    ax.text(lim * 0.12, lim * 0.75, "ricevitore netto\n(sink > source)",
            fontsize=7, color="#2471a3", alpha=0.7, ha="center")

    ax.set_xlim(-lim * 0.05, lim)
    ax.set_ylim(-lim * 0.05, lim)
    ax.set_xlabel("Source strength  Σ_B K_AB", fontsize=11)
    ax.set_ylabel("Sink strength    Σ_A K_AB", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.show()
    return fig, ax


# ═══════════════════════════════════════════════════════════════════════════
# MASTER RUNNER
# ═══════════════════════════════════════════════════════════════════════════

def run_transcompartment_analysis(
    sol,
    W,
    labels:   np.ndarray,
    beta,
    gamma,
    coords:   np.ndarray | None = None,
    seed_idx: int | None        = None,
    plot:     bool              = True,
) -> dict:
    """
    Compute all seven trans-compartment invasion metrics from a SIR solution.

    Parameters
    ----------
    beta, gamma : float scalar OR ndarray (N,) per-node parameters.
                  Using per-node values is strongly recommended when the
                  model has heterogeneous β/γ (e.g. PT vs immune compartments).

    Returns
    -------
    dict with keys:
        K_dyn, K_struct, K_norm, source_sink, ret_disp,
        entropy, propagation, flux
    """
    S, I, t = _extract_SIR(sol)

    print("Computing dynamic next-generation matrix (vectorised) …")
    K_dyn = next_generation_matrix_dynamic(S, I, t, W, labels, beta)

    print("Computing structural invasion matrix (per-node β/γ) …")
    K_struct, K_norm = structural_invasion_matrix(W, labels, beta, gamma)

    print("Computing source / sink scores …")
    ss = source_sink_scores(K_dyn, labels)

    print("Computing retention / dispersal …")
    rd = retention_dispersal(K_dyn)

    print("Computing invasion entropy …")
    ent = invasion_entropy(I, t, labels)

    prop = None
    if coords is not None:
        print("Computing spatial propagation distance …")
        prop = spatial_propagation_distance(I, t, coords, seed_idx=seed_idx)
    else:
        print("  [SKIP] spatial propagation distance — coords not provided.")

    print("Computing cross-compartment flux (vectorised) …")
    flux = cross_compartment_flux(S, I, t, W, labels, beta)

    results = dict(
        K_dyn       = K_dyn,
        K_struct    = K_struct,
        K_norm      = K_norm,
        source_sink = ss,
        ret_disp    = rd,
        entropy     = ent,
        propagation = prop,
        flux        = flux,
    )

    if plot:
        print("\nPlotting …")
        plot_invasion_matrix(K_dyn,    title="Dynamic invasion matrix  K_AB^dyn")
        plot_invasion_matrix(K_struct, title="Structural invasion matrix  K_AB (per-node R0)")
        plot_invasion_matrix(K_norm,   title="Normalized structural matrix  K_AB^norm")
        plot_invasion_matrix(flux["integrated"], title="Integrated flux  F_{A→B}")
        plot_source_sink(ss)
        plot_entropy(ent)

    return results


# ═══════════════════════════════════════════════════════════════════════════
# PRINT SUMMARY
# ═══════════════════════════════════════════════════════════════════════════

def print_transcompartment_summary(results: dict) -> None:
    sep = "═" * 70
    print(f"\n{sep}")
    print("TRANS-COMPARTMENT INVASION METRICS — SUMMARY")
    print(sep)

    print("\n── Source / Sink scores ─────────────────────────────────────────")
    print(results["source_sink"].to_string(index=False,
                                           float_format="{:.4f}".format))

    print("\n── Retention / Dispersal ────────────────────────────────────────")
    print(results["ret_disp"].to_string(index=False,
                                        float_format="{:.4f}".format))

    ent = results["entropy"]
    print("\n── Invasion Entropy ─────────────────────────────────────────────")
    print(f"  Mean H  : {ent['mean_H']:.4f}")
    print(f"  Peak H  : {ent['peak_H']:.4f}  @  t = {ent['t_peak_H']:.2f}")

    if results["propagation"] is not None:
        prop = results["propagation"]
        print("\n── Spatial Propagation Distance ─────────────────────────────────")
        print(f"  Seed node : {prop['seed_idx']}")
        print(f"  Max d(t)  : {prop['max_d']:.4f}  @  t = {prop['t_max_d']:.2f}")

    print(f"\n{sep}\n")