"""
fix_interlayer_flux.py
======================
Patch for the inter-layer flux = 0 bug in run_transcompartment_analysis.

ROOT CAUSE
----------
The original code computes the inter-layer flux matrix using W_ml, which is
built from comp["Wc2"] sliced to the multilayer global_index:

    W_ml = sp.csr_matrix(comp["Wc2"])[np.ix_(global_index, global_index)]

However, run_transcompartment_analysis receives this W and iterates over
layer pairs using the *local* (0-based) indices within the multilayer ODE
solution, while looking up rows/cols of W using the *global* compartment
indices. The index mismatch produces zero flux everywhere.

FIX
---
Replace the inter-layer flux computation block with a version that:
1. Correctly maps from local ODE slice indices to W_ml row/col indices.
2. Uses the actual sol_ml trajectory (I block) rather than the single-layer sol.
3. Computes flux in the exposure-driven formulation (Eq. 4):
   the infection flux from layer B to layer A is the time-integrated
   exposure of A's susceptible nodes to B's diseased nodes:
       F_{B→A} = ∫ Σ_{i∈A} Σ_{j∈B} β_i · S_i(t) · w_ij · I_j(t) dt
   This is consistent with the exposure-driven multilayer model
   (infect_i = β_i · S_i · λ_i, where λ_i includes W_inter @ I).

HOW TO APPLY
------------
Call ``patch_interlayer_flux(tc_results, sol_ml, net, W_ml, diff_params)``
after run_transcompartment_analysis to overwrite the zero-flux entries.

Alternatively, replace the flux block in run_transcompartment_analysis
as described in the inline comments below.
"""

from __future__ import annotations
import numpy as np
import scipy.sparse as sp

try:
    _trapz = np.trapezoid
except AttributeError:
    _trapz = np.trapezoid


def compute_interlayer_flux(
    sol_ml,
    net:        dict,
    W_ml,
    sir_params: dict,
) -> dict[str, float]:
    """
    Compute the time-integrated inter-layer diffusion flux between
    every pair of layers (PT, DCT, TAL).

    Parameters
    ----------
    sol_ml     : ODE solution from run_SIR_multilayer (sol.y shape: 3*n_total × T)
    net        : net dict from run_SIR_multilayer, keys: layers, node_ranges,
                 global_index, n_total
    W_ml       : sparse coupling matrix in *multilayer local* indices
                 (n_total × n_total), i.e. the same W passed to the multilayer ODE
    sir_params : dict with key "beta" (length = total compartments in full model)

    Returns
    -------
    dict  {  "DCT↔PT": float,  "PT↔TAL": float,  "DCT↔TAL": float  }
    """
    layers       = net["layers"]
    node_ranges  = net["node_ranges"]
    global_index = net["global_index"]
    n_total      = net["n_total"]
    macros       = [m for m in ("PT", "DCT", "TAL") if m in layers]

    beta_g = np.asarray(sir_params["beta"], dtype=float)

    t      = sol_ml.t
    # S and I in multilayer-local indices (0 … n_total-1)
    S_traj = sol_ml.y[:n_total, :]           # shape (n_total, T)
    I_traj = sol_ml.y[n_total:2*n_total, :]  # shape (n_total, T)

    # Beta vector in multilayer-local indexing
    beta_local = beta_g[global_index]         # shape (n_total,)

    W_arr = _to_dense(W_ml)  # (n_total, n_total)

    # Build layer → local index mapping
    layer_local: dict[str, np.ndarray] = {}
    for m in macros:
        s, e = node_ranges[m]
        layer_local[m] = np.arange(s, e)

    flux = {}
    for i_m, ma in enumerate(macros):
        for mb in macros[i_m + 1:]:
            idx_a = layer_local[ma]
            idx_b = layer_local[mb]

            # Sub-block of W between layer A and layer B
            W_ab = W_arr[np.ix_(idx_a, idx_b)]  # (|A|, |B|)
            W_ba = W_arr[np.ix_(idx_b, idx_a)]  # (|B|, |A|)

            # Flux A→B (exposure-driven, Eq. 4):
            #   node i∈A is exposed to disease of node j∈B via w_ij.
            #   The infection flux from B to A is:
            #     Σ_i∈A Σ_j∈B  β_i · S_i(t) · w_ij · I_j(t)  integrated over time
            #   i.e. A's susceptibility × cross-layer exposure to B's disease.
            flux_ab = 0.0
            for k_a, i_a in enumerate(idx_a):
                for k_b, i_b in enumerate(idx_b):
                    w_ij = W_ab[k_a, k_b]
                    if w_ij <= 0:
                        continue
                    integrand = beta_local[i_a] * S_traj[i_a, :] * w_ij * I_traj[i_b, :]
                    flux_ab += float(_trapz(integrand, t))

            # Flux B→A (symmetric: B exposed to A's disease)
            flux_ba = 0.0
            for k_b, i_b in enumerate(idx_b):
                for k_a, i_a in enumerate(idx_a):
                    w_ji = W_ba[k_b, k_a]
                    if w_ji <= 0:
                        continue
                    integrand = beta_local[i_b] * S_traj[i_b, :] * w_ji * I_traj[i_a, :]
                    flux_ba += float(_trapz(integrand, t))

            key = f"{ma}↔{mb}"
            flux[key] = flux_ab + flux_ba

    return flux


def _to_dense(W):
    if sp.issparse(W):
        return W.toarray().astype(np.float64)
    return np.asarray(W, dtype=np.float64)


def patch_interlayer_flux(
    tc_results: dict,
    sol_ml,
    net:        dict,
    W_ml,
    sir_params: dict,
    verbose:    bool = True,
) -> dict:
    """
    Recompute inter-layer flux and overwrite the zero entries in tc_results.

    Usage (add after run_transcompartment_analysis in main.py):

        from fix_interlayer_flux import patch_interlayer_flux
        tc_results = patch_interlayer_flux(
            tc_results, ml_result["sol"], ml_result["net"],
            W_ml, diff_params
        )

    Returns the patched tc_results dict.
    """
    flux = compute_interlayer_flux(sol_ml, net, W_ml, sir_params)

    if "interlayer_flux" not in tc_results:
        tc_results["interlayer_flux"] = {}
    tc_results["interlayer_flux"].update(flux)

    if verbose:
        print("\n  Inter-layer flux (corrected):")
        for k, v in flux.items():
            print(f"    {k}: {v:.5f}")

    return tc_results


# ─────────────────────────────────────────────────────────────────────────
# HOW TO FIX run_transcompartment_analysis DIRECTLY
# ─────────────────────────────────────────────────────────────────────────
#
# In sir_transcompartment_metrics.py, find the block that computes
# inter-layer flux (look for "interlayer_flux").
# Replace the W indexing from:
#
#   OLD (BUGGY):
#       W_block = W[global_index[s_a:e_a], :][:, global_index[s_b:e_b]]
#
#   NEW (CORRECT):
#       # node_ranges gives LOCAL indices into the multilayer ODE
#       W_block = W[s_a:e_a, s_b:e_b]   # W is already in local coords
#
# Also make sure sol passed in is sol_ml (the multilayer solution),
# not sol_diff (the single-layer solution).
#
# See compute_interlayer_flux() above for a self-contained reference impl.
# ─────────────────────────────────────────────────────────────────────────
