"""
model_core.py — single canonical implementation of the H/D/R model (R3).

Every simulation in the manuscript must go through this module. Before R3
the model was implemented separately in five places (sir_compartments,
sir_multilayer, sir_drug, sir_section7, rebuttal/diffusion_model), with
different solvers and different ad hoc corrections (clipping inside the
right-hand side, renormalisation of final states, p_R = 1 - p_H - p_D).
That is the origin of the inconsistent tables flagged by Reviewer 2.

Model (Eqs. 3-5 of the revised manuscript)
------------------------------------------
For each compartment i, p_i = (p^H_i, p^D_i, p^R_i) lies on the probability
simplex. Spatial coupling acts *only* through the exposure field

    lambda_i(t) = sum_j  E_ij  p^D_j(t),                        (Eq. 2 / 6)

where E is the exposure matrix (row-normalised intra-layer adjacency, plus
the inter-layer coupling block in the multilayer case). The dynamics are

    dp^H_i/dt = - beta_i p^H_i lambda_i
    dp^D_i/dt = + beta_i p^H_i lambda_i - rho_i p^D_i
    dp^R_i/dt = + rho_i p^D_i

Every flux leaving one state enters another state of the *same*
compartment, so d/dt (p^H_i + p^D_i + p^R_i) = 0 identically and, because
lambda_i >= 0, each component stays in [0, 1]. The Laplacian transport terms
-D_H (L p^H)_i and -D_D (L p^D)_i of the R1/R2 formulation are removed: they
moved probability mass between different compartments, violated the
per-compartment normalisation and produced p^R > 1 (Table 2 of R2).
Linear invariants are preserved exactly by BDF, so the conservation check
below holds to round-off.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.integrate import solve_ivp

SIMPLEX_TOL = 1e-8          # floor for max |sum - 1| and max negative excursion
# The actual tolerance used by simulate() is max(SIMPLEX_TOL, 100 * atol):
# BDF preserves linear invariants up to the Newton-iteration error, which
# scales with the solver tolerance (1e-7 with the canonical atol = 1e-9).
DEFAULT_RTOL = 1e-6
DEFAULT_ATOL = 1e-9


# ─────────────────────────────────────────────────────────────────────────
# Exposure matrices
# ─────────────────────────────────────────────────────────────────────────

def row_normalise(W) -> np.ndarray:
    """Row-normalised adjacency w~_ij = w_ij / sum_k w_ik (zero diagonal)."""
    W = np.asarray(sp.csr_matrix(W).toarray() if sp.issparse(W) else W,
                   dtype=float).copy()
    np.fill_diagonal(W, 0.0)
    rs = W.sum(axis=1, keepdims=True)
    return np.where(rs > 1e-12, W / np.where(rs > 1e-12, rs, 1.0), 0.0)


def row_normalised_from_normed_laplacian(L) -> np.ndarray:
    """Exact D^-1 W from a symmetrically normalised Laplacian L = I - D^-1/2 W D^-1/2.

    A = D^-1/2 W D^-1/2 has sqrt(d) as eigenvector with eigenvalue 1 on each
    connected component, so D^-1 W = diag(1/v) A diag(v) with v the Perron
    vector of the component (the scale of v cancels). Isolated nodes get a
    zero row. This makes the exposure of Eq. (2) identical whether a module
    receives the adjacency W or its normalised Laplacian.
    """
    from scipy.sparse.csgraph import connected_components
    L = np.asarray(sp.csr_matrix(L).toarray() if sp.issparse(L) else L,
                   dtype=float)
    A = -L.copy()
    np.fill_diagonal(A, 0.0)
    A = np.clip((A + A.T) / 2, 0.0, None)
    n_comp, lab = connected_components(sp.csr_matrix(A > 0), directed=False)
    Wt = np.zeros_like(A)
    for c in range(n_comp):
        idx = np.where(lab == c)[0]
        if len(idx) < 2:
            continue
        Ac = A[np.ix_(idx, idx)]
        vals, vecs = np.linalg.eigh(Ac)
        v = np.abs(vecs[:, -1])
        Wt[np.ix_(idx, idx)] = Ac * v[None, :] / v[:, None]
    return Wt


def adjacency_from_laplacian(L) -> np.ndarray:
    """Recover the (off-diagonal) weighted adjacency from a Laplacian."""
    L = np.asarray(sp.csr_matrix(L).toarray() if sp.issparse(L) else L,
                   dtype=float)
    W = -L.copy()
    np.fill_diagonal(W, 0.0)
    return np.clip(W, 0.0, None)


def multilayer_exposure(blocks: list[tuple[int, int, np.ndarray]],
                        W_inter, n_total: int) -> np.ndarray:
    """E = blockdiag(W~[alpha]) + W_inter, in the global node ordering.

    blocks : list of (start, end, W_tilde_alpha) for each layer.
    """
    E = np.zeros((n_total, n_total))
    for s, e, Wt in blocks:
        E[s:e, s:e] = np.asarray(Wt.toarray() if sp.issparse(Wt) else Wt)
    if W_inter is not None:
        E += np.asarray(W_inter.toarray() if sp.issparse(W_inter) else W_inter)
    return E


# ─────────────────────────────────────────────────────────────────────────
# Right-hand side and integration
# ─────────────────────────────────────────────────────────────────────────

def hdr_rhs(E: np.ndarray, beta: np.ndarray, rho: np.ndarray):
    """Return f(t, y) for y = [p^H | p^D | p^R]. No clipping, no transport."""
    n = len(beta)

    def f(t, y):
        pH, pD = y[:n], y[n:2 * n]
        lam = E @ pD
        infect = beta * pH * lam
        recover = rho * pD
        return np.concatenate([-infect, infect - recover, recover])

    return f


@dataclass
class Trajectory:
    t: np.ndarray           # (T,)
    pH: np.ndarray          # (n, T)
    pD: np.ndarray          # (n, T)
    pR: np.ndarray          # (n, T)
    sol: object = None      # raw OdeResult, for code that needs .t / .y

    @property
    def total(self) -> np.ndarray:
        return self.pH + self.pD + self.pR


def check_simplex(*states: np.ndarray, tol: float = SIMPLEX_TOL,
                  where: str = "") -> float:
    """Raise if the states leave the probability simplex. Returns max error."""
    total = sum(states)
    err_sum = float(np.max(np.abs(total - 1.0)))
    err_neg = float(-min(0.0, min(np.min(s) for s in states)))
    err = max(err_sum, err_neg)
    if err > tol:
        raise AssertionError(
            f"[{where}] probability simplex violated: "
            f"max|sum-1| = {err_sum:.2e}, max negative = {err_neg:.2e} "
            f"(tol {tol:.0e})")
    return err


def initial_state(n: int, seed_mask, seed_D: float = 0.05,
                  pR0: np.ndarray | None = None) -> np.ndarray:
    """Seeded compartments: p^D = seed_D, p^H = 1 - seed_D; others healthy."""
    pD0 = np.where(np.asarray(seed_mask, dtype=bool), float(seed_D), 0.0)
    pR0 = np.zeros(n) if pR0 is None else np.asarray(pR0, dtype=float)
    pH0 = 1.0 - pD0 - pR0
    return np.concatenate([pH0, pD0, pR0])


def simulate(E, beta, rho, y0, t_end: float, n_eval: int = 601,
             method: str = "BDF", rtol: float = DEFAULT_RTOL,
             atol: float = DEFAULT_ATOL, check: bool = True,
             where: str = "simulate") -> Trajectory:
    """Integrate the canonical H/D/R system. All manuscript results use this."""
    E = np.asarray(E.toarray() if sp.issparse(E) else E, dtype=float)
    beta = np.asarray(beta, dtype=float)
    rho = np.asarray(rho, dtype=float)
    y0 = np.asarray(y0, dtype=float)
    n = len(beta)
    if y0.shape != (3 * n,):
        raise ValueError(f"y0 must have shape ({3*n},), got {y0.shape}")
    if check:
        check_simplex(y0[:n], y0[n:2*n], y0[2*n:], where=f"{where}: y0")

    t_eval = np.linspace(0.0, float(t_end), int(n_eval))
    sol = solve_ivp(hdr_rhs(E, beta, rho), (0.0, float(t_end)), y0,
                    t_eval=t_eval, method=method, rtol=rtol, atol=atol)
    if not sol.success:
        warnings.warn(f"{where}: solver failed — {sol.message}",
                      RuntimeWarning, stacklevel=2)
    tr = Trajectory(sol.t, sol.y[:n], sol.y[n:2*n], sol.y[2*n:], sol)
    if check:
        check_simplex(tr.pH, tr.pD, tr.pR, tol=max(SIMPLEX_TOL, 100 * atol),
                      where=where)
    return tr


def warn_ignored_diffusion(DS=None, DD=None, where: str = ""):
    """D_H / D_D no longer enter the model. Warn once if non-zero values are passed."""
    vals = [v for v in (DS, DD) if v is not None]
    if any(np.any(np.asarray(v, dtype=float) != 0) for v in vals):
        warnings.warn(
            f"[{where}] D_H / D_D are ignored since R3: spatial coupling acts "
            "only through the exposure field (see model_core docstring).",
            DeprecationWarning, stacklevel=3)
