"""
R3 (Reviewer 2, point 1): every entry point of the model must keep each
compartment on the probability simplex: sum of states = 1, no state < 0,
so no p^R > 1. Runs on synthetic graphs; no data needed.
"""
import os
import sys
import warnings

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.sparse.csgraph import laplacian

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "experiments", "rebuttal"))

import model_core as mc  # noqa: E402

TOL = 1e-8


@pytest.fixture
def graph():
    rng = np.random.default_rng(1)
    n = 17
    xy = rng.uniform(size=(n, 2))
    W = np.exp(-((xy[:, None] - xy[None]) ** 2).sum(-1) / 0.05)
    np.fill_diagonal(W, 0)
    W[W < np.quantile(W, 0.6)] = 0
    W = np.maximum(W, W.T)
    beta = rng.uniform(0.15, 0.5, n)
    rho = rng.uniform(0.07, 0.14, n)
    seed = np.zeros(n, bool); seed[:6] = True
    return dict(n=n, W=W, beta=beta, rho=rho, seed=seed,
                Lc=sp.csr_matrix(laplacian(W, normed=True)))


def _assert_simplex(*states):
    total = sum(states)
    assert np.max(np.abs(total - 1)) < TOL
    assert min(np.min(s) for s in states) > -TOL


def test_core_conserves(graph):
    y0 = mc.initial_state(graph["n"], graph["seed"], 0.05)
    tr = mc.simulate(mc.row_normalise(graph["W"]), graph["beta"], graph["rho"], y0, 60)
    _assert_simplex(tr.pH, tr.pD, tr.pR)
    assert tr.pR.max() <= 1 + TOL


def test_core_rejects_violation():
    with pytest.raises(AssertionError):
        mc.check_simplex(np.array([0.5]), np.array([0.3]), np.array([0.4]))


def test_multilayer_core_conserves(graph):
    # two layers + strong inter-layer coupling: exposure row sums > 1 must not
    # break normalisation (lambda is a pressure, not a probability)
    n = graph["n"]; h = n // 2
    Wt1 = mc.row_normalise(graph["W"][:h, :h]); Wt2 = mc.row_normalise(graph["W"][h:, h:])
    Wi = np.zeros((n, n)); Wi[:h, h:] = 0.5 * graph["W"][:h, h:]; Wi = Wi + Wi.T
    E = mc.multilayer_exposure([(0, h, Wt1), (h, n, Wt2)], Wi, n)
    tr = mc.simulate(E, graph["beta"], graph["rho"],
                     mc.initial_state(n, graph["seed"], 0.05), 60)
    _assert_simplex(tr.pH, tr.pD, tr.pR)


def test_sir_compartments_ivp_and_euler(graph):
    from sir_compartments import simulate_SIR_ivp, simulate_SIR_euler
    n = graph["n"]
    I0 = np.where(graph["seed"], 0.05, 0.0)
    params = dict(beta=graph["beta"], gamma=graph["rho"],
                  D_S=np.full(n, 0.05), D_I=np.full(n, 0.20))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        H, D, R, sol = simulate_SIR_ivp(graph["Lc"], 1 - I0, I0, np.zeros(n), params, t_end=60)
        _assert_simplex(sol.y[:n], sol.y[n:2*n], sol.y[2*n:])
        H, D, R, d = simulate_SIR_euler(graph["Lc"], 1 - I0, I0, np.zeros(n), params,
                                        t_end=60, n_steps=3000)
        _assert_simplex(d["S"], d["I"], d["R"])


def test_drug_extension(graph):
    from sir_drug import simulate_SIRD_drug
    n = graph["n"]; z = np.zeros(n)
    I0 = np.where(graph["seed"], 0.05, 0.0)
    params = dict(beta=graph["beta"], gamma=graph["rho"], delta=np.full(n, 0.02),
                  gamma2=1.5 * graph["rho"], D_S=np.full(n, 0.05), D_I=np.full(n, 0.2),
                  k_F=0.5, k_e=0.15, D_F=0.3, IC50_beta=0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        sol, fin = simulate_SIRD_drug(graph["Lc"], 1 - I0, I0, z, z, z, z, params,
                                      t_end=60, t_dose=12.0)
    _assert_simplex(*(sol.y[k*n:(k+1)*n] for k in range(5)))


def test_rebuttal_model(graph):
    from diffusion_model import CompartmentDiffusionModel
    m = CompartmentDiffusionModel(graph["W"], graph["beta"], graph["rho"])
    tr = m.simulate(np.where(graph["seed"], 0.05, 0.0), t_end=60.0, dt=0.05) \
        if hasattr(m, "simulate") else m.run(np.where(graph["seed"], 0.05, 0.0), t_end=60.0, dt=0.05)
    _assert_simplex(tr["pH"], tr["pD"], tr["pR"])
