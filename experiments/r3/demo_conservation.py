"""
demo_conservation.py — R3, Reviewer 2 point 1.

Shows on a synthetic compartment graph that the R2 equations (Laplacian
transport acting on state probabilities) leave the probability simplex,
whereas the revised canonical model (src/model_core.py) conserves
p^H + p^D + p^R = 1 in every compartment to round-off.

Needs no data. Usage:  python experiments/r3/demo_conservation.py
"""
import os
import sys

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import model_core as mc  # noqa: E402

rng = np.random.default_rng(0)
n = 17                                   # as in the V1 kidney case study
xy = rng.uniform(size=(n, 2))
d2 = ((xy[:, None] - xy[None]) ** 2).sum(-1)
W = np.exp(-d2 / 0.05)
np.fill_diagonal(W, 0)
W[W < np.quantile(W, 0.6)] = 0
W = np.maximum(W, W.T)

# Table 1-like parameters; compartment 0..5 seeded at p^D = 0.05
beta = rng.choice([0.40, 0.28, 0.25, 0.30, 0.22], n) * rng.choice([1.25, 1.0, 0.75], n)
rho = rng.choice([0.08, 0.10, 0.12], n)
DH = np.full(n, 0.05)
DD = rng.choice([0.20, 0.35], n)
seed = np.zeros(n, bool); seed[:6] = True
y0 = mc.initial_state(n, seed, 0.05)
E = mc.row_normalise(W)
T = 60.0

# --- R2 formulation (as in sir_compartments.simulate_SIR_ivp, without clipping)
deg = W.sum(1)
Dm12 = np.diag(1 / np.sqrt(np.where(deg > 0, deg, 1)))
Lsym = np.eye(n) - Dm12 @ W @ Dm12


def rhs_r2(t, y):
    pH, pD = y[:n], y[n:2 * n]
    inf = beta * pH * (E @ pD)
    return np.concatenate([-inf - DH * (Lsym @ pH),
                           inf - rho * pD - DD * (Lsym @ pD),
                           rho * pD])


s2 = solve_ivp(rhs_r2, (0, T), y0, method="BDF", rtol=1e-6, atol=1e-9,
               t_eval=np.linspace(0, T, 601))
tot2 = s2.y[:n] + s2.y[n:2*n] + s2.y[2*n:]

# --- R3 canonical model
tr = mc.simulate(E, beta, rho, y0, T, check=False)
tot3 = tr.total

print(f"{'':28s}{'R2 (Laplacian on p)':>22s}{'R3 (canonical)':>18s}")
print(f"{'max |sum - 1|':28s}{np.abs(tot2-1).max():22.3e}{np.abs(tot3-1).max():18.3e}")
print(f"{'max final p^R':28s}{s2.y[2*n:, -1].max():22.3f}{tr.pR[:, -1].max():18.3f}")
print(f"{'compartments with p^R > 1':28s}{int((s2.y[2*n:, -1] > 1).sum()):22d}{int((tr.pR[:, -1] > 1).sum()):18d}")
print(f"{'min of any state':28s}{min(s2.y.min(), 0):22.3e}{min(tr.pH.min(), tr.pD.min(), tr.pR.min()):18.3e}")
mc.check_simplex(tr.pH, tr.pD, tr.pR, where="R3 demo")
print("R3 conservation check passed.")
