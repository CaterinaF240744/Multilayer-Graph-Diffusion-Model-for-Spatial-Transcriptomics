"""
diffusion_model.py
-------------------
Reusable implementation of the probabilistic three-state (H, D, R) diffusion-reaction
model described in Section 5 and Appendix A.1 of the manuscript. Deterministic
mean-field regime (death state emptied, mu_i = 0), exactly as used in Section 6.2.

This module is dataset-agnostic: it only needs a compartment graph (weighted
adjacency) and per-compartment parameters (beta, rho). It is used both to
reproduce the original V1 Mouse Kidney results and to run the same model on
the new GSE182939 time-course dataset for temporal validation (Experiment C).

Equations implemented (mean-field, continuous-time Euler integration):
    lambda_i(t) = sum_j w_tilde_ij * p_D_j(t)                      (Eq. 2 / 17)
    dpH_i/dt = - beta_i * lambda_i(t) * pH_i  - D_{H,i} * (L @ pH)_i   (Eq. 2 / 18)
    dpD_i/dt = + beta_i * lambda_i(t) * pH_i  - rho_i * pD_i
                - D_{D,i} * (L @ pD)_i                                (Eq. 3 / 19)
    dpR_i/dt = + rho_i * pD_i                                         (Eq. 4 / 20)
where w_tilde is the row-normalised compartment adjacency (Eq. 17) and
L = diag(row_sums) - W is the graph Laplacian. D_H and D_D are the
per-compartment diffusion coefficients from Table 1 (default 0 if not
provided, which disables the spatial diffusion term).
"""

import numpy as np
import pandas as pd


class CompartmentDiffusionModel:
    def __init__(self, W, beta, rho, D_H=None, D_D=None, compartment_names=None):
        """
        Parameters
        ----------
        W : (n, n) array-like
            Compartment-level adjacency/weight matrix (symmetric, non-negative).
            Use the same coarse-graining as Section 6.1: average spot-level
            edge weights between all spot pairs belonging to different
            compartments.
        beta : (n,) array-like
            Susceptibility parameter per compartment.
        rho : (n,) array-like
            Recovery rate per compartment.
        D_H : (n,) array-like or float, optional
            Diffusion coefficient for the H (healthy) state per compartment.
            If None, defaults to 0 (no spatial diffusion of H).
            Paper Table 1 uses D_H = 0.20 for most cell types, 0.35 for vascular.
        D_D : (n,) array-like or float, optional
            Diffusion coefficient for the D (diseased) state per compartment.
            If None, defaults to 0 (no spatial diffusion of D).
            Paper Table 1 uses D_D = 0.20 for most cell types, 0.35 for vascular.
        compartment_names : list of str, optional
            Labels for reporting; defaults to integer indices.
        """
        W = np.asarray(W, dtype=float)
        assert W.shape[0] == W.shape[1], "W must be square"
        n = W.shape[0]

        row_sums = W.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0  # avoid div by zero for isolated nodes
        self.W_tilde = W / row_sums  # Eq. 17 row-normalisation

        # Laplacian for spatial diffusion term (Eq. 3/19: D_D * L @ pD)
        self.L = np.diag(row_sums.flatten()) - W

        self.beta = np.asarray(beta, dtype=float).reshape(n)
        self.rho = np.asarray(rho, dtype=float).reshape(n)

        # Diffusion coefficients: accept scalar or per-compartment vector
        if D_H is None:
            self.D_H = np.zeros(n)
        elif np.isscalar(D_H):
            self.D_H = np.full(n, float(D_H))
        else:
            self.D_H = np.asarray(D_H, dtype=float).reshape(n)

        if D_D is None:
            self.D_D = np.zeros(n)
        elif np.isscalar(D_D):
            self.D_D = np.full(n, float(D_D))
        else:
            self.D_D = np.asarray(D_D, dtype=float).reshape(n)

        self.n = n
        self.names = list(compartment_names) if compartment_names is not None else [
            f"c{i}" for i in range(n)
        ]

    def simulate(self, p_D0, p_R0=None, t_end=60.0, dt=0.05):
        """
        Forward-simulate the mean-field ODE system with explicit Euler steps.

        Parameters
        ----------
        p_D0 : (n,) array-like
            Initial diseased probability per compartment (Section 6.3 uses 0.05
            in seed compartments, 0 elsewhere).
        p_R0 : (n,) array-like, optional
            Initial recovered probability (defaults to zero).
        t_end : float
            Simulation horizon (paper default: 60 time units).
        dt : float
            Euler step size.

        Returns
        -------
        traj : dict with keys 'time', 'pH', 'pD', 'pR', each an array
            time: (T,)   pH/pD/pR: (T, n)
        """
        n = self.n
        p_D0 = np.asarray(p_D0, dtype=float).reshape(n)
        p_R0 = np.zeros(n) if p_R0 is None else np.asarray(p_R0, dtype=float).reshape(n)
        p_H0 = 1.0 - p_D0 - p_R0
        assert np.all(p_H0 >= -1e-9), "p_D0 + p_R0 must not exceed 1"

        steps = int(np.ceil(t_end / dt)) + 1
        time = np.linspace(0.0, t_end, steps)
        pH = np.zeros((steps, n))
        pD = np.zeros((steps, n))
        pR = np.zeros((steps, n))
        pH[0], pD[0], pR[0] = p_H0, p_D0, p_R0

        for k in range(steps - 1):
            lam = self.W_tilde @ pD[k]                       # Eq. 2 / 17
            # Eq. 2-4 / 18-20: exposure-driven infection + spatial diffusion
            dH = -self.beta * lam * pH[k] - self.D_H * (self.L @ pH[k])
            dD = self.beta * lam * pH[k] - self.rho * pD[k] - self.D_D * (self.L @ pD[k])
            dR = self.rho * pD[k]
            pH[k + 1] = np.clip(pH[k] + dt * dH, 0.0, 1.0)
            pD[k + 1] = np.clip(pD[k] + dt * dD, 0.0, 1.0)
            pR[k + 1] = np.clip(pR[k] + dt * dR, 0.0, 1.0)
            # renormalise for numerical drift
            s = pH[k + 1] + pD[k + 1] + pR[k + 1]
            s[s == 0] = 1.0
            pH[k + 1] /= s
            pD[k + 1] /= s
            pR[k + 1] /= s

        return {"time": time, "pH": pH, "pD": pD, "pR": pR}

    def metrics(self, traj, activation_threshold=0.10):
        """
        Compute the same per-compartment scalar metrics used in Table 2/3/13:
        activation time, peak time, AUC, final recovered fraction.
        """
        time, pD, pR = traj["time"], traj["pD"], traj["pR"]
        rows = []
        for i, name in enumerate(self.names):
            series = pD[:, i]
            above = np.where(series > activation_threshold)[0]
            t_act = time[above[0]] if len(above) else np.nan
            t_peak = time[int(np.argmax(series))]
            trapz_fn = getattr(np, "trapezoid", None) or np.trapz
            auc = trapz_fn(series, time)
            r_fin = pR[-1, i]
            rows.append(
                dict(compartment=name, t_act=t_act, t_peak=t_peak,
                     AUC_D=auc, p_R_fin=r_fin, peak_D=series.max())
            )
        return pd.DataFrame(rows)
