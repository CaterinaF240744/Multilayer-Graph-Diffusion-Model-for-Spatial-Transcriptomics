"""
benchmark_synthetic.py
======================
Synthetic ground-truth recovery benchmark (manuscript Section
"Ground-truth recovery benchmark"; response letter R2.3).

Reconstructed standalone version: reproduces the methods described in the
manuscript and the exact output schemas of the released results
(benchmark_aggregated.csv, experimentA_noise_response.csv,
variance_decomposition.json, table_variance_decomp.tex).

Design (fixed by the manuscript text)
-------------------------------------
Synthetic tissue generator:
  - jittered hexagonal grid, N = 2,704 spots
  - 3 anatomical zones (cortex / outer_medulla / inner_medulla) with noisy
    radial boundaries
  - 6 cell types (PT, DCT, TAL, vascular, immune, other) assigned by
    region-dependent composition with spatially autocorrelated identity fields
  - known spot graph G* : Gaussian-weighted kNN (k = 6), same construction
    as the real pipeline (graph_utils.build_knn_graph)
  - known parameters theta* follow Table 1 (macro-type x zone)
  - known seed S*: all outer-medulla spots, p^D_0 = 0.05
  - ground truth: exposure-driven dynamics integrated at SPOT level with
    explicit Euler (dt = 0.01) -- deliberately a different integrator and
    resolution from the analysis pipeline (BDF on the coarse-grained
    compartment graph)

Expression generation:
  X_ig ~ NB(mu_ig, phi_g),  log mu_ig = alpha_g + delta_{g,c_i} + gamma_g D*_i(t_obs)
  t_obs = 2.0; injury program (Havcr1, Lcn2, Vcam1) coupled to the latent
  state; zone-marker programs for blind anatomical zoning; measurement noise
  sigma through 3 channels: ambient contamination (background 1 -> 1+9*sigma),
  per-spot count-depth variation, dropout.

Blind pipeline (receives counts + coordinates only):
  marker-based NN annotation, marker-based zoning, kNN graph,
  macro x zone coarse-graining, Table 1 parameters, BDF integration
  (sir_compartments.simulate_SIR_ivp) -- identical in every step to the
  real-data analysis.

Experiments
-----------
  A: correctly specified model, noise sigma in {0, 0.1, 0.25, 0.5, 1.0},
     50 replicates each  -> 250 runs  -> experimentA_noise_response.csv
  B: one-factor-at-a-time misspecification, 30 replicates each, sigma = 0:
     kinetics eps in {+-10, +-25, +-50}%, graph k in {4,8,10,15},
     edge deletion in {5,10,20,30}%, annotation error in {5,10,20,30}%,
     seed (remote inner-medulla), model structure (uniform beta analysed,
     heterogeneous beta* generated with strength 0.25 / 0.5)
     -> 21 cells x 30 = 630 runs -> benchmark_aggregated.csv
  Total = 880 runs.

  Variance decomposition: regression-based attribution of recovery loss
  across all benchmark conditions -> variance_decomposition.json,
  table_variance_decomp.tex

Repository placement
---------------------
Lives in experiments/rebuttal/, alongside the other rebuttal scripts; imports
the model code from src/ (two levels up), or from MODEL_SRC_DIR if set.

Usage
-----
  python benchmark_synthetic.py [--out-dir OUT] [--n-jobs 8] [--seed 0]
  python benchmark_synthetic.py --quick     # 5 replicates per cell (smoke test)

Outputs (in --out-dir, default ./benchmark_outputs):
  experimentA_noise_response.csv, benchmark_aggregated.csv,
  variance_decomposition.json, table_variance_decomp.tex, runs_raw.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy import sparse as sp
from scipy.sparse.csgraph import laplacian as sp_laplacian
from scipy.stats import spearmanr, kendalltau

# ── repository model code (src/) ─────────────────────────────────────────────
# This script is expected to live in experiments/rebuttal/; src/ is two
# levels up. Override with the MODEL_SRC_DIR env var if the layout differs.
_SRC_DIR = os.environ.get(
    "MODEL_SRC_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"),
)
sys.path.insert(0, os.path.abspath(_SRC_DIR))
from graph_utils import build_knn_graph                      # noqa: E402
from compartments import coarse_grain_graph                  # noqa: E402
from sir_compartments import (                               # noqa: E402
    build_default_params_SIR, simulate_SIR_ivp,
)

# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS (Table 1 of the manuscript / sir_compartments defaults)
# ═══════════════════════════════════════════════════════════════════════════

TYPES = ["PT", "DCT", "TAL", "vascular", "immune", "other"]

MACRO_PARAMS = {                      # Table 1 (beta, rho) + diffusion
    "PT":       dict(beta=0.40, rho=0.08, D_D=0.20),
    "DCT":      dict(beta=0.28, rho=0.10, D_D=0.20),
    "TAL":      dict(beta=0.25, rho=0.12, D_D=0.20),
    "vascular": dict(beta=0.30, rho=0.10, D_D=0.35),
    "immune":   dict(beta=0.20, rho=0.25, D_D=0.20),
    "other":    dict(beta=0.22, rho=0.10, D_D=0.20),
}
REGION_MULT = {                       # anatomical zone modulation
    "cortex":        dict(beta=1.00, rho=1.00),
    "outer_medulla": dict(beta=1.25, rho=1.10),
    "inner_medulla": dict(beta=0.75, rho=0.85),
}

D_H_BASE = 0.05                       # D_H — healthy-state diffusion
T_END = 15.0
T_OBS = 2.0
ACT_THRESHOLD = 0.1                   # activation threshold p^D = 0.1
SEED_P0 = 0.05

N_SPOTS = 2704                        # 52 x 52 hexagonal grid
HEX_SIDE = 52

ZONE_MARKERS = {                      # same panels as compartments.py
    "cortex":        ["slc34a1", "lrp2", "cubn", "nphs1", "nphs2", "pecam1"],
    "outer_medulla": ["umod", "slc12a1", "cldn16", "slc12a3"],
    "inner_medulla": ["aqp2", "aqp3", "avpr2", "atp6v1b1"],
}
INJURY_GENES = ["Havcr1", "Lcn2", "Vcam1"]

# marker panels used by the BLIND annotation (per cell type)
TYPE_MARKERS = {
    "PT":       ["slc34a1", "lrp2", "cubn", "slc27a2", "ass1"],
    "DCT":      ["slc12a3", "atp6v0d2", "slc8a1"],
    "TAL":      ["umod", "slc12a1", "cldn16"],
    "vascular": ["pecam1", "kdr", "flt1", "vwf"],
    "immune":   ["ptprc", "cd74", "h2-ab1"],
    "other":    ["aqp2", "aqp3", "avpr2", "col1a1", "dcn"],
}

NOISE_LEVELS = [0.0, 0.1, 0.25, 0.5, 1.0]
N_REPS_A = 50
N_REPS_B = 30
MISSPEC_GRID = [                      # (factor, level) — Experiment B
    ("kinetics", 0.1), ("kinetics", -0.1),
    ("kinetics", 0.25), ("kinetics", -0.25),
    ("kinetics", 0.5), ("kinetics", -0.5),
    ("graph_k", 4.0), ("graph_k", 8.0), ("graph_k", 10.0), ("graph_k", 15.0),
    ("edge_del", 0.05), ("edge_del", 0.1), ("edge_del", 0.2), ("edge_del", 0.3),
    ("annotation", 0.05), ("annotation", 0.1),
    ("annotation", 0.2), ("annotation", 0.3),
    ("seed", 1.0),
    ("model", 0.25), ("model", 0.5),
]


# ═══════════════════════════════════════════════════════════════════════════
# SYNTHETIC TISSUE GENERATOR
# ═══════════════════════════════════════════════════════════════════════════

def make_hex_grid(n_spots=N_SPOTS, side=HEX_SIDE, jitter=0.15, rng=None):
    """Jittered hexagonal grid of spot coordinates."""
    rng = rng or np.random.default_rng()
    rows, cols = [], []
    for r in range(side):
        for c in range(side):
            if len(rows) >= n_spots:
                break
            rows.append(r * np.sqrt(3) / 2.0)
            cols.append(c + (r % 2) / 2.0)
    pts = np.stack([cols, rows], axis=1).astype(float)
    pts += rng.normal(0, jitter, pts.shape)
    return pts[:n_spots]


def assign_zones(coords, rng):
    """Three anatomical zones by noisy radial boundaries."""
    center = coords.mean(axis=0)
    r = np.linalg.norm(coords - center, axis=1)
    r_noisy = r * (1.0 + rng.normal(0, 0.05, len(r)))
    q1, q2 = np.quantile(r_noisy, [1 / 3, 2 / 3])
    zones = np.where(r_noisy <= q1, "inner_medulla",
                     np.where(r_noisy <= q2, "outer_medulla", "cortex"))
    return zones


def assign_cell_types(zones, coords, rng):
    """Region-dependent composition + spatially autocorrelated identity."""
    # region-dependent composition (rows: zone, cols: type)
    comp = {
        "inner_medulla": [0.15, 0.05, 0.30, 0.05, 0.05, 0.40],
        "outer_medulla": [0.35, 0.15, 0.35, 0.05, 0.02, 0.08],
        "cortex":        [0.55, 0.25, 0.05, 0.10, 0.02, 0.03],
    }
    # spatially autocorrelated per-type preference fields
    # (short correlation length -> intercalated types, as in the real Visium
    #  tissue where PT/DCT/TAL alternate at spot level)
    from scipy.ndimage import gaussian_filter
    pref = {}
    for t in TYPES:
        f = gaussian_filter(rng.normal(0, 1, (HEX_SIDE, HEX_SIDE)), sigma=1.5).ravel()[:len(coords)]
        pref[t] = f / (f.std() + 1e-12)

    labels = []
    for i, z in enumerate(zones):
        scores = np.log(np.array(comp[z])) + 0.8 * np.array(
            [pref[t][i] for t in TYPES])
        scores += rng.gumbel(0, 0.1, len(TYPES))
        labels.append(TYPES[int(np.argmax(scores))])
    return np.array(labels)


def spot_graph(coords, k=6):
    """Known spot graph G* — same Gaussian kNN as the real pipeline."""
    return build_knn_graph(coords.astype(np.float32), k=k)


def row_normalise(W):
    W = sp.csr_matrix(W, dtype=float)
    rs = np.asarray(W.sum(axis=1)).ravel()
    inv = np.where(rs > 0, 1.0 / np.maximum(rs, 1e-12), 0.0)
    return sp.diags(inv) @ W


def spot_level_truth(Wstar, cell_types, zones, rng,
                     beta_het=None, t_end=T_END, dt=0.01):
    """
    Spot-level exposure-driven ground truth with explicit Euler (dt = 0.01).

    States H, D, R per spot; lambda_i = sum_j W~*_ij D_j;
    dH = -beta H lambda; dD = beta H lambda - rho D; dR = rho D.
    Seed: all outer-medulla spots, D_0 = 0.05.
    Returns trajectories of D on a common time grid + per-spot activation times.
    """
    n = Wstar.shape[0]
    Wt = row_normalise(Wstar)
    beta = np.array([MACRO_PARAMS[t]["beta"] for t in cell_types], dtype=float)
    rho = np.array([MACRO_PARAMS[t]["rho"] for t in cell_types], dtype=float)
    if beta_het is not None:
        beta = beta * beta_het

    H = np.ones(n)
    D = np.where(zones == "outer_medulla", SEED_P0, 0.0)
    R = np.zeros(n)

    n_steps = int(round(t_end / dt))
    rec_every = max(1, n_steps // 100)          # ~100 recorded frames
    D_traj = np.zeros((n_steps // rec_every + 1, n))
    t_traj = np.zeros(n_steps // rec_every + 1)
    act_t = np.full(n, np.nan)                  # spot activation times
    frame = 0
    for s in range(1, n_steps + 1):
        lam = Wt @ D
        inf = beta * H * lam
        H = H - dt * inf
        D = D + dt * (inf - rho * D)
        R = R + dt * (rho * D)
        # enforce conservation
        tot = np.clip(H + D + R, 1e-12, None)
        H, D, R = H / tot, D / tot, R / tot
        newly = (D >= ACT_THRESHOLD) & np.isnan(act_t)
        act_t[newly] = s * dt
        if s % rec_every == 0:
            D_traj[frame] = D
            t_traj[frame] = s * dt
            frame += 1
    return D_traj[:frame], t_traj[:frame], act_t


# ═══════════════════════════════════════════════════════════════════════════
# EXPRESSION GENERATION
# ═══════════════════════════════════════════════════════════════════════════

def build_gene_panel():
    """Gene panel: type markers + zone markers + injury program."""
    genes, role = [], []
    for t, mk in TYPE_MARKERS.items():
        for g in mk:
            if g not in genes:
                genes.append(g)
                role.append(("type", t))
    for z, mk in ZONE_MARKERS.items():
        for g in mk:
            if g not in genes:
                genes.append(g)
                role.append(("zone", z))
    for g in INJURY_GENES:
        if g not in genes:
            genes.append(g)
            role.append(("injury", g))
    return genes, role


def generate_expression(D_obs, cell_types, zones, sigma, rng,
                        n_housekeeping=60):
    """NB count matrix coupled to the latent state (manuscript Eq.)."""
    genes, role = build_gene_panel()
    n, G = len(D_obs), len(genes) + n_housekeeping
    alpha = np.exp(rng.normal(0.0, 0.3, G))              # baseline expression
    delta = np.zeros((len(TYPES), G))
    for j, (r, tgt) in enumerate(role):
        if r == "type":
            delta[TYPES.index(tgt), j] = 3.0
    for j, (r, tgt) in enumerate(role):
        if r == "zone":
            pass  # zone markers already in panel; add zone-specific boost below
    zone_delta = np.zeros((3, G))
    for j, (r, tgt) in enumerate(role):
        if r == "zone":
            zone_delta[["inner_medulla", "outer_medulla", "cortex"].index(tgt), j] = 2.5
    gamma_g = np.zeros(G)
    for j, (r, tgt) in enumerate(role):
        if r == "injury":
            gamma_g[j] = 2.0

    mu = np.zeros((n, G))
    type_idx = np.array([TYPES.index(t) for t in cell_types])
    zone_idx = np.array([["inner_medulla", "outer_medulla", "cortex"].index(z)
                         for z in zones])
    for i in range(n):
        log_mu = (np.log(alpha) + delta[type_idx[i]] + zone_delta[zone_idx[i]]
                  + gamma_g * D_obs[i])
        mu[i] = np.exp(log_mu)
    # ambient contamination: background mean 1 -> 1 + 9*sigma
    mu = mu + (1.0 + 9.0 * sigma)
    phi = rng.gamma(2.0, 2.0, G) + 0.1                   # dispersion per gene
    p = 1.0 / (1.0 + mu / phi)
    X = rng.negative_binomial(phi[None, :], p)
    # per-spot count-depth variation
    depth = rng.lognormal(0.0, 0.3 * sigma + 0.05, n)
    X = rng.poisson(X * depth[:, None])
    # dropout, increasing with sigma
    drop_p = np.clip((sigma / 2.0) * (1.0 - X / (X.max() + 1)), 0, 0.5)
    X = X * (rng.random(X.shape) > drop_p)
    return X, genes + [f"hk_{k}" for k in range(n_housekeeping)]


# ═══════════════════════════════════════════════════════════════════════════
# BLIND PIPELINE (counts + coordinates only)
# ═══════════════════════════════════════════════════════════════════════════

def blind_annotation(X, genes, rng, err_level=0.0, ref_profiles=None,
                     err_mode="adjacent"):
    """Marker-based nearest-neighbour annotation against noise-free profiles."""
    genes_l = {g.lower(): i for i, g in enumerate(genes)}
    marker_cols, marker_types = [], []
    for t, mk in TYPE_MARKERS.items():
        for g in mk:
            if g.lower() in genes_l:
                marker_cols.append(genes_l[g.lower()])
                marker_types.append(t)
    sub = X[:, marker_cols].astype(float)
    sub = np.log1p(sub)
    sub = sub / np.maximum(sub.sum(axis=1, keepdims=True), 1e-9)

    # noise-free reference profiles per type (from the generator's panels)
    ref = np.zeros((len(TYPES), len(marker_cols)))
    for k, t in enumerate(marker_types):
        ref[TYPES.index(t), k] = 1.0
    ref = ref / np.maximum(ref.sum(axis=1, keepdims=True), 1e-9)

    # cosine nearest-neighbour
    num = sub @ ref.T
    den = (np.linalg.norm(sub, axis=1, keepdims=True)
           * np.linalg.norm(ref, axis=1)[None, :])
    pred = np.array(TYPES)[np.argmax(num / np.maximum(den, 1e-12), axis=1)]

    # injected annotation error: swap to anatomically adjacent classes
    # (Reviewer 5: "more realistic: only swap anatomically adjacent classes")
    if err_level > 0:
        n_err = int(round(err_level * len(pred)))
        idx = rng.choice(len(pred), n_err, replace=False)
        if err_mode == "uniform":
            choices = [rng.choice([t for t in TYPES if t != pred[i]])
                       for i in idx]
        else:
            ADJ = {"PT": ["DCT", "vascular"], "DCT": ["PT", "TAL"],
                   "TAL": ["DCT", "other"], "vascular": ["PT", "other"],
                   "immune": ["other", "PT"], "other": ["TAL", "vascular"]}
            choices = [rng.choice(ADJ[pred[i]]) for i in idx]
        for i, c in zip(idx, choices):
            pred[i] = c
    return pred


def blind_zoning(X, genes, coords):
    """Marker-based anatomical zoning with radial fallback (as the real pipeline)."""
    genes_l = {g.lower(): i for i, g in enumerate(genes)}
    scores = {}
    for zone, mk in ZONE_MARKERS.items():
        cols = [genes_l[g.lower()] for g in mk if g.lower() in genes_l]
        if not cols:
            scores[zone] = np.zeros(len(X))
            continue
        sub = X[:, cols].astype(float)
        sub = sub / np.maximum(sub.max(axis=0)[None, :], 1e-9)
        scores[zone] = sub.mean(axis=1)
    S = np.stack([scores[z] for z in ["inner_medulla", "outer_medulla", "cortex"]], axis=1)
    best = S.argmax(axis=1)
    zone_lab = np.array(["inner_medulla", "outer_medulla", "cortex"])[best]
    zero = S.max(axis=1) <= 0
    if zero.any():  # radial fallback
        center = coords.mean(axis=0)
        d = np.linalg.norm(coords - center, axis=1)
        q1, q2 = np.quantile(d, [0.33, 0.66])
        zone_lab[zero] = np.where(d[zero] <= q1, "core",
                                  np.where(d[zero] <= q2, "mid", "boundary"))
    return zone_lab


def run_blind_pipeline(X, genes, coords, rng,
                       k=6, edge_del=0.0, kinetics_eps=0.0,
                       uniform_beta=False, err_level=0.0,
                       err_mode="adjacent", oracle_labels=None,
                       exposure="rownorm"):
    """Compartment construction via the REPOSITORY build_compartment2 +
    BDF integration (identical construction to the real-data analysis).

    The blind annotations (pred_type) are passed as cell_identity; zoning is
    the repository's anatomical zoning (ZONE_MARKERS + radial fallback) run
    on the synthetic expression. Edge deletion (graph misspecification) is
    injected by temporarily wrapping the module-level build_knn_graph that
    build_compartment2 calls.

    Annotation error injection (factor 'annotation'): a fraction `err_level`
    of spots has its label swapped AFTER the marker-based classification.
    err_mode='adjacent' swaps only between anatomically adjacent classes
    (Reviewer 5 suggestion); err_mode='uniform' swaps to any of the 6 types
    (diagnostic contrast).

    oracle_labels (diagnostics only): per-spot compartment labels
    (type_zone) from the ground truth — bypasses annotation AND zoning but
    uses the identical graph construction (kNN k=6, 'avg' coarse-graining,
    self-weight patch), so the only difference vs the blind path is label
    quality.

    exposure (diagnostics only): 'rownorm' (canonical, lambda = row-normalised
    weighted adjacency @ D, via sir_compartments._build_W_tilde) or 'binary'
    (lambda from the row-normalised BINARY contact adjacency).

    Same documented patch as the Sobol script: the within-compartment
    self-weight removed by coarse_grain_graph is restored on Wc2 (without it
    the compartment wave cannot sustain itself).
    """
    import anndata as ad
    import compartments as _cmod
    from compartments import build_compartment2

    if oracle_labels is not None:
        pred_type = np.array([l.split("_")[0] for l in oracle_labels])
        zone_lab = np.array(["_".join(l.split("_")[1:]) for l in oracle_labels])
    else:
        pred_type = blind_annotation(X, genes, rng, err_level=err_level,
                                     err_mode=err_mode)

    if oracle_labels is not None:
        # diagnostics: identical graph construction, oracle labels only
        comp_lab = np.asarray(oracle_labels)
        Wsp = build_knn_graph(coords.astype(np.float32), k=k)
        if edge_del > 0:
            Wsp = Wsp.tolil()
            rows, cols = Wsp.nonzero()
            upper = [(r, c) for r, c in zip(rows, cols) if r < c]
            n_del = int(round(edge_del * len(upper)))
            for r, c in rng.permutation(upper)[:n_del]:
                Wsp[r, c] = Wsp[c, r] = 0.0
            Wsp = Wsp.tocsr()
        comps, inv2 = np.unique(comp_lab, return_inverse=True)
        C = len(comps)
        N = len(comp_lab)
        S_mat = sp.csr_matrix((np.ones(N), (np.arange(N), inv2)), shape=(N, C))
        Wc_full = (S_mat.T @ Wsp @ S_mat).tocsr()
        Nc = np.asarray(S_mat.sum(axis=0)).reshape(-1)
        Wc = Wc_full.multiply(1.0 / np.maximum(Nc[:, None] * Nc[None, :], 1.0)).tocsr()
        self_w = np.asarray(Wc_full.diagonal()) / np.maximum(Nc ** 2, 1.0)
        Wc = (Wc + sp.diags(self_w)).tocsr()
        macro_of = np.array([c.split("_")[0] for c in comps])
        region_of = np.array(["_".join(c.split("_")[1:]) for c in comps])
        zone_lab = np.array(["_".join(l.split("_")[1:]) for l in oracle_labels])
    else:
        adata = ad.AnnData(
            X=sp.csr_matrix(X),
            obs=pd.DataFrame({"cell_identity": pred_type}),
            var=pd.DataFrame(index=[g.lower() for g in genes]),
        )
        adata.obsm["spatial"] = coords.astype(np.float32)
        macro_map = {t: t for t in TYPES}      # synthetic labels are already macro

        orig_knn = _cmod.build_knn_graph
        if edge_del > 0:
            def _knn_del(c, k=6):
                W = orig_knn(c.astype(np.float32), k=k)
                W = W.tolil()
                rows, cols = W.nonzero()
                upper = [(r, c2) for r, c2 in zip(rows, cols) if r < c2]
                n_del = int(round(edge_del * len(upper)))
                for r, c2 in rng.permutation(upper)[:n_del]:
                    W[r, c2] = W[c2, r] = 0.0
                return W.tocsr()
            _cmod.build_knn_graph = _knn_del
        try:
            g = build_compartment2(adata, macro_map=macro_map, k_knn=k,
                                   use_anatomical_zones=True, fallback_radial=True)
        finally:
            _cmod.build_knn_graph = orig_knn

        Wsp, Wc = g["Wsp"], g["Wc2"]
        comps, inv2 = g["comps2"], g["inv2"]
        macro_of, region_of = g["macro_of"], g["region_of"]
        C = len(comps)

        # restore the self-weight zeroed by coarse_grain_graph ('avg'):
        # self_i = (S^T Wsp S)[i,i] / N_i^2
        N = Wsp.shape[0]
        S_mat = sp.csr_matrix((np.ones(N), (np.arange(N), inv2)), shape=(N, C))
        Wc_full = (S_mat.T @ Wsp @ S_mat).tocsr()
        Nc = np.asarray(S_mat.sum(axis=0)).reshape(-1)
        self_w = np.asarray(Wc_full.diagonal()) / np.maximum(Nc ** 2, 1.0)
        Wc = (Wc + sp.diags(self_w)).tocsr()
        zone_lab = adata.obs["region"].values

    params = build_default_params_SIR(comps, macro_of, region_of)
    if kinetics_eps != 0.0:
        params["beta"] = params["beta"] * (1.0 + kinetics_eps)
        params["gamma"] = params["gamma"] * (1.0 + kinetics_eps)
    if uniform_beta:
        params["beta"] = np.full(C, 0.30)

    seed_mask = np.isin(region_of, ["outer_medulla", "boundary"])
    S0 = np.ones(C)
    I0 = np.where(seed_mask, SEED_P0, 0.0)
    R0 = np.zeros(C)
    if exposure == "binary":
        # diagnostics: exposure from the BINARY contact adjacency (same
        # row-normalisation inside _build_W_tilde, but on W_bin = Wc > 0)
        W_bin = sp.csr_matrix((Wc > 0).astype(float))
        L_exposure = sp_laplacian(W_bin, normed=True).tocsr()
    else:
        L_exposure = sp_laplacian(sp.csr_matrix(Wc), normed=True).tocsr()
    H_fin, D_fin, R_fin, sol = simulate_SIR_ivp(
        L_exposure, S0, I0, R0, params, t_end=T_END, n_steps=300)
    return dict(comps=comps, inv2=inv2, D_fin=D_fin, sol=sol,
                pred_type=pred_type, zone_lab=zone_lab,
                Wsp=Wsp, Wc=Wc, params=params, seed_mask=seed_mask)


def _laplacian(W):
    W = sp.csr_matrix(W, dtype=float)
    d = np.asarray(W.sum(axis=1)).ravel()
    return sp.diags(d) - W


# ═══════════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════════

def compute_metrics(truth, sim):
    """Common recovery metrics (manuscript 'Common recovery metrics')."""
    comps = sim["comps"]
    C = len(comps)
    D_traj, t_traj, act_t = truth["D_traj"], truth["t_traj"], truth["act_t"]
    inv2 = sim["inv2"]

    # compartment-level truth trajectory (mean over spots of each compartment)
    S_mat = sp.csr_matrix((np.ones(len(inv2)), (np.arange(len(inv2)), inv2)),
                          shape=(len(inv2), C))
    Nc = np.asarray(S_mat.sum(axis=0)).ravel()
    D_true_c = (D_traj @ S_mat.toarray()) / np.maximum(Nc[None, :], 1)

    D_sim_c = sim["sol"].y[C:2 * C]                      # (C, n_steps)
    t_sim = sim["sol"].t

    # 1. RMSE_D over compartments and time (common grid)
    grid = np.linspace(0, T_END, 100)
    D_true_g = np.stack([np.interp(grid, t_traj, D_true_c[:, k])
                         for k in range(C)], axis=1)
    D_sim_g = np.stack([np.interp(grid, t_sim, D_sim_c[k, :])
                        for k in range(C)], axis=1)
    rmse_D = float(np.sqrt(np.mean((D_true_g - D_sim_g) ** 2)))

    # 2. MAE_t — activation time at p^D = 0.1 per compartment
    def act_times(D_c, tt):
        act = np.full(D_c.shape[1], np.nan)
        for k in range(D_c.shape[1]):
            hit = np.where(D_c[:, k] >= ACT_THRESHOLD)[0]
            if len(hit):
                act[k] = tt[hit[0]]
        return act
    a_true = act_times(D_true_g, grid)
    a_sim = act_times(D_sim_g, t_sim)
    ok = np.isfinite(a_true) & np.isfinite(a_sim)
    mae_t = float(np.mean(np.abs(a_true[ok] - a_sim[ok]))) if ok.any() else np.nan

    # 3. r_spatial — correlation of final state across spots
    d_true_spot = D_traj[-1]
    d_sim_spot = D_sim_c[:, -1][inv2]
    r_spatial = float(np.corrcoef(d_true_spot, d_sim_spot)[0, 1])

    # 4. rho_source — Spearman of final-state ranking (compartments)
    d_true_fin_c = D_true_g[-1]
    d_sim_fin_c = D_sim_g[-1]
    rho_source = float(spearmanr(d_true_fin_c, d_sim_fin_c).statistic)

    # 5. tau — Kendall of activation ordering
    ok2 = np.isfinite(a_true) & np.isfinite(a_sim)
    tau = float(kendalltau(a_true[ok2], a_sim[ok2]).statistic) if ok2.sum() >= 3 else np.nan

    # 6. AUROC / AUPRC — true high-flux edges (top quartile) recovery
    Wstar = truth["Wstar"]
    # compartment-pair flux, avg-normalised like the coarse-grained graph:
    #   truth: mean over spot pairs of w*_ij * D*_i(t_obs) * D*_j(t_obs)
    #   sim:   Wc_ab * D_fin,a * D_fin,b
    S_mat = sp.csr_matrix((np.ones(len(inv2)), (np.arange(len(inv2)), inv2)),
                          shape=(len(inv2), C))
    F_true_cc = np.asarray((S_mat.T @ sp.diags(truth["D_obs"]) @ Wstar
                            @ sp.diags(truth["D_obs"]) @ S_mat).todense())
    Nc_mat = np.maximum(np.asarray(S_mat.sum(axis=0)).ravel()[None, :]
                        * np.asarray(S_mat.sum(axis=0)).ravel()[:, None], 1.0)
    F_true_cc = F_true_cc / Nc_mat
    F_sim_cc = sim["Wc"].toarray() * np.outer(sim["D_fin"], sim["D_fin"])
    iu = ~np.eye(C, dtype=bool)
    y_true = (F_true_cc[iu] >= np.quantile(F_true_cc[iu], 0.75)).astype(int)
    score = F_sim_cc[iu]
    from sklearn.metrics import roc_auc_score, average_precision_score
    if y_true.sum() in (0, len(y_true)):
        auroc_edge = auprc_edge = np.nan
    else:
        auroc_edge = float(roc_auc_score(y_true, score))
        auprc_edge = float(average_precision_score(y_true, score))

    # 7. annotation accuracy
    ann = float(np.mean(sim["pred_type"] == truth["cell_types"]))

    return dict(rmse_D=rmse_D, mae_t=mae_t, r_spatial=r_spatial,
                rho_source=rho_source, tau=tau, auroc_edge=auroc_edge,
                auprc_edge=auprc_edge, annotation_accuracy=ann)


# ═══════════════════════════════════════════════════════════════════════════
# SINGLE BENCHMARK RUN
# ═══════════════════════════════════════════════════════════════════════════

def run_one(args):
    (factor, level, noise, rep, base_seed, quick) = args
    rng = np.random.default_rng(base_seed + 1000 * hash((factor, level)) % 100000 + rep)

    coords = make_hex_grid(rng=rng)
    zones = assign_zones(coords, rng)
    cell_types = assign_cell_types(zones, coords, rng)
    Wstar = spot_graph(coords, k=6)

    # factor-specific ground-truth modifications
    beta_het = None
    uniform_beta = False
    seed_zone = "outer_medulla"
    if factor == "model":
        # heterogeneous beta* generated (autocorrelated field), uniform analysed
        from scipy.ndimage import gaussian_filter
        f = gaussian_filter(rng.normal(0, 1, (HEX_SIDE, HEX_SIDE)), 6.0).ravel()[:N_SPOTS]
        f = f / (f.std() + 1e-12)
        beta_het = np.exp(level * f)
        uniform_beta = True
    if factor == "seed":
        seed_zone = "inner_medulla"

    zones_true = zones if factor != "seed" else zones  # zones unchanged
    D_traj, t_traj, act_t = spot_level_truth(
        Wstar, cell_types, zones, rng, beta_het=beta_het)
    D_obs = D_traj[np.searchsorted(t_traj, T_OBS)]
    D_obs = np.minimum(D_obs, 1.0)

    X, genes = generate_expression(D_obs, cell_types, zones, noise, rng)

    sim = run_blind_pipeline(
        X, genes, coords, rng,
        k=6 if factor != "graph_k" else int(level),
        edge_del=level if factor == "edge_del" else 0.0,
        kinetics_eps=level if factor == "kinetics" else 0.0,
        uniform_beta=uniform_beta,
        err_level=level if factor == "annotation" else 0.0)

    # seed misspecification: analyse with remote inner-medulla seed
    if factor == "seed":
        region_of = np.array(["_".join(c.split("_")[1:]) for c in sim["comps"]])
        sim["seed_mask"] = region_of == "inner_medulla"
        C = len(sim["comps"])
        S0 = np.ones(C)
        I0 = np.where(sim["seed_mask"], SEED_P0, 0.0)
        R0 = np.zeros(C)
        Hf, Df, Rf, sol = simulate_SIR_ivp(
            sp_laplacian(sp.csr_matrix(sim["Wc"]), normed=True).tocsr(),
            S0, I0, R0, sim["params"], t_end=T_END, n_steps=300)
        sim["D_fin"], sim["sol"] = Df, sol

    m = compute_metrics(
        dict(D_traj=D_traj, t_traj=t_traj, act_t=act_t, Wstar=Wstar,
             D_obs=D_obs, cell_types=cell_types),
        sim)
    m.update(factor=factor, level=level, noise=noise, rep=rep)
    return m


# ═══════════════════════════════════════════════════════════════════════════
# AGGREGATION + VARIANCE DECOMPOSITION
# ═══════════════════════════════════════════════════════════════════════════

METRICS = ["rmse_D", "mae_t", "r_spatial", "rho_source", "tau",
           "auroc_edge", "auprc_edge", "annotation_accuracy"]


def aggregate(runs_df):
    """Multi-index (factor, level, noise) x (metric, mean/std/count) table."""
    rows = []
    for (f, l, nz), g in runs_df.groupby(["factor", "level", "noise"]):
        row = [f, l, nz]
        for m in METRICS:
            v = g[m].dropna()
            row += [v.mean() if len(v) else np.nan,
                    v.std() if len(v) else np.nan, len(v)]
        rows.append(row)
    cols = pd.MultiIndex.from_tuples(
        [("", "", ""), ("", "", ""), ("", "", "")]
        + [(m, s, "") for m in METRICS for s in ("mean", "std", "count")])
    return pd.DataFrame(rows, columns=cols)


def variance_decomposition(runs_df):
    """
    Regression-based decomposition of recovery loss across all benchmark
    conditions (per-factor strength regressors; partial R^2 shares).
    """
    df = runs_df.copy()
    # composite recovery loss: mean of z-scored metric losses
    loss_parts = []
    for m, sign in [("rmse_D", 1), ("mae_t", 1), ("r_spatial", -1),
                    ("rho_source", -1), ("tau", -1), ("auroc_edge", -1)]:
        v = pd.to_numeric(df[m], errors="coerce")
        z = (v - v.mean()) / (v.std() + 1e-12)
        loss_parts.append(sign * z)
    df["loss"] = pd.concat(loss_parts, axis=1).mean(axis=1)

    X = pd.DataFrame({
        "kinetics_s": df["factor"].eq("kinetics") * df["level"].abs(),
        "annotation_s": df["factor"].eq("annotation") * df["level"],
        "graph_s": (df["factor"].eq("graph_k") * (df["level"] - 6).abs()
                    + df["factor"].eq("edge_del") * df["level"]),
        "seed_s": df["factor"].eq("seed").astype(float),
        "model_s": df["factor"].eq("model") * df["level"],
        "noise_s": df["noise"],
    })
    ok = df["loss"].notna() & X.notna().all(axis=1)
    y = df.loc[ok, "loss"].values
    Xa = X.loc[ok].values
    Xa = np.column_stack([np.ones(ok.sum()), Xa])

    beta_hat, *_ = np.linalg.lstsq(Xa, y, rcond=None)
    resid = y - Xa @ beta_hat
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    full_R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    shares = {}
    for j, name in enumerate(X.columns):
        Xr = np.delete(Xa, j + 1, axis=1)
        b_r, *_ = np.linalg.lstsq(Xr, y, rcond=None)
        r_r = y - Xr @ b_r
        r2_r = 1.0 - float(r_r @ r_r) / ss_tot
        shares[name] = max(full_R2 - r2_r, 0.0)
    tot = sum(shares.values())
    shares = {k: v / tot for k, v in shares.items()} if tot > 0 else shares
    return dict(partial_R2_shares=shares, full_R2=full_R2)


def variance_decomp_tex(vd):
    labels = dict(kinetics_s="Kinetic parameters", annotation_s="Annotation error",
                  graph_s="Graph construction", seed_s="Seed placement",
                  model_s="Model structure", noise_s="Measurement noise")
    rows = "\n".join(
        f"{labels[k]} & {100*v:.1f}\\% \\\\" for k, v in vd["partial_R2_shares"].items())
    return (
        "\\begin{table}[!h]\n\\centering\n"
        "\\caption{\\textbf{Variance decomposition of benchmark recovery loss.} "
        f"Full-model $R^2 = {vd['full_R2']:.2f}$; shares are partial $R^2$ fractions.}}\n"
        "\\label{tab:variance-decomp}\n"
        "\\begin{tabular}{lc}\\toprule Factor & Share of explained loss \\\\\\midrule\n"
        + rows + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="benchmark_outputs")
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true",
                    help="smoke test: 5 replicates per cell")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    nA = 5 if a.quick else N_REPS_A
    nB = 5 if a.quick else N_REPS_B

    jobs = []
    # Experiment A: correctly specified, increasing noise
    for nz in NOISE_LEVELS:
        for rep in range(nA):
            jobs.append(("none", 0.0, nz, rep, a.seed, a.quick))
    # Experiment B: one-factor-at-a-time misspecification (noise = 0)
    for f, l in MISSPEC_GRID:
        for rep in range(nB):
            jobs.append((f, l, 0.0, rep, a.seed, a.quick))
    print(f"Total benchmark runs: {len(jobs)}")

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.n_jobs) as ex:
        results = list(ex.map(run_one, jobs, chunksize=4))
    print(f"Done in {time.time()-t0:.0f}s")
    runs = pd.DataFrame(results)
    runs.to_csv(os.path.join(a.out_dir, "runs_raw.csv"), index=False)

    # Experiment A table — release format: 2 header rows (metric / mean,std),
    # a 'noise' label row, then one row per noise level
    import csv
    expA = runs[runs["factor"] == "none"]
    with open(os.path.join(a.out_dir, "experimentA_noise_response.csv"),
              "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([""] + [m for m in METRICS for _ in (0, 1)])
        w.writerow([""] + ["mean", "std"] * len(METRICS))
        w.writerow(["noise"] + [""] * (2 * len(METRICS)))
        for nz in sorted(expA["noise"].unique()):
            g = expA[expA["noise"] == nz]
            row = [nz]
            for m in METRICS:
                row += [g[m].mean(), g[m].std()]
            w.writerow(row)

    # Experiment B aggregated (includes the 'none' rows, as in the release) —
    # release format: 2 header rows, a 'factor,level,noise' label row, data
    agg = aggregate(runs)
    with open(os.path.join(a.out_dir, "benchmark_aggregated.csv"),
              "w", newline="") as fh:
        w = csv.writer(fh)
        # release file omits the trailing 'count' of the last metric
        w.writerow(["", "", ""] + [m for m in METRICS[:-1] for _ in (0, 0, 0)]
                   + [METRICS[-1], METRICS[-1]])
        w.writerow(["", "", ""] + ["mean", "std", "count"] * (len(METRICS) - 1)
                   + ["mean", "std"])
        w.writerow(["factor", "level", "noise"] + [""] * (3 * len(METRICS) - 1))
        for _, r in agg.iterrows():
            w.writerow([r.iloc[i] for i in range(len(r) - 1)])

    # Variance decomposition
    vd = variance_decomposition(runs)
    with open(os.path.join(a.out_dir, "variance_decomposition.json"), "w") as fh:
        json.dump(vd, fh, indent=1)
    with open(os.path.join(a.out_dir, "table_variance_decomp.tex"), "w") as fh:
        fh.write(variance_decomp_tex(vd))

    print("Experiment A (sigma=0):",
          expA[expA["noise"] == 0][METRICS].mean().to_dict())
    print("Variance decomposition:", json.dumps(vd, indent=1))


if __name__ == "__main__":
    main()
