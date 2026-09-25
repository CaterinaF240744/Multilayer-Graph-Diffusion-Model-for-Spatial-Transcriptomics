"""
diagnose_model_first.py
=======================
Diagnostic for the "model-first" vs "annotation-first" hierarchy in the
variance decomposition of the synthetic benchmark (Reviewer 2, B6/B7
follow-up request). Three blocks, all at sigma=0 unless noted otherwise:

1. staged_mae_decomposition — recovery error per pipeline stage:
     A_oracle   : true labels + true zones (identical build_compartment2 graph)
     B_blindann : blind annotation, true zones
     C_fullblind: blind annotation + blind zoning (full pipeline)
   30 replicates; metrics: mae_t, rho_source, tau, rmse_D, r_spatial.

2. annotation_error_mechanism — contrast between injecting error only
   between anatomically adjacent classes (Reviewer 5) vs uniformly over the
   6 cell types; error levels 5/10/20/30%, 30 replicates, sigma=0.

3. exposure_normalization — stage A (oracle) with the canonical
   row-normalised exposure vs a binary contact adjacency; 30 replicates.

Repository placement
---------------------
Lives in experiments/rebuttal/, next to benchmark_synthetic.py (imported
directly, same directory).

Usage
-----
  python diagnose_model_first.py [--out-dir OUT] [--n-jobs 8] [--reps 30] [--seed 0]

Outputs (CSV, in --out-dir, default ./diagnosis_outputs):
  staged_mae_decomposition.csv, annotation_error_mechanism.csv,
  exposure_normalization.csv
"""
from __future__ import annotations
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

# This script is expected to live in experiments/rebuttal/, alongside
# benchmark_synthetic.py, which it reuses directly.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchmark_synthetic as bs  # noqa: E402


def one_truth(seed):
    """Ground truth + observed expression at sigma=0 for one replicate."""
    rng = np.random.default_rng(seed)
    coords = bs.make_hex_grid(rng=rng)
    zones = bs.assign_zones(coords, rng)
    cell_types = bs.assign_cell_types(zones, coords, rng)
    Wstar = bs.spot_graph(coords, k=6)
    D_traj, t_traj, act_t = bs.spot_level_truth(Wstar, cell_types, zones, rng)
    D_obs = np.minimum(D_traj[np.searchsorted(t_traj, bs.T_OBS)], 1.0)
    X, genes = bs.generate_expression(D_obs, cell_types, zones, 0.0, rng)
    oracle_labels = np.array([f"{t}_{z}" for t, z in zip(cell_types, zones)])
    return dict(coords=coords, zones=zones, cell_types=cell_types,
                Wstar=Wstar, D_traj=D_traj, t_traj=t_traj, act_t=act_t,
                D_obs=D_obs, X=X, genes=genes, oracle_labels=oracle_labels)


def metrics(truth, sim):
    return bs.compute_metrics(
        dict(D_traj=truth["D_traj"], t_traj=truth["t_traj"],
             act_t=truth["act_t"], Wstar=truth["Wstar"],
             D_obs=truth["D_obs"], cell_types=truth["cell_types"]), sim)


def staged_row(seed, variant):
    tr = one_truth(seed)
    rng = np.random.default_rng(10_000 + seed)
    kw = dict(k=6)
    if variant == "A_oracle":
        sim = bs.run_blind_pipeline(tr["X"], tr["genes"], tr["coords"], rng,
                                    oracle_labels=tr["oracle_labels"], **kw)
    elif variant == "B_blindann":
        # blind annotation, oracle zones: oracle labels built from predicted
        # types + true zones (same graph construction path as A_oracle)
        sim_full_labels = bs.blind_annotation(tr["X"], tr["genes"], rng)
        lab = np.array([f"{t}_{z}" for t, z in
                        zip(sim_full_labels, tr["zones"])])
        sim = bs.run_blind_pipeline(tr["X"], tr["genes"], tr["coords"], rng,
                                    oracle_labels=lab, **kw)
    elif variant == "C_fullblind":
        sim = bs.run_blind_pipeline(tr["X"], tr["genes"], tr["coords"], rng,
                                    **kw)
    else:
        raise ValueError(variant)
    m = metrics(tr, sim)
    m.update(stage=variant, seed=seed)
    return m


def mechanism_row(seed, level, mode):
    tr = one_truth(seed)
    rng = np.random.default_rng(20_000 + seed)
    sim = bs.run_blind_pipeline(tr["X"], tr["genes"], tr["coords"], rng,
                                err_level=level, err_mode=mode, k=6)
    m = metrics(tr, sim)
    m.update(level=level, mode=mode, seed=seed)
    return m


def exposure_row(seed, exposure):
    tr = one_truth(seed)
    rng = np.random.default_rng(30_000 + seed)
    sim = bs.run_blind_pipeline(tr["X"], tr["genes"], tr["coords"], rng,
                                oracle_labels=tr["oracle_labels"], k=6,
                                exposure=exposure)
    m = metrics(tr, sim)
    m.update(exposure=exposure, seed=seed)
    return m


def _run_job(j):
    """Module-level dispatcher (picklable) for the process pool."""
    kind, s, arg = j
    if kind == "staged":
        return {**staged_row(s, arg), "block": "staged"}
    if kind == "mech":
        lvl, mode = arg
        return {**mechanism_row(s, lvl, mode), "block": "mechanism"}
    return {**exposure_row(s, arg), "block": "exposure"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="diagnosis_outputs")
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    seeds = [a.seed * 1_000_000 + i for i in range(a.reps)]
    jobs, tag = [], f"[{pd.Timestamp.now().strftime('%H:%M:%S')}]"

    # 1. staged decomposition
    for v in ["A_oracle", "B_blindann", "C_fullblind"]:
        jobs += [("staged", s, v) for s in seeds]
    # 2. annotation mechanism contrast
    for lvl in [0.05, 0.1, 0.2, 0.3]:
        for mode in ["adjacent", "uniform"]:
            jobs += [("mech", s, (lvl, mode)) for s in seeds]
    # 3. exposure normalisation
    for exp in ["rownorm", "binary"]:
        jobs += [("expo", s, exp) for s in seeds]

    print(f"{tag} {len(jobs)} diagnostic runs")
    with ProcessPoolExecutor(max_workers=a.n_jobs) as ex:
        rows = list(ex.map(_run_job, jobs, chunksize=2))
    df = pd.DataFrame(rows)

    staged = df[df.block == "staged"]
    staged.groupby("stage")[["mae_t", "rho_source", "tau", "rmse_D",
                             "r_spatial"]].mean().round(3).to_csv(
        os.path.join(a.out_dir, "staged_mae_decomposition.csv"))
    mech = df[df.block == "mechanism"]
    mech.groupby(["mode", "level"])[["mae_t", "rho_source", "tau",
                                     "annotation_accuracy"]].mean().round(3) \
        .to_csv(os.path.join(a.out_dir, "annotation_error_mechanism.csv"))
    expo = df[df.block == "exposure"]
    expo.groupby("exposure")[["mae_t", "rho_source", "tau", "rmse_D"]] \
        .mean().round(3).to_csv(
            os.path.join(a.out_dir, "exposure_normalization.csv"))

    print("\n=== STAGED (sigma=0) ===")
    print(staged.groupby("stage")[["mae_t", "rho_source", "tau", "rmse_D"]]
          .mean().round(3))
    print("\n=== ANNOTATION MECHANISM (sigma=0) ===")
    print(mech.groupby(["mode", "level"])[["mae_t", "annotation_accuracy"]]
          .mean().round(3))
    print("\n=== EXPOSURE (sigma=0, oracle) ===")
    print(expo.groupby("exposure")[["mae_t", "rho_source", "rmse_D"]]
          .mean().round(3))


if __name__ == "__main__":
    main()
