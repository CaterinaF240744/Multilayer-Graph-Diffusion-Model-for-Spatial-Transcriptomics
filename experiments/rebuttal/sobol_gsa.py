"""
sobol_gsa.py -- Global sensitivity analysis (Sobol) of the CANONICAL case-study
model (Section "Global sensitivity analysis"; responses R2.2 and R3).

Same model as the case study (Table 4): multilayer exposure-driven ODE
(sir_multilayer.run_SIR_multilayer) on the compartment graph of the V1 Mouse
Kidney pipeline run, canonical PT/DCT/TAL layers and inter-layer topology,
seed p^D_0 = 0.05 in the canonical seed compartments, t_end = 60, threshold 0.10.

Parameters (9), each varied as a multiplier in [0.6, 1.4] (+-40%) of its
case-study value, so that zone-specific multipliers are preserved:
  beta_PT, beta_DCT, beta_TAL, rho_PT, rho_DCT, rho_TAL, D_H, D_D, kappa
Saltelli sampling, N = 256, no second-order terms -> N (D + 2) = 2,816 runs.

Outputs per run:
  t_inv      global invasion time (multilayer metrics)
  peak_D     global peak diseased fraction
  auc_D      integral of the mean diseased fraction over nodes
  t_inv_PT   invasion time of the PT layer
Ranking stability: Spearman correlation between the per-compartment peak
burden of each run and of the baseline (mean, SD, 5th/95th percentiles).

Requires the model context saved by main.py (QUICK_EXPORT=1 RUN_TAG=NN):
  results/annotation_runs/NN/model_context.pkl

Usage (repository root, environment with SALib):
  python experiments/rebuttal/sobol_gsa.py [--N 256] [--n-jobs 4] [--quick]
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.show = lambda *args, **kwargs: None
import scanpy as sc
import scanpy.plotting as _scpl
sc.pl.spatial = _scpl.spatial = lambda *args, **kwargs: None
import argparse, contextlib, io, json, os, pickle, sys, time, warnings
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.stats import spearmanr
from SALib.sample import saltelli
from SALib.analyze import sobol as sobol_analyze

sys.path.insert(0, "src")
from sir_multilayer import run_SIR_multilayer
from sir_transcompartment_metrics import _extract_SIR

CTX_PATH = "results/annotation_runs/NN/model_context.pkl"
SEED_I, D_INTER, T_END, N_STEPS, THR = 0.05, 0.05, 60.0, 800, 0.10
NAMES = ["beta_PT", "beta_DCT", "beta_TAL", "rho_PT", "rho_DCT", "rho_TAL", "D_H", "D_D", "kappa"]
PROBLEM = dict(num_vars=len(NAMES), names=NAMES, bounds=[[0.6, 1.4]] * len(NAMES))
OUTPUTS = ["t_inv", "peak_D", "auc_D", "t_inv_PT"]
_trapz = getattr(np, "trapezoid", None) or np.trapz
_CTX = {}


def _init(ctx):
    _CTX.update(ctx)


def run(m):
    """m = vector of 9 multipliers."""
    comp, base, adata = _CTX["comp"], _CTX["params"], _CTX["adata"]
    macro_of = np.asarray(comp["macro_of"])
    p = {k: np.array(v, dtype=float, copy=True) for k, v in base.items()}
    for j, mt in enumerate(["PT", "DCT", "TAL"]):
        sel = macro_of == mt
        p["beta"][sel] *= m[j]
        p["gamma"][sel] *= m[3 + j]
    p["D_S"] *= m[6]
    tub = np.isin(macro_of, ["PT", "DCT", "TAL"])
    p["D_I"][tub] *= m[7]
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        ml = run_SIR_multilayer(comp=comp, sir_params=p, adata_st=adata, seed_I=SEED_I,
                                D_inter=D_INTER * m[8], t_end=T_END, n_steps=N_STEPS,
                                threshold_I=THR, plot=False)
    
    plt.close("all")
    
    met = ml["metrics"]
    S, I, t = _extract_SIR(ml["sol"])
    I = np.asarray(I)
    if I.shape[0] != len(t):
        I = I.T
    pt = met["by_layer"].get("PT", {}).get("invasion_time")
    gi = met.get("global_inv_time")
    out = [np.nan if gi is None else float(gi), float(met.get("global_peak_I", np.nan)),
           float(_trapz(I.mean(axis=1), t)), np.nan if pt is None else float(pt)]
    return out, I.max(axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=256)
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--out-dir", default="results/gsa")
    ap.add_argument("--quick", action="store_true", help="N = 8 smoke test")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    N = 8 if a.quick else a.N
    ctx = pickle.load(open(CTX_PATH, "rb"))
    _init(ctx)

    base_out, base_peak = run(np.ones(len(NAMES)))
    print("Baseline (must match Table 4):", dict(zip(OUTPUTS, np.round(base_out, 3))), flush=True)

    try:
        from SALib.sample import sobol as sobol_sample
        X = sobol_sample.sample(PROBLEM, N, calc_second_order=False, seed=0)
    except (ImportError, TypeError):
        np.random.seed(0)
        X = saltelli.sample(PROBLEM, N, calc_second_order=False)
    print(f"Sobol: {len(X)} model evaluations", flush=True)
    Y = np.zeros((len(X), len(OUTPUTS)))
    peaks = np.zeros((len(X), len(base_peak)))
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.n_jobs, initializer=_init, initargs=(ctx,)) as ex:
        for i, (o, pk) in enumerate(ex.map(run, list(X), chunksize=8)):
            Y[i], peaks[i] = o, pk
            if (i + 1) % 200 == 0 or i + 1 == len(X):
                print(f"  {i+1}/{len(X)} ({time.time()-t0:.0f}s)", flush=True)

    n_nan = {o: int(np.isnan(Y[:, j]).sum()) for j, o in enumerate(OUTPUTS)}
    Yimp = np.where(np.isnan(Y), np.nanmean(Y, axis=0), Y)
    idx = {}
    for j, o in enumerate(OUTPUTS):
        si = sobol_analyze.analyze(PROBLEM, Yimp[:, j], calc_second_order=False, seed=0)
        idx[o] = {k: dict(zip(NAMES, np.round(si[k], 3).tolist()))
                  for k in ["S1", "S1_conf", "ST", "ST_conf"]}
    rho = np.array([spearmanr(base_peak, peaks[i]).statistic for i in range(len(X))])
    rho = rho[~np.isnan(rho)]
    res = dict(model="canonical case-study multilayer model", N=N, n_eval=int(len(X)),
               bounds_multiplier=[0.6, 1.4], seed_I=SEED_I, kappa_base=D_INTER, t_end=T_END,
               baseline=dict(zip(OUTPUTS, base_out)), n_nan_imputed=n_nan, indices=idx,
               rank_stability=dict(mean=float(rho.mean()), sd=float(rho.std(ddof=1)),
                                   p05=float(np.percentile(rho, 5)), p95=float(np.percentile(rho, 95))))
    json.dump(res, open(f"{a.out_dir}/sobol_results.json", "w"), indent=1)
    print(json.dumps({o: idx[o]["ST"] for o in OUTPUTS}, indent=1))
    print("rank stability:", res["rank_stability"])


if __name__ == "__main__":
    main()
