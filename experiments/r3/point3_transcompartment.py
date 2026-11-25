"""
point3_transcompartment.py — R3, Reviewer 2 point 3.

Recomputes the trans-compartment metrics of Section 6.5 (Table 5) with the
corrected transmission matrix, and adds the two analyses the reviewer asks
for to support (or drop) the "absorber" interpretation:

1. K_dyn with the model's own flux, beta_j * E_ji * pD_i * pH_j (receiver
   susceptibility, exposure matrix actually simulated). Column sums equal the
   infected mass of each layer (checked). Source/receiver strengths are also
   reported per node and as net inter-layer export.
2. Pass-through test (operational definition of an absorber): each layer X is
   made neutral (rho_X := beta_X, local R_diff = 1) and the change in disease
   burden (AUC of p^D) of the OTHER layers is measured.
      burden falls  -> X was AMPLIFYING the spread it received
      burden rises  -> X was ABSORBING it
3. Local R_diff per layer (beta/rho, node mean).

Input : results/annotation_runs/NN/model_context.pkl  (01b_context step, or
        QUICK_EXPORT=1 RUN_TAG=NN python experiments/run_main.py)
Output: results/r3/point3/  (K_dyn.csv, K_struct.csv, source_receiver.csv,
        retention_dispersal.csv, pass_through_test.csv, summary.json)
Usage : MPLBACKEND=Agg python experiments/r3/point3_transcompartment.py
"""
import contextlib
import io
import json
import os
import pickle
import sys
import warnings

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from sir_multilayer import run_SIR_multilayer  # noqa: E402
from sir_transcompartment_metrics import (  # noqa: E402
    _extract_SIR, next_generation_matrix_dynamic, structural_invasion_matrix,
    source_sink_scores, retention_dispersal)

CTX = os.path.join(ROOT, "results", "annotation_runs", "NN", "model_context.pkl")
OUT = os.path.join(ROOT, "results", "r3", "point3")
KAPPA, SEED, T_END, N_STEPS = 0.01, 0.05, 60.0, 2401
TYPES = ["PT", "DCT", "TAL"]


def run_ml(comp, params, adata):
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        return run_SIR_multilayer(comp=comp, sir_params=params, adata_st=adata,
                                  seed_I=SEED, D_inter=KAPPA, t_end=T_END,
                                  n_steps=N_STEPS, threshold_I=0.10, plot=False)


def layer_auc(ml):
    n = ml["net"]["n_total"]
    lab = np.asarray(ml["net"]["layer_of_global"])
    t, pD = ml["sol"].t, ml["sol"].y[n:2 * n]
    auc = np.trapezoid(pD, t, axis=1)
    return {L: float(auc[lab == L].sum()) for L in TYPES}


def main():
    os.makedirs(OUT, exist_ok=True)
    ctx = pickle.load(open(CTX, "rb"))
    comp, params, adata = ctx["comp"], ctx["params"], ctx["adata"]
    macro_of = np.asarray(comp["macro_of"])

    # ---- 1. corrected transmission matrix --------------------------------
    ml = run_ml(comp, params, adata)
    gi = np.asarray(ml["net"]["global_index"])
    labels = np.asarray(ml["net"]["layer_of_global"])
    beta, rho = params["beta"][gi], params["gamma"][gi]
    S, I, t = _extract_SIR(ml["sol"])
    K = next_generation_matrix_dynamic(S, I, t, ml["E"], labels, beta).loc[TYPES, TYPES]
    lost_H = pd.Series({L: float((S[0, labels == L] - S[-1, labels == L]).sum())
                        for L in TYPES})
    attribution_err = float((K.sum(axis=0) - lost_H).abs().max() / lost_H.max())
    K_struct, K_norm = structural_invasion_matrix(ml["E"], labels, beta, rho)
    ss = source_sink_scores(K, labels)
    rd = retention_dispersal(K)
    rdiff = pd.Series({L: float(np.mean(beta[labels == L] / rho[labels == L]))
                       for L in TYPES}, name="R_diff_mean")
    ss = ss.merge(rdiff.rename_axis("compartment").reset_index(), on="compartment")

    # ---- 2. pass-through test ---------------------------------------------
    base = layer_auc(ml)
    rows = []
    for X in TYPES:
        p = {k: (np.array(v, dtype=float, copy=True) if isinstance(v, np.ndarray) else v)
             for k, v in params.items()}
        m = macro_of == X
        p["gamma"][m] = p["beta"][m]                  # local R_diff = 1
        auc = layer_auc(run_ml(comp, p, adata))
        others = [L for L in TYPES if L != X]
        b0 = sum(base[L] for L in others); b1 = sum(auc[L] for L in others)
        rows.append(dict(layer=X, R_diff_mean=rdiff[X],
                         burden_others_baseline=b0, burden_others_neutral=b1,
                         delta_pct=100 * (b1 - b0) / b0,
                         role="amplifier" if b1 < b0 else "absorber"))
    pt = pd.DataFrame(rows)

    # ---- outputs ----------------------------------------------------------
    K.round(5).to_csv(f"{OUT}/K_dyn.csv")
    K_struct.loc[TYPES, TYPES].round(5).to_csv(f"{OUT}/K_struct.csv")
    ss.round(5).to_csv(f"{OUT}/source_receiver.csv", index=False)
    rd.round(4).to_csv(f"{OUT}/retention_dispersal.csv", index=False)
    pt.round(4).to_csv(f"{OUT}/pass_through_test.csv", index=False)
    json.dump(dict(kappa=KAPPA, seed=SEED, t_end=T_END, n_steps=N_STEPS,
                   K_column_sum_vs_infected_mass_max_rel_err=attribution_err,
                   infected_mass=lost_H.round(5).to_dict(),
                   baseline_burden=base),
              open(f"{OUT}/summary.json", "w"), indent=1)

    pd.set_option("display.width", 140)
    print("K_dyn (rows = source, columns = receiver):\n", K.round(4), "\n")
    print(f"Attribution check (column sums vs infected mass): max rel. error "
          f"{100*attribution_err:.2f}%\n")
    print(ss.round(4).to_string(index=False), "\n")
    print("Pass-through test:\n", pt.round(3).to_string(index=False))
    print(f"\nOutputs in {OUT}")


if __name__ == "__main__":
    main()
