"""
predictive_test_gse269622.py — R3, Reviewer 2 point 3.

Implements the pre-specified plan in PREREGISTRATION_predictive_test.md
(committed before this analysis was run). Do not change thresholds, markers,
predictors or tests here without reporting the deviation.

Stage 1 (one sample per process, memory-lean, like datasetB_lean.py):
    python experiments/r3/predictive_test_gse269622.py sample sham
    python experiments/r3/predictive_test_gse269622.py sample hour4
    python experiments/r3/predictive_test_gse269622.py sample hour12
    python experiments/r3/predictive_test_gse269622.py sample day2
Stage 2 (tests, no heavy data):
    python experiments/r3/predictive_test_gse269622.py test

Outputs in results/r3/predictive/.
"""
import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "experiments", "rebuttal"))
import model_core as mc  # noqa: E402

OUT = os.path.join(ROOT, "results", "r3", "predictive")
MARKERS = ["Havcr1", "Lcn2", "Vcam1", "Spp1"]       # = INJURY_MARKERS (R2)
MIN_SPOTS, SEED_D, T_END, THR, N_EVAL = 5, 0.05, 60.0, 0.10, 1201
N_PERM, RNG_SEED = 10_000, 0
PRIMARY, SECONDARY = "hour12", ["hour4", "day2"]


# ─────────────────────────────────────────────────────────────────────────
# Pure functions (unit-testable without data)
# ─────────────────────────────────────────────────────────────────────────

def activation_times(comp, params):
    """Simulate the canonical single-layer model on a compartment graph and
    return per-compartment t_act (first p^D >= THR; T_END if never)."""
    Wc2 = comp["Wc2"]
    Wc2 = Wc2.toarray() if hasattr(Wc2, "toarray") else np.asarray(Wc2)
    n = Wc2.shape[0]
    tr = mc.simulate(mc.row_normalise(Wc2), params["beta"], params["gamma"],
                     mc.initial_state(n, comp["boundary_mask"], SEED_D),
                     T_END, n_eval=N_EVAL, where="predictive_test")
    t_act = np.full(n, T_END)
    for i in range(n):
        hit = np.where(tr.pD[i] >= THR)[0]
        if len(hit):
            t_act[i] = tr.t[hit[0]]
    auc = np.trapezoid(tr.pD, tr.t, axis=1)
    return t_act, auc


def compartment_scores(X_markers, inv2, n_comp):
    """Mean per-spot injury score (mean over markers) per compartment."""
    spot_score = np.asarray(X_markers).mean(axis=1)
    sums = np.bincount(inv2, weights=spot_score, minlength=n_comp)
    counts = np.bincount(inv2, minlength=n_comp)
    with np.errstate(invalid="ignore", divide="ignore"):
        return sums / counts, counts


def spearman(x, y):
    from scipy.stats import rankdata
    rx, ry = rankdata(x), rankdata(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def perm_test(x, y, strata=None, n_perm=N_PERM, seed=RNG_SEED):
    """One-sided (rho > 0) permutation test of Spearman rho. With strata,
    ranks are residualised within strata and y is permuted within strata."""
    from scipy.stats import rankdata
    x, y = np.asarray(x, float), np.asarray(y, float)
    rx, ry = rankdata(x), rankdata(y)
    if strata is not None:
        strata = np.asarray(strata)
        for s in np.unique(strata):
            m = strata == s
            rx[m] -= rx[m].mean(); ry[m] -= ry[m].mean()
        keep = np.array([np.sum(strata == s) > 1 for s in strata])
        rx, ry, strata = rx[keep], ry[keep], strata[keep]
        if len(rx) < 4 or np.allclose(rx, 0) or np.allclose(ry, 0):
            return dict(rho=None, p=None, n=int(len(rx)))
    obs = float(np.corrcoef(rx, ry)[0, 1])
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for k in range(n_perm):
        if strata is None:
            yp = rng.permutation(ry)
        else:
            yp = ry.copy()
            for s in np.unique(strata):
                m = np.where(strata == s)[0]
                yp[m] = ry[rng.permutation(m)]
        null[k] = np.corrcoef(rx, yp)[0, 1]
    p = (1 + np.sum(null >= obs - 1e-12)) / (n_perm + 1)
    return dict(rho=round(obs, 3), p=round(float(p), 4), n=int(len(rx)))


def holm(pvals):
    order = np.argsort(pvals)
    adj = np.empty(len(pvals)); running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = running
    return adj


# ─────────────────────────────────────────────────────────────────────────
# Stage 1: one sample
# ─────────────────────────────────────────────────────────────────────────

def stage_sample(name, data_dir, repo_data_dir):
    import gc
    import scipy.sparse as sp
    from datasetB_lean import (SAMPLE_SUBPATHS, build_reference, prep_sample,
                               knn_transfer)
    from compartments import build_compartment2
    from sir_compartments import build_default_params_SIR

    warnings.filterwarnings("ignore")
    ref = build_reference(repo_data_dir)
    st = prep_sample(os.path.join(data_dir, SAMPLE_SUBPATHS[name]))
    st = knn_transfer(ref, st); del ref; gc.collect()
    comp = build_compartment2(st)
    comps = np.asarray(comp["comps2"])
    params = build_default_params_SIR(comps, comp["macro_of"], comp["region_of"])
    t_act, auc = activation_times(comp, params)

    used = [g for g in MARKERS if g in st.var_names]
    X = st[:, used].X
    X = X.toarray() if sp.issparse(X) else np.asarray(X)
    score, counts = compartment_scores(X, np.asarray(comp["inv2"]), len(comps))

    df = pd.DataFrame(dict(
        compartment=comps, macro=comp["macro_of"], zone=comp["region_of"],
        n_spots=counts, seeded=comp["boundary_mask"],
        beta=params["beta"], rho=params["gamma"],
        R_diff=params["beta"] / params["gamma"],
        t_act=t_act, auc_D=auc, injury_score=score))
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(f"{OUT}/{name}_compartments.csv", index=False)
    json.dump(dict(sample=name, markers_used=used, n_spots=int(st.n_obs)),
              open(f"{OUT}/{name}_meta.json", "w"), indent=1)
    print(df.round(3).to_string(index=False))
    print(f"[{name}] markers used: {used} -> {OUT}/{name}_compartments.csv")


# ─────────────────────────────────────────────────────────────────────────
# Stage 2: pre-specified tests
# ─────────────────────────────────────────────────────────────────────────

def compare(sham, inj):
    m = sham.merge(inj, on="compartment", suffixes=("_sham", "_inj"))
    keep = (m.n_spots_sham >= MIN_SPOTS) & (m.n_spots_inj >= MIN_SPOTS)
    excluded = m.loc[~keep, "compartment"].tolist() + sorted(
        set(sham.compartment) ^ set(inj.compartment))
    m = m[keep].copy()
    m["delta_injury"] = m.injury_score_inj - m.injury_score_sham
    m["pred_sham"] = -m.t_act_sham          # primary predictor
    m["pred_own"] = -m.t_act_inj            # secondary predictor
    return m, excluded


def stage_test():
    tabs = {s: pd.read_csv(f"{OUT}/{s}_compartments.csv")
            for s in ["sham", PRIMARY] + SECONDARY}
    res, merged = {}, {}
    for tp in [PRIMARY] + SECONDARY:
        m, excl = compare(tabs["sham"], tabs[tp])
        merged[tp] = m
        res[tp] = dict(
            n=int(len(m)), excluded=excl,
            model_sham_graph=perm_test(m.pred_sham, m.delta_injury),
            model_own_graph=perm_test(m.pred_own, m.delta_injury),
            within_macro=perm_test(m.pred_sham, m.delta_injury, strata=m.macro_sham),
            baseline_R_diff_only=perm_test(m.R_diff_sham, m.delta_injury),
        )
    p_sec = np.array([res[s]["model_sham_graph"]["p"] for s in SECONDARY])
    for s, pa in zip(SECONDARY, holm(p_sec)):
        res[s]["model_sham_graph"]["p_holm"] = round(float(pa), 4)

    h1 = res[PRIMARY]["model_sham_graph"]
    h3 = res[PRIMARY]["within_macro"]
    h1_sig = h1["p"] is not None and h1["p"] < 0.05
    h3_sig = h3["p"] is not None and h3["p"] < 0.05
    verdict = ("The model predicts compartment-level injury beyond its macro-type priors."
               if h1_sig and h3_sig else
               "The predictive ordering is explained by macro-type susceptibility "
               "(literature priors), not by spatial propagation."
               if h1_sig else
               "No evidence of predictive validity on GSE269622 (inconclusive at this n).")
    summary = dict(plan="experiments/r3/PREREGISTRATION_predictive_test.md",
                   primary=PRIMARY, markers=MARKERS, min_spots=MIN_SPOTS,
                   n_perm=N_PERM, results=res,
                   H1_significant=h1_sig, H3_significant=h3_sig,
                   prespecified_conclusion=verdict)
    json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1, default=str)
    pd.concat({k: v for k, v in merged.items()}, names=["timepoint"]).to_csv(
        f"{OUT}/merged_compartments.csv")
    _plot(merged, res)

    for tp in [PRIMARY] + SECONDARY:
        r = res[tp]
        print(f"\n[{tp}] n = {r['n']}  (excluded: {r['excluded']})")
        for k in ["model_sham_graph", "model_own_graph", "within_macro",
                  "baseline_R_diff_only"]:
            print(f"   {k:22s} {r[k]}")
    print(f"\nH1 significant: {h1_sig} | H3 significant: {h3_sig}")
    print(f"Pre-specified conclusion: {verdict}")


def _plot(merged, res):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    tps = [PRIMARY] + SECONDARY
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
    for ax, tp in zip(axes, tps):
        m = merged[tp]
        for mac, g in m.groupby("macro_sham"):
            ax.scatter(g.t_act_sham, g.delta_injury, label=mac, s=40)
        r = res[tp]["model_sham_graph"]
        ax.set_title(f"{tp}: ρ = {r['rho']}, p = {r['p']}, n = {r['n']}")
        ax.set_xlabel("simulated t_act on sham graph (earlier = more vulnerable)")
        ax.axhline(0, color="grey", lw=0.5)
    axes[0].set_ylabel("Δ injury score vs sham")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/predictive_test_scatter.png", dpi=200)
    fig.savefig(f"{OUT}/predictive_test_scatter.svg")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("name", choices=["sham", "hour4", "hour12", "day2"])
    s.add_argument("--data-dir", default="data/GSE269622/extracted")
    s.add_argument("--repo-data-dir", default="data")
    sub.add_parser("test")
    a = ap.parse_args()
    if a.cmd == "sample":
        stage_sample(a.name, a.data_dir, a.repo_data_dir)
    else:
        stage_test()


if __name__ == "__main__":
    main()
