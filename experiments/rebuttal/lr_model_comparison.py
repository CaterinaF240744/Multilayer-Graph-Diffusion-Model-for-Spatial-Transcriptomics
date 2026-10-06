"""
lr_model_comparison.py -- Comparison between the simulation and the
ligand-receptor methods (Section "Comparison with ligand-receptor communication
inference"): Tables lr-rankings and panel-vs-full (agreement rows), Figure
fig:commot-compartment and all rho / p values quoted in the text.

Inputs
  results/annotation_runs/NN/model_context.pkl    (main.py, QUICK_EXPORT=1 RUN_TAG=NN)
  results/cellchat_full_v2/cellchat_aggregated_weight.csv, cellchat_signaling_roles.csv
  results/commot_full_v2/spot_level_disThr{150,250,344,400}.csv
  results/commot_full_v2/uns_commot_cluster-macro_type-cellchat-total-total-communication_matrix_disThr{...}.csv

Outputs (results/lr_comparison/)
  summary.json, K_dyn_macro.csv, K_dyn_compartment.csv, spot_compartment.csv,
  compartment_table_disThr{thr}.csv, fig_commot_compartment.pdf/.png

Usage (repository root, environment stpipeline):
  MPLBACKEND=Agg python experiments/rebuttal/lr_model_comparison.py
"""
import contextlib, io, itertools, json, os, pickle, sys, warnings
import numpy as np, pandas as pd, scipy.sparse as sp
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, rankdata

sys.path.insert(0, "src")
from sir_multilayer import run_SIR_multilayer
from sir_transcompartment_metrics import _extract_SIR, next_generation_matrix_dynamic

CTX = "results/annotation_runs/NN/model_context.pkl"
CC, CM = "results/cellchat_full_v2", "results/commot_full_v2"
OUT = "results/lr_comparison"
THRS = [250, 150, 344, 400]                     # first = main threshold
TYPES = ["DCT", "PT", "TAL"]
MIN_SPOTS, N_PERM = 10, 200000
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(0)


def qap_offdiag(A, B):
    """Spearman of off-diagonal entries and exact one-sided QAP p (all 3! permutations)."""
    m = ~np.eye(3, dtype=bool)
    r = spearmanr(A[m], B[m]).statistic
    null = [spearmanr(A[m], B[np.ix_(p, p)][m]).statistic for p in itertools.permutations(range(3))]
    return dict(rho=round(float(r), 3), p_one_sided=round(float(np.mean(np.array(null) >= r - 1e-12)), 3))


def perm_spearman(x, y):
    rx, ry = rankdata(x), rankdata(y)
    r = np.corrcoef(rx, ry)[0, 1]
    null = np.array([np.corrcoef(rx, rng.permutation(ry))[0, 1] for _ in range(N_PERM)])
    return float(r), float((np.abs(null) >= abs(r) - 1e-12).mean())


# ---- model: K_dyn at macro and compartment level, from the case-study run ----
ctx = pickle.load(open(CTX, "rb"))
comp, params, adata = ctx["comp"], ctx["params"], ctx["adata"]
with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
    warnings.simplefilter("ignore")
    ml = run_SIR_multilayer(comp=comp, sir_params=params, adata_st=adata, seed_I=0.05,
                            D_inter=0.05, t_end=60.0, n_steps=800, threshold_I=0.10, plot=False)
gi = np.asarray(ml["net"]["global_index"])
comps2 = np.asarray(comp["comps2"])
W = sp.csr_matrix(comp["Wc2"])[np.ix_(gi, gi)]
S, I, t = _extract_SIR(ml["sol"])
K_mac = next_generation_matrix_dynamic(S, I, t, W, np.asarray(ml["net"]["layer_of_global"]),
                                       params["beta"][gi]).loc[TYPES, TYPES]
K_cmp = next_generation_matrix_dynamic(S, I, t, W, comps2[gi], params["beta"][gi])
K_mac.to_csv(f"{OUT}/K_dyn_macro.csv"); K_cmp.to_csv(f"{OUT}/K_dyn_compartment.csv")
spot_comp = pd.Series(np.asarray(adata.obs["compartment2"]), index=adata.obs_names, name="compartment")
spot_comp.to_csv(f"{OUT}/spot_compartment.csv")
src_m, snk_m = K_mac.sum(axis=1), K_mac.sum(axis=0)

summ = dict(K_dyn_macro=dict(source=src_m.round(3).to_dict(), sink=snk_m.round(3).to_dict()))

# ---- CellChat ----
Wcc = pd.read_csv(f"{CC}/cellchat_aggregated_weight.csv", index_col=0).loc[TYPES, TYPES]
roles = pd.read_csv(f"{CC}/cellchat_signaling_roles.csv").set_index("macro_type")
summ["CellChat"] = dict(outgoing=roles["outgoing_strength"].round(3).to_dict(),
                        incoming=roles["incoming_strength"].round(3).to_dict(),
                        offdiag_vs_Kdyn=qap_offdiag(K_mac.values, Wcc.values))

# ---- COMMOT, macro-type level ----
summ["COMMOT"] = {}
for thr in THRS:
    s = pd.read_csv(f"{CM}/spot_level_disThr{thr}.csv")
    g = s[s.macro_type.isin(TYPES)].groupby("macro_type")[["sent_s_total_total", "received_r_total_total"]].mean()
    M = pd.read_csv(f"{CM}/uns_commot_cluster-macro_type-cellchat-total-total-communication_matrix_disThr{thr}.csv",
                    index_col=0).loc[TYPES, TYPES]
    summ["COMMOT"][f"disThr{thr}"] = dict(
        sent=g.iloc[:, 0].round(3).to_dict(), received=g.iloc[:, 1].round(3).to_dict(),
        offdiag_vs_Kdyn=qap_offdiag(K_mac.values, M.values),
        offdiag_vs_CellChat=qap_offdiag(Wcc.values, M.values))

# ---- COMMOT, compartment level (two pre-specified primary tests) ----
srcC, snkC = K_cmp.sum(axis=1), K_cmp.sum(axis=0)
summ["compartment_level"] = {}
for thr in THRS:
    s = pd.read_csv(f"{CM}/spot_level_disThr{thr}.csv").set_index("barcode")
    s["compartment"] = spot_comp.reindex(s.index)
    c = s.groupby("compartment").agg(n=("macro_type", "size"), sent=("sent_s_total_total", "mean"),
                                     recv=("received_r_total_total", "mean"))
    c = c[c.index.isin(K_cmp.index) & (c.n >= MIN_SPOTS)]
    c["K_source"], c["K_sink"] = srcC[c.index], snkC[c.index]
    c["macro"] = [i.split("_")[0] for i in c.index]
    c.round(5).to_csv(f"{OUT}/compartment_table_disThr{thr}.csv")
    r_snk, p_snk = perm_spearman(c.recv, c.K_sink)
    r_src, p_src = perm_spearman(c.sent, c.K_source)
    loo = [spearmanr(c.recv.drop(i), c.K_sink.drop(i)).statistic for i in c.index]
    within = {m: round(float(spearmanr(c.recv[c.macro == m], c.K_sink[c.macro == m]).statistic), 3)
              for m in ["PT", "DCT"] if (c.macro == m).sum() >= 3}
    summ["compartment_level"][f"disThr{thr}"] = dict(
        n=int(len(c)), sink=dict(rho=round(r_snk, 3), p=round(p_snk, 4), p_bonferroni=round(min(1, 2 * p_snk), 4),
                                 loo_range=[round(min(loo), 2), round(max(loo), 2)], within_macro=within),
        source=dict(rho=round(r_src, 3), p=round(p_src, 4)))
    if thr == THRS[0]:
        main = c
json.dump(summ, open(f"{OUT}/summary.json", "w"), indent=1)
print(json.dumps(summ, indent=1))

# ---- Figure fig:commot-compartment (main threshold) ----
col = {"PT": "#0072B2", "DCT": "#E69F00", "TAL": "#009E73"}
cl = summ["compartment_level"][f"disThr{THRS[0]}"]
fig, axs = plt.subplots(1, 2, figsize=(12, 5.2))
for ax, (x, y, xl, yl, key, title) in zip(axs, [
        ("sent", "K_source", "COMMOT sent signal (per-spot mean)", r"$K^{\mathrm{dyn}}$ row sum (source strength)", "source", "A  Source"),
        ("recv", "K_sink", "COMMOT received signal (per-spot mean)", r"$K^{\mathrm{dyn}}$ column sum (sink strength)", "sink", "B  Sink")]):
    for m in ["PT", "DCT", "TAL"]:
        d = main[main.macro == m]
        ax.scatter(d[x], d[y], s=30 + d.n * 0.6, c=col[m], alpha=.85, edgecolor="k", lw=.5, label=m)
    for i, r in main.iterrows():
        ax.annotate(i.replace("_", " ").replace("medulla", "med."), (r[x], r[y]),
                    xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel(xl); ax.set_ylabel(yl); ax.grid(alpha=.3)
    ax.set_title(f"{title}   ($\\rho$ = {cl[key]['rho']:.2f}, permutation p = {cl[key]['p']:.3g}, n = {cl['n']})",
                 loc="left", fontsize=10)
axs[1].legend(title="Macro-type", loc="lower right", markerscale=.6)
fig.tight_layout()
for e in ["pdf", "png"]:
    fig.savefig(f"{OUT}/fig_commot_compartment.{e}", dpi=300)
print("Saved", OUT)
