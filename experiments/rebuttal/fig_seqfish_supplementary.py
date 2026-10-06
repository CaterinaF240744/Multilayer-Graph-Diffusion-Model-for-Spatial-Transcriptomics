"""
fig_seqfish_supplementary.py -- Supplementary Figure S1 (seqFISH perturbation
analysis) from the output of polonsky_seqfish_test.py.

Usage (repository root):
  MPLBACKEND=Agg python experiments/rebuttal/fig_seqfish_supplementary.py
Output: results/polonsky_seqfish/figS_seqfish_compartments.pdf/.png
"""
import json
import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "results/polonsky_seqfish"
t = pd.read_csv(f"{D}/compartment_table_R100.csv", index_col=0)
p1 = json.load(open(f"{D}/polonsky_summary.json"))["results"]["R100"]["tests"]["P1_peak_vs_dHavcr1_all"]
col = {"PT": "#0072B2", "DCT": "#E69F00", "TAL": "#009E73"}
mk = {False: "o", True: "s"}
fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
for m in ["PT", "DCT", "TAL"]:
    for sd in [False, True]:
        d = t[(t.macro == m) & (t.seed == sd)]
        if len(d):
            ax[0].scatter(d.peak_I, d.d_havcr1, s=60, c=col[m], marker=mk[sd], edgecolor="k", lw=.5,
                          label=m + (" (seed)" if sd else ""))
for i, r in t[t.macro == "PT"].iterrows():
    ax[0].annotate(i.replace("PT_", "").replace("_", " "), (r.peak_I, r.d_havcr1),
                   xytext=(-10, 7), textcoords="offset points", fontsize=8)
ax[0].axhline(0, c="grey", lw=.6, ls="--")
ax[0].set_xlabel("Predicted peak diseased fraction (control sections)")
ax[0].set_ylabel(r"Observed $\Delta$Havcr1 (IRI $-$ control)")
ax[0].set_title(f"A  Pre-specified test P1: $\\rho$ = {p1['rho']:.2f}, permutation p = {p1['p']:.2g}, n = {p1['n']}",
                loc="left", fontsize=10)
ax[0].legend(fontsize=8, loc="center left")
pt = t[t.macro == "PT"].sort_values("t_act")
ax[1].bar([i.replace("PT_", "").replace("_", " ") for i in pt.index], pt.frac_injured * 100,
          color="#0072B2", edgecolor="k", lw=.5)
for k, (i, r) in enumerate(pt.iterrows()):
    ax[1].text(k, r.frac_injured * 100 + 1, f"predicted t_act = {r.t_act:.1f}" + ("\n(seed)" if r.seed else ""),
               ha="center", fontsize=8)
ax[1].set_ylabel("Injured PT cells in IRI sections (%)"); ax[1].set_ylim(0, 60)
ax[1].set_title(f"B  PT compartments (test P2 not computable, n = {len(pt)})", loc="left", fontsize=10)
fig.tight_layout()
for e in ["pdf", "png"]:
    fig.savefig(f"{D}/figS_seqfish_compartments.{e}", dpi=300)
print("Saved", D)
