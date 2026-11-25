"""
fig2_hdr_panel.py — source of the H/D/R panel of Fig. 2 (R3, Reviewer 2 point 4).

The curves are model output, not a drawing: the canonical model of
src/model_core.py is integrated and p^H, p^D, p^R are plotted for two
compartments, together with their sum (which must equal 1).

Default: the single-layer case study (V1 Mouse Kidney), read from
results/annotation_runs/NN/model_context.pkl; shown are the seeded
PT_outer_medulla and the last-activated TAL_inner_medulla.
--illustrative: a 3-compartment chain with Table 1 PT / DCT / TAL rates
(no data needed), used only to preview the layout.

Usage:
    MPLBACKEND=Agg python experiments/r3/fig2_hdr_panel.py [--illustrative]
Output: results/r3/fig2/fig2_hdr_panel.{png,svg,csv}
"""
import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
import model_core as mc  # noqa: E402

OUT = os.path.join(ROOT, "results", "r3", "fig2")
# N_EVAL = 801 points: the same time grid as Table 2 (n_steps = 800), so the
# t_act shown in the figure equals t_cross in Table 2.
T_END, N_EVAL, SEED_D, THR = 60.0, 801, 0.05, 0.10


def case_study():
    ctx = pickle.load(open(os.path.join(ROOT, "results", "annotation_runs",
                                        "NN", "model_context.pkl"), "rb"))
    comp, p = ctx["comp"], ctx["params"]
    W = comp["Wc2"]; W = W.toarray() if hasattr(W, "toarray") else np.asarray(W)
    names = list(comp["comps2"])
    tr = mc.simulate(mc.row_normalise(W), p["beta"], p["gamma"],
                     mc.initial_state(len(names), comp["boundary_mask"], SEED_D),
                     T_END, n_eval=N_EVAL, where="fig2")
    return tr, names, ["PT_outer_medulla", "TAL_inner_medulla"], "V1 Mouse Kidney case study"


def illustrative():
    W = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], float)
    names = ["PT (seeded)", "DCT", "TAL"]
    tr = mc.simulate(mc.row_normalise(W), [0.40, 0.28, 0.25], [0.08, 0.10, 0.12],
                     mc.initial_state(3, [True, False, False], SEED_D),
                     T_END, n_eval=N_EVAL, where="fig2-illustrative")
    return tr, names, ["PT (seeded)", "TAL"], "illustrative 3-compartment chain"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--illustrative", action="store_true")
    a = ap.parse_args()
    tr, names, show, label = illustrative() if a.illustrative else case_study()
    err = mc.check_simplex(tr.pH, tr.pD, tr.pR, where="fig2")

    import matplotlib.pyplot as plt
    os.makedirs(OUT, exist_ok=True)
    fig, axes = plt.subplots(1, len(show), figsize=(4.2 * len(show), 3.3), sharey=True)
    rows = []
    for ax, nm in zip(np.atleast_1d(axes), show):
        i = names.index(nm)
        for arr, lab, col in [(tr.pH, "$p^H$ healthy", "#2b7bba"),
                              (tr.pD, "$p^D$ diseased", "#d7301f"),
                              (tr.pR, "$p^R$ recovered", "#1a9850")]:
            ax.plot(tr.t, arr[i], lw=2, color=col, label=lab)
        ax.plot(tr.t, tr.total[i], ls=":", color="k", lw=1.2, label="sum")
        ax.axhline(THR, color="grey", lw=0.6, ls="--")
        hit = np.where(tr.pD[i] >= THR)[0]
        if len(hit):
            ax.axvline(tr.t[hit[0]], color="grey", lw=0.6)
            ax.text(tr.t[hit[0]] + 0.8, 0.9, f"$t_{{act}}$ = {tr.t[hit[0]]:.1f}", fontsize=8)
        ax.set_title(nm, fontsize=10); ax.set_xlabel("simulated time (model units)")
        ax.set_ylim(-0.02, 1.05); ax.set_xlim(0, T_END)
        rows.append(pd.DataFrame({"compartment": nm, "t": tr.t, "pH": tr.pH[i],
                                  "pD": tr.pD[i], "pR": tr.pR[i]}))
    np.atleast_1d(axes)[0].set_ylabel("probability")
    np.atleast_1d(axes)[0].legend(fontsize=8, frameon=False, loc="center right")
    fig.suptitle(f"Canonical H/D/R dynamics ({label}); max |sum - 1| = {err:.1e}", fontsize=9)
    fig.tight_layout()
    stem = os.path.join(OUT, "fig2_hdr_panel" + ("_illustrative" if a.illustrative else ""))
    fig.savefig(stem + ".png", dpi=300); fig.savefig(stem + ".svg")
    pd.concat(rows).to_csv(stem + ".csv", index=False)
    print(f"saved {stem}.png/.svg/.csv  (max |sum-1| = {err:.2e})")


if __name__ == "__main__":
    main()
