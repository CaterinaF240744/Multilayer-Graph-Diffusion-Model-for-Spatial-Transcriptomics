"""
graphical_abstract.py — R3: Figure 2 (overview) generated entirely from data
and model output, replacing the schematic of the previous version.

Panel 1: V1 Mouse Kidney Visium spots coloured by macro-type and by
         anatomical zone (the inputs of the compartment graph).
Panel 2: the 17-node compartment graph used in the case study (rows =
         macro-type, columns = zone; edge width ~ adjacency weight; ringed
         nodes = seeds) and the canonical equations of Section 5.2.
Panel 3: simulated activation time of every spot (that of its compartment)
         and the H/D/R trajectories of the seeded PT outer-medulla and the
         TAL inner-medulla compartments, with their sum.

Every number comes from src/model_core.py run on the case-study context
(results/annotation_runs/NN/model_context.pkl, written by run_main.py),
with the same settings as Table 2 (t_end = 60, 801 output points,
seed p^D(0) = 0.05). The values plotted are saved next to the figure.

Usage (repository root):  python experiments/r3/graphical_abstract.py
Output: results/r3/fig2/graphical_abstract.{pdf,png,svg} + _data.csv
"""
import os, sys, pickle
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
import model_core as mc  # noqa: E402

CTX = os.environ.get("CTX", os.path.join(ROOT, "results", "annotation_runs", "NN", "model_context.pkl"))
OUT = os.path.join(ROOT, "results", "r3", "fig2")
T_END, N_EVAL, SEED_D, THR = 60.0, 801, 0.05, 0.10

MACROS = ["PT", "DCT", "TAL", "vascular", "other"]
MCOL = {"PT": "#3b7dd8", "DCT": "#f08c3a", "TAL": "#4caf50", "vascular": "#d64545", "other": "#9e9e9e"}
ZONES = ["boundary", "outer_medulla", "mid", "cortex", "core", "inner_medulla"]
ZLAB = {"boundary": "boundary", "outer_medulla": "outer medulla", "mid": "mid",
        "cortex": "cortex", "core": "core", "inner_medulla": "inner medulla"}
ZCOL = {"boundary": "#6a3d9a", "outer_medulla": "#e31a1c", "mid": "#fdbf6f",
        "cortex": "#1f78b4", "core": "#b2df8a", "inner_medulla": "#33a02c"}
SCOL = {"H": "#2b7bba", "D": "#d7301f", "R": "#1a9850"}


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, Circle
    from matplotlib.lines import Line2D

    ctx = pickle.load(open(CTX, "rb"))
    comp, p, ad = ctx["comp"], ctx["params"], ctx["adata"]
    W = comp["Wc2"]; W = W.toarray() if hasattr(W, "toarray") else np.asarray(W)
    names = list(comp["comps2"]); n = len(names)
    seed = np.asarray(comp["boundary_mask"], bool)
    tr = mc.simulate(mc.row_normalise(W), p["beta"], p["gamma"],
                     mc.initial_state(n, seed, SEED_D), T_END, n_eval=N_EVAL, where="graphical_abstract")
    err = mc.check_simplex(tr.pH, tr.pD, tr.pR, where="graphical_abstract")
    t_act = np.array([tr.t[np.argmax(tr.pD[i] >= THR)] if (tr.pD[i] >= THR).any() else np.nan for i in range(n)])

    xy = np.asarray(ad.obsm["spatial"], float); xy[:, 1] *= -1
    macro = ad.obs["macro_type"].astype(str).values
    zone = ad.obs["region"].astype(str).values
    inv = np.asarray(comp["inv2"], int)
    spot_tact = t_act[inv]
    nspots = np.bincount(inv, minlength=n)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig = plt.figure(figsize=(15, 8.4))
    # column geometry (figure fractions)
    cols = [(0.02, 0.30), (0.355, 0.30), (0.69, 0.30)]
    titles = ["Spatial transcriptomics", "Compartment graph and\ncanonical model", "Simulated propagation"]
    for k, ((x0, w), ttl) in enumerate(zip(cols, titles)):
        fig.text(x0 + 0.005, 0.955, str(k + 1), fontsize=15, fontweight="bold", color="white", ha="center", va="center",
                 bbox=dict(boxstyle="circle,pad=0.35", fc="#1565c0", ec="none"))
        fig.text(x0 + 0.03, 0.955, ttl, fontsize=14, fontweight="bold", va="center")
    for xa in (0.335, 0.67):
        fig.patches.append(matplotlib.patches.FancyArrow(xa - 0.012, 0.6, 0.02, 0, width=0.018, head_width=0.04,
                                                         head_length=0.008, transform=fig.transFigure, color="#8a8a8a"))

    ms = 9
    # ---- panel 1 -----------------------------------------------------------
    a1 = fig.add_axes([0.02, 0.47, 0.30, 0.43])
    for m in MACROS:
        s = macro == m
        a1.scatter(xy[s, 0], xy[s, 1], s=ms * (3 if m in ("vascular", "other") else 1), marker="h",
                   c=MCOL[m], lw=0, label=f"{m} ({s.sum()})")
    a1.set_title("V1 Mouse Kidney Visium: macro-type per spot (1,438 spots)", fontsize=10)
    a1.legend(loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=5, fontsize=8, frameon=False,
              handletextpad=0.2, columnspacing=0.8, markerscale=1.6)
    a1.set_aspect("equal"); a1.axis("off")

    a1b = fig.add_axes([0.02, 0.06, 0.30, 0.33])
    for z in ZONES:
        s = zone == z
        a1b.scatter(xy[s, 0], xy[s, 1], s=ms * 0.7, marker="h", c=ZCOL[z], lw=0, label=ZLAB[z])
    a1b.set_title("Anatomical zones (marker-based)", fontsize=10)
    a1b.legend(loc="center left", bbox_to_anchor=(0.86, 0.5), fontsize=8, frameon=False, markerscale=1.8)
    a1b.set_aspect("equal"); a1b.axis("off")

    # ---- panel 2 -----------------------------------------------------------
    a2t = fig.add_axes([0.355, 0.70, 0.30, 0.20]); a2t.axis("off")
    a2t.add_patch(FancyBboxPatch((0.02, 0.05), 0.96, 0.9, boxstyle="round,pad=0.01,rounding_size=0.04",
                                 fc="white", ec="#1565c0", lw=1.5, transform=a2t.transAxes))
    eq = (r"$\dfrac{dp_i^H}{dt}=-\beta_i\,p_i^H\lambda_i,\quad"
          r"\dfrac{dp_i^D}{dt}=\beta_i\,p_i^H\lambda_i-\rho_i\,p_i^D,\quad"
          r"\dfrac{dp_i^R}{dt}=\rho_i\,p_i^D$")
    a2t.text(0.5, 0.66, eq, ha="center", va="center", fontsize=11, transform=a2t.transAxes)
    a2t.text(0.5, 0.25, r"exposure $\lambda_i=\sum_j \tilde w_{ij}\,p_j^D$ ;  $p_i^H+p_i^D+p_i^R=1$"
             f"  (max error {err:.0e})", ha="center", va="center", fontsize=10, transform=a2t.transAxes)

    a2 = fig.add_axes([0.39, 0.15, 0.265, 0.48])
    zpos = {z: k for k, z in enumerate(ZONES)}
    mpos = {m: k for k, m in enumerate(MACROS)}
    pos = {}
    for i, nm in enumerate(names):
        m, z = comp["macro_of"][i], comp["region_of"][i]
        pos[i] = (zpos[z], -mpos[m])
    wmax = W.max()
    iu = np.argwhere(np.triu(W, 1) > 0)
    for i, j in sorted(iu.tolist(), key=lambda e: W[e[0], e[1]]):
        (x1, y1), (x2, y2) = pos[i], pos[j]
        lw = 0.3 + 4.0 * np.sqrt(W[i, j] / wmax)
        rad = 0.25 if (x1 == x2 or y1 == y2) and abs(x1 - x2) + abs(y1 - y2) > 1 else 0.0
        a2.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-", lw=lw, color="#7f7f7f", alpha=0.55,
                                    connectionstyle=f"arc3,rad={rad}"))
    for i, nm in enumerate(names):
        x, y = pos[i]; m = comp["macro_of"][i]
        r = 0.10 + 0.17 * np.sqrt(nspots[i] / nspots.max())
        a2.add_patch(Circle((x, y), r, fc=MCOL[m], ec="black" if seed[i] else "white",
                            lw=2.2 if seed[i] else 0.8, zorder=3))
        a2.text(x + r + 0.04, y - r, f"{nspots[i]}", ha="left", va="top", fontsize=7, color="#444", zorder=4)
    a2.set_xticks(range(len(ZONES))); a2.set_xticklabels([ZLAB[z] for z in ZONES], rotation=30, ha="right", fontsize=8)
    a2.set_yticks([-k for k in range(len(MACROS))]); a2.set_yticklabels(MACROS, fontsize=9)
    for sp in a2.spines.values(): sp.set_visible(False)
    a2.tick_params(length=0)
    a2.set_xlim(-0.6, len(ZONES) - 0.4); a2.set_ylim(-len(MACROS) + 0.45, 0.55)
    a2.set_title("17 macro-type $\\times$ zone compartments (72 edges)", fontsize=10)
    a2.legend(handles=[Line2D([], [], marker="o", ls="", mfc="white", mec="black", mew=2, ms=9, label="seed, $p^D(0)=0.05$"),
                       Line2D([], [], color="#7f7f7f", lw=3, label="edge width ~ adjacency")],
              loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=2, fontsize=8, frameon=False)

    # ---- panel 3 -----------------------------------------------------------
    a3 = fig.add_axes([0.69, 0.47, 0.28, 0.43])
    sc = a3.scatter(xy[:, 0], xy[:, 1], s=ms, marker="h", c=spot_tact, cmap="RdYlBu_r", lw=0)
    a3.set_title("Activation time ($p^D \\geq 0.10$) of each spot's compartment", fontsize=10)
    a3.set_aspect("equal"); a3.axis("off")
    cb = fig.colorbar(sc, ax=a3, orientation="horizontal", fraction=0.05, pad=0.02, aspect=35)
    cb.set_label("simulated activation time (earlier $\\rightarrow$ later)", fontsize=8); cb.ax.tick_params(labelsize=8)

    a3b = fig.add_axes([0.715, 0.07, 0.265, 0.25])
    show = [("PT_outer_medulla", "-", "PT outer medulla (seed)"), ("TAL_inner_medulla", "--", "TAL inner medulla")]
    rows = []
    for nm, ls, lab in show:
        i = names.index(nm)
        for arr, s in [(tr.pH, "H"), (tr.pD, "D"), (tr.pR, "R")]:
            a3b.plot(tr.t, arr[i], ls=ls, lw=1.8, color=SCOL[s])
        a3b.axvline(t_act[i], color="grey", lw=0.6, ls=ls)
        rows.append(pd.DataFrame({"compartment": nm, "t": tr.t, "pH": tr.pH[i], "pD": tr.pD[i], "pR": tr.pR[i]}))
    a3b.plot(tr.t, tr.total[names.index("PT_outer_medulla")], ":", color="k", lw=1)
    a3b.axhline(THR, color="grey", lw=0.5, ls=":")
    a3b.set_xlim(0, T_END); a3b.set_ylim(-0.02, 1.05)
    a3b.set_xlabel("simulated time (model units)"); a3b.set_ylabel("probability")
    a3b.set_title("State trajectories", fontsize=10, pad=34)
    h = [Line2D([], [], color=SCOL[s], lw=2, label=l) for s, l in [("H", "$p^H$ healthy"), ("D", "$p^D$ diseased"), ("R", "$p^R$ recovered")]]
    h += [Line2D([], [], color="k", ls=ls, lw=1.2, label=lab) for _, ls, lab in show]
    h += [Line2D([], [], color="k", ls=":", lw=1, label="sum")]
    a3b.legend(handles=h, fontsize=7, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, columnspacing=0.9)
    a3b.spines[["top", "right"]].set_visible(False)

    os.makedirs(OUT, exist_ok=True)
    stem = os.path.join(OUT, "graphical_abstract")
    for e in ("pdf", "png", "svg"):
        fig.savefig(f"{stem}.{e}", dpi=300 if e == "png" else None)
    pd.concat(rows).to_csv(stem + "_trajectories.csv", index=False)
    pd.DataFrame({"compartment": names, "macro": comp["macro_of"], "zone": comp["region_of"], "n_spots": nspots,
                  "seed": seed, "t_act": t_act}).to_csv(stem + "_compartments.csv", index=False)
    print(f"saved {stem}.pdf/.png/.svg (max |sum-1| = {err:.1e}); t_act PT_OM = "
          f"{t_act[names.index('PT_outer_medulla')]:.2f}, TAL_IM = {t_act[names.index('TAL_inner_medulla')]:.2f}")


if __name__ == "__main__":
    main()
