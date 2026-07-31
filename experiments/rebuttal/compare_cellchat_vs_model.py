"""
compare_cellchat_vs_model.py
------------------------------
Final step of Experiment A. Joins CellChat's per-macro-type signaling role
scores (sender/outgoing, receiver/incoming) against the manuscript's own
quantities:
  - R_diff = beta/rho per macro-type (Table 1)
  - row/column sums of the dynamic invasion matrix K_dyn (Table 6, single-layer
    13-compartment analysis, aggregated up to the 6 macro-types to match
    CellChat's grouping)

Produces:
  - table_cellchat_vs_model.csv  (ready to drop into the rebuttal / SI)
  - fig_cellchat_vs_model.png    (scatter with Spearman rho annotated)

Usage: run AFTER cellchat_analysis.R has produced
  cellchat_outputs/cellchat_signaling_roles.csv
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# 1. Manuscript quantities (copied from Table 1 and from the K_dyn matrix
#    of Table 6, aggregated to the 6 macro-types used by CellChat).
#    NOTE: Table 6 in the manuscript covers only PT/DCT/TAL (the multilayer
#    analysis). For vascular and immune, use the single-layer flux figures
#    reported in Section 6.5 / Figure 6 (Phi values) instead -- edit below
#    once you have re-extracted the full 6x6 single-layer flux matrix from
#    your own simulation code (Equation 14).
# ---------------------------------------------------------------------
R_DIFF = {
    "PT": 5.04, "DCT": 2.82, "TAL": 2.10,
    "vascular": 3.12, "immune": 0.80, "other": 2.20,
}

# Row/column sums of K_dyn (Table 6) for PT/DCT/TAL; placeholders (np.nan)
# for vascular/immune/other -- fill in from your own re-run of Eq. 11/14 on
# the full 6-macro-type single-layer graph.
K_DYN_ROW_SUM = {"PT": 0.178, "DCT": 0.100, "TAL": 0.054,
                  "vascular": np.nan, "immune": np.nan, "other": np.nan}
K_DYN_COL_SUM = {"PT": 0.095, "DCT": 0.170, "TAL": 0.067,
                  "vascular": np.nan, "immune": np.nan, "other": np.nan}


def main(cellchat_roles_csv="cellchat_outputs/cellchat_signaling_roles.csv"):
    roles = pd.read_csv(cellchat_roles_csv)
    roles = roles.set_index("macro_type")

    merged = pd.DataFrame({
        "R_diff": pd.Series(R_DIFF),
        "K_dyn_row_sum": pd.Series(K_DYN_ROW_SUM),
        "K_dyn_col_sum": pd.Series(K_DYN_COL_SUM),
    }).join(roles, how="inner")

    merged.to_csv("table_cellchat_vs_model.csv")
    print(merged)

    # Primary comparison: does CellChat's independent outgoing-strength
    # ranking agree with the model's own amplifier ranking (R_diff)?
    valid = merged.dropna(subset=["R_diff", "outgoing_strength"])
    rho_out, p_out = spearmanr(valid["R_diff"], valid["outgoing_strength"])
    print(f"\nSpearman(R_diff, CellChat outgoing strength) = {rho_out:.3f}, "
          f"p = {p_out:.4f}, n = {len(valid)}")

    valid2 = merged.dropna(subset=["K_dyn_row_sum", "outgoing_strength"])
    if len(valid2) >= 3:
        rho_kdyn, p_kdyn = spearmanr(valid2["K_dyn_row_sum"], valid2["outgoing_strength"])
        print(f"Spearman(K_dyn row-sum, CellChat outgoing strength) = "
              f"{rho_kdyn:.3f}, p = {p_kdyn:.4f}, n = {len(valid2)}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].scatter(valid["R_diff"], valid["outgoing_strength"])
    for name, row in valid.iterrows():
        axes[0].annotate(name, (row["R_diff"], row["outgoing_strength"]), fontsize=8)
    axes[0].set_xlabel("Model R_diff = beta/rho (Table 1)")
    axes[0].set_ylabel("CellChat outgoing (sender) strength")
    axes[0].set_title(f"Amplifier ranking agreement (rho = {rho_out:.2f})")

    axes[1].bar(merged.index, merged["incoming_strength"])
    axes[1].set_ylabel("CellChat incoming (receiver) strength")
    axes[1].set_title("Independent check: immune should be lowest sender /\nhighest relative receiver")
    fig.tight_layout()
    fig.savefig("fig_cellchat_vs_model.png", dpi=200)
    print("\nSaved: table_cellchat_vs_model.csv, fig_cellchat_vs_model.png")


if __name__ == "__main__":
    main()
