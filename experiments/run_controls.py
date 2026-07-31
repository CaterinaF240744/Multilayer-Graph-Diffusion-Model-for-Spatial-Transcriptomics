"""
run_controls.py — Entry point for control experiments (R2 point 6, R1 point 5b).

Runs all 5 controls on the Visium-derived compartment graph:
  1. Randomized edge null model (Maslov-Sneppen rewiring)
  2. Uniform parameters
  3. Vascular edge removal
  4. Alternative seed locations
  5. R-state sensitivity analysis (R1 point 5b)

Usage:
    python experiments/run_controls.py                    # full Visium pipeline
    python experiments/run_controls.py --simplified       # Section 7 simplified model
    python experiments/run_controls.py --n-rewires 50     # more rewiring replicates

Prerequisites (full pipeline):
    - data/Mouse_kidney_single_cell_datamatrix.txt (from GSE107585) [optional]
    - Internet connection (for Visium auto-download)

If GSE107585 scRNA-seq data is NOT available, the script falls back to
marker-based Leiden cluster annotation (less accurate than full label
transfer, but sufficient for control experiments).
"""

import sys
import os
import argparse
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Change to data directory for SC file lookup
os.chdir(os.path.join(os.path.dirname(__file__), "..", "data"))


# ═══════════════════════════════════════════════════════════════════════════
# FALLBACK: Marker-based Leiden cluster annotation
# ═══════════════════════════════════════════════════════════════════════════

# Marker gene sets for kidney cell types (gene names in mouse Title-Case)
_CELLTYPE_MARKERS = {
    "PT":          ["Slc34a1", "Lrp2", "Cubn", "Dpp4"],
    "TAL":         ["Umod", "Slc12a1", "Cldn16"],
    "DCT":         ["Slc12a3", "Calb1", "Pvalb"],
    "CD":          ["Aqp2", "Aqp3", "Avpr2"],
    "Endothelial": ["Pecam1", "Kdr", "Emcn"],
    "Podocyte":    ["Nphs1", "Nphs2", "Wt1"],
    "Immune":      ["Cd68", "Csf1r", "Adgre1"],
}

# Map cell-type labels to macro_types used by the SIR model
_CELLTYPE_TO_MACRO = {
    "PT": "PT", "TAL": "TAL", "DCT": "DCT",
    "CD": "other", "Endothelial": "vascular",
    "Podocyte": "other", "Immune": "immune",
}


def _annotate_leiden_by_markers(adata, resolution=1.5):
    """
    Annotate Leiden clusters using marker gene set scores.

    This is a fallback when GSE107585 scRNA-seq data is unavailable.
    Uses scanpy's score_genes to compute per-cluster cell-type scores,
    then assigns each cluster to the highest-scoring cell type.

    Parameters
    ----------
    adata      : AnnData (preprocessed, log-normalized)
    resolution : Leiden clustering resolution

    Returns
    -------
    adata with obs['cell_identity'] set to cell-type labels
    """
    import scanpy as sc

    # Re-cluster at higher resolution for better cell-type separation
    if "leiden" not in adata.obs.columns or adata.obs["leiden"].nunique() < 8:
        sc.tl.leiden(adata, resolution=resolution, flavor="igraph",
                     n_iterations=2, random_state=42, key_added="leiden")

    # Score each cell type
    for ct, genes in _CELLTYPE_MARKERS.items():
        avail = [g for g in genes if g in adata.var_names]
        if avail:
            sc.tl.score_genes(adata, avail, score_name=f"_score_{ct}",
                              use_raw=False)

    score_cols = [f"_score_{ct}" for ct in _CELLTYPE_MARKERS
                  if f"_score_{ct}" in adata.obs.columns]

    # Assign each cluster to the best-scoring cell type
    cluster_map = {}
    for cluster in sorted(adata.obs["leiden"].unique()):
        mask = adata.obs["leiden"] == cluster
        scores = {col.replace("_score_", ""): float(adata.obs.loc[mask, col].mean())
                  for col in score_cols}
        best_ct = max(scores, key=scores.get) if scores else "other"
        cluster_map[cluster] = best_ct

    # Map cluster → cell type → macro_type
    adata.obs["cell_identity"] = (
        adata.obs["leiden"].map(cluster_map).fillna("other").astype(str)
    )

    print("  [Marker annotation] Leiden cluster → cell type → macro_type:")
    for cluster, ct in sorted(cluster_map.items()):
        macro = _CELLTYPE_TO_MACRO.get(ct, "other")
        n = int((adata.obs["leiden"] == cluster).sum())
        print(f"    Cluster {cluster:2s} ({n:3d} spots): {ct:12s} → {macro}")

    # Clean up score columns
    adata.obs = adata.obs.drop(columns=score_cols, errors="ignore")

    return adata


# ═══════════════════════════════════════════════════════════════════════════
# MAIN CONTROL RUNNERS
# ═══════════════════════════════════════════════════════════════════════════

def run_visium_controls(n_rewires=20, seed_I=0.05, t_end=60.0, n_steps=400):
    """Run controls on the full Visium pipeline (Steps 1-5 + controls)."""
    from preprocessing import load_and_preprocess_visium, align_sc_st
    from compartments import build_compartment2
    from sir_compartments import build_default_params_SIR
    from sir_controls import run_all_controls_visium

    print("=" * 60)
    print("SETTING UP VISIUM PIPELINE FOR CONTROLS")
    print("=" * 60)

    # Step 1: Load Visium
    adata_st = load_and_preprocess_visium()

    # Step 3: Load scRNA-seq (if available) for label transfer
    sc_file = "Mouse_kidney_single_cell_datamatrix.txt"
    if os.path.exists(sc_file):
        from preprocessing import (
            convert_sc_to_h5ad, annotate_sc, preprocess_sc_for_integration,
        )
        adata_sc = convert_sc_to_h5ad(sc_file)
        adata_sc = annotate_sc(adata_sc)
        adata_sc = preprocess_sc_for_integration(adata_sc)
        # Step 4: SC-ST alignment + compartments
        adata_st2 = align_sc_st(adata_sc, adata_st, obs_key="cell_identity")
        print("  [Label transfer] Using GSE107585 scRNA-seq reference")
    else:
        print(f"  [INFO] SC file not found: {sc_file}")
        print("  Falling back to marker-based Leiden cluster annotation...")
        adata_st2 = _annotate_leiden_by_markers(adata_st, resolution=1.5)

    comp = build_compartment2(adata_st2, use_anatomical_zones=True,
                              fallback_radial=True)

    # Step 5: Build SIR params
    sir_params = build_default_params_SIR(
        comp["comps2"], comp["macro_of"], comp["region_of"]
    )

    # Run all controls
    results = run_all_controls_visium(
        comp=comp,
        sir_params=sir_params,
        seed_I=seed_I,
        t_end=t_end,
        n_steps=n_steps,
        n_rewires=n_rewires,
        save_dir="./results/figures",
    )

    return results


def run_simplified_controls():
    """Run controls on the Section 7 simplified model (supplementary)."""
    from sir_controls import run_all_controls_section7
    return run_all_controls_section7(save_dir="./results/figures")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run control experiments")
    parser.add_argument("--simplified", action="store_true",
                        help="Use Section 7 simplified model instead of Visium pipeline")
    parser.add_argument("--n-rewires", type=int, default=20,
                        help="Number of Maslov-Sneppen rewiring replicates (default 20)")
    parser.add_argument("--seed", type=float, default=0.05,
                        help="Initial diseased fraction (default 0.05)")
    parser.add_argument("--t-end", type=float, default=60.0,
                        help="Simulation horizon (default 60.0)")
    parser.add_argument("--n-steps", type=int, default=400,
                        help="ODE solver steps (default 400)")
    args = parser.parse_args()

    if args.simplified:
        run_simplified_controls()
    else:
        run_visium_controls(
            n_rewires=args.n_rewires,
            seed_I=args.seed,
            t_end=args.t_end,
            n_steps=args.n_steps,
        )
