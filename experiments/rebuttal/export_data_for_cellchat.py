"""
export_data_for_cellchat.py
----------------------------
Helper script: exports the V1 Mouse Kidney Visium data from the existing
Scanpy/AnnData pipeline into the three files that cellchat_analysis.R needs:

  1. v1_kidney_raw_counts.mtx  — sparse Matrix Market format (genes x spots)
  2. v1_kidney_meta.csv        — barcode, macro_type, compartment, zone
  3. v1_kidney_spatial_coords.csv — barcode, x, y (in pixel units from Visium)

Usage:
    # After running the main pipeline (which produces adata_st2 with
    # macro_type labels), run this script to export:
    python experiments/rebuttal/export_data_for_cellchat.py

    # Or import and call directly:
    from export_data_for_cellchat import export_for_cellchat
    export_for_cellchat(adata_st2, out_dir="cellchat_inputs/")
"""

import sys
import os
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


def export_for_cellchat(adata, out_dir="cellchat_inputs/"):
    """
    Export AnnData to CellChat-compatible files.

    Parameters
    ----------
    adata     : AnnData with raw counts in .X (or .raw.X), obs['macro_type'],
                and obsm['spatial']
    out_dir   : output directory for the three files
    """
    
    def export_for_cellchat(adata, out_dir="cellchat_inputs/"):
        os.makedirs(out_dir, exist_ok=True)

    # --- TEMP CHECK ---
    X_check = adata.raw.X if adata.raw is not None else adata.X
    genes_check = adata.raw.var_names if adata.raw is not None else adata.var_names
    print("X shape:", X_check.shape)
    print("n genes:", len(genes_check))
    # --- END TEMP CHECK ---

   

    # 1. Raw counts as Matrix Market (.mtx)
    # CellChat expects genes x spots (transposed from AnnData's spots x genes)
    X = adata.raw.X if adata.raw is not None else adata.X
    if not sp.issparse(X):
        X = sp.csr_matrix(X)
    # Transpose to genes x spots
    X_genes_x_spots = X.T.tocoo()
    mtx_path = os.path.join(out_dir, "v1_kidney_raw_counts.mtx")
    sio.mmwrite(mtx_path, X_genes_x_spots)
    print(f"  Wrote counts matrix: {mtx_path} ({X_genes_x_spots.shape[0]} genes x {X_genes_x_spots.shape[1]} spots)")

     # 1b. Gene names and barcodes (needed because .mtx alone has no labels)
    gene_names = adata.raw.var_names if adata.raw is not None else adata.var_names
    genes_path = os.path.join(out_dir, "v1_kidney_genes.csv")
    pd.Series(gene_names).to_csv(genes_path, index=False, header=False)
    print(f"  Wrote gene names: {genes_path} ({len(gene_names)} genes)")

    barcodes_path = os.path.join(out_dir, "v1_kidney_barcodes.csv")
    pd.Series(adata.obs_names).to_csv(barcodes_path, index=False, header=False)
    print(f"  Wrote barcodes: {barcodes_path} ({len(adata.obs_names)} spots)")

    # 2. Metadata CSV (barcode as row index, macro_type as column)
    meta = adata.obs[["macro_type"]].copy()
    if "compartment" in adata.obs.columns:
        meta["compartment"] = adata.obs["compartment"]
    if "zone" in adata.obs.columns:
        meta["zone"] = adata.obs["zone"]
    meta.index.name = "barcode"
    meta_path = os.path.join(out_dir, "v1_kidney_meta.csv")
    meta.to_csv(meta_path)
    print(f"  Wrote metadata: {meta_path} ({len(meta)} spots)")
    print(f"    macro_type distribution: {meta['macro_type'].value_counts().to_dict()}")

    # 3. Spatial coordinates CSV (barcode, x, y)
    coords = adata.obsm["spatial"]
    coord_df = pd.DataFrame(coords, columns=["x", "y"], index=adata.obs_names)
    coord_df.index.name = "barcode"
    coords_path = os.path.join(out_dir, "v1_kidney_spatial_coords.csv")
    coord_df.to_csv(coords_path)
    print(f"  Wrote spatial coords: {coords_path} ({len(coord_df)} spots)")

    print(f"\n  Next step: edit paths in cellchat_analysis.R to point to:")
    print(f"    counts_path <- \"{os.path.abspath(mtx_path)}\"")
    print(f"    meta_path   <- \"{os.path.abspath(meta_path)}\"")
    print(f"    coords_path <- \"{os.path.abspath(coords_path)}\"")

    return dict(mtx=mtx_path, meta=meta_path, coords=coords_path)


def export_from_pipeline():
    """
    Run the main pipeline up to Step 4 (label transfer) and export.
    This is the easiest way to get the data in the right format.
    """
    from preprocessing import load_and_preprocess_visium, convert_sc_to_h5ad, annotate_sc, preprocess_sc_for_integration, align_sc_st, MACRO_MAP

    print("Loading Visium data...")
    adata_st = load_and_preprocess_visium()

    sc_file = "Mouse_kidney_single_cell_datamatrix.txt"
    if os.path.exists(sc_file):
        print(f"Loading scRNA-seq reference: {sc_file}")
        adata_sc = convert_sc_to_h5ad(sc_file)
        adata_sc = annotate_sc(adata_sc)
        adata_sc = preprocess_sc_for_integration(adata_sc)
        print("Running label transfer...")
        adata_st2 = align_sc_st(adata_sc, adata_st, obs_key="cell_identity")
        print("[diagnostic] cell_identity distribution (post label-transfer):")
        print(adata_st2.obs["cell_identity"].value_counts())
        # Map cell_identity to macro_type
        adata_st2.obs["macro_type"] = (
            adata_st2.obs["cell_identity"].astype(str)
            .map(MACRO_MAP).fillna("other")
        )
    else:
        print(f"  [WARNING] SC file not found: {sc_file}")
        print("  Using Leiden clusters as macro_type proxy...")
        adata_st2 = adata_st.copy()
        adata_st2.obs["macro_type"] = adata_st2.obs.get(
            "leiden", "unknown"
        ).astype(str)

    # IMPORTANT: adata_st2 has been scaled for ingest.
    # Rebuild an AnnData using the original log-normalised expression.
    common_genes = adata_st2.var_names
    common_barcodes = adata_st2.obs_names

    # adata_st (pre-integration) may have gene names in a different case
    # (e.g. 'Sulf1' vs 'sulf1'), so match case-insensitively.
    st_gene_lookup = {g.lower(): g for g in adata_st.var_names}
    matched_genes = [st_gene_lookup[g] for g in common_genes if g in st_gene_lookup]
    missing = [g for g in common_genes if g not in st_gene_lookup]
    if missing:
        print(f"  [WARNING] {len(missing)} genes from common_genes not found "
              f"in adata_st (case-insensitive): {missing[:10]}...")
    adata_export = adata_st[common_barcodes, matched_genes].copy()
    adata_export.var_names = [g.lower() for g in adata_export.var_names]

    adata_export.obs["macro_type"] = adata_st2.obs["macro_type"].values

    if "compartment" in adata_st2.obs.columns:
        adata_export.obs["compartment"] = adata_st2.obs["compartment"].values

    if "zone" in adata_st2.obs.columns:
        adata_export.obs["zone"] = adata_st2.obs["zone"].values

    adata_export.obsm["spatial"] = adata_st2.obsm["spatial"]

    print(
        f"  [sanity check] expression range: "
        f"{adata_export.X.min():.3f} -> {adata_export.X.max():.3f}"
    )

    export_for_cellchat(adata_export, out_dir="cellchat_inputs/")


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
    export_from_pipeline()
