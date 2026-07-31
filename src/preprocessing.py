"""
preprocessing.py
================
Loading and preprocessing of Visium and single-cell data for murine kidney.

Main functions
--------------
load_and_preprocess_visium   : load Visium data, normalise, Leiden clustering
convert_sc_to_h5ad           : convert .txt matrix (GSE107585) to AnnData
annotate_sc                  : add cell_identity, compartment, state columns
align_sc_st                  : gene intersection and label transfer via sc.tl.ingest
"""
import os
import warnings
import numpy as np
import scipy.sparse as sp
import scanpy as sc

# ── Annotation dictionaries ──────────────────────────────────────────────────

CLUSTER_ANNOTATION: dict = {
    "1":  "Endothelial",      "2":  "Podocyte",
    "3":  "Proximal_Tubule",  "4":  "TAL",
    "5":  "DCT",              "6":  "CD_Principal",
    "7":  "Unknown_minor",    "8":  "CD_like",
    "9":  "PT_Cycling",       "10": "Macrophage",
    "11": "Immune_minor",     "12": "Immune_minor",
    "13": "Immune_minor",     "14": "Ribosomal/stress",
    "15": "Immune_minor",     "16": "Cycling_cells",
}

COMPARTMENT_MAP: dict = {
    "Proximal_Tubule": "tubular",   "PT_Cycling":       "tubular",
    "TAL":             "tubular",   "DCT":              "tubular",
    "CD_Principal":    "tubular",   "CD_like":          "tubular",
    "Endothelial":     "vascular",  "Podocyte":         "glomer",
    "Macrophage":      "immune",    "Immune_minor":     "immune",
    "Ribosomal/stress":"stress",    "Cycling_cells":    "cycling",
    "Unknown_minor":   "other",     "Unassigned":       "other",
}

MACRO_MAP: dict = {
    "Proximal_Tubule": "PT",       "PT_Cycling":      "PT",
    "DCT":             "DCT",      "TAL":             "TAL",
    "Endothelial":     "vascular",
    "Podocyte":        "other",    "CD_Principal":    "other",
    "CD_like":         "other",    "Unknown_minor":   "other",
    "Ribosomal/stress":"other",    "Cycling_cells":   "other",
    "Macrophage":      "immune",   "Immune_minor":    "immune",
    "Unassigned":      "other",
}

#: Marker genes for dotplot and violin validation plots
KIDNEY_MARKERS: dict = {
    "Podocyte":       ["nphs1", "nphs2", "wt1", "mafb", "podxl"],
    "PT":             ["slc34a1", "lrp2", "cubn", "enpep", "dpp4"],
    "TAL":            ["umod", "slc12a1", "cldn16"],
    "DCT":            ["slc12a3", "pvalb", "calb1"],
    "CD_principal":   ["aqp2", "aqp3", "avpr2"],
    "CD_intercalated":["atp6v1b1", "slc4a1", "foxi1"],
    "Endothelial":    ["pecam1", "kdr", "emcn", "vwf"],
    "Macrophage":     ["adgre1", "cd68", "csf1r", "fcer1g"],
}

# ═══════════════════════════════════════════════════════════════════════════
# VISIUM
# ═══════════════════════════════════════════════════════════════════════════

def load_and_preprocess_visium(
    dataset_name: str = "V1_Mouse_Kidney",
    n_top_genes: int  = 2000,
    resolution:  float = 0.5,
    random_state: int  = 42,
) -> sc.AnnData:
    """
    Load the Visium demonstration dataset, apply standard preprocessing,
    and compute Leiden clustering.

    Normalisation is applied only once: a flag ``adata.uns['_preprocessed']``
    prevents double application if the function is called multiple times in
    the same session.

    Parameters
    ----------
    dataset_name  : squidpy dataset name (default "V1_Mouse_Kidney")
    n_top_genes   : number of highly variable genes for dimensionality reduction
    resolution    : Leiden clustering resolution
    random_state  : random seed for reproducibility

    Returns
    -------
    adata : preprocessed AnnData with Leiden clustering and spatial neighbourhood
    """
    import squidpy as sq

    adata = sq.datasets.visium(dataset_name, include_hires_tiff=False)
    adata.var_names_make_unique()

    if not adata.uns.get("_preprocessed", False):
        sc.pp.filter_genes(adata, min_counts=1)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        adata.uns["_preprocessed"] = True

    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes,
                                 flavor="seurat")
    adata = adata[:, adata.var["highly_variable"]].copy()

    sc.pp.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.leiden(adata, resolution=resolution, flavor="igraph",
                 n_iterations=2, random_state=random_state)

    sq.gr.spatial_neighbors(adata, coord_type="grid", n_neighs=6)

    return adata


# ═══════════════════════════════════════════════════════════════════════════
# SINGLE-CELL
# ═══════════════════════════════════════════════════════════════════════════

def convert_sc_to_h5ad(
    infile:      str = "Mouse_kidney_single_cell_datamatrix.txt",
    outfile:     str = "GSE107585_sc.h5ad",
    cluster_row: str = "Cluster_Number",
    dtype=np.float32,
) -> sc.AnnData:
    """
    Convert the GSE107585 counts matrix (.txt, transposed format) to AnnData.

    Handles rows with variable token counts: pads with zeros if missing,
    truncates if in excess. The ``Cluster_Number`` row is extracted as a
    cell label and excluded from the count matrix.

    Parameters
    ----------
    infile      : path to the input .txt file
    outfile     : output path for the .h5ad file
    cluster_row : name of the row containing cell cluster labels
    dtype       : numpy dtype for counts (default float32)

    Returns
    -------
    adata : AnnData (cells × genes) with ``obs['cell_type']``
    """
    with open(infile, "r") as f:
        header   = f.readline().rstrip("\n").split("\t")
        cell_ids = header[1:]
        if cell_ids and cell_ids[-1] == "":
            cell_ids = cell_ids[:-1]
        n_cells = len(cell_ids)
        print(f"[convert_sc_to_h5ad] detected cells: {n_cells}")

        data, indices, indptr = [], [], [0]
        genes      = []
        cell_types = None

        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            gene, rest = line.split("\t", 1)
            toks = rest.split("\t")
            if toks and toks[-1] == "":
                toks = toks[:-1]

            # Normalise token length
            if len(toks) > n_cells:
                toks = toks[:n_cells]
            elif len(toks) < n_cells:
                toks = toks + ["0"] * (n_cells - len(toks))

            if gene == cluster_row:
                cell_types = np.array(toks, dtype=str)
                continue

            arr = np.asarray(toks, dtype=dtype)
            nz  = np.nonzero(arr)[0]
            if nz.size:
                data.extend(arr[nz].tolist())
                indices.extend(nz.tolist())
            indptr.append(len(data))
            genes.append(gene)

    if cell_types is None:
        raise ValueError(f"Row '{cluster_row}' not found in {infile}.")

    X_sp = sp.csr_matrix(
        (np.array(data,    dtype=np.float32),
         np.array(indices, dtype=np.int32),
         np.array(indptr,  dtype=np.int64)),
        shape=(len(genes), n_cells),
    )
    adata = sc.AnnData(X=X_sp.T)
    adata.obs_names = cell_ids
    adata.var_names = genes
    adata.obs["cell_type"] = cell_types
    adata.write_h5ad(outfile, compression="gzip")
    print(f"[convert_sc_to_h5ad] saved to: {os.path.abspath(outfile)}")
    return adata


def annotate_sc(adata_sc: sc.AnnData) -> sc.AnnData:
    """
    Add biological annotation columns to the single-cell AnnData:
    ``cell_identity``, ``compartment``, ``state``, ``macro_type``.

    Parameters
    ----------
    adata_sc : AnnData with ``obs['cell_type']`` (cluster number as string)

    Returns
    -------
    adata_sc modified in-place and returned
    """
    adata_sc.obs["cell_identity"] = (
        adata_sc.obs["cell_type"].astype(str)
        .map(CLUSTER_ANNOTATION)
        .fillna("Unassigned")
    )
    adata_sc.obs["compartment"] = (
        adata_sc.obs["cell_identity"]
        .map(COMPARTMENT_MAP)
        .fillna("other")
    )

    def _state(x: str) -> str:
        if x in ("PT_Cycling", "Cycling_cells"):  return "cycling"
        if x == "Ribosomal/stress":               return "stress"
        if x in ("Macrophage", "Immune_minor"):   return "immune"
        return "baseline"

    adata_sc.obs["state"]      = adata_sc.obs["cell_identity"].map(_state)
    adata_sc.obs["macro_type"] = (
        adata_sc.obs["cell_identity"].astype(str)
        .map(MACRO_MAP).fillna("other")
    )
    return adata_sc


def preprocess_sc_for_integration(adata_sc: sc.AnnData,
                                   n_top_genes: int = 2000,
                                   n_pcs: int = 50,
                                   n_neighbors: int = 15) -> sc.AnnData:
    """
    Preprocess single-cell data for integration with spatial data.

    Raw counts are saved to ``layers['counts']`` before normalisation.
    Pipeline: normalize → log1p → scale → PCA → neighbors → UMAP.

    NOTE: HVG selection is intentionally deferred to align_sc_st so that
    genes present in the spatial data are not discarded prematurely.

    Returns
    -------
    adata_sc preprocessed in-place and returned
    """
    adata_sc.layers["counts"] = adata_sc.X.copy()

    sc.pp.normalize_total(adata_sc, target_sum=1e4)
    sc.pp.log1p(adata_sc)

    sc.pp.scale(adata_sc, max_value=10)
    sc.tl.pca(adata_sc, svd_solver="arpack")
    sc.pp.neighbors(adata_sc, n_neighbors=n_neighbors, n_pcs=30)
    sc.tl.umap(adata_sc)
    return adata_sc


def align_sc_st(adata_sc: sc.AnnData,
                adata_st: sc.AnnData,
                obs_key:  str = "cell_identity",
                n_pcs:    int = 50) -> sc.AnnData:
    """
    Align single-cell and spatial data via ``sc.tl.ingest`` (label transfer).

    Gene names are lowercased on *copies* to maximise the intersection
    without modifying the original objects.

    Parameters
    ----------
    adata_sc  : preprocessed single-cell AnnData (with PCA and neighbors)
    adata_st  : spatial AnnData
    obs_key   : column of ``adata_sc.obs`` to transfer
    n_pcs     : PCA components for the spatial data

    Returns
    -------
    adata_st2 : spatial AnnData with predicted ``obs[obs_key]`` and
                preserved ``obsm['spatial']``
    """
    # Working copies with lowercase var_names (originals unchanged)
    sc_lower = adata_sc.copy()
    st_lower = adata_st.copy()
    sc_lower.var_names = sc_lower.var_names.str.lower()
    st_lower.var_names = st_lower.var_names.str.lower()

    # Remove duplicate var_names that may arise after lowercasing
    if sc_lower.var_names.has_duplicates:
        sc_lower = sc_lower[:, ~sc_lower.var_names.duplicated()].copy()

    if st_lower.var_names.has_duplicates:
        st_lower = st_lower[:, ~st_lower.var_names.duplicated()].copy()

    common = sc_lower.var_names.intersection(st_lower.var_names)
    if len(common) == 0:
        raise ValueError(
            "No genes in common between SC and ST after lowercasing. "
            "Check gene name formats."
        )
    print(f"[align_sc_st] common genes (total): {len(common)}")

    adata_sc2 = sc_lower[:, common].copy()
    adata_st2 = st_lower[:, common].copy()

    # HVG selection on the intersection; flavor="seurat" because data are
    # already log-normalised (seurat_v3 requires raw integer counts)
    sc.pp.highly_variable_genes(adata_sc2, n_top_genes=min(2000, len(common)),
                                 flavor="seurat")
    hvg_common = adata_sc2.var_names[adata_sc2.var["highly_variable"]]
    adata_sc2  = adata_sc2[:, hvg_common].copy()
    adata_st2  = adata_st2[:, hvg_common].copy()
    print(f"[align_sc_st] HVG genes on intersection: {len(hvg_common)}")

    # PCA + neighbours on SC subset
    if "X_pca" not in adata_sc2.obsm:
        sc.tl.pca(adata_sc2, n_comps=n_pcs)
    if "neighbors" not in adata_sc2.uns:
        sc.pp.neighbors(adata_sc2, n_neighbors=15, n_pcs=30)

    sc.pp.scale(adata_st2, max_value=10)
    sc.tl.pca(adata_st2, n_comps=n_pcs)

    sc.tl.ingest(adata_st2, adata_sc2, obs=obs_key)

    # Preserve original spatial coordinates
    adata_st2.obsm["spatial"] = adata_st.obsm["spatial"].copy()
    return adata_st2


def compute_transfer_confidence(adata_st2, adata_sc, obs_key="cell_identity"):
    """
    Estimate label-transfer confidence by computing the KL divergence between
    the cell-type distribution in the SC reference and the ST query.

    A KL value close to 0 indicates that the transferred labels reproduce the
    original distribution well; values below 0.5 are generally acceptable.

    Returns
    -------
    kl_div : float — KL(SC || ST) label distribution divergence
    """
    sc_dist = adata_sc.obs[obs_key].value_counts(normalize=True)
    st_dist = adata_st2.obs[obs_key].value_counts(normalize=True)
    common  = sc_dist.index.intersection(st_dist.index)
    kl_div  = float(np.sum(sc_dist[common] *
                            np.log(sc_dist[common] / (st_dist[common] + 1e-9))))
    print(f"  KL divergence SC→ST label distribution: {kl_div:.3f}")
    print(f"  (0 = identical distribution, <0.5 = acceptable)")
    return kl_div
