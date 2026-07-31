"""
preprocess_compartments.py
---------------------------
Reproduces the Section 6.1 preprocessing pipeline (normalisation, HVG/PCA,
label transfer, marker-based anatomical zoning, spot-level kNN graph,
coarse-graining to the compartment graph) and generalises it so the SAME
function can be applied to every timepoint of GSE182939 (sham, 4h, 12h, 48h,
6 weeks), producing one compartment graph + compartment feature table per
timepoint.

Requirements: scanpy, squidpy (optional, for kNN spatial graph), scikit-learn.

NOTE ON DATA ACCESS
--------------------
This sandbox's network egress is restricted to a small allow-list of package
registries (pypi, npm, github, etc.) and does NOT include GEO/NCBI, so the
actual GSE182939 files could not be downloaded or inspected from here. This
script is written against the standard 10x Space Ranger output layout
(filtered_feature_bc_matrix.h5 + spatial/tissue_positions.csv or .parquet +
spatial/scalefactors_json.json), which is how Visium GEO submissions are
almost always deposited. You will likely need to adjust file names once you
download the actual GSM supplementary files for GSE182939 -- check the
"Supplementary file" list on the GEO record page for each GSM (one per
timepoint) and update SAMPLE_PATHS below accordingly.
"""

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from scipy.stats import entropy

# ---------------------------------------------------------------------------
# 0. Configuration -- EDIT THESE PATHS after downloading GSE182939 locally
# ---------------------------------------------------------------------------
SAMPLE_PATHS = {
    "sham": "data/GSE182939/sham/fsham_137_processed/outs",
    "4h": "data/GSE182939/4h/f4hr_115_processed/outs",
    "12h": "data/GSE182939/12h/f12hr_140_processed/outs",
    "48h": "data/GSE182939/48h/f2dps_158_processed/outs",
    "6wk": "data/GSE182939/6wk/f6wks_110_processed/outs",
}

SC_REFERENCE_H5AD = "data/GSE107585_sc.h5ad"  # same reference as the paper

# Marker genes reused verbatim from Section 6.1 / caption of Figure 3
ZONE_MARKERS = {
    "cortex": ["Slc34a1", "Lrp2", "Nphs1", "Pecam1"],
    "outer_medulla": ["Umod", "Slc12a1", "Cldn16"],
    "inner_medulla": ["Aqp2", "Aqp3", "Avpr2"],
}

MACRO_TYPES = ["PT", "DCT", "TAL", "vascular", "immune", "other"]

# Injury/recovery marker panel used later for temporal validation
# (Experiment C) -- classic murine AKI/IRI markers with well-documented
# induction kinetics, so real timepoint expression can be compared against
# simulated p_D(t) ordering.
INJURY_MARKERS = ["Havcr1", "Vcam1", "Lcn2", "Il6", "Ccl2"]  # Kim1, Vcam1, Ngal, IL6, MCP1


# ---------------------------------------------------------------------------
# 1. Standard preprocessing (Section 6.1, first paragraph)
# ---------------------------------------------------------------------------
def load_and_normalise(spaceranger_dir):
    adata = sc.read_visium(spaceranger_dir)
    adata.var_names_make_unique()
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)  # pseudocount of 1, as in the paper
    adata.raw = adata
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    adata_hvg = adata[:, adata.var.highly_variable].copy()
    sc.pp.scale(adata_hvg, max_value=10)
    sc.tl.pca(adata_hvg, n_comps=50)
    adata.obsm["X_pca"] = adata_hvg.obsm["X_pca"]
    sc.pp.neighbors(adata, use_rep="X_pca")
    sc.tl.leiden(adata, resolution=0.5)
    return adata


# ---------------------------------------------------------------------------
# 2. Label transfer from the single-cell reference (Section 6.1, second para)
# ---------------------------------------------------------------------------
def label_transfer(adata, reference_h5ad, macro_types=MACRO_TYPES, n_neighbors=15):
    """
    Nearest-neighbour label transfer in shared PCA space, plus KL-divergence
    QC exactly as reported in the paper (KL < 0.5 indicates good agreement).
    """
    ref = sc.read_h5ad(reference_h5ad)
    ref.var_names_make_unique()
    shared_genes = adata.var_names.intersection(ref.var_names)
    ref_sub = ref[:, shared_genes].copy()
    adata_sub = adata[:, shared_genes].copy()

    sc.pp.normalize_total(ref_sub, target_sum=1e4)
    sc.pp.log1p(ref_sub)
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    import scipy.sparse as sp

    ref_X = ref_sub.X.toarray() if sp.issparse(ref_sub.X) else np.asarray(ref_sub.X)
    query_X = adata_sub.X.toarray() if sp.issparse(adata_sub.X) else np.asarray(adata_sub.X)

    # Fit scaling + PCA on the REFERENCE only, then project the spatial
    # (query) data into the SAME PCA space -- this is the step that was
    # missing. Previously PCA was fit independently on each side, producing
    # two unrelated coordinate systems and making nearest-neighbour matching
    # meaningless (this is the standard approach used by sc.tl.ingest).
    scaler = StandardScaler(with_mean=True).fit(ref_X)
    ref_X_scaled = scaler.transform(ref_X)
    query_X_scaled = scaler.transform(query_X)

    pca_model = PCA(n_components=50, random_state=0).fit(ref_X_scaled)
    ref_pca = pca_model.transform(ref_X_scaled)
    query_pca = pca_model.transform(query_X_scaled)

    nn = NearestNeighbors(n_neighbors=n_neighbors).fit(ref_pca)
    _, idx = nn.kneighbors(query_pca)
    # Translate raw numeric cluster codes into biological identity, then
    # into macro_type, exactly as done in src/preprocessing.py::annotate_sc
    CLUSTER_ANNOTATION = {
        "1": "Endothelial", "2": "Podocyte", "3": "Proximal_Tubule",
        "4": "TAL", "5": "DCT", "6": "CD_Principal", "7": "Unknown_minor",
        "8": "CD_like", "9": "PT_Cycling", "10": "Macrophage",
        "11": "Immune_minor", "12": "Immune_minor", "13": "Immune_minor",
        "14": "Ribosomal/stress", "15": "Immune_minor", "16": "Cycling_cells",
    }
    FULL_MACRO_MAP = {
        "Proximal_Tubule": "PT", "PT_Cycling": "PT",
        "DCT": "DCT", "TAL": "TAL",
        "Endothelial": "vascular",
        "Podocyte": "other", "CD_Principal": "other", "CD_like": "other",
        "Macrophage": "immune", "Immune_minor": "immune",
        "Ribosomal/stress": "other", "Cycling_cells": "other",
        "Unknown_minor": "other", "Unassigned": "other",
    }
    ref_identity = ref_sub.obs["cell_type"].astype(str).map(CLUSTER_ANNOTATION).fillna("Unassigned")
    ref_macro = ref_identity.map(FULL_MACRO_MAP).fillna("other")
    neighbour_labels = ref_macro.values[idx]  # majority vote, now on macro_type
    transferred = pd.Series(neighbour_labels.tolist()).apply(
        lambda row: pd.Series(row).value_counts().idxmax()
    )
    adata.obs["macro_type"] = transferred.values
    # KL divergence QC between reference and spatial label distributions
    ref_dist = ref_macro.value_counts(normalize=True)
    sp_dist = adata.obs["macro_type"].value_counts(normalize=True)
    common = ref_dist.index.union(sp_dist.index)
    p = ref_dist.reindex(common, fill_value=1e-9)
    q = sp_dist.reindex(common, fill_value=1e-9)
    kl = entropy(p, q)
    print(f"[label_transfer] KL(reference || spatial) = {kl:.3f} "
          f"({'OK, < 0.5' if kl < 0.5 else 'WARNING: poor agreement'})")
    return adata, kl


# ---------------------------------------------------------------------------
# 3. Marker-based anatomical zoning (radial fallback as in the paper)
# ---------------------------------------------------------------------------
def assign_zones(adata, zone_markers=ZONE_MARKERS):
    # Build a case-insensitive lookup so marker genes work regardless of
    # whether var_names are Title-Case (Visium default) or lowercase.
    var_lower_map = {v.lower(): v for v in adata.var_names}

    scores = {}
    for zone, genes in zone_markers.items():
        genes_present = []
        for g in genes:
            if g in adata.var_names:
                genes_present.append(g)
            elif g.lower() in var_lower_map:
                genes_present.append(var_lower_map[g.lower()])
        if not genes_present:
            print(f"  [WARNING] No marker genes found for zone '{zone}' "
                  f"(tried: {genes})")
        sc.tl.score_genes(adata, genes_present, score_name=f"score_{zone}")
        scores[zone] = adata.obs[f"score_{zone}"].values
    score_df = pd.DataFrame(scores, index=adata.obs_names)
    zone_assignment = score_df.idxmax(axis=1)

    # Fallback: spots with all-negative/near-zero scores -> radial distance
    weak = (score_df.max(axis=1) < 0.05)
    if weak.any():
        coords = adata.obsm["spatial"]
        centroid = coords.mean(axis=0)
        radial = np.linalg.norm(coords - centroid, axis=1)
        thirds = np.quantile(radial, [1 / 3, 2 / 3])
        radial_zone = np.where(radial < thirds[0], "cortex",
                       np.where(radial < thirds[1], "outer_medulla", "inner_medulla"))
        zone_assignment[weak.values] = radial_zone[weak.values]

    adata.obs["zone"] = zone_assignment.values
    adata.obs["compartment"] = (
        adata.obs["macro_type"].astype(str) + "_" + adata.obs["zone"].astype(str)
    )
    return adata


# ---------------------------------------------------------------------------
# 4. Spot-level kNN graph + coarse-graining to compartment graph (Section 6.1
#    last paragraph)
# ---------------------------------------------------------------------------
def build_compartment_graph(adata, k=6, sigma=None):
    coords = adata.obsm["spatial"]
    nn = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    dist, idx = nn.kneighbors(coords)
    if sigma is None:
        sigma = np.median(dist[:, 1:])  # typical nearest-neighbour distance

    n_spots = coords.shape[0]
    W_sp = np.zeros((n_spots, n_spots))
    for i in range(n_spots):
        for jj in range(1, k + 1):
            j = idx[i, jj]
            w = np.exp(-dist[i, jj] ** 2 / sigma ** 2)
            W_sp[i, j] = w
            W_sp[j, i] = w  # symmetrise

    compartments = adata.obs["compartment"].astype("category")
    cats = compartments.cat.categories
    n_c = len(cats)
    W_c = np.zeros((n_c, n_c))
    counts = np.zeros((n_c, n_c))
    comp_idx = compartments.cat.codes.values

    # average edge weights between spot pairs belonging to different
    # compartments, exactly as described in Section 6.1
    rows, cols = np.nonzero(W_sp)
    for r, c in zip(rows, cols):
        a, b = comp_idx[r], comp_idx[c]
        W_c[a, b] += W_sp[r, c]
        counts[a, b] += 1
    with np.errstate(invalid="ignore"):
        W_c = np.divide(W_c, counts, out=np.zeros_like(W_c), where=counts > 0)

    return W_c, list(cats), adata


def run_pipeline(spaceranger_dir, reference_h5ad):
    adata = load_and_normalise(spaceranger_dir)
    adata, kl = label_transfer(adata, reference_h5ad)
    adata = assign_zones(adata)
    W_c, compartment_names, adata = build_compartment_graph(adata)
    return {"W_c": W_c, "names": compartment_names, "adata": adata, "kl": kl}


if __name__ == "__main__":
    results = {}
    for tp, path in SAMPLE_PATHS.items():
        print(f"--- Processing timepoint {tp} ---")
        results[tp] = run_pipeline(path, SC_REFERENCE_H5AD)
        np.save(f"compartment_graph_{tp}.npy", results[tp]["W_c"])
        pd.Series(results[tp]["names"]).to_csv(f"compartment_names_{tp}.csv", index=False)
        print(f"  -> {len(results[tp]['names'])} compartments, KL={results[tp]['kl']:.3f}")
