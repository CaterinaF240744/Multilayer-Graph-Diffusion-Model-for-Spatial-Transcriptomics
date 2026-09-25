"""
datasetC_pipeline.py -- Human kidney transplant-rejection Visium (GSE304669):
generality test of the simulation platform on human data (Reviewer 2,
cross-species independent dataset).

Annotation: NN label transfer from the Susztak human kidney sc/sn reference
(GSE211785). Compartments: human macro-types x radial zones (marker-based
zoning uses mouse markers, so the human analysis uses radial tertiles,
stated explicitly). Parameters: the SAME literature-informed murine IRI
values (PT/DCT/TAL/vascular/immune/other) -- no refitting; claims are
restricted to structural quantities (source/sink ranking, activation
ordering, flux edges, vulnerability) with marker-gene corroboration.

Usage:
    python datasetC_pipeline.py [--repo-src PATH] [--data-dir PATH]
                                 [--ref-dir PATH] [--out-dir PATH]
"""
import argparse
import os
import sys
import json
import warnings

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.io as sio

SAMPLE_SUBPATHS = {
    "control":            "GSM9155022_control_58055",
    "active_AMR":         "GSM9155023_active_AMR_58056",
    "acute_TCMR":         "GSM9155024_acute_TCMR_58057",
    "chronic_active_AMR": "GSM9155025_chronic_active_AMR_58058",
}

# Human macro mapping from Susztak cell types (substring match on lowercase)
HUMAN_MACRO_RULES = [
    ("PT", ["pt_s", "ipt", "proximal tubule", "fr-pt"]),
    ("DCT", ["dct", "connecting", "cnt", "macula densa"]),
    ("TAL", ["tal", "loh", "loop of henle", "thick ascending"]),
    ("vascular", ["endo", "vascular", "pericyte", "smooth muscle", "vsmc", "mes"]),
    ("immune", ["mono", "cd4t", "cd8t", "nk", "mac", "dc", "pdc", "cdc",
                "neutroph", "plasma", "baso", "mast", "lym", "b_naive", "b_memory",
                "immune", "leuk", "t cell", "b cell"]),
    ("other", ["pc", "ic_a", "ic_b", "podo", "pec", "fibroblast", "myofib",
               "stroma", "glomerul", "mesangial", "neural", "rbc", "descending",
               "ascending", "collecting", "principal", "intercalated"]),
]

INJURY_MARKERS_HUMAN = ["HAVCR1", "LCN2", "VCAM1", "SPP1", "UMOD", "SLC12A1", "SLC12A3"]


def map_macro(celltype: str) -> str:
    ct = str(celltype).lower()
    for macro, keys in HUMAN_MACRO_RULES:
        if any(k in ct for k in keys):
            return macro
    return "other"


def load_sample(prefix):
    """Load one GSE304669 CytAssist Visium sample from its GSM file prefix."""
    X = sio.mmread(f"{prefix}_matrix.mtx.gz").T.tocsr()  # spots x genes
    import gzip
    with gzip.open(f"{prefix}_barcodes.tsv.gz", "rt") as f:
        barcodes = [l.strip() for l in f if l.strip()]
    with gzip.open(f"{prefix}_features.tsv.gz", "rt") as f:
        feats = pd.read_csv(f, sep="\t", header=None)
    genes = feats.iloc[:, 1].astype(str).values
    with gzip.open(f"{prefix}_tissue_positions.csv.gz", "rt") as f:
        pos = pd.read_csv(f)
    # Visium CytAssist columns: barcode, in_tissue, array_row, array_col,
    # pxl_row_in_fullres, pxl_col_in_fullres (last two used as coordinates)
    pos = pos.set_index(pos.columns[0])
    common = [b for b in barcodes if b in pos.index]
    X = X[[barcodes.index(b) for b in common]]
    coords = pos.loc[common, [pos.columns[-2], pos.columns[-1]]].values.astype(float)
    a = sc.AnnData(X, obs=pd.DataFrame(index=common), var=pd.DataFrame(index=genes))
    a.obsm["spatial"] = coords
    a.var_names_make_unique()
    return a


def sp_laplacian_norm(W):
    import scipy.sparse as sp
    deg = np.asarray(W.sum(axis=1)).reshape(-1)
    inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(np.maximum(deg, 1e-12)), 0.0)
    Dm = sp.diags(inv_sqrt)
    return sp.csr_matrix(sp.identity(W.shape[0]) - Dm @ W @ Dm)


def row_normalise(W):
    import scipy.sparse as sp
    deg = np.asarray(W.sum(axis=1)).reshape(-1)
    inv = np.where(deg > 0, 1.0 / deg, 0.0)
    return sp.diags(inv) @ W


def build_reference(ref_dir):
    counts = sio.mmread(os.path.join(ref_dir, "sc_counts.mtx")).T.tocsr()
    genes = [l.strip() for l in open(os.path.join(ref_dir, "sc_genes.txt"))]
    barcodes = [l.strip() for l in open(os.path.join(ref_dir, "sc_barcodes.txt"))]
    meta = pd.read_csv(os.path.join(ref_dir, "sc_meta.txt"), sep="\t")
    meta = meta.set_index(meta.columns[0]).loc[barcodes]
    ct_col = "Cluster_Idents" if "Cluster_Idents" in meta.columns else meta.columns[-1]
    adata_sc = sc.AnnData(
        counts,
        obs=pd.DataFrame({"cell_identity": meta[ct_col].astype(str).values}, index=barcodes),
        var=pd.DataFrame(index=genes),
    )
    adata_sc.var_names_make_unique()
    # Subsample to <= 30k cells for speed
    if adata_sc.n_obs > 30000:
        rng = np.random.default_rng(0)
        keep = rng.choice(adata_sc.n_obs, 30000, replace=False)
        adata_sc = adata_sc[keep].copy()
    sc.pp.filter_genes(adata_sc, min_counts=1)
    sc.pp.normalize_total(adata_sc, target_sum=1e4)
    sc.pp.log1p(adata_sc)
    sc.pp.highly_variable_genes(adata_sc, n_top_genes=2000, flavor="seurat")
    sc.pp.pca(adata_sc)
    sc.pp.neighbors(adata_sc, n_neighbors=15, n_pcs=30)
    adata_sc.obs["macro_type"] = adata_sc.obs["cell_identity"].map(map_macro)
    return adata_sc


def process_sample(name, prefix, adata_sc):
    adata_st = load_sample(prefix)
    sc.pp.filter_genes(adata_st, min_counts=1)
    sc.pp.normalize_total(adata_st, target_sum=1e4)
    sc.pp.log1p(adata_st)
    sc.pp.highly_variable_genes(adata_st, n_top_genes=2000, flavor="seurat")
    sc.pp.pca(adata_st)

    # kNN label transfer on common HVGs (ingest-free, version-robust)
    import sklearn.neighbors as skn
    common = [g for g in adata_sc.var_names if g in adata_st.var_names]
    ref = adata_sc[:, common].X
    q = adata_st[:, common].X
    if hasattr(ref, "toarray"):
        ref = ref.toarray()
    if hasattr(q, "toarray"):
        q = q.toarray()
    nn = skn.NearestNeighbors(n_neighbors=15).fit(ref)
    idx = nn.kneighbors(q, return_distance=False)
    lab = pd.DataFrame([adata_sc.obs["cell_identity"].values[i] for i in idx])
    adata_st.obs["cell_identity"] = lab.mode(axis=1)[0].values

    macro_map = {ct: map_macro(ct) for ct in adata_st.obs["cell_identity"].unique()}
    adata_st.obs["macro_type"] = adata_st.obs["cell_identity"].astype(str).map(macro_map)

    # Radial zones (human zoning; mouse anatomical marker genes are not valid here)
    coords = adata_st.obsm["spatial"].astype(np.float32)
    d = np.linalg.norm(coords - coords.mean(0), axis=1)
    q1, q2 = np.quantile(d, [0.33, 0.66])
    zone = np.where(d > q2, "cortex", np.where(d > q1, "outer_medulla", "inner_medulla"))
    adata_st.obs["zone_radial"] = zone

    # Compartment graph: macro type x radial zone
    from graph_utils import build_knn_graph
    from compartments import coarse_grain_graph
    Wsp = build_knn_graph(coords, k=6)
    comp_lab = (adata_st.obs["macro_type"].astype(str) + "_" +
                adata_st.obs["zone_radial"].astype(str)).values
    Wc, comps, _ = coarse_grain_graph(Wsp, comp_lab, normalize="avg")

    # Parameters: murine IRI Table 1 values per macro-type (no refitting)
    from sir_compartments import SIR_DEFAULTS, _MACRO_SIR, _REGION_SIR
    C = len(comps)
    beta_c = np.empty(C)
    rho_c = np.empty(C)
    DI_c = np.empty(C)
    for i, cname in enumerate(comps):
        m, r = cname.rsplit("_", 1)
        p = _MACRO_SIR.get(m, SIR_DEFAULTS)
        rm = _REGION_SIR.get(r, {})
        beta_c[i] = p.get("beta", 0.3) * rm.get("beta_mult", 1.0)
        rho_c[i] = p.get("gamma", 0.1) * rm.get("gamma_mult", 1.0)
        DI_c[i] = p.get("D_I", 0.2)

    # Canonical exposure-driven ODE, single-layer coarse graph
    from scipy.integrate import solve_ivp
    Lc = sp_laplacian_norm(Wc)
    Wn = row_normalise(Wc)
    boundary = [i for i, c in enumerate(comps) if c.endswith("cortex")]
    D0 = np.zeros(C)
    D0[boundary] = 0.05
    y0 = np.concatenate([1 - D0, D0, np.zeros(C)])
    t_eval = np.linspace(0, 15, 61)

    def rhs(t, y):
        H = np.clip(y[:C], 0, 1)
        D = np.clip(y[C:2 * C], 0, 1)
        lam = Wn @ D
        inf = beta_c * H * lam
        dH = -inf - 0.05 * (Lc @ H)
        dD = +inf - rho_c * D - DI_c * (Lc @ D)
        dR = rho_c * D
        return np.concatenate([dH, dD, dR])

    sol = solve_ivp(rhs, (0, 15), y0, t_eval=t_eval, method="BDF", rtol=1e-6, atol=1e-9)
    D_traj = sol.y[C:2 * C]
    tact = np.where((D_traj > 0.1).any(1),
                    t_eval[np.argmax(D_traj > 0.1, axis=1)], np.inf)

    enrich = {}
    for g in INJURY_MARKERS_HUMAN:
        if g in adata_st.var_names:
            v = adata_st[:, g].X.toarray().ravel()
            enrich[g] = pd.Series(v).groupby(adata_st.obs["macro_type"].values).mean().to_dict()

    return dict(
        n_spots=int(adata_st.n_obs),
        compartments=list(comps),
        D_fin=D_traj[:, -1].tolist(),
        tact=tact.tolist(),
        beta=beta_c.tolist(), rho=rho_c.tolist(),
        marker_enrichment=enrich,
        macro_counts=adata_st.obs["macro_type"].value_counts().to_dict(),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-src", default="src",
                    help="Path to the repository's src/ directory")
    ap.add_argument("--data-dir", default="data/GSE304669",
                    help="Directory containing the GSE304669 GSM_* files "
                         "(matrix.mtx.gz, barcodes.tsv.gz, features.tsv.gz, "
                         "tissue_positions.csv.gz per sample)")
    ap.add_argument("--ref-dir", default="data/human_ref",
                    help="Directory with the Susztak human kidney reference "
                         "(sc_counts.mtx, sc_genes.txt, sc_barcodes.txt, "
                         "sc_meta.txt; GSE211785)")
    ap.add_argument("--out-dir", default="results/datasetC",
                    help="Output directory for datasetC_summary.json")
    a = ap.parse_args()

    sys.path.insert(0, a.repo_src)
    os.makedirs(a.out_dir, exist_ok=True)

    print("Building human reference...", flush=True)
    adata_sc = build_reference(a.ref_dir)

    summary = {}
    for name, subpath in SAMPLE_SUBPATHS.items():
        print(f"===== {name} =====", flush=True)
        prefix = os.path.join(a.data_dir, subpath)
        try:
            summary[name] = process_sample(name, prefix, adata_sc)
            print(f"  {name}: {len(summary[name]['compartments'])} comps, "
                  f"{summary[name]['n_spots']} spots", flush=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            summary[name] = dict(error=str(e))

    out_path = os.path.join(a.out_dir, "datasetC_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=1)
    print(f"saved {out_path}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
