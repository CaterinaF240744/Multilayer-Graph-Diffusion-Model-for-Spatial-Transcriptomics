"""datasetB_lean.py -- memory-lean Dataset B pipeline (GSE269622).

Independent murine IRI Visium cohort used to test reproducibility of the
qualitative propagation structure (source/sink ranking, activation
ordering, marker corroboration) reported for the main V1 Mouse Kidney case
study, with the SAME model parameters and no refitting.

One sample per process invocation:
    python datasetB_lean.py <sample> [--repo-src PATH] [--data-dir PATH]
                                      [--out-dir PATH]
Appends the result to <out-dir>/summary_<sample>.json and exits, so a
driver script can run each sample in a fresh process (a kill loses at most
one sample).

Memory strategy (16 GB budget):
- reference subsampled to 20k cells; no scale/neighbors/UMAP on reference
- sparse kNN label transfer (brute force on sparse matrices)
- no dense conversion of full matrices; only the HVG-restricted blocks
"""
import argparse
import json
import os
import sys
import warnings

import numpy as np
import scanpy as sc
import scipy.sparse as sp
import sklearn.neighbors as skn

warnings.filterwarnings("ignore")

SAMPLE_SUBPATHS = {
    "sham":   "mouse_sham/outs",
    "hour4":  "mouse_hour4/outs",
    "hour12": "mouse_hour12/outs",
    "day2":   "mouse_day2/outs",
}
INJURY_MARKERS = ["Havcr1", "Lcn2", "Vcam1", "Spp1"]


def build_reference(repo_data_dir):
    a = sc.read_h5ad(os.path.join(repo_data_dir, "GSE107585_sc.h5ad"))
    a.var_names_make_unique()
    from preprocessing import annotate_sc
    a = annotate_sc(a)
    if a.n_obs > 20000:
        rng = np.random.default_rng(0)
        a = a[rng.choice(a.n_obs, 20000, replace=False)].copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    sc.pp.highly_variable_genes(a, n_top_genes=2000, flavor="seurat")
    a.var["highly_variable"] = a.var["highly_variable"].fillna(False)
    sc.pp.pca(a, n_comps=30)
    return a


def prep_sample(path):
    a = sc.read_visium(path)
    a.var_names_make_unique()
    sc.pp.filter_genes(a, min_counts=1)
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    return a


def knn_transfer(adata_sc, adata_st, n_neighbors=15):
    common = [g for g in adata_sc.var_names
              if g in adata_st.var_names and adata_sc.var.loc[g, "highly_variable"]]
    if len(common) < 200:
        common = [g for g in adata_sc.var_names if g in adata_st.var_names]
    ref = adata_sc[:, common].X
    q = adata_st[:, common].X
    nn = skn.NearestNeighbors(n_neighbors=n_neighbors, algorithm="brute").fit(ref)
    idx = nn.kneighbors(q, return_distance=False)
    import pandas as pd
    lab = pd.DataFrame([adata_sc.obs["cell_identity"].values[i] for i in idx])
    adata_st.obs["cell_identity"] = lab.mode(axis=1)[0].values
    return adata_st


def run_sample(name, sample_path, repo_data_dir, out_dir):
    print(f"[{name}] building reference...", flush=True)
    adata_sc = build_reference(repo_data_dir)
    print(f"[{name}] reference ready: {adata_sc.shape}", flush=True)

    adata_st = prep_sample(sample_path)
    print(f"[{name}] sample loaded: {adata_st.shape}", flush=True)
    adata_st = knn_transfer(adata_sc, adata_st)
    del adata_sc
    import gc
    gc.collect()
    print(f"[{name}] transfer done", flush=True)

    from compartments import build_compartment2
    from sir_compartments import build_default_params_SIR
    comp = build_compartment2(adata_st)
    comps2 = list(comp["comps2"])
    params = build_default_params_SIR(np.asarray(comps2),
                                      comp["macro_of"], comp["region_of"])
    print(f"[{name}] compartments: {len(comps2)}", flush=True)

    from sir_multilayer import run_SIR_multilayer
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = run_SIR_multilayer(
            comp=comp, sir_params=params, adata_st=adata_st,
            seed_I=0.05, D_inter=0.05, t_end=15.0, n_steps=800,
            threshold_I=0.10, plot=False)

    # marker enrichment per sample (mean expression of injury markers)
    markers = {}
    for g in INJURY_MARKERS:
        if g in adata_st.var_names:
            col = adata_st[:, g].X
            markers[g] = float(col.mean()) if sp.issparse(col) else float(np.asarray(col).mean())

    metrics = res["metrics"]
    out = {
        "sample": name,
        "n_spots": int(adata_st.n_obs),
        "n_compartments": len(comps2),
        "compartments": comps2,
        "global_inv_time": metrics.get("global_inv_time"),
        "by_layer": metrics.get("by_layer"),
        "interlayer_flow": metrics.get("interlayer_flow"),
        "most_vulnerable": metrics.get("most_vulnerable"),
        "markers": markers,
    }

    def _strkeys(o):
        if isinstance(o, dict):
            return {str(k): _strkeys(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_strkeys(v) for v in o]
        return o

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"summary_{name}.json")
    with open(out_path, "w") as f:
        json.dump(_strkeys(out), f, indent=1, default=float)
    print(f"[{name}] saved -> {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sample", choices=sorted(SAMPLE_SUBPATHS),
                    help="Which GSE269622 sample to process")
    ap.add_argument("--repo-src", default="src",
                    help="Path to the repository's src/ directory "
                         "(contains preprocessing.py, compartments.py, "
                         "sir_compartments.py, sir_multilayer.py)")
    ap.add_argument("--data-dir", default="data/GSE269622/extracted",
                    help="Directory containing the extracted GSE269622 "
                         "Space Ranger output folders (one per sample)")
    ap.add_argument("--repo-data-dir", default="data",
                    help="Directory containing GSE107585_sc.h5ad "
                         "(the single-cell reference used elsewhere in "
                         "this work)")
    ap.add_argument("--out-dir", default="results/datasetB",
                    help="Output directory for summary_<sample>.json")
    a = ap.parse_args()

    sys.path.insert(0, a.repo_src)
    sample_path = os.path.join(a.data_dir, SAMPLE_SUBPATHS[a.sample])
    run_sample(a.sample, sample_path, a.repo_data_dir, a.out_dir)


if __name__ == "__main__":
    main()
