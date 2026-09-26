#!/usr/bin/env python
"""COMMOT on the full-transcriptome V1 Mouse Kidney Visium matrix (1,438 spots).

Manuscript: Section 8.6 (Table 15, Figure 12).

NOTE ON UNITS: spot coordinates are used in full-resolution PIXELS. At
0.7266 um/px, dis_thr = 150 / 250 / 400 correspond to ~110 / ~180 / ~290 um,
which are the distances reported in the manuscript.

Usage (from repository root):
  python experiments/rebuttal/commot_full_transcriptome.py [input_dir] [output_dir]

Pre-specified settings (reviewer R2.5 protocol) — no tuning:
  normalize_total(1e4) + log1p; CellChatDB mouse; filter min_cell_pct=0.05;
  spatial_communication(dis_thr=250, heteromeric=True, pathway_sum=True);
  cluster_communication(clustering='macro_type', n_permutations=1000, random_seed=0);
  sensitivity dis_thr=150 and 400 (250 remains the main result).
"""
import json, sys, time
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import commot

IN = sys.argv[1] if len(sys.argv) > 1 else "data/full_matrix_inputs"
OUT = sys.argv[2] if len(sys.argv) > 2 else "results/commot_full"
import os; os.makedirs(OUT, exist_ok=True)

t0 = time.time()
adata = ad.AnnData(
    X=None, obs=pd.DataFrame(index=pd.read_csv(f"{IN}/v1_kidney_full_barcodes.csv", header=None)[0].astype(str).values))
import scipy.io as scio
Xg = scio.mmread(f"{IN}/v1_kidney_full_counts.mtx").tocsr()          # genes x spots
genes = pd.read_csv(f"{IN}/v1_kidney_full_genes.csv", header=None)[0].astype(str).values
adata = ad.AnnData(X=Xg.T.tocsr(), obs=pd.DataFrame(index=pd.read_csv(f"{IN}/v1_kidney_full_barcodes.csv", header=None)[0].astype(str).values))
adata.var_names = pd.Index(genes)
adata.var_names_make_unique()
coords = pd.read_csv(f"{IN}/v1_kidney_full_coords.csv")
adata.obsm["spatial"] = coords[["x", "y"]].values.astype(np.float32)
meta = pd.read_csv(f"{IN}/v1_kidney_full_meta.csv", dtype=str).set_index("barcode")
adata.obs["macro_type"] = meta.loc[adata.obs_names, "macro_type"].values
print(f"adata: {adata.shape}", flush=True)

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

from commot.pp import ligand_receptor_database, filter_lr_database
df_ligrec = ligand_receptor_database(database="CellChat", species="mouse")
print(f"CellChatDB mouse pairs (Secreted Signaling): {len(df_ligrec)}", flush=True)
df_filtered = filter_lr_database(df_ligrec, adata, min_cell_pct=0.05)
n_pairs = len(df_filtered)
n_pathways = df_filtered.iloc[:, 2].nunique() if df_filtered.shape[1] > 2 else None
print(f"pairs after min_cell_pct=0.05: {n_pairs} | pathways: {n_pathways}", flush=True)

from commot.tl import spatial_communication, cluster_communication
results = {}
for thr in [250, 150, 400]:
    t = time.time()
    spatial_communication(adata, database_name="cellchat", df_ligrec=df_filtered,
                          dis_thr=thr, heteromeric=True, pathway_sum=True)
    print(f"dis_thr={thr}: spatial_communication done in {time.time()-t:.0f}s", flush=True)
    t = time.time()
    cluster_communication(adata, database_name="cellchat", clustering="macro_type",
                          n_permutations=1000, random_seed=0)
    print(f"dis_thr={thr}: cluster_communication done in {time.time()-t:.0f}s", flush=True)
    # spot-level totals (COMMOT stores marginal sums as DataFrames in obsm)
    skey = "commot-cellchat-sum-sender"
    rkey = "commot-cellchat-sum-receiver"
    df_spot = pd.DataFrame({
        "barcode": adata.obs_names,
        "macro_type": adata.obs["macro_type"].values,
        "sent_s_total_total": np.asarray(adata.obsm[skey]["s-total-total"]).ravel(),
        "received_r_total_total": np.asarray(adata.obsm[rkey]["r-total-total"]).ravel(),
    })
    df_spot.to_csv(f"{OUT}/spot_level_disThr{thr}.csv", index=False)
    # group-level cluster communication results
    cck = [k for k in adata.uns.keys() if "commot" in str(k) and "cluster" in str(k)]
    results[f"disThr{thr}"] = dict(n_pairs=n_pairs, n_pathways=n_pathways,
                                   commot_uns_keys=cck)
    # dump cluster-communication uns frames
    for k in cck:
        v = adata.uns[k]
        if hasattr(v, "to_csv"):
            v.to_csv(f"{OUT}/uns_{k}_disThr{thr}.csv")
        elif isinstance(v, dict):
            for kk, vv in v.items():
                if hasattr(vv, "to_csv"):
                    vv.to_csv(f"{OUT}/uns_{k}-{kk}_disThr{thr}.csv")
    print(f"dis_thr={thr}: saved spot-level + cluster frames", flush=True)

# pathway-level spot signals (main run, 250) for pathway analysis
w = adata.uns.get("commot-cellchat-weights") if "commot-cellchat-weights" in adata.uns else None
info = dict(n_pairs=int(n_pairs), n_pathways=(int(n_pathways) if n_pathways else None),
            versions=dict(python=sys.version.split()[0],
                          commot=commot.__version__ if hasattr(commot, "__version__") else "0.0.3",
                          scanpy=sc.__version__, anndata=ad.__version__,
                          numpy=np.__version__, pandas=pd.__version__),
            obsm_keys=[str(k) for k in adata.obsm.keys()],
            uns_keys=[str(k) for k in adata.uns.keys()])
json.dump(info, open(f"{OUT}/run_info.json", "w"), indent=1)
adata.write_h5ad(f"{OUT}/commot_full_matrix_250_with_sensitivity.h5ad", compression="gzip")
print(f"ALL DONE in {time.time()-t0:.0f}s", flush=True)
