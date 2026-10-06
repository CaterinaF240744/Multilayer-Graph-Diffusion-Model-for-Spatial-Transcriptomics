"""run_c2l.py -- cell2location annotation for the annotation-robustness
analysis (Section 8.3), V1 Mouse Kidney.

Usage (from the repository root, environment with cell2location):
    python experiments/rebuttal/run_c2l.py [--in-dir data/rctd_io] [--out-dir results/annotation]

Inputs (written by export_annotation_inputs.py): st_counts.mtx, st_barcodes.csv,
st_genes.csv, st_coords.csv, sc_counts.mtx, sc_barcodes.csv, sc_genes.csv,
sc_meta.csv (column "cell_identity").
Outputs: ref_model/, labels_c2l.csv (barcode, c2l_label, c2l_entropy),
c2l_abundance.csv (q05 cell abundance per reference type; used by
compare_annotations.py to derive the dominant macro-type).
"""
import argparse
import os

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.io as sio
import cell2location
from cell2location.models import RegressionModel

ap = argparse.ArgumentParser()
ap.add_argument("--in-dir", default="data/rctd_io")
ap.add_argument("--out-dir", default="results/annotation")
args = ap.parse_args()
IN, OUT = args.in_dir, args.out_dir
os.makedirs(OUT, exist_ok=True)

# ---- ST --------------------------------------------------------------------
X = sio.mmread(f"{IN}/st_counts.mtx").T.tocsr()  # spots x genes
st_bc = pd.read_csv(f"{IN}/st_barcodes.csv", header=None)[0].astype(str).values
st_genes = pd.read_csv(f"{IN}/st_genes.csv", header=None)[0].astype(str).values
adata = sc.AnnData(X, obs=pd.DataFrame(index=st_bc), var=pd.DataFrame(index=st_genes))
adata.var_names_make_unique()
sc.pp.filter_genes(adata, min_counts=3)
sc.pp.filter_cells(adata, min_counts=3)
print("ST:", adata.shape, flush=True)

# ---- reference -------------------------------------------------------------
refX = sio.mmread(f"{IN}/sc_counts.mtx").T.tocsr()
ref_bc = pd.read_csv(f"{IN}/sc_barcodes.csv", header=None)[0].astype(str).values
ref_genes = pd.read_csv(f"{IN}/sc_genes.csv", header=None)[0].astype(str).values
meta = pd.read_csv(f"{IN}/sc_meta.csv", index_col=0)
labels = meta.loc[ref_bc, "cell_identity"].astype(str).values
adata_ref = sc.AnnData(refX, obs=pd.DataFrame({"cell_identity": labels}, index=ref_bc),
                       var=pd.DataFrame(index=ref_genes))
adata_ref.var_names_make_unique()
rng = np.random.default_rng(0)
if adata_ref.n_obs > 15000:                       # same cap as RCTD
    keep = rng.choice(adata_ref.n_obs, 15000, replace=False)
    adata_ref = adata_ref[keep].copy()
tab = adata_ref.obs["cell_identity"].value_counts()
adata_ref = adata_ref[adata_ref.obs["cell_identity"].isin(tab[tab >= 25].index)].copy()
adata_ref.obs["cell_identity"] = pd.Categorical(adata_ref.obs["cell_identity"])
print("ref:", adata_ref.shape, adata_ref.obs["cell_identity"].nunique(), "types", flush=True)

sc.pp.filter_genes(adata_ref, min_counts=3)
RegressionModel.setup_anndata(adata_ref, labels_key="cell_identity")
if os.path.exists(f"{OUT}/ref_model/model.pt") or os.path.exists(f"{OUT}/ref_model/params.pt"):
    print("loading saved reference model...", flush=True)
    model_ref = RegressionModel.load(f"{OUT}/ref_model", adata=adata_ref)
else:
    model_ref = RegressionModel(adata_ref)
    model_ref.train(max_epochs=250, batch_size=2500, train_size=1.0)
    model_ref.save(f"{OUT}/ref_model", overwrite=True)
adata_ref = model_ref.export_posterior(model_ref.adata, sample_kwargs={"batch_size": 2500})
vkey = "q05_per_cluster_mu_fg" if "q05_per_cluster_mu_fg" in adata_ref.varm else "q05_mu"
inf_aver = pd.DataFrame(np.asarray(adata_ref.varm[vkey]), index=adata_ref.var_names,
                        columns=adata_ref.uns["mod"]["factor_names"])
inf_aver = inf_aver[inf_aver.index.isin(adata.var_names)]
print("signatures:", inf_aver.shape, flush=True)
if inf_aver.shape[0] < 1000:
    raise ValueError("Too few shared genes between ST and reference (check gene-symbol case)")

# ---- spatial mapping -------------------------------------------------------
adata = adata[:, inf_aver.index].copy()
cell2location.models.Cell2location.setup_anndata(adata, batch_key=None)
model = cell2location.models.Cell2location(
    adata, cell_state_df=inf_aver, N_cells_per_location=8, detection_alpha=20)
model.train(max_epochs=2000, batch_size=None, train_size=1.0)
adata = model.export_posterior(model.adata, sample_kwargs={"batch_size": None})
abund = pd.DataFrame(np.asarray(adata.obsm["q05_cell_abundance_w_sf"]), index=adata.obs_names,
                     columns=adata.uns["mod"]["factor_names"])
abund.to_csv(f"{OUT}/c2l_abundance.csv")
p = abund.div(abund.sum(axis=1), axis=0)
out = pd.DataFrame({"c2l_label": abund.idxmax(axis=1),
                    "c2l_entropy": -(p * np.log(p + 1e-12)).sum(axis=1)})
out.to_csv(f"{OUT}/labels_c2l.csv")
print("saved", f"{OUT}/labels_c2l.csv and c2l_abundance.csv", flush=True)
print(out["c2l_label"].value_counts().head(), flush=True)
