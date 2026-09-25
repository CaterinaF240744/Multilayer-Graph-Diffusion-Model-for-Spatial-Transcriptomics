"""run_c2l.py -- cell2location annotation for the annotation-robustness
experiment (Reviewer 2, annotation comparison: NN vs RCTD vs cell2location).

Note: this is a heavy-compute step (deep generative model training,
GPU recommended); it was run separately from the lighter RCTD/NN steps.

Usage:
    python run_c2l.py [--in-dir PATH] [--out-dir PATH]

Inputs (in --in-dir, default "data/rctd_io" -- same export used by
run_rctd.R): st_counts.mtx, st_barcodes.csv, st_genes.csv, st_coords.csv,
sc_counts.mtx, sc_barcodes.csv, sc_genes.csv, sc_meta.csv (with a
"cell_identity" column).

Output (in --out-dir, default "results/annotation"):
    ref_model/            saved cell2location reference regression model
    labels_c2l.csv        (barcode, c2l_label, c2l_entropy)
"""
import argparse
import os

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.io as sio
import cell2location


def load_st(in_dir):
    X = sio.mmread(os.path.join(in_dir, "st_counts.mtx")).T.tocsr()  # spots x genes
    barcodes = pd.read_csv(os.path.join(in_dir, "st_barcodes.csv"), header=None)[0].astype(str).values
    genes = pd.read_csv(os.path.join(in_dir, "st_genes.csv"), header=None)[0].astype(str).values
    adata = sc.AnnData(X, obs=pd.DataFrame(index=barcodes), var=pd.DataFrame(index=genes))
    adata.var_names_make_unique()
    sc.pp.filter_genes(adata, min_counts=3)
    sc.pp.filter_cells(adata, min_counts=3)
    print("ST:", adata.shape, flush=True)
    return adata


def load_reference(in_dir):
    X = sio.mmread(os.path.join(in_dir, "sc_counts.mtx")).T.tocsr()
    barcodes = pd.read_csv(os.path.join(in_dir, "sc_barcodes.csv"), header=None)[0].astype(str).values
    genes = pd.read_csv(os.path.join(in_dir, "sc_genes.csv"), header=None)[0].astype(str).values
    meta = pd.read_csv(os.path.join(in_dir, "sc_meta.csv"), index_col=0)
    labels = meta.loc[barcodes, "cell_identity"].astype(str).values
    adata_ref = sc.AnnData(X, obs=pd.DataFrame({"cell_identity": labels}, index=barcodes),
                           var=pd.DataFrame(index=genes))
    adata_ref.var_names_make_unique()

    # subsample to 15k cells, keeping only types with >= 25 cells
    # (matches the RCTD protocol in run_rctd.R for a fair comparison)
    rng = np.random.default_rng(0)
    if adata_ref.n_obs > 15000:
        keep = rng.choice(adata_ref.n_obs, 15000, replace=False)
        adata_ref = adata_ref[keep].copy()
    tab = adata_ref.obs["cell_identity"].value_counts()
    keep_types = tab[tab >= 25].index
    adata_ref = adata_ref[adata_ref.obs["cell_identity"].isin(keep_types)].copy()
    adata_ref.obs["cell_identity"] = pd.Categorical(adata_ref.obs["cell_identity"])
    print("ref:", adata_ref.shape, adata_ref.obs["cell_identity"].nunique(), "types", flush=True)
    return adata_ref


def fit_reference_signatures(adata_ref, out_dir):
    from cell2location.models import RegressionModel
    sc.pp.filter_genes(adata_ref, min_counts=3)
    RegressionModel.setup_anndata(adata_ref, labels_key="cell_identity")

    model_dir = os.path.join(out_dir, "ref_model")
    if os.path.exists(os.path.join(model_dir, "model.pt")):
        print("loading saved reference model...", flush=True)
        model_ref = RegressionModel.load(model_dir, adata=adata_ref)
    else:
        model_ref = RegressionModel(adata_ref)
        model_ref.train(max_epochs=250, batch_size=2500, train_size=1.0)
        model_ref.save(model_dir, overwrite=True)

    adata_ref = model_ref.export_posterior(model_ref.adata, sample_kwargs={"batch_size": 2500})
    vkey = "q05_per_cluster_mu_fg" if "q05_per_cluster_mu_fg" in adata_ref.varm else "q05_mu"
    inf_aver = adata_ref.varm[vkey]
    inf_aver = pd.DataFrame(np.asarray(inf_aver), index=adata_ref.var_names,
                            columns=adata_ref.uns["mod"]["factor_names"])
    return inf_aver


def map_spots(adata, inf_aver):
    inf_aver = inf_aver[inf_aver.index.isin(adata.var_names)]
    print("signatures:", inf_aver.shape, flush=True)
    adata = adata[:, inf_aver.index].copy()

    cell2location.models.Cell2location.setup_anndata(adata, batch_key=None)
    model = cell2location.models.Cell2location(
        adata, cell_state_df=inf_aver, N_cells_per_location=8, detection_alpha=20)
    model.train(max_epochs=2000, batch_size=None, train_size=1.0)
    adata = model.export_posterior(model.adata, sample_kwargs={"batch_size": None})

    abund = adata.obsm["q05_cell_abundance_w_sf"]
    abund = pd.DataFrame(np.asarray(abund), index=adata.obs_names,
                         columns=adata.uns["mod"]["factor_names"])
    labels = abund.idxmax(axis=1)
    # abundance entropy as a per-spot confidence proxy (lower = more confident)
    p = abund.div(abund.sum(axis=1), axis=0)
    entropy = -(p * np.log(p + 1e-12)).sum(axis=1)
    return labels, entropy


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", default="data/rctd_io")
    ap.add_argument("--out-dir", default="results/annotation")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    adata_st = load_st(a.in_dir)
    adata_ref = load_reference(a.in_dir)
    inf_aver = fit_reference_signatures(adata_ref, a.out_dir)
    labels, entropy = map_spots(adata_st, inf_aver)

    out = pd.DataFrame({"c2l_label": labels, "c2l_entropy": entropy})
    out_path = os.path.join(a.out_dir, "labels_c2l.csv")
    out.to_csv(out_path)
    print("saved", out_path)
    print(out["c2l_label"].value_counts().head(), flush=True)


if __name__ == "__main__":
    main()
