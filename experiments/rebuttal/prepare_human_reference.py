"""
prepare_human_reference.py -- Converte il riferimento Susztak (GSE211785,
PreSCVI h5ad) nel formato letto da datasetC_canonical.py e scrive la
mappatura esplicita tipo cellulare -> macro-tipo.

Scelte (fissate prima del run):
  * solo cellule RNA (tech = SC_RNA o SN_RNA); SN_ATAC escluse;
  * counts dal layer 'counts' (X contiene dati gia' normalizzati);
  * tutte le condizioni (Control + Disease), cosi' il riferimento contiene
    anche il PT danneggiato (iPT), atteso nelle biopsie di rigetto;
  * sottocampione casuale proporzionale di 20.000 cellule (seed 0).

Uso (dalla cartella python_files, ambiente stpipeline):
  python prepare_human_reference.py
"""
import os
import numpy as np
import pandas as pd
import anndata as ad
import scipy.io as sio
import scipy.sparse as sp

REF = "data/human_ref/GSE211785_Susztak_SC_SN_ATAC_merged_PreSCVI_final.h5ad"
OUT = "data/human_ref"
N_CELLS, SEED = 20000, 0

# Mappatura esplicita (Cluster_Idents -> macro-tipo del modello)
MACRO = {
    # Proximal tubule (incluso PT danneggiato)
    "PT_S1": "PT", "PT_S2": "PT", "PT_S3": "PT", "iPT": "PT",
    # Distal convoluted tubule
    "DCT1": "DCT", "DCT2": "DCT",
    # Thick ascending limb (la macula densa e' il tratto terminale del cTAL)
    "M_TAL": "TAL", "C_TAL": "TAL", "Macula_Densa": "TAL",
    # Endotelio (come "Endothelial" -> vascular nel riferimento murino)
    "Endo_Peritubular": "vascular", "Endo_GC": "vascular", "Endo_Lymphatic": "vascular",
    # Immunitario
    "CD8T": "immune", "CD4T": "immune", "NK": "immune", "CD16_Mono": "immune",
    "CD14_Mono": "immune", "B_Naive": "immune", "B_memory": "immune", "Mac": "immune",
    "Neutrophil": "immune", "cDC": "immune", "pDC": "immune", "Plasma_Cells": "immune",
    "Baso/Mast": "immune", "Prolif_Lym": "immune",
    # Altro: segmenti e tipi non modellati come layer
    "CNT": "other", "PC": "other", "IC_A": "other", "IC_B": "other",
    "Des-Thin_Limb": "other", "Ascending_Thin_LOH": "other",
    "Podo": "other", "PEC": "other", "Mes": "other", "GS_Stromal": "other",
    "Fibroblast_1": "other", "Fibroblast_2": "other", "MyoFib/VSMC": "other",
    "Neural_Cells": "other", "RBC": "other",
}

a = ad.read_h5ad(REF, backed="r")
obs = a.obs[["tech", "Cluster_Idents"]].copy()
rna = np.where(obs["tech"].isin(["SC_RNA", "SN_RNA"]).values)[0]
labels = set(obs["Cluster_Idents"].iloc[rna].astype(str))
missing = labels - set(MACRO)
if missing:
    raise ValueError(f"Etichette senza mappatura: {sorted(missing)}")
print(f"Cellule RNA: {len(rna)} (escluse {a.n_obs - len(rna)} SN_ATAC)")

rng = np.random.default_rng(SEED)
idx = np.sort(rng.choice(rna, size=min(N_CELLS, len(rna)), replace=False))
sub = a[idx].to_memory()
counts = sub.layers["counts"]
counts = counts.tocsr() if sp.issparse(counts) else sp.csr_matrix(counts)
vals = counts.data[:100000]
print(f"Sottocampione: {sub.n_obs} cellule x {sub.n_vars} geni | "
      f"counts interi: {np.allclose(vals, np.round(vals))}")

sio.mmwrite(os.path.join(OUT, "sc_counts.mtx"), counts.T.tocoo())    # geni x cellule
pd.Series(sub.var_names).to_csv(os.path.join(OUT, "sc_genes.txt"), index=False, header=False)
pd.Series(sub.obs_names).to_csv(os.path.join(OUT, "sc_barcodes.txt"), index=False, header=False)
meta = sub.obs[["Cluster_Idents", "tech"]].copy()
meta.insert(0, "barcode", sub.obs_names)
meta[["barcode", "tech", "Cluster_Idents"]].to_csv(os.path.join(OUT, "sc_meta.txt"),
                                                    sep="\t", index=False)

counts_by_type = sub.obs["Cluster_Idents"].astype(str).value_counts()
mp = pd.DataFrame({"cell_identity": counts_by_type.index,
                   "n_cells_subsample": counts_by_type.values,
                   "macro_type": [MACRO[c] for c in counts_by_type.index]})
mp.to_csv(os.path.join(OUT, "human_macro_map.csv"), index=False)
print("\nCellule per macro-tipo nel sottocampione:")
print(mp.groupby("macro_type")["n_cells_subsample"].sum().to_string())
print(f"\nFile scritti in {OUT}: sc_counts.mtx, sc_genes.txt, sc_barcodes.txt, "
      f"sc_meta.txt, human_macro_map.csv")
