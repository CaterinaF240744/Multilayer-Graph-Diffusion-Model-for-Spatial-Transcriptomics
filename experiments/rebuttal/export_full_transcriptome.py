"""
Scarica V1_Mouse_Kidney (10x Genomics, Space Ranger 1.1.0) ed esporta gli
input full-transcriptome per cellchat_full_transcriptome.R.

Uso (dalla cartella python_files, ambiente stpipeline attivo):
    python export_full_transcriptome.py

Le annotazioni macro_type vengono riprese dal file del pannello
(data/cellchat_inputs/v1_kidney_meta.csv), cosi' le etichette per spot
sono identiche a quelle del manoscritto.
"""
import os
import pandas as pd
import scipy.io
import scipy.sparse as sp
import scanpy as sc

OUT_DIR = "data/full_matrix_inputs"
PANEL_META = "data/cellchat_inputs/v1_kidney_meta.csv"
os.makedirs(OUT_DIR, exist_ok=True)

# 1. Download dal sito 10x (salvato in ./data/V1_Mouse_Kidney/)
adata = sc.datasets.visium_sge(sample_id="V1_Mouse_Kidney")
adata.var_names_make_unique()
print("Matrice completa:", adata.n_obs, "spot x", adata.n_vars, "geni")

# 2. Annotazioni macro_type dall'analisi originale
meta = pd.read_csv(PANEL_META)
assert {"barcode", "macro_type"} <= set(meta.columns), meta.columns
meta["barcode"] = meta["barcode"].astype(str)
found = meta["barcode"].isin(adata.obs_names)
print(f"Barcode del meta trovati nella matrice 10x: {found.sum()}/{len(meta)}")
if not found.all():
    print("ATTENZIONE, esempi non trovati:", meta.loc[~found, "barcode"].head().tolist())
meta = meta[found].reset_index(drop=True)

# 3. Stesso ordine di spot per tutti i file
adata = adata[meta["barcode"].values].copy()

# 4. Esporta (nomi dei geni con la capitalizzazione originale)
X = sp.csr_matrix(adata.X)
scipy.io.mmwrite(os.path.join(OUT_DIR, "v1_kidney_full_counts.mtx"), X.T)  # geni x spot, counts raw
pd.Series(adata.var_names).to_csv(os.path.join(OUT_DIR, "v1_kidney_full_genes.csv"),
                                  index=False, header=False)
pd.Series(adata.obs_names).to_csv(os.path.join(OUT_DIR, "v1_kidney_full_barcodes.csv"),
                                  index=False, header=False)
meta[["barcode", "macro_type"]].to_csv(os.path.join(OUT_DIR, "v1_kidney_full_meta.csv"),
                                       index=False)
coords = pd.DataFrame(adata.obsm["spatial"], columns=["x", "y"])  # pixel full-resolution
coords.insert(0, "barcode", adata.obs_names.values)
coords.to_csv(os.path.join(OUT_DIR, "v1_kidney_full_coords.csv"), index=False)

# 5. Controlli
sf = adata.uns["spatial"]["V1_Mouse_Kidney"]["scalefactors"]
print("spot_diameter_fullres:", sf["spot_diameter_fullres"], "(nello script R: 89.45675017406688)")
print("macro_type:\n", meta["macro_type"].value_counts())
for g in ["Adgre1", "Cd68", "Csf1r", "Ptprc"]:
    if g in adata.var_names:
        print(f"{g}: presente, UMI totali = {int(adata[:, g].X.sum())}")
    else:
        print(f"{g}: assente")
print("Fatto. File in", OUT_DIR)
