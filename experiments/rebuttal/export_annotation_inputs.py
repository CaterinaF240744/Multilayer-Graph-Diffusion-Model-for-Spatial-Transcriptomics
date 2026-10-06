"""
export_annotation_inputs.py -- Prepara data/rctd_io/ per RCTD e cell2location
sul V1 Mouse Kidney (analisi di robustezza all'annotazione, Sezione 8.3).

  ST : trascrittoma completo dei 1.438 spot (data/full_matrix_inputs/, gia'
       esportato per CellChat), counts grezzi.
  Ref: GSE107585 (data/GSE107585_sc.h5ad), counts grezzi, etichette
       cell_identity assegnate con la stessa funzione annotate_sc del pipeline.
  Scrive anche macro_map.csv (cell_identity -> macro-tipo, da MACRO_MAP).

Uso (cartella python_files, ambiente stpipeline):
  python experiments/rebuttal/export_annotation_inputs.py
"""
import os, sys
import numpy as np, pandas as pd, scanpy as sc, scipy.io as sio, scipy.sparse as sp

sys.path.insert(0, "src")
from preprocessing import annotate_sc, MACRO_MAP

OUT, ST = "data/rctd_io", "data/full_matrix_inputs"
os.makedirs(OUT, exist_ok=True)

# --- riferimento single-cell ---
ref = sc.read_h5ad("data/GSE107585_sc.h5ad")
ref.var_names_make_unique()
X = ref.X.tocsr() if sp.issparse(ref.X) else sp.csr_matrix(ref.X)
if not np.allclose(X.data[:100000], np.round(X.data[:100000])):
    raise ValueError("GSE107585_sc.h5ad: .X non contiene counts grezzi")
ref = annotate_sc(ref)
# simboli del riferimento allineati a quelli del Visium (confronto senza maiuscole)
st_genes = pd.read_csv(f"{ST}/v1_kidney_full_genes.csv", header=None)[0].astype(str)
low2st = {g.lower(): g for g in st_genes}
ref.var_names = [low2st.get(g.lower(), g) for g in ref.var_names.astype(str)]
ref.var_names_make_unique()
lab = ref.obs["cell_identity"].astype(str)
print("Riferimento:", ref.shape, "\n", lab.value_counts().to_string())
sio.mmwrite(f"{OUT}/sc_counts.mtx", X.T.tocoo().astype(np.int64), field="integer")   # geni x cellule
pd.Series(ref.var_names).to_csv(f"{OUT}/sc_genes.csv", index=False, header=False)
pd.Series(ref.obs_names).to_csv(f"{OUT}/sc_barcodes.csv", index=False, header=False)
pd.DataFrame({"cell_identity": lab.values}, index=ref.obs_names).to_csv(f"{OUT}/sc_meta.csv")
pd.Series(MACRO_MAP, name="macro_type").rename_axis("cell_identity").to_csv(f"{OUT}/macro_map.csv")

# --- Visium, trascrittoma completo ---
C = sio.mmread(f"{ST}/v1_kidney_full_counts.mtx").tocoo()                     # geni x spot
sio.mmwrite(f"{OUT}/st_counts.mtx", C.astype(np.int64), field="integer")
genes = pd.read_csv(f"{ST}/v1_kidney_full_genes.csv", header=None)[0]
bcs = pd.read_csv(f"{ST}/v1_kidney_full_barcodes.csv", header=None)[0]
genes.to_csv(f"{OUT}/st_genes.csv", index=False, header=False)
bcs.to_csv(f"{OUT}/st_barcodes.csv", index=False, header=False)
xy = pd.read_csv(f"{ST}/v1_kidney_full_coords.csv").set_index("barcode").loc[bcs, ["x", "y"]]
xy.to_csv(f"{OUT}/st_coords.csv")
common = len(set(genes) & set(ref.var_names))
print(f"Spot: {len(bcs)} | geni in comune ST/riferimento: {common}")
if common < 5000:
    print("ATTENZIONE: pochi geni in comune, probabile differenza di maiuscole/minuscole")
print("Scritto", OUT)
