"""
export_rctd_human.py -- Esporta i campioni Visium umani (GSE304669, GSE183456)
nel formato letto da run_rctd_human.R: counts grezzi (geni x spot), geni,
barcode e coordinate, in data/rctd_human/<dataset>/<campione>/.
Usa gli stessi caricatori di datasetC_canonical.py, quindi gli stessi spot.

Uso (dalla cartella python_files, ambiente stpipeline):
  python experiments/rebuttal/export_rctd_human.py
"""
import os, importlib.util
import numpy as np, pandas as pd, scipy.io as sio, scipy.sparse as sp

spec = importlib.util.spec_from_file_location("dc", "experiments/rebuttal/datasetC_canonical.py")
dc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dc)

for ds, cfg in dc.DATASETS.items():
    loader = dc.load_sample if cfg["loader"] == "geo_files" else dc.load_spaceranger
    for name, sub in cfg["samples"].items():
        a = loader(os.path.join(cfg["data_dir"], sub))
        out = os.path.join("data/rctd_human", ds, name)
        os.makedirs(out, exist_ok=True)
        X = sp.csr_matrix(a.X)
        if not np.allclose(X.data[:100000], np.round(X.data[:100000])):
            raise ValueError(f"{ds}/{name}: counts non interi")
        sio.mmwrite(os.path.join(out, "st_counts.mtx"), X.T.tocoo().astype(np.int64), field="integer")
        pd.Series(a.var_names).to_csv(os.path.join(out, "st_genes.txt"), index=False, header=False)
        pd.Series(a.obs_names).to_csv(os.path.join(out, "st_barcodes.txt"), index=False, header=False)
        pd.DataFrame(a.obsm["spatial"], index=a.obs_names, columns=["x", "y"]).to_csv(
            os.path.join(out, "st_coords.csv"))
        print(f"{ds}/{name}: {a.n_obs} spot, {a.n_vars} geni", flush=True)
print("Export completato in data/rctd_human/")
