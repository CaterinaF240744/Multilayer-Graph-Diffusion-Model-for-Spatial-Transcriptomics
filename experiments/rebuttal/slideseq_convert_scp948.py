"""slideseq_convert_scp948.py -- build data/slideseq_<tissue>.h5ad from the
Single Cell Portal study SCP948 (Stickels et al. 2021, Slide-seqV2).

Download from https://singlecell.broadinstitute.org/single_cell/study/SCP948
(Download tab, sign-in required):
    Cerebellum_MappedDGEForR.csv     raw counts (genes x beads)
    Cerebellum_BeadLocationsForR.csv bead coordinates (barcodes, xcoord, ycoord)
(or the Hippocampus_* pair) and place them in data/slideseq_scp948/.

Usage
-----
python experiments/rebuttal/slideseq_convert_scp948.py --tissue cerebellum
-> data/slideseq_cerebellum.h5ad  (X = raw integer counts, CSR;
   obsm["spatial"] = bead coordinates in the units of the CSV)
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def read_locations(path):
    loc = pd.read_csv(path)
    cols = {c.lower(): c for c in loc.columns}
    bc = cols.get("barcodes", cols.get("barcode", loc.columns[0]))
    x = cols.get("xcoord", cols.get("x"))
    y = cols.get("ycoord", cols.get("y"))
    if x is None or y is None:
        raise ValueError(f"no x/y columns in {path}: {list(loc.columns)}")
    return loc.set_index(bc)[[x, y]].astype(float)


def read_counts(path, barcodes, chunksize=2000):
    """Dense genes x beads CSV -> sparse beads x genes (read in chunks of genes)."""
    head = pd.read_csv(path, nrows=0)
    cols = list(head.columns)
    first = cols[0]
    beads = cols[1:]
    if len(set(beads) & set(barcodes)) < 0.5 * len(beads):
        raise ValueError("bead barcodes in the count matrix do not match the locations "
                         "file; is the matrix beads x genes instead of genes x beads?")
    genes, blocks = [], []
    for ch in pd.read_csv(path, chunksize=chunksize, index_col=0):
        genes.extend(ch.index.astype(str))
        blocks.append(sp.csr_matrix(ch.to_numpy(dtype=np.int32).T))
        print(f"  {len(genes)} genes read", flush=True)
    X = sp.hstack(blocks).tocsr()
    return X, beads, genes, first


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tissue", default="cerebellum", choices=["cerebellum", "hippocampus"])
    ap.add_argument("--indir", default=os.path.join(ROOT, "data", "slideseq_scp948"))
    ap.add_argument("--out", default=None)
    a_ = ap.parse_args()
    T = a_.tissue.capitalize()
    loc = read_locations(os.path.join(a_.indir, f"{T}_BeadLocationsForR.csv"))
    print(f"{len(loc)} bead locations", flush=True)
    X, beads, genes, _ = read_counts(os.path.join(a_.indir, f"{T}_MappedDGEForR.csv"),
                                     set(loc.index))
    keep = [i for i, b in enumerate(beads) if b in loc.index]
    X = X[keep]
    beads = [beads[i] for i in keep]

    import anndata as ad
    a = ad.AnnData(X=X.astype(np.float32),
                   obs=pd.DataFrame(index=pd.Index(beads, name="barcode")),
                   var=pd.DataFrame(index=pd.Index(genes, name="gene")))
    a.obsm["spatial"] = loc.loc[beads].to_numpy()
    a.uns["source"] = f"SCP948 (Stickels et al. 2021, Slide-seqV2), {T} puck"
    out = a_.out or os.path.join(ROOT, "data", f"slideseq_{a_.tissue}.h5ad")
    a.write_h5ad(out)
    print(f"{a.n_obs} beads x {a.n_vars} genes, {X.nnz} non-zeros -> {out}", flush=True)


if __name__ == "__main__":
    main()
