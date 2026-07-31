"""
generality_second_tissue.py
------------------------------
Experiment D (lowest priority, cut first if time runs out). Demonstrates
that the multilayer diffusion machinery itself -- not the IRI-specific
biology -- generalises to a second tissue, addressing Reviewer 1 point 3
("single dataset, single tissue") at minimal cost.

Scope, deliberately kept narrow: NO attempt to calibrate biologically
meaningful beta/rho parameters for the new tissue (that would require a
literature review as deep as the kidney IRI one, which is out of scope for
a 3-week rebuttal). Instead:
  - generic, tissue-agnostic parameters (uniform beta, uniform rho) are used
  - a boundary/edge region is seeded, mirroring the "outer medulla seed"
    logic of Section 6.3
  - the only claim made is structural: propagation follows the anatomical
    connectivity of the new tissue in a qualitatively sensible way
    (a monotone front, not random/disordered activation).

Recommended dataset: any public 10x Visium mouse brain (sagittal) section,
already widely used and well annotated with anatomical regions -- this
minimises the amount of new preprocessing/QC needed relative to a less
common dataset.
"""

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

from diffusion_model import CompartmentDiffusionModel
from preprocess_compartments import build_compartment_graph

# ---------------------------------------------------------------------
# 0. Configuration -- point this at any public Visium brain dataset with
#    an existing region/cluster annotation (e.g. Allen-CCF-mapped regions,
#    or just Leiden clusters if no anatomical annotation is available).
# ---------------------------------------------------------------------
BRAIN_VISIUM_DIR = "data/visium_mouse_brain_raw.h5ad"
REGION_OBS_COLUMN = "region"  # or "leiden" if no anatomical labels exist

# Generic, non-calibrated parameters -- same value for every compartment.
# The point of this experiment is NOT to recover tissue-specific biology,
# only to show the framework produces a coherent propagation front.
GENERIC_BETA = 0.30
GENERIC_RHO = 0.10

# Which region(s) to use as the seed -- edit after inspecting the actual
# region labels of your chosen dataset. For a sagittal brain section, a
# natural choice is an outer/peripheral structure (e.g. cortical layer 1,
# or meninges-adjacent regions) to mirror the "boundary seeding" logic used
# for the kidney outer medulla.
SEED_REGIONS = ["1"]


def run_generality_check():
    adata = sc.read_h5ad(BRAIN_VISIUM_DIR)
    adata.var_names_make_unique()
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    if REGION_OBS_COLUMN not in adata.obs.columns:
        # fall back to unsupervised clustering if no annotation is provided
        sc.pp.highly_variable_genes(adata, n_top_genes=2000)
        sc.pp.pca(adata[:, adata.var.highly_variable])
        sc.pp.neighbors(adata)
        sc.tl.leiden(adata, resolution=0.5, key_added="leiden")
        adata.obs[REGION_OBS_COLUMN] = adata.obs["leiden"]

    adata.obs["compartment"] = adata.obs[REGION_OBS_COLUMN].astype(str)

    W_c, names, adata = build_compartment_graph(adata, k=6)

    n = len(names)
    model = CompartmentDiffusionModel(
        W_c, beta=[GENERIC_BETA] * n, rho=[GENERIC_RHO] * n,
        D_H=0.02, D_D=0.02, compartment_names=names
    )

    p_D0 = np.array([0.05 if n_ in SEED_REGIONS else 0.0 for n_ in names])
    if p_D0.sum() == 0:
        raise ValueError(
            f"No compartment matched SEED_REGIONS={SEED_REGIONS}. "
            f"Available compartments: {names}"
        )

    traj = model.simulate(p_D0, t_end=60.0, dt=0.05)
    metrics = model.metrics(traj)
    metrics = metrics.sort_values("t_act")
    metrics.to_csv("generality_second_tissue_metrics.csv", index=False)

    # Minimal sanity plot: activation-time ranking, to visually confirm a
    # coherent (non-random) propagation order from the seed outward.
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(metrics["compartment"], metrics["t_act"])
    ax.set_xlabel("Simulated activation time")
    ax.set_title("Generality check: propagation order on second tissue "
                 "(generic, non-calibrated parameters)")
    fig.tight_layout()
    fig.savefig("generality_second_tissue_activation_order.png", dpi=200)

    print(metrics)
    print("\nIf this ordering looks anatomically coherent (seed region "
          "first, progressively more distant regions later), that is "
          "sufficient evidence for the generality claim in the rebuttal -- "
          "no further biological interpretation is needed for this "
          "minimal-scope experiment.")
    return metrics


if __name__ == "__main__":
    run_generality_check()
