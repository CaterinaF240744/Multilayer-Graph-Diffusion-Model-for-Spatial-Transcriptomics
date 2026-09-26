# Multilayer Graph Diffusion Model for Spatial Transcriptomics

A multilayer graph diffusion framework that simulates compartment-level perturbation propagation as an exposure-driven diffusion-reaction process on spatial graphs derived from spatial transcriptomics data.
Applied to the V1 Mouse Kidney 10x Visium dataset in the context of
ischaemia-reperfusion injury (IRI).

## Authors

Caterina Francesca Perri¹, Annamaria Defilippo¹, Federico Manuel Giorgi³,
Pietro Hiram Guzzi¹, Pierangelo Veltri²

1. Department of Surgical and Medical Sciences, University of Catanzaro
2. Department of Computer Science Modelling and Systems (DIMES), University of Calabria
3. Department of Pharmacy and Biotechnology, University of Bologna

## Overview

The framework represents tissue as a spatial graph where nodes are Visium spots
annotated by transcriptomic state and phenotype. A multilayer architecture
separates distinct cell populations (PT, DCT, TAL) into dedicated layers
connected through inter-layer coupling. State transitions
(Healthy -> Diseased -> Recovered, with Dead and attenuated states in the
pharmacological extension) are driven by local exposure to diseased neighbours,
following an exposure-driven diffusion-reaction formulation.

### Key features

- **Exposure-driven dynamics**: activation term is beta*H*lambda (not bilinear
  beta*H*D), where lambda is the local disease exposure field
- **Multilayer graph**: PT/DCT/TAL layers with node-to-node inter-layer coupling
  derived from spot-level kNN adjacency
- **Anatomical zoning**: compartments defined by macro cell-type x anatomical
  zone (cortex, outer medulla, inner medulla) using marker gene expression
- **Pharmacological extension**: H/D/D2/R/X + drug concentration field with
  Hill kinetics
- **Control experiments**: randomized edge null model, uniform parameters,
  vascular edge removal, alternative seeds, R-state sensitivity
- **Temporal validation**: out-of-sample forecast against the GSE182939 IRI
  time-course (sham, 4h, 12h, 48h, 6wk)
- **Cross-tissue generality**: uncalibrated application to mouse brain Visium

## Installation

```bash
pip install -r requirements.txt
```

## Data download

### Visium (V1 Mouse Kidney)
The dataset is automatically downloaded by squidpy on first run:
```python
import squidpy as sq
adata = sq.datasets.visium("V1_Mouse_Kidney")
```

### Single-cell reference (GSE107585)
Download from GEO: [GSE107585](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE107585)

Place the file `Mouse_kidney_single_cell_datamatrix.txt` in the `data/` directory.

### Temporal validation data (GSE182939)
Download from GEO: [GSE182939](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE182939)

Place the 5 Space Ranger output directories under `data/GSE182939/` with the
structure defined in `experiments/rebuttal/preprocess_compartments.py`
(`SAMPLE_PATHS`).

## Quick start

```bash
# Run the full analysis pipeline (Sections 5-7)
python experiments/run_main.py

# Run control experiments (Section 7.4 / R2.6)
python experiments/run_controls.py

# Run sensitivity analysis (Section 8.1)
python experiments/run_sensitivity.py
```

## Repository structure

```
kidney-graph-diffusion/
├── src/                        # Source modules
│   ├── preprocessing.py            # Visium + scRNA-seq loading and preprocessing
│   ├── graph_utils.py              # Spatial graph construction and random walk
│   ├── compartments.py             # Compartmental graph (macro_type x anatomical zone)
│   ├── sir_compartments.py         # H/D/R exposure-driven model (single-layer)
│   ├── sir_multilayer.py           # H/D/R multilayer model (exposure-driven)
│   ├── sir_drug.py                 # H/D/D2/R/X + drug pharmacological model
│   ├── sir_invasion_metrics.py     # Per-compartment invasion metrics
│   ├── sir_transcompartment_metrics.py  # Trans-compartment flux, entropy, NGM
│   ├── sir_sensitivity.py          # OAT, Morris, Monte Carlo sensitivity analysis
│   ├── sir_section7.py             # Mechanistic analysis (edge flux, ablation, seeding)
│   ├── sir_controls.py             # Control experiments (R2 point 6)
│   ├── sir_transition_graphs.py    # Transition graph visualizations
│   ├── sir_compartment_graphs.py   # Per-compartment graph visualizations
│   ├── sir_drug_graphs.py          # Drug model graph visualizations
│   ├── sir_spatial_plots.py        # Spatial map visualizations for publication
│   ├── fix_interlayer_flux.py      # Patch for inter-layer flux computation
│   └── main.py                     # Main analysis pipeline
├── experiments/                # Entry-point scripts
│   ├── run_main.py                 # Full pipeline entry point
│   ├── run_controls.py             # Control experiments entry point
│   ├── run_sensitivity.py          # Sensitivity analysis entry point
│   └── rebuttal/                   # Rebuttal experiments
│       ├── preprocess_compartments.py    # Label transfer + compartment graph for GSE182939
│       ├── diffusion_model.py            # Reusable H/D/R diffusion model (Euler integrator)
│       ├── temporal_validation_FINAL.py  # Pre-registered temporal validation (Section 8.2)
│       ├── temporal_validation_gse182939.py  # Deprecated (redirects to FINAL)
│       ├── generality_second_tissue.py   # Cross-tissue generality control (Section 8.3)
│       ├── export_data_for_cellchat.py   # Export Visium data for CellChat
│       ├── cellchat_analysis.R           # CellChat ligand-receptor analysis (R)
│       └── compare_cellchat_vs_model.py  # Compare CellChat scores vs model metrics
├── tests/                      # Unit tests
│   └── test_coherence.py           # Code-article coherence tests (17 tests)
├── docs/                       # Documentation
│   └── NOMENCLATURE.md             # H/D/R/X <-> S/I/R mapping
├── data/                       # Data files (not committed; download separately)
├── results/                    # Generated figures and tables
├── requirements.txt
├── LICENSE
└── README.md
```

## Nomenclature

The manuscript uses **H/D/R/X** (Healthy, Diseased, Recovered, Dead). The
codebase uses **S/I/R** as internal variable names for backward compatibility.
See [docs/NOMENCLATURE.md](docs/NOMENCLATURE.md) for the complete mapping.

## Reproducibility

### Important: `sir_section7.py` initialisation

The module `src/sir_section7.py` (manuscript Section 7 -- mechanistic
analysis) **requires** an explicit initialisation call before any of its
functions can be used:

```python
from compartments import build_compartment2
from sir_section7 import init_from_compartments

comp = build_compartment2(adata_st2, use_anatomical_zones=True)
init_from_compartments(comp)   # MUST be called before any Section 7 function
```

This call loads the real Visium-derived compartment graph (`comp["Wc2"]`,
`comp["comps2"]`, macro-type and zone labels) into the module's globals.
A runtime guard (`_check_initialised()`) raises `RuntimeError` if any
Section 7 function (`build_params`, `run_edge_importance`, `run_edge_ablation`,
`run_seeding_scenarios`, etc.) is called before initialisation. This was
previously a silent failure mode -- the module would run on a fictional
hardcoded graph with no error. In `main.py`, the call is placed in STEP 13,
immediately before `run_section7_analysis()`.

### Note on baseline invasion time across modules

The baseline mean invasion time differs between `sir_controls.py`
(Controls 1-5, t_inv ~ 8.35) and `sir_section7.py` (Section 7,
t_inv ~ 6.45). Both modules use the same real 13-compartment Visium graph,
the same correctly-separated D_H/D_D equations, and the same seed region
(`boundary_mask` = boundary + outer_medulla compartments). The difference
arises from **Laplacian normalisation**:

- `sir_compartments.py` (used by `sir_controls.py`) computes the
  **symmetric normalised Laplacian** `L = I - D^{-1/2} W D^{-1/2}`
  (`scipy.sparse.csgraph.laplacian(normed=True)`) and derives the
  exposure matrix W_tilde from it.
- `sir_section7.py` uses **row-normalised adjacency** `W_norm = W / row_sum`
  as both the exposure matrix and the basis for an unnormalised Laplacian
  `L = diag(W_norm * 1) - W_norm`.

These two normalisations produce different off-diagonal weights in both the
exposure term (lambda = W_tilde @ pD) and the diffusion term (D * L @ p),
leading to systematically different propagation speeds. This is a documented
methodological difference between the two analyses, not a bug. The manuscript
reports each module's baseline in its respective section.

### Manuscript output mapping

| Manuscript output | Code source |
|---|---|
| Table 1 (baseline parameters) | `sir_compartments.py::_MACRO_SIR` + `_REGION_SIR` |
| Tables 2-3 (temporal metrics) | `main.py` Step 5 + `sir_invasion_metrics.py` |
| Tables 4-5 (multilayer metrics) | `main.py` Step 7 + `sir_multilayer.py` |
| Table 6 (Kdyn matrix) | `main.py` Step 9 + `sir_transcompartment_metrics.py` |
| Tables 7-8 (drug modulation) | `main.py` Step 10b + `sir_drug.py` |
| Tables 9-11 (sensitivity) | `main.py` Step 11 + `sir_sensitivity.py` |
| Table 12 (edge ablation) | `sir_section7.py::run_edge_ablation` |
| Fig 4 (single-layer graph) | `sir_transition_graphs.py` |
| Fig 5 (multilayer snapshots) | `sir_transition_graphs.py` |
| Fig 6 (drug comparison) | `sir_drug_graphs.py` |
| Figs 7-8 (edge flux/ablation) | `sir_section7.py` |
| Figs 9-11 (seeding scenarios) | `sir_section7.py::run_seeding_scenarios` |

## License

MIT License (see LICENSE file).
