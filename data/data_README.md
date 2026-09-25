# Data

All datasets used in this study are publicly available. This folder is
intentionally empty in the repository (only this README and `.gitkeep`);
download the data as described below before running the pipelines.
All paths are relative to the repository root.

## 1. V1 Mouse Kidney — primary case study

- **Source**: 10x Genomics, *V1 Mouse Kidney* demonstration dataset
  (https://www.10xgenomics.com/datasets/v1-mouse-kidney-demonstration-1-stds-1),
  1,438 tissue spots after filtering (full cortico-papillary axis).
- **Single-cell reference for label transfer**: GSE107585
  (Park et al., 2018 — murine kidney scRNA-seq atlas, 16 cluster identities).
- **Expected files** (consumed by `experiments/rebuttal/sobol_gsa.py`):

  | File | Content |
  |---|---|
  | `v1_kidney_raw_counts.mtx` | raw counts, genes × spots (MatrixMarket) |
  | `v1_kidney_genes.csv` | gene symbols, one per line |
  | `v1_kidney_barcodes.csv` | spot barcodes, one per line |
  | `v1_kidney_meta.csv` | index `barcode`, column `macro_type` |
  | `v1_kidney_spatial_coords.csv` | index `barcode`, spot coordinates |

- **How to generate**: these files are written by
  `experiments/rebuttal/export_data_for_cellchat.py` from the raw 10x
  download plus the label-transfer annotation (`src/preprocessing.py`).
  Run it once and the same files serve the Sobol GSA, the CellChat
  analysis, and the RCTD/cell2location comparison.

## 2. GSE269622 — independent murine IRI cohort (Dataset B)

- **Source**: GEO https://doi.org/10.1093/... (accession **GSE269622**);
  male bilateral IRI, sham / 4 h / 12 h / day 2.
- **Expected layout** (standard Space Ranger outputs, loaded with
  `sc.read_visium`):

  ```
  data/GSE269622/extracted/mouse_sham/outs
  data/GSE269622/extracted/mouse_hour4/outs
  data/GSE269622/extracted/mouse_hour12/outs
  data/GSE269622/extracted/mouse_day2/outs
  ```

- **Reference**: `data/GSE107585_sc.h5ad` — AnnData with
  `obs["cell_identity"]` (same reference as the paper; subsampled to
  20k cells inside the script).
- **Run** (from the repository root, one process per sample):
  `python datasetB_lean.py sham` (then `hour4`, `hour12`, `day2`).

## 3. GSE304669 — human kidney transplant rejection (Dataset C)

- **Source**: GEO accession **GSE304669**; FFPE Visium CytAssist biopsies
  (non-rejection control, active AMR, acute TCMR, chronic active AMR).
- **Expected layout** — per-sample GSM prefixes, each containing the
  CytAssist files (`*_matrix.mtx.gz`, `*_barcodes.tsv.gz`,
  `*_features.tsv.gz`, `*_tissue_positions.csv.gz`):

  ```
  data/GSE304669/GSM9155022_control_58055/
  data/GSE304669/GSM9155023_active_AMR_58056/
  data/GSE304669/GSM9155024_acute_TCMR_58057/
  data/GSE304669/GSM9155025_chronic_active_AMR_58058/
  ```

- **Reference**: `data/human_ref/` — Susztak human kidney sc/sn reference
  (**GSE211785**) exported as `sc_counts.mtx`, `sc_genes.txt`,
  `sc_barcodes.txt`, `sc_meta.txt` (tab-separated; cell-type column
  `Cluster_Idents`). Subsampled to 30k cells inside the script.
- **Run** (from the repository root): `python datasetC_pipeline.py`.

## 4. GSE182939 — longitudinal murine IRI (temporal validation)

- **Source**: GEO accession **GSE182939** (Dixon et al., 2022); sham,
  4 h, 12 h, 48 h, 6 weeks.
- **Expected layout** (edit `SAMPLE_PATHS` in
  `experiments/rebuttal/preprocess_compartments.py` if your local
  folder names differ):

  ```
  data/GSE182939/sham/fsham_137_processed/outs
  data/GSE182939/4h/f4hr_115_processed/outs
  data/GSE182939/12h/f12hr_140_processed/outs
  data/GSE182939/48h/f2dps_158_processed/outs
  data/GSE182939/6wk/f6wks_110_processed/outs
  ```

- **Reference**: `data/GSE107585_sc.h5ad` (as above).
- **Run**: `python experiments/rebuttal/temporal_validation_FINAL.py`
  (the pre-registered version; `temporal_validation_gse182939.py` is
  deprecated and kept for historical reference only).

## 5. `data/rctd_io/` — annotation robustness (RCTD, cell2location)

- **How to generate**: `experiments/rebuttal/export_data_for_cellchat.py`
  also writes the format-agnostic export consumed by the two alternative
  annotation methods:

  ```
  data/rctd_io/st_counts.mtx      # genes × spots
  data/rctd_io/st_barcodes.csv
  data/rctd_io/st_genes.csv
  data/rctd_io/sc_counts.mtx      # genes × cells
  data/rctd_io/sc_meta.csv        # reference cell-type labels
  ```

- **Consumed by**: `run_rctd.R` (spacexr/RCTD) and `run_c2l.py`
  (cell2location).

## 6. Slide-seqV2 mouse cerebellum — cross-technology generality

- **Source**: Single Cell Portal, dataset **SCP948** (Stickels et al.,
  2021, Slide-seqV2).
- Convert the downloaded matrix to AnnData and place it as
  `data/slideseq_cerebellum.h5ad` (see the regeneration script
  `slideseq_regen.py` for the expected marker-based macro-type annotation
  and folial zoning).

## 7. Visium mouse brain — second-tissue generality

- `data/visium_mouse_brain_raw.h5ad` — raw Visium mouse brain AnnData,
  consumed by `experiments/rebuttal/generality_second_tissue.py`.

---

**Note on sizes.** The raw spatial datasets total several GB and cannot be
hosted in the repository. Regenerate everything with the export scripts
above; the derived compartment graphs and simulation outputs are fully
deterministic given the data and the parameters in `src/sir_compartments.py`.
