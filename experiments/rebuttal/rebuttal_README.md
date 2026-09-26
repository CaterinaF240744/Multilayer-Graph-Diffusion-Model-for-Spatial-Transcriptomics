# Rebuttal experiments — script map

This folder contains the analyses added during revision. Each script
documents its own usage; data paths are described in `data/README.md`.
All scripts are run from the repository root unless stated otherwise.

## Robustness and sensitivity (Sections: benchmark, GSA, variance decomposition)

| Script | What it does | Manuscript output |
|---|---|---|
| `benchmark_synthetic.py` | Synthetic ground-truth benchmark: spot-level latent propagation → ST-like expression → blind pipeline (annotation → zoning → coarse-graining → simulation); noise dose-response (Experiment A) and one-factor misspecification (Experiment B) | Benchmark tables (noise dose-response, partial-R² variance decomposition), benchmark landscape figure |
| `sobol_gsa.py` | Global sensitivity analysis (Saltelli, N = 256 base samples, 2,816 evaluations, ±40% ranges) of the 9 kinetic parameters on the real V1 Kidney multilayer graph (18 compartments) | Sobol total-order indices table, rank-stability analysis |
| `diagnose_model_first.py` | Staged diagnostic (oracle → blind annotation → full blind) locating the benchmark recovery bottleneck | Diagnostic supporting the variance-decomposition interpretation |

## Ligand–receptor comparison (Section 8.6: Tables 15–16, Figure 12)

| Script | What it does |
|---|---|
| `cellchat_full_transcriptome.R` | CellChat v2, spatially constrained (interaction.range 250 µm), on the **full-transcriptome** V1 Kidney matrix (32,285 genes; 913 L–R pairs). Main CellChat result (Table 15). |
| `commot_full_transcriptome.py` | COMMOT on the same full-transcriptome matrix and labels (299 L–R pairs, 47 pathways); dis_thr 250 px (≈180 µm), sensitivity 150/400 px (≈110/290 µm). Table 15, Figure 12. |
| `cellchat_analysis_spatial.R` | Same CellChat configuration on the 1,460-gene label-transfer panel (81/687 CellChatDB genes, 16 L–R pairs). **Superseded**; kept only for the panel-vs-full comparison (Table 16). |
| `cellchat_analysis_nonspatial.R` | Non-spatial configuration used in an earlier version. **Not used in the manuscript**; kept for transparency. |
| `export_data_for_cellchat.py` | Writes the panel exports (`v1_kidney_*`) and `data/rctd_io/`. |


## Independent datasets

| Script | What it does |
|---|---|
| `datasetB_lean.py` | Independent murine IRI cohort GSE269622 (sham/4h/12h/day2): label transfer, compartment graph, simulation with unchanged Table 1 parameters; memory-lean, one sample per process |
| `datasetC_pipeline.py` | Human kidney transplant rejection GSE304669 (control/AMR/TCMR/chronic): Susztak GSE211785 reference, radial zoning, unchanged murine parameters; structural-quantities-only comparison |
| `temporal_validation_FINAL.py` | Pre-registered temporal validation against the GSE182939 IRI time course (five timepoints); reports the observed null result|
| `generality_second_tissue.py` | Second-tissue generality test on Visium mouse brain |

## Annotation robustness

| Script | What it does |
|---|---|
| `run_rctd.R` | RCTD (spacexr) annotation on the shared `data/rctd_io/` export |
| `run_c2l.py` | cell2location annotation on the same export |
| (agreement/simulation summaries) | Agreement metrics (fraction, ARI, NMI) and per-annotation simulation are computed from the label outputs; see manuscript annotation-robustness section |

## Suggested run order

1. `export_data_for_cellchat.py` (writes all shared exports)
2. `benchmark_synthetic.py` → `sobol_gsa.py` → `diagnose_model_first.py`
3. `cellchat_analysis_spatial.R` → `compare_cellchat_vs_model.py`
4. `run_rctd.R`, `run_c2l.py`
5. `datasetB_lean.py` (×4 samples), `datasetC_pipeline.py`
6. `temporal_validation_FINAL.py`, `generality_second_tissue.py`
