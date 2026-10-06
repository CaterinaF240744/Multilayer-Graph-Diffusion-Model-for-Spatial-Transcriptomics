# Revision analyses (R2): scripts and commands

All commands are run from the repository root. Paths to input data are those
described in `data/data_README.md`. Each analysis writes to its own folder under
`results/`. Exact package versions are given in `environment/` (see the end of this file).

## 1. Case study (Sections 6–7)

| Manuscript item | Script | Command (environment `stpipeline`) |
|---|---|---|
| Tables 2, 4, 5, 6, pharmacology tables, sensitivity tables (OAT, Monte Carlo, Spearman, ranking stability), Section 7 (edge importance, ablation, seeding); all case-study figures | `src/main.py` + `src/export_tables.py` | `python experiments/run_main.py` → `results/manuscript_tables/` |
| Zoning sensitivity (without *Slc12a3*) | `src/main.py` | `ZONING_NO_SLC12A3=1 QUICK_EXPORT=1 RUN_TAG=no_slc12a3 python experiments/run_main.py` |
| Model context for the analyses below | `src/main.py` | `QUICK_EXPORT=1 RUN_TAG=NN python experiments/run_main.py` → `results/annotation_runs/NN/` |

Options of `src/main.py` (environment variables; without them the script runs the full case study):
- `QUICK_EXPORT=1 RUN_TAG=<tag>`: stops after the trans-compartmental metrics and writes `summary.json`, `labels.csv` and `model_context.pkl` to `results/annotation_runs/<tag>/`;
- `ANNOTATION_LABELS=<csv>`: replaces the spot labels (index barcode, column `cell_identity`) after label transfer;
- `ZONING_NO_SLC12A3=1`: zoning without *Slc12a3* in the outer-medulla markers.

## 2. Validation (Section 8)

| Section | Script(s) | Command |
|---|---|---|
| 8.1 Synthetic benchmark | `benchmark_synthetic.py` | `MPLBACKEND=Agg python experiments/rebuttal/benchmark_synthetic.py --out-dir results/benchmark --n-jobs 4` |
| 8.2 Global sensitivity (Sobol) | `sobol_gsa.py` (requires `results/annotation_runs/NN/model_context.pkl`) | `MPLBACKEND=Agg python experiments/rebuttal/sobol_gsa.py --n-jobs 4` |
| 8.3 Annotation robustness | `export_annotation_inputs.py` → `run_rctd_mouse.R` (env `rctd`) → `run_c2l.py` (env `c2l`) → `compare_annotations.py` | see block below |
| 8.4 GSE269622 | `datasetB_lean.py` | `for s in sham hour4 hour12 day2; do python experiments/rebuttal/datasetB_lean.py $s; done` |
| 8.5 Ligand–receptor comparison | `export_full_transcriptome.py` → `cellchat_full_transcriptome.R` → `commot_full_transcriptome.py` (env `commot`) → `lr_model_comparison.py`; figure: `fig_cellchat_circles.R` | see block below |
| 8.6 Slide-seq cerebellum | `generality_second_tissue.py` | as documented in the script |
| 8.7 Structural generality | `generality_second_tissue.py` | as documented in the script |
| 8.8 Temporal validation (GSE182939) | `temporal_validation_FINAL.py` | as documented in the script |
| 8.9 Applicability: human Visium | `prepare_human_reference.py` → `datasetC_canonical.py` (ingest) → `export_rctd_human.py` → `run_rctd_human.R` (env `rctd`) → `datasetC_canonical.py --annotation rctd` | see block below |
| 8.9 Applicability: seqFISH | `polonsky_seqfish_test.py` → `fig_seqfish_supplementary.py` | `MPLBACKEND=Agg python experiments/rebuttal/polonsky_seqfish_test.py && python experiments/rebuttal/fig_seqfish_supplementary.py` |

### Annotation robustness (8.3)
```bash
python experiments/rebuttal/export_annotation_inputs.py
~/miniforge3/envs/rctd/bin/Rscript experiments/rebuttal/run_rctd_mouse.R
conda activate c2l && python experiments/rebuttal/run_c2l.py && conda activate stpipeline
QUICK_EXPORT=1 RUN_TAG=NN python experiments/run_main.py
python experiments/rebuttal/compare_annotations.py prepare
for t in rctd c2l consensus; do
  T=$(echo $t | sed 's/rctd/RCTD/')
  QUICK_EXPORT=1 RUN_TAG=$T ANNOTATION_LABELS=$PWD/results/annotation/labels_for_sim_$t.csv python experiments/run_main.py
done
python experiments/rebuttal/compare_annotations.py summarize
```

### Ligand–receptor comparison (8.5)
```bash
python experiments/rebuttal/export_full_transcriptome.py
Rscript experiments/rebuttal/cellchat_full_transcriptome.R data/full_matrix_inputs results/cellchat_full_v2/
conda activate commot
python experiments/rebuttal/commot_full_transcriptome.py data/full_matrix_inputs results/commot_full_v2
conda activate stpipeline
MPLBACKEND=Agg python experiments/rebuttal/lr_model_comparison.py
Rscript experiments/rebuttal/fig_cellchat_circles.R
```
The ligand–receptor analyses use the macro-type labels of `data/cellchat_inputs/v1_kidney_meta.csv`
(exported by `export_data_for_cellchat.py`; 881/437/110 PT/DCT/TAL spots); the current pipeline run
assigns 878/437/112.

### Human Visium applicability (8.9)
```bash
python experiments/rebuttal/prepare_human_reference.py
python experiments/rebuttal/datasetC_canonical.py --dataset GSE304669
python experiments/rebuttal/datasetC_canonical.py --dataset GSE183456
python experiments/rebuttal/export_rctd_human.py
~/miniforge3/envs/rctd/bin/Rscript experiments/rebuttal/run_rctd_human.R
python experiments/rebuttal/datasetC_canonical.py --dataset GSE304669 --annotation rctd
python experiments/rebuttal/datasetC_canonical.py --dataset GSE183456 --annotation rctd
```

## 3. Environments

| Environment | Used for | Lock file |
|---|---|---|
| `stpipeline` (Python 3.11) | pipeline, benchmark, Sobol, human and seqFISH analyses | `environment/requirements_stpipeline.txt` |
| `commot` (Python 3.9, numpy < 2) | COMMOT | `environment/requirements_commot.txt` |
| `c2l` (Python 3.10, scipy 1.12) | cell2location | `environment/requirements_c2l.txt` |
| `rctd` (R 4.3.3, spacexr 2.2.1, bioconda) | RCTD | `environment/rctd_environment.yml` |
| system R 4.3 (CellChat 2.2.0.9001) | CellChat | `environment/sessionInfo_cellchat.txt` |
