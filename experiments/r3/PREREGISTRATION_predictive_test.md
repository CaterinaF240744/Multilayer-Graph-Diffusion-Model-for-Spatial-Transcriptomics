# Pre-specified predictive test on GSE269622 (R3, Reviewer 2 point 3)

This plan was written and committed **before** the analysis was run on the
data. The commit date and hash of this file are the record of
pre-specification. Any deviation from this plan must be reported as such in
the manuscript.

## Question

Does the model, with the parameters of the V1 Mouse Kidney case study
(Table 1, no refitting), predict **which kidney compartments show the
strongest injury response** in an independent murine IRI Visium cohort
(GSE269622: sham, 4 h, 12 h, 2 days)?

## Data and preprocessing (fixed)

- Samples: `sham`, `hour4`, `hour12`, `day2`; same Space Ranger outputs,
  reference (GSE107585, 20,000-cell subsample, seed 0), kNN label transfer
  and `build_compartment2` as in `experiments/rebuttal/datasetB_lean.py`.
- Compartments: macro-type × anatomical zone, as in the case study.
- Compartments with fewer than **5 spots** in either of the two samples being
  compared are excluded.

## Model prediction (fixed)

- Single-layer compartment model of `src/model_core.py` (Eqs. 2–5),
  parameters from `build_default_params_SIR` (Table 1 with zone multipliers),
  seeds = boundary / outer-medulla compartments (case-study rule),
  p^D(0) = 0.05, t_end = 60, threshold 0.10.
- **Primary predictor:** activation time `t_act` (first time p^D ≥ 0.10)
  simulated on the **sham** graph, i.e. on uninjured tissue structure only.
  Earlier activation = higher predicted vulnerability; compartments that
  never activate get t_act = t_end. The model sees no injured-sample data.
- Secondary predictor: `t_act` simulated on each injured sample's own graph.

## Observed outcome (fixed)

- Injury score per spot: mean log1p-normalised expression of the injury
  markers already defined in R2 (`INJURY_MARKERS` in `datasetB_lean.py`):
  **Havcr1, Lcn2, Vcam1, Spp1** (markers absent from a sample are dropped and
  reported).
- Compartment injury response: mean spot score in the injured sample **minus**
  mean spot score of the same compartment in sham (Δ vs sham), which removes
  cell-type baseline expression.

## Hypotheses and tests

- **H1 (primary):** at **12 h**, Spearman ρ between predicted vulnerability
  (−t_act, sham graph) and Δ injury is positive. One-sided permutation test,
  10,000 permutations, seed 0, α = 0.05.
- **H2 (secondary):** the same at 4 h and 2 days. Holm correction across the
  two secondary tests.
- **H3 (key control, model-specific value):** the association in H1 persists
  **within macro-types**: Spearman of rank residuals after removing
  macro-type means, with permutations stratified by macro-type. This tests
  the spatial/graph component of the model, because macro-type susceptibility
  is a literature prior that would predict PT injury on its own.
- **Baseline reported alongside H1:** ρ of the parameter-only predictor
  R_diff = β/ρ (no graph, no dynamics).
- Secondary predictor (own-graph t_act) is reported for all time points.

## Interpretation (fixed in advance)

| H1 | H3 | Conclusion that will be written |
|----|----|----|
| significant | significant | The model predicts compartment-level injury beyond its macro-type priors. |
| significant | not significant | The predictive ordering is explained by macro-type susceptibility (literature priors), not by spatial propagation. |
| not significant | – | No evidence of predictive validity on GSE269622. |

Power note: with n ≈ 10–14 compartments, a one-sided Spearman test at
α = 0.05 detects only |ρ| ≳ 0.5. A non-significant result is therefore
weak evidence and will be reported as inconclusive, not as evidence against
the model.

## Outputs

`results/r3/predictive/`: per-sample compartment tables, `summary.json`
(all ρ, p, n, excluded compartments, markers used) and a scatter figure.
