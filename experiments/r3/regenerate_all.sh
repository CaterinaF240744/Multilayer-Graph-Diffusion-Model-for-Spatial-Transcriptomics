#!/usr/bin/env bash
# regenerate_all.sh — R3: regenerate every number in the manuscript from one
# commit of the canonical model (src/model_core.py), so that no table or figure
# mixes runs from different code versions (Reviewer 2, points 1-2).
#
# Usage (repository root, data in data/ as described in data/data_README.md):
#   bash experiments/r3/regenerate_all.sh            # Python analyses
#   WITH_R=1 bash experiments/r3/regenerate_all.sh   # also RCTD / CellChat steps
#
# Everything is logged under results/r3/<commit>/ ; the commit hash and
# package versions are written next to the outputs. Refuses to run on a dirty
# working tree, so every output can be traced to a commit.
set -euo pipefail
cd "$(dirname "$0")/../.."

if [ -n "$(git status --porcelain -- src experiments)" ]; then
  echo "Uncommitted changes in src/ or experiments/: commit first." >&2
  exit 1
fi
REV=$(git rev-parse --short HEAD)
OUT=results/r3/$REV
mkdir -p "$OUT/logs"
git rev-parse HEAD > "$OUT/COMMIT"
python -m pip freeze > "$OUT/pip_freeze.txt"
export MPLBACKEND=Agg PYTHONWARNINGS="ignore::DeprecationWarning"

step () {  # step <name> <command...>
  local name=$1; shift
  echo "==> $name"
  ( "$@" ) > "$OUT/logs/$name.log" 2>&1 || { echo "FAILED: $name (see $OUT/logs/$name.log)"; exit 1; }
}

step 00_tests         python -m pytest -q tests/
step 00_conservation  python experiments/r3/demo_conservation.py

# Sections 6-7: full case study, Tables 1-12, Figs 3-11 (all 12 steps)
step 01_main          python experiments/run_main.py
# Model context for Sec. 8 (Sobol, annotation, L-R): stops after step 9
step 01b_context      env QUICK_EXPORT=1 RUN_TAG=NN python experiments/run_main.py
step 02_controls      python experiments/run_controls.py          # Sec. 7.4, ex-Fig. 17
step 05_point3        python experiments/r3/point3_transcompartment.py   # Table 5, pass-through test
step 03_sensitivity   python experiments/run_sensitivity.py
step 04_zoning        env ZONING_NO_SLC12A3=1 QUICK_EXPORT=1 RUN_TAG=no_slc12a3 python experiments/run_main.py

# Section 8
step 10_benchmark     python experiments/rebuttal/benchmark_synthetic.py --out-dir "$OUT/benchmark" --n-jobs 4
step 11_sobol         python experiments/rebuttal/sobol_gsa.py --n-jobs 4 --out-dir "$OUT/gsa"
for s in sham hour4 hour12 day2; do
  step 12_gse269622_$s python experiments/rebuttal/datasetB_lean.py "$s"
done
step 13_temporal      python experiments/rebuttal/temporal_validation_FINAL.py
step 14_generality    python experiments/rebuttal/generality_second_tissue.py

if [ "${WITH_R:-0}" = "1" ]; then
  step 20_annot_export  python experiments/rebuttal/export_annotation_inputs.py
  step 21_rctd_mouse    Rscript experiments/rebuttal/run_rctd_mouse.R
  step 22_c2l           python experiments/rebuttal/run_c2l.py
  step 23_annot_prepare python experiments/rebuttal/compare_annotations.py prepare
  for t in rctd c2l consensus; do
    step 24_annot_$t env QUICK_EXPORT=1 RUN_TAG=$t \
      ANNOTATION_LABELS="$PWD/results/annotation/labels_for_sim_$t.csv" python experiments/run_main.py
  done
  step 25_annot_summary python experiments/rebuttal/compare_annotations.py summarize   # single source for Table 13
  step 30_lr_compare    python experiments/rebuttal/lr_model_comparison.py
fi

echo "All outputs for commit $REV are in $OUT (tables: results/manuscript_tables/)."
