#!/usr/bin/env bash
# End-to-end run. On Windows, run this in Git Bash (not cmd). Usage:  ./run_all.sh            (real freMTPL2 data, needs internet)
#                         ./run_all.sh --synthetic (offline pipeline test; results are NOT findings)
set -euo pipefail
cd "$(dirname "$0")"

python src/00_get_data.py "$@"
python src/01_build_db.py                 # SQL: load, clean, validate, split, DQ report
if command -v Rscript >/dev/null; then
  Rscript r/02_glm.R                      # baseline Gamma GLM (the "official" baseline)
else
  echo "!! Rscript not found - skipping R GLM; comparison will use the Python cross-check"
fi
python src/02b_glm_crosscheck.py          # same GLM in Python: validates R, and backs it up
python src/03_gbm.py                      # LightGBM, Gamma objective
python src/04_compare.py                  # deviance, bootstrap CI, calibration, double lift
python src/05_explain.py                  # SHAP + GLM-vs-GBM effect plots (with GLM confidence intervals)
python src/06_cv.py                       # 5-fold CV, selection nested in folds, vs a constant benchmark

# ---- frequency model (Poisson, exposure offset) ----
if command -v Rscript >/dev/null; then
  Rscript r/07_freq_glm.R                 # baseline Poisson GLM in R
fi
python src/07b_freq_glm_crosscheck.py     # same GLM in Python (validates R)
python src/08_freq_gbm.py                 # LightGBM, Poisson objective
python src/09_freq_compare.py             # deviance, bootstrap CIs, calibration, double lift, constant benchmark
python src/10_freq_cv.py                  # 5-fold CV, selection nested in folds
python src/11_freq_explain.py             # SHAP + GLM-vs-GBM effects
echo "Done. See results/results.md, cv_results.md, freq_results.md, freq_cv_results.md"
