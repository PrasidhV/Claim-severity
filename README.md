# Claim frequency and severity: actuarial GLM vs gradient boosting (SQL + R + Python)

**Question:** for French motor third-party liability claims, can a gradient-boosted model predict
*claim severity* (average cost per claim) and *claim frequency* (claims per exposure-year) better than the GLM an
actuary would build, and which one would I trust to price with? Pure premium is frequency x severity, so both are modelled the same way and compared on the same terms.

**Data:** freMTPL2freq + freMTPL2sev, public (OpenML 41214 / 41215), ~678k policies, ~25k with claims.

| Step | Tool | File | What it does |
|---|---|---|---|
| 1 | **SQL** (SQLite) | `sql/01_schema.sql`, `02_clean.sql`, `03_checks.sql` | Load raw tables untouched; type, trim, validate, de-duplicate, cap large losses, band rating factors, assign a deterministic train/valid/test split (`idpol % 5`); emit a data-quality report |
| 2 | **R** | `r/02_glm.R` | Gamma GLM, log link, weights = claim count, banded factors; relativities with CIs; per-term F-tests using the Pearson dispersion (see Methodological note) |
| 3 | **Python** | `src/03_gbm.py` | LightGBM, Gamma objective, small grid, early stopping on the validation split |
| 4 | Python | `src/04_compare.py` | Test deviance and bias for GLM, GBM and a **constant-prediction benchmark**; paired-bootstrap CIs on the differences; calibration and double-lift charts |
| 5 | Python | `src/05_explain.py` | SHAP importance and beeswarm; GBM effects overlaid on GLM relativities with confidence intervals |
| 7 | **SQL** + **R** | `sql/04_frequency.sql`, `r/07_freq_glm.R` | Frequency table (all policies, claim count, exposure); Poisson GLM with a log-exposure offset, banded factors, quasi-Poisson F-tests and CIs (Pearson dispersion) |
| 7b | Python | `src/07b_freq_glm_crosscheck.py` | Same Poisson GLM in numpy; validates R |
| 8 | Python | `src/08_freq_gbm.py` | LightGBM, Poisson objective on the claim rate with exposure weights |
| 9 | Python | `src/09_freq_compare.py` | Test Poisson deviance vs a constant claim rate, bootstrap CIs, calibration, double lift |
| 10 | Python | `src/10_freq_cv.py` | Frequency 5-fold CV with selection nested inside each fold |
| 11 | Python | `src/11_freq_explain.py` | SHAP and GLM-vs-GBM effects for frequency |
| 12 | **R** + Python | `r/12_freq_glm_plus.R`, `src/glm_plus.py`, `src/12_freq_glm_plus.py` | A better-specified frequency GLM (piecewise-linear splines, bonus-malus = 50 indicator, two interactions chosen by a Rao score test on the training data only) scored on the same nested folds, to test whether the GBM's edge is just a basic GLM |
| 6 | Python | `src/06_cv.py` | 5-fold cross-validation with model selection nested inside each fold: constant, driver-age-only GLM, full GLM, F-test-selected GLM, GBM |
| – | Python | `src/02b_glm_crosscheck.py` | Same GLM re-fit in scikit-learn. Test deviance should match R to ~3 decimals; if it doesn't, one of them has a bug |

## Run it
```bash
pip install -r requirements.txt     # + in R: install.packages(c("DBI","RSQLite"))
./run_all.sh                        # real data (needs internet for OpenML)
./run_all.sh --synthetic            # offline smoke test on fake data with the same schema
```
Outputs: `outputs/dq_report.txt`, **`results/results.md`** (single train/test split), **`results/cv_results.md`** (severity 5-fold cross-validation), **`results/freq_results.md`** and **`results/freq_cv_results.md`** (the same two views for frequency), `results/*.png`.
The frequency steps need the `model_frequency` table: `src/01_build_db.py` builds it, so rebuild the database once after pulling these files.
On Windows run the script in Git Bash, not `cmd`.

## Design decisions worth defending
- **Target and weights.** Per-policy mean claim cost, weighted by the number of claims. This is
  the correct Gamma likelihood for an average of *n* claims and is used identically by both models.
  Claim count comes from the severity table itself, so numerator and denominator always agree
  (freq `ClaimNb` and severity rows disagree for some policies; the DQ report counts them).
- **Large-loss cap at 100,000 per claim.** A few multi-million claims would otherwise dominate the fit.
  In production, capped severity is modelled here and the excess is loaded separately.
- **Same data, same split, same metric.** GLM fits on train+valid; GBM trains on train and early-stops
  on valid; both are scored once on test. Metric is weighted mean Gamma deviance, the loss both
  models actually optimise.
- **GLM gets banded factors, GBM gets raw numerics.** Banding is the actuary's job; letting the GBM
  find its own cut-points is the point of the comparison.
- **Don't trust a point estimate on ~5k claims.** Test claims are few, so `04_compare.py` reports a
  paired bootstrap CI for the deviance gap rather than just "model A < model B".
- **Frequency uses an exposure offset.** Claim counts are modelled as Poisson with log(exposure) as an offset (GBM: the
  claim rate with exposure as the sample weight, which is the same likelihood). ClaimNb is capped at 4. Severity is modelled
  only on policies with claims; frequency on all valid policies.
- **Give the GLM a fair fight.** A GBM beating a crude GLM is not evidence the GBM is better. `12_freq_glm_plus.py` gives the GLM the shapes and interactions a good actuary would try, selecting the interactions with a score test inside each fold so the GBM never influences the GLM, and reports the share of the GBM's advantage the GLM recovers.
- **Always include a constant benchmark.** "Predict the average severity for everyone" is the bar a model
  must clear. Severity is noisy enough that a model can look sophisticated and still lose to it.
- **Cross-validate with selection inside the folds.** `06_cv.py` re-runs the factor-selection step on each
  fold's training data, so choosing factors cannot leak information from the held-out fold.

## Methodological note: which F-test
For the Poisson frequency GLM the issue is related: R's `summary()` fixes the Poisson dispersion at 1, so its standard errors and p-values assume no overdispersion. `r/07_freq_glm.R` estimates the Pearson dispersion and uses it for standard errors and F-tests (quasi-Poisson inference).

For the severity GLM, R's `drop1(fit, test = "F")` scales the test by the *deviance-based* dispersion (residual deviance divided by residual degrees of freedom), whereas `summary()` reports the *Pearson* dispersion. For well-behaved data these are close. For heavy-tailed claim severities they can differ several-fold, which makes every F-statistic look far more impressive than it is and nearly every rating factor appear significant. `r/02_glm.R` therefore refits the model without each term and computes the F-test with the Pearson dispersion (output: `outputs/glm_term_tests.csv`). `src/06_cv.py` uses the same test when it selects factors inside each fold.

## Results
Run the pipeline and read `results/results.md` (one fixed train/test split) and `results/cv_results.md`
(5-fold cross-validation). The numbers are generated by running the code, not asserted in this README.

## Which model I'd trust in practice
The answer depends on the result, so here is the decision rule I'd apply, in order:

1. **Is the GBM better by more than noise?** If the bootstrap CI on the deviance difference includes
   zero, the models are statistically tied and the GLM wins by default: it is auditable, stable and
   explainable to a regulator. Severity on this dataset has weak signal (most predictive power sits in claim *frequency*),
   so a tie is a plausible outcome. That is a prior to check, not a finding.
2. **Where does it win?** Use the double-lift chart. A GBM that wins only on a thin slice (e.g. young
   drivers in powerful cars) has found an interaction the GLM is missing.
3. **Is the win real structure or noise?** Overlay GBM SHAP effects on GLM relativities
   (`glm_vs_gbm_effects.png`). If they agree on main effects, the GBM's edge is in interactions and shape,
   and the GLM is a sound base. If the GBM disagrees on a major factor, find out why before trusting either.
4. **Does either model beat the constant at all?** If neither beats the constant prediction in
   cross-validation by more than the fold-to-fold SD, severity should be modelled with a very small model and the
   modelling effort should go to claim frequency.
5. **Can it be deployed and defended?** Calibration (bias near 0 across deciles), stability on a
   later time period, no proxy-discrimination problems, and a rating structure that can be filed.
   Gradient boosting scores lower on all of these than a GLM unless it is constrained and monitored.

**Likely best answer if the GBM wins modestly:** price with the GLM, and *add the interaction or non-linearity
the GBM found* as an explicit term, or use the GBM prediction as an offset/challenger in a
monitored second stage. If the GBM wins decisively and calibrates well, use it, with SHAP-based
reason codes and the GLM kept as the benchmark.

## Limitations
- Frequency and severity are modelled and evaluated separately; combining them into a pure-premium model and scoring that on actual loss cost is not done here.
- Random hash split, not out-of-time, so it flatters both models relative to deployment.
- SQLite rather than Postgres; the SQL is standard (window functions, CTEs) and ports directly.
- No monotonic constraints, fairness checks or full hyper-parameter search on the GBM.

## Status of this repo
The R GLM, the LightGBM model and the SHAP step have been run on the real freMTPL2 data. The Python GLM
cross-check reproduces the R GLM's test deviance (1.3333), which validates both implementations. Current results are in
`results/results.md` and `results/cv_results.md`. `data/claims.db` and `data/raw/` are not committed: rebuild them with
`src/00_get_data.py` and `src/01_build_db.py`.
