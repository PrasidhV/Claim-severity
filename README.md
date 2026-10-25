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
| 13 | Python | `src/13_freq_gbm_additive.py` | GBM complexity ladder (stumps = additive, up to 32 leaves) to show whether the GBM's edge is interactions or main-effect shape |
| 14 | Python | `src/14_freq_interactions.py` | Which interactions the frequency GBM relies on: Friedman H-statistic on the log-rate scale (validated on a planted-interaction test) plus SHAP interaction values, with pattern heatmaps for the top pairs; validation split only |
| 15 | Python | `src/15_freq_glm_from_gbm.py` | Adds the GBM-identified interactions (brand x vehicle age, region x density, region x driver age) to the GLM on the single split and measures how much of the gap they close, with parameter counts and bootstrap CIs; an upper-bound diagnostic, not a nested CV |
| 16 | Python | `src/16_freq_hybrid.py` | Hybrid frequency model: GLM+ with brand x vehicle-age as the backbone, plus a GBM fitted on the ratio claims / GLM-expected-claims (same likelihood as an offset), compared with the pure GBM in nested CV; reports how hard the GBM correction pushes the GLM |
| 17 | Python | `src/17_freq_hybrid_analysis.py` | Accuracy as a function of a cap on the GBM's correction (how much of the lift survives a +/-5..50% limit), and a shallow-tree segment profile of where the GLM is wrong |
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
Numbers below are from the author's run on the real freMTPL2 data (deviance explained is relative to predicting one constant for everyone; severity and frequency use different deviances, so compare models within a table, not across). Full tables and plots are in `results/`; rerun the pipeline to reproduce them.

**Severity** (average cost per claim; 24,944 policies with claims; `results/cv_results.md`, 5-fold CV)

| Model | Beats the constant? |
|---|---|
| Full banded GLM | No: worse than the constant in 4 of 5 folds |
| GLM with F-test factor selection (nested in folds) | No: also worse |
| GLM on driver age only | Small gain (about 0.4% of deviance), not distinguishable from noise |
| GBM | Borderline (about 0.5% of deviance, 4 of 5 folds); discounted for tuning on overlapping data |

**Frequency** (claims per exposure-year; 678,013 policies; `results/freq_cv_results.md`, `results/freq_glm_plus_results.md`, `results/freq_gbm_ladder.md`)

| Model | Deviance explained (5-fold CV) |
|---|---|
| Banded GLM (R) | about 5.2% |
| GLM+ (splines and two screened interactions) | about 5.6% |
| GBM (LightGBM, Poisson) | about 8.7% |

The GBM beats the banded GLM in all five folds by roughly 0.0118 mean Poisson deviance, against a fold-to-fold SD of well under 0.001. The better-specified GLM+ recovers only about 12-13% of that gap. A complexity ladder (GBM with 2, 3, 4, 8, 16, 32 leaves) shows that a purely additive GBM (no interactions) performs about the same as GLM+, while allowing one interaction per tree closes most of the remaining gap. So the GBM's advantage comes from interactions, not from smoother main effects.

Which interactions? `results/freq_interactions.md` (Friedman H-statistic on the log-rate scale, validation split only) ranks vehicle age x brand first by a wide margin: that interaction is several times larger than brand's own main effect, i.e. brand matters mainly through how its risk changes with vehicle age. Region interacts moderately with many variables. Adding vehicle age x brand to the GLM on the single test split (`results/freq_glm_from_gbm.md`, GBM-guided, so an upper bound) lifts the share of the gap closed from about 13% to about 32% for roughly 50 extra parameters, with a gain whose bootstrap CI is clearly above zero. The region interactions did not generalise, and the other terms added almost nothing, so about two thirds of the GBM's advantage remains unexplained by any hand-built term tried.

## Which model I'd trust in practice
**Severity: neither GBM nor full GLM.** On this data the rating factors carry almost no usable signal for cost per claim. The full GLM overfits (about 60 parameters, worse than a constant in most folds), and its apparent significance came from an F-test that used the wrong dispersion (see Methodological note). I would use a constant severity, or at most a driver-age adjustment (the one factor selected in every fold), and treat any GBM gain as unproven. The honest finding is that severity is mostly noise here.

**Frequency: the GBM, with conditions.** It is clearly and consistently better than the GLM, and giving the GLM splines and screened interactions did not close the gap, so "the GLM was just crude" is not the explanation. Since severity barely varies by risk, pure premium differences across policies are driven almost entirely by frequency, so this choice largely decides the pricing model. I would deploy it only with:
1. **SHAP-based reason codes** for every score, and SHAP interaction values reviewed to see *which* interactions it relies on (the ladder says how much, not which).
2. **The GLM kept as the benchmark**, with the GBM monitored against it (decile calibration and double lift on new data), because a GLM can be filed, audited and explained to a regulator more easily.
3. **Out-of-time validation before use.** This dataset has no dates, so stability over time is untested. A random split flatters any flexible model.
4. **A check that the gain is broad**, not concentrated in thin cells (rare region x brand combinations), where a GBM can extrapolate badly.

If governance rules out a GBM, the fallback is the GLM+ with a vehicle age x brand interaction (GLM+ beats the banded GLM in every fold, and the brand x age term gains out of sample), accepting that it still leaves roughly two thirds of the GBM's lift unused. A middle path worth testing is a GBM fitted on top of the GLM's predictions as an offset, so the GLM carries the auditable structure and the GBM only learns the residual.

## Where this could be wrong
- Single hash split plus 5-fold CV on one dataset; no time dimension.
- The GBM's tuning grid was small and its chosen configuration sat at the edge of the grid, so the GBM's lead is probably understated.
- The brand x vehicle-age term and the other GBM-identified interactions were chosen using the GBM's behaviour on the validation split and scored once on the test split; this is not a nested CV, so treat that result as an upper bound for a hand-built GLM.
- GLM+ interaction candidates in the nested CV were limited to bilinear products of six numeric variables; a larger, hand-built GLM could narrow the gap further (the ladder suggests many moderate interactions, so it would need many terms).
- Severity is capped at 100,000 per claim and the cap choice was not tested for sensitivity.

## Limitations
- Frequency and severity are modelled and evaluated separately; combining them into pure premium and scoring that on actual loss cost is not done here.
- Random hash split, not out-of-time, so it flatters both models relative to deployment.
- SQLite rather than Postgres; the SQL is standard (window functions, CTEs) and ports directly.
- No monotonic constraints, fairness or proxy-discrimination checks, or large hyper-parameter search on the GBM.

## Status of this repo
The R GLM, the LightGBM model and the SHAP step have been run on the real freMTPL2 data. The Python GLM
cross-check reproduces the R GLM's test deviance (1.3333), which validates both implementations. Current results are in
`results/results.md` and `results/cv_results.md`. `data/claims.db` and `data/raw/` are not committed: rebuild them with
`src/00_get_data.py` and `src/01_build_db.py`.
