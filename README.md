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
| 18 | Python | `src/18_freq_bm_levels.py` | Level-by-level bonus-malus calibration of the GLM and the GBM's correction (looks for a saw-tooth the smooth splines miss), then a nested CV of the GLM with one indicator per well-populated BM level; validated with a planted-spike control |
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
Numbers are from the author's run on the real freMTPL2 data (rounded; full tables and plots in `results/`, reproducible with `run_all.sh`). Deviance explained is relative to predicting one constant for everyone; severity and frequency use different deviances, so compare models within a table, not across.

### Severity (cost per claim; 24,944 policies with claims; `results/cv_results.md`, 5-fold CV)

| Model | Beats the constant? |
|---|---|
| Full banded GLM | No: worse than the constant in 4 of 5 folds |
| GLM with F-test factor selection (nested in folds) | No: also worse |
| GLM on driver age only | Small gain (about 0.4% of deviance), not distinguishable from noise |
| GBM | Borderline (about 0.5% of deviance, 4 of 5 folds); discounted for tuning on overlapping data |

### Frequency (claims per exposure-year; 678,013 policies; 5-fold CV)

| Model | Parameters (approx.) | Deviance explained | Share of banded-GLM-to-GBM gap closed |
|---|--:|--:|--:|
| Banded GLM (R) | 60 | 5.2% | 0% |
| GLM+ (splines + 2 Rao-screened interactions) | 77 | 5.6% | 13% |
| GLM* (GLM+ + vehicle age x brand) | 127 | 6.3% | 31% |
| GLM* + one indicator per bonus-malus level | 150 | about 7.0% | 50% |
| Hybrid (GLM* + GBM correction) | n/a | 8.6% | 97% |
| GBM (LightGBM, Poisson) | n/a | 8.7% | 100% |

### What the investigation found
1. **The GBM's lead is real but was overstated by a crude baseline.** It beats the banded GLM by about 0.0118 mean Poisson deviance in all five folds (fold SD well under 0.001). Giving the GLM smooth splines recovered little: an additive GBM (stumps, no interactions) performs about the same as GLM+. Allowing one interaction per tree closes most of the rest, so the gap is interactions and level-specific structure, not smoother curves (`freq_gbm_ladder.md`).
2. **Vehicle age x brand is a real interaction.** It is the strongest by far in the GBM (several times larger than brand's own main effect, `freq_interactions.md`), and adding it to the GLM gains out of sample with a bootstrap CI clearly above zero (`freq_glm_from_gbm.md`). The region interactions did not generalise.
3. **Bonus-malus has level-specific effects a smooth curve misses.** The segment analysis of the hybrid's corrections split almost entirely on specific bonus-malus values: levels 62-63 were under-predicted by the GLM by up to about 2.3x (older vehicles), while neighbouring levels 60-61 and 64 were over-predicted. That saw-tooth is plausibly the French malus step (a floor-50 driver with one at-fault claim lands near 62.5); this README does not verify that mechanism. One indicator per well-populated level (23 per fold, levels chosen on training folds) improves every fold and lifts the share of the gap closed from about 31% to about 50% (`freq_bm_levels.md`). The idea was found in held-out predictions from these same folds, so treat this as an upper bound.
4. **The GBM correction on top of a GLM is large, not a light top-up.** The hybrid matches the pure GBM to within 0.0003 (pure GBM better in all five folds), but the GBM moves about 70% of policies' GLM predictions by more than 10% (5th to 95th percentile factor roughly 0.58 to 1.58). Capping the correction costs accuracy steadily: a +/-20% cap keeps about two thirds of the lift (`freq_hybrid.md`, `freq_hybrid_analysis.md`). A depth-3 tree explains only about 11% of the correction's variance, so beyond the bonus-malus levels it is a diffuse web.

## Which model I'd trust in practice
**Severity: neither GBM nor full GLM.** The rating factors carry almost no usable signal for cost per claim here. The full GLM overfits (about 60 parameters, worse than a constant in most folds), and its apparent significance came from an F-test that used the wrong dispersion (see Methodological note). I would use a constant severity, or at most a driver-age adjustment (the one factor selected in every fold). The honest finding is that severity is mostly noise.

**Frequency: the GBM for accuracy, but the margin is narrower than it first looked.** The pure GBM is the most accurate model and I would deploy it only with:
1. **SHAP-based reason codes** for every score and the interaction analysis reviewed, not just importances.
2. **A well-built GLM kept as the monitored benchmark.** The GLM* with bonus-malus levels is a credible one: it closes about half the gap with roughly 150 interpretable parameters.
3. **Out-of-time validation before use.** The dataset has no dates, so stability over time is untested, and a random split flatters any flexible model.
4. **A check that the gain is broad,** not concentrated in thin cells (rare brand x age or region combinations), where a GBM can extrapolate badly.

**If governance requires a GLM,** use GLM* with bonus-malus level indicators. It leaves about half of the GBM's lift unused, and the earlier bonus-malus lesson suggests other integer variables (driver age, vehicle age) may hide similar level-specific effects that I have not tested; a GLM that treats them as levels might close more. The hybrid is not a good compromise: its correction is too large to count as a bounded adjustment, and it is slightly less accurate than the pure GBM.

**The main lesson:** a GBM beating a GLM is weak evidence for the GBM until the GLM has been given a fair chance. Here, about half of the apparent GBM advantage turned out to be two specification problems (a discrete variable treated as smooth, and one additive-only interaction), found by using the GBM as a diagnostic.

## Where this could be wrong
- Single hash split plus 5-fold CV on one dataset; no time dimension.
- The GBM's tuning grid was small and its chosen configuration sat at the edge of the grid, so the GBM's lead is probably understated.
- The vehicle age x brand term and the bonus-malus levels were found by looking at the GBM or at held-out predictions from these same folds. Terms and levels are fitted on training data only, but the choice of *which* structure to add was informed by the answer, so the GLM figures above are optimistic. There is no fresh data to confirm them on.
- The bonus-malus explanation (French malus step) is a hypothesis consistent with the pattern, not something verified here.
- GLM+ interaction candidates in the nested CV were limited to bilinear products of six numeric variables.
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
