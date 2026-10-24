# 5-fold cross-validation (selection nested inside each fold)

GBM backend: `lightgbm`; folds = `idpol % 5` over all 24,944 policies with claims (26,444 claims).  Lower deviance is better.

| model        |   mean_deviance |   sd_across_folds |   deviance_explained_pooled |   mean_gain_vs_constant |   sd_of_gain | folds_beating_constant   | clearly_better_than_constant   |
|:-------------|----------------:|------------------:|----------------------------:|------------------------:|-------------:|:-------------------------|:-------------------------------|
| constant     |          1.3160 |            0.0295 |                      0.0000 |                nan      |     nan      | <NA>                     | n/a                            |
| glm_age_only |          1.3107 |            0.0278 |                      0.0041 |                  0.0053 |       0.0075 | 3                        | no                             |
| glm_full     |          1.3266 |            0.0363 |                     -0.0081 |                 -0.0106 |       0.0074 | 1                        | no                             |
| glm_selected |          1.3197 |            0.0332 |                     -0.0029 |                 -0.0037 |       0.0118 | 1                        | no                             |
| gbm          |          1.3093 |            0.0322 |                      0.0051 |                  0.0067 |       0.0055 | 4                        | yes                            |

`mean_gain_vs_constant` = constant deviance minus model deviance, per fold (positive = better than predicting the average for everyone); `sd_of_gain` is its standard deviation across the 5 folds.

**Reading rule:** if a model's mean gain over the constant is smaller than the fold-to-fold SD of that gain, it should not be considered better. (`clearly_better_than_constant` applies exactly this rule; with 5 folds it is roughly a p < 0.1 bar.)

## Terms kept by `glm_selected` (F-test, Pearson dispersion, p < 0.10)

- fold 0: driv_age_band
- fold 1: driv_age_band
- fold 2: driv_age_band, area, veh_power_g
- fold 3: driv_age_band, veh_power_g
- fold 4: driv_age_band, bm_band, region

## Known mild optimism (so read gains as upper bounds)

- The GBM's `num_leaves` / `min_child_samples` were chosen earlier using the original validation split, which overlaps these folds.
- `glm_age_only` was chosen after looking at full-sample results.
- `glm_selected` is the only model whose selection is fully nested inside each fold.
