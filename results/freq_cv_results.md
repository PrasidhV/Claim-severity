# Frequency: 5-fold cross-validation (selection nested inside each fold)

GBM backend: `lightgbm`; folds = `idpol % 5` over all 678,013 policies (36,056 claims). Metric: mean Poisson deviance per policy, lower is better.

| model        |   mean_deviance |   sd_across_folds |   deviance_explained_pooled |   mean_gain_vs_constant |   sd_of_gain | folds_beating_constant   | clearly_better_than_constant   |
|:-------------|----------------:|------------------:|----------------------------:|------------------------:|-------------:|:-------------------------|:-------------------------------|
| constant     |         0.33028 |           0.00134 |                     0.00000 |               nan       |    nan       | <NA>                     | n/a                            |
| glm_full     |         0.31319 |           0.00116 |                     0.05175 |                 0.01709 |      0.00058 | 5                        | yes                            |
| glm_selected |         0.31337 |           0.00110 |                     0.05121 |                 0.01691 |      0.00059 | 5                        | yes                            |
| gbm          |         0.30142 |           0.00105 |                     0.08739 |                 0.02886 |      0.00076 | 5                        | yes                            |

`mean_gain_vs_constant` = constant deviance minus model deviance, per fold (positive = better than a single claims-per-year rate for everyone); `sd_of_gain` is its standard deviation over the 5 folds.

**Reading rule:** a model whose mean gain over the constant is smaller than the fold-to-fold SD of that gain should not be considered better.

## Terms kept by `glm_selected` (quasi-Poisson F-test, p < 0.10)

- fold 0: driv_age_band, veh_age_band, bm_band, veh_gas, veh_power_g, veh_brand, region
- fold 1: driv_age_band, veh_age_band, bm_band, veh_gas, veh_power_g, veh_brand, region
- fold 2: driv_age_band, veh_age_band, bm_band, veh_gas, veh_power_g, veh_brand, region, log_density
- fold 3: driv_age_band, veh_age_band, bm_band, veh_gas, veh_power_g, veh_brand, region
- fold 4: driv_age_band, veh_age_band, bm_band, veh_gas, veh_power_g, veh_brand, region

## Known mild optimism (read gains as upper bounds)

- The GBM's `num_leaves` / `min_child_samples` were chosen earlier on a validation split that overlaps these folds.
- `glm_selected` is the only model whose selection is fully nested inside each fold.
