# Frequency: does a better-specified GLM close the gap to the GBM?

Data: **real freMTPL2 (public)**  
GBM backend: `lightgbm`. Metric: mean Poisson deviance per policy (lower is better).

## 5-fold cross-validation (interaction screening nested inside each fold)

| model          |   mean_deviance |   deviance_explained_pooled |   gain_vs_banded_glm |   sd_of_gain | folds_beating_banded   |   share_of_gbm_gap_closed |
|:---------------|----------------:|----------------------------:|---------------------:|-------------:|:-----------------------|--------------------------:|
| constant       |         0.33028 |                     0.00000 |            nan       |    nan       | <NA>                   |                 nan       |
| glm_banded     |         0.31319 |                     0.05175 |            nan       |    nan       | <NA>                   |                 nan       |
| glm_splines    |         0.31223 |                     0.05465 |              0.00096 |      0.00009 | 5                      |                   0.08132 |
| glm_splines_ix |         0.31171 |                     0.05621 |              0.00147 |      0.00014 | 5                      |                   0.12508 |
| gbm            |         0.30142 |                     0.08739 |              0.01177 |      0.00035 | 5                      |                   1.00000 |

`gain_vs_banded_glm` = banded-GLM deviance minus model deviance per fold (positive = better); `sd_of_gain` is its SD over the 5 folds. `share_of_gbm_gap_closed` = (banded GLM - model) / (banded GLM - GBM), using pooled deviance: 0 = no better than the banded GLM, 1 = as good as the GBM (left blank if the GBM does not clearly beat the banded GLM).

Interactions chosen per fold (top 2 by Rao score, training folds only):

- fold 0: bonus_malus_c x veh_age_c, veh_age_c x gas_diesel
- fold 1: veh_age_c x gas_diesel, bonus_malus_c x veh_age_c
- fold 2: bonus_malus_c x veh_age_c, veh_age_c x gas_diesel
- fold 3: bonus_malus_c x veh_age_c, veh_age_c x gas_diesel
- fold 4: bonus_malus_c x veh_age_c, veh_age_c x gas_diesel

## Single train/valid -> test split

| model      |   test_deviance |
|:-----------|----------------:|
| banded GLM |         0.31145 |
| GLM+ (R)   |         0.31035 |
| GBM        |         0.30000 |

- banded GLM minus GLM+ (R): **+0.00110** (95% CI +0.00065 to +0.00152) -> GLM+ (R) is better by more than sampling noise
- GLM+ (R) minus GBM: **+0.01035** (95% CI +0.00923 to +0.01157) -> GBM is better by more than sampling noise
