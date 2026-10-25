# Frequency hybrid: what the GBM corrects

Data: **real freMTPL2 (public)**; held-out predictions from the 5 CV folds (678,013 policies). Mean Poisson deviance per policy.

## 1. How much accuracy survives a cap on the GBM's correction?

| max_correction     |   deviance |   share_of_gap_closed |   policies_hitting_cap |
|:-------------------|-----------:|----------------------:|-----------------------:|
| 0% (GLM* only)     |    0.30948 |               0.31452 |                1.00000 |
| +/-5%              |    0.30821 |               0.42302 |                0.84608 |
| +/-10%             |    0.30719 |               0.50961 |                0.70864 |
| +/-20%             |    0.30571 |               0.63560 |                0.49208 |
| +/-30%             |    0.30472 |               0.71972 |                0.34037 |
| +/-50%             |    0.30354 |               0.81938 |                0.16784 |
| none (full hybrid) |    0.30173 |               0.97355 |                0.00000 |

Reference: banded GLM 0.31319, GLM* 0.30948, full hybrid 0.30173, pure GBM 0.30142. `share_of_gap_closed` is relative to banded GLM -> pure GBM. Totals are rescaled per fold so capping does not change portfolio-level calibration.

## 2. Where is the GLM* wrong?

Correction factor quantiles (5/25/50/75/95%): 0.583, 0.796, 0.952, 1.129, 1.579. A depth-3 tree on simple policy attributes explains **11.2%** of the (weighted) variance in log-correction.

| segment                                            |   share_of_policies |   share_of_expected_claims |   mean_correction |   actual_vs_GLM_star |   actual_vs_hybrid |
|:---------------------------------------------------|--------------------:|---------------------------:|------------------:|---------------------:|-------------------:|
| 61.5 < bonus_malus <= 63.5 AND veh_age > 5.5       |               0.007 |                      0.010 |             1.928 |                2.303 |              1.114 |
| 61.5 < bonus_malus <= 63.5 AND veh_age <= 5.5      |               0.008 |                      0.012 |             1.268 |                1.358 |              0.984 |
| bonus_malus > 64.5 AND veh_age > 0.5               |               0.231 |                      0.269 |             0.965 |                1.026 |              1.007 |
| bonus_malus <= 59.5                                |               0.680 |                      0.605 |             0.956 |                0.999 |              1.000 |
| 59.5 < bonus_malus <= 61.5                         |               0.028 |                      0.032 |             0.763 |                0.748 |              0.930 |
| bonus_malus > 64.5 AND veh_age <= 0.5              |               0.018 |                      0.043 |             0.762 |                0.839 |              0.961 |
| 63.5 < bonus_malus <= 64.5 AND log_density > 6.16  |               0.014 |                      0.015 |             0.752 |                0.776 |              0.991 |
| 63.5 < bonus_malus <= 64.5 AND log_density <= 6.16 |               0.013 |                      0.013 |             0.651 |                0.566 |              0.833 |

`mean_correction` = average multiplicative factor the GBM applies in that segment; `actual_vs_GLM_star` and `actual_vs_hybrid` are actual claims / predicted claims (1.0 = calibrated). A segment with `actual_vs_GLM_star` well above 1 is one the GLM under-prices. A low tree R-squared means the correction is a diffuse web rather than a few nameable segments.
