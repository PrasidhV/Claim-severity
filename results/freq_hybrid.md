# Frequency: hybrid (GLM backbone + GBM correction) vs pure GBM

Data: **real freMTPL2 (public)**; GBM backend `lightgbm`; nested 5-fold CV, mean Poisson deviance per policy (lower is better).

| model      |   mean_deviance |   deviance_explained_pooled |   share_of_gbm_gap_closed |   gain_vs_banded |   sd_of_gain |
|:-----------|----------------:|----------------------------:|--------------------------:|-----------------:|-------------:|
| constant   |         0.33028 |                     0.00000 |                 nan       |        nan       |    nan       |
| glm_banded |         0.31319 |                     0.05175 |                   0.00000 |        nan       |    nan       |
| glm_star   |         0.30948 |                     0.06296 |                   0.31449 |          0.00370 |      0.00020 |
| hybrid     |         0.30173 |                     0.08644 |                   0.97355 |          0.01146 |      0.00039 |
| gbm        |         0.30142 |                     0.08739 |                   1.00000 |          0.01177 |      0.00035 |

- GBM correction on top of GLM*: gain +0.00776 per fold (SD 0.00047, better in 5/5 folds).
- Hybrid minus pure GBM: hybrid better by -0.00031 (SD 0.00017, better in 0/5 folds) -> **pure GBM is better than the hybrid** (rule: mean difference larger than its fold-to-fold SD).

## How hard does the GBM push the GLM? (held-out folds, averaged)

- typical size of the correction: mean |log factor| = 0.233 (~23.3%)
- share of policies whose GLM prediction is moved by more than 10%: 70.9%
- correction factor quantiles: 5% = 0.583, median = 0.952, 95% = 1.579

A small, tightly-centred correction means the GLM already carries the structure and the GBM is a monitored top-up; a wide one means the GBM is effectively the model.

Caveat: brand x vehicle-age band was chosen using GBM behaviour on the validation split (see `freq_glm_from_gbm.md`), so GLM* and the hybrid have a small, documented advantage over a fully blind pipeline.
