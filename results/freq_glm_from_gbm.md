# Frequency: can GBM-identified interactions close the gap?  (single split, diagnostic)

Data: **real freMTPL2 (public)**. Mean Poisson deviance per policy on the test set (lower is better). `share_of_gap_closed` = (banded GLM - model) / (banded GLM - GBM).

| model                           |    params |   test_deviance |   share_of_gap_closed | gain_vs_GLM+ (95% CI)         |
|:--------------------------------|----------:|----------------:|----------------------:|:------------------------------|
| banded GLM                      | nan       |         0.31145 |               0.00000 | nan                           |
| GBM                             | nan       |         0.30000 |               1.00000 | nan                           |
| GLM+ (screened numeric pairs)   |  77.00000 |         0.30994 |               0.13201 | nan                           |
| GLM+ + brand x vehicle-age band | 127.00000 |         0.30782 |               0.31676 | +0.00212 (+0.00161, +0.00269) |
| GLM+ + region x log-density     |  98.00000 |         0.31004 |               0.12294 | -0.00010 (-0.00019, -0.00001) |
| GLM+ + region x driver-age band | 203.00000 |         0.30975 |               0.14844 | +0.00019 (-0.00006, +0.00044) |
| GLM+ + driver age x bonus-malus |  78.00000 |         0.30985 |               0.14030 | +0.00010 (+0.00003, +0.00017) |
| GLM+ + ALL four added           | 275.00000 |         0.30763 |               0.33353 | +0.00231 (+0.00174, +0.00293) |

Positive `gain_vs_GLM+` with a CI above zero means the added interaction helps on unseen data by more than sampling noise. The terms were chosen from the GBM's behaviour on the validation split, so this is an upper-bound diagnostic, not a nested-CV estimate. Parameter counts show the price paid: a hand-built GLM needs many extra terms.
