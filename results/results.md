# Test-set results

Data: **real freMTPL2 (public)**  
Test policies with claims: 4,997; claims: 5,234  
GBM backend: `lightgbm`, 137 trees, params {'leaves': 4, 'mcs': 50}

| model          |   test_deviance |   deviance_explained |    bias |
|:---------------|----------------:|---------------------:|--------:|
| GLM (R)        |          1.3333 |              -0.0056 |  0.0039 |
| GBM (lightgbm) |          1.3247 |               0.0009 | -0.0090 |

Paired bootstrap, GLM minus GBM deviance: **+0.0086** (95% CI -0.0157 to +0.0339) -> the difference is within sampling noise (95% CI includes 0).

![calibration and lift](calibration_and_lift.png)
