# Frequency: GBM complexity ladder (test set, single split)

Data: **real freMTPL2 (public)**; backend `lightgbm`; mean Poisson deviance per policy, lower is better.

| model                                             |      trees |   test_deviance |
|:--------------------------------------------------|-----------:|----------------:|
| banded GLM                                        |  nan       |         0.31145 |
| GLM+ (python check)                               |  nan       |         0.30994 |
| GLM+ (R)                                          |  nan       |         0.30994 |
| GBM, 2 leaves: additive (stumps, no interactions) | 5991.00000 |         0.30953 |
| GBM, 3 leaves: up to 1 interaction per tree       | 5578.00000 |         0.30336 |
| GBM, 4 leaves: small trees                        | 5187.00000 |         0.30198 |
| GBM, 8 leaves: medium trees                       | 2515.00000 |         0.30097 |
| GBM, 16 leaves: larger trees                      | 1963.00000 |         0.30047 |
| GBM, 32 leaves: full GBM (as chosen earlier)      |  865.00000 |         0.30015 |

Reading it: the first GBM row is purely additive. If it is close to GLM+, the GBM's advantage is interactions; if it is already close to the full GBM, the advantage is main-effect shape that the GLM's splines miss. (Single split: differences of a few thousandths between neighbouring rows are within noise; see freq_results.md for bootstrap CIs.)
