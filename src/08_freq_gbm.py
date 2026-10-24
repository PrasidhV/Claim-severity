"""Gradient boosting FREQUENCY model (Poisson objective, log link).

Exposure is handled the standard way: model the claim RATE (claims / exposure) with sample weight = exposure.
That is mathematically identical to Poisson on counts with a log-exposure offset. Predicted counts = rate x exposure.
LightGBM if installed, else scikit-learn HistGradientBoosting (loud fallback). Train on 'train', early-stop and
select on 'valid', report on untouched 'test'.
"""
import itertools
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from common import FEATURES, OUT, SEED, load_frequency_frame, poisson_deviance

warnings.filterwarnings("ignore", message=".*eval_set.*deprecated.*")
try:
    import lightgbm as lgb
    BACKEND = "lightgbm"
except ImportError:
    from sklearn.ensemble import HistGradientBoostingRegressor
    BACKEND = "sklearn-HistGradientBoosting (FALLBACK: lightgbm not installed)"

GRID = [dict(leaves=l, mcs=m) for l, m in itertools.product([8, 16, 32], [200, 1000])]


def rate(d):
    return d.claim_nb / d.exposure


def fit_one(p, tr, va):
    X, y, w = tr[FEATURES], rate(tr), tr.exposure
    Xv, yv, wv = va[FEATURES], rate(va), va.exposure
    if BACKEND == "lightgbm":
        m = lgb.LGBMRegressor(objective="poisson", n_estimators=3000, learning_rate=0.03,
                              num_leaves=p["leaves"], min_child_samples=p["mcs"],
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                              reg_lambda=5.0, random_state=SEED, verbose=-1)
        m.fit(X, y, sample_weight=w, eval_set=[(Xv, yv)], eval_sample_weight=[wv],
              eval_metric="poisson", callbacks=[lgb.early_stopping(100, verbose=False)])
        return m, int(m.best_iteration_)
    kw = dict(loss="poisson", learning_rate=0.05, max_leaf_nodes=p["leaves"], min_samples_leaf=p["mcs"],
              l2_regularization=5.0, categorical_features="from_dtype", random_state=SEED)
    probe = HistGradientBoostingRegressor(max_iter=800, **kw).fit(X, y, sample_weight=w)
    curve = [poisson_deviance(va.claim_nb, r * va.exposure) for r in probe.staged_predict(Xv)]
    best = int(np.argmin(curve)) + 1
    return HistGradientBoostingRegressor(max_iter=best, **kw).fit(X, y, sample_weight=w), best


def predict_counts(model, d):
    return model.predict(d[FEATURES]) * d.exposure.values


if __name__ == "__main__":
    df = load_frequency_frame()
    tr, va, te = (df[df.split == s] for s in ("train", "valid", "test"))
    print(f"backend: {BACKEND}")
    best = None
    for p in GRID:
        m, n_iter = fit_one(p, tr, va)
        dev = poisson_deviance(va.claim_nb, predict_counts(m, va))
        print(f"  leaves={p['leaves']:>2} min_leaf={p['mcs']:>4} trees={n_iter:>4}  valid Poisson deviance={dev:.5f}")
        if best is None or dev < best[0]:
            best = (dev, p, n_iter, m)
    dev_v, p, n_iter, model = best
    pred = predict_counts(model, te)
    print(f"chosen {p}, {n_iter} trees | GBM test mean Poisson deviance: {poisson_deviance(te.claim_nb, pred):.5f}")
    OUT.mkdir(exist_ok=True)
    pd.DataFrame(dict(idpol=te.idpol, y=te.claim_nb, exposure=te.exposure, pred=pred)
                 ).to_csv(OUT / "freq_gbm_test_preds.csv", index=False)
    joblib.dump(model, OUT / "freq_gbm_model.joblib")
    (OUT / "freq_gbm_meta.json").write_text(json.dumps(dict(backend=BACKEND, params=p, n_trees=n_iter,
                                                            valid_deviance=dev_v)))
