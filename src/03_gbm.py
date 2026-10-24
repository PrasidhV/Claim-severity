"""Gradient boosting severity model (Gamma objective, log link) with a small tuned grid.

Uses LightGBM if installed; otherwise falls back to scikit-learn's HistGradientBoostingRegressor
(same idea, same Gamma loss) and says so loudly. Train on 'train', early-stop + select on 'valid',
report on untouched 'test'. Weights = number of claims, exactly as in the GLM.
"""
import itertools
import json

import joblib
import numpy as np
import pandas as pd

from common import FEATURES, OUT, SEED, gamma_deviance, load_model_frame

try:
    import lightgbm as lgb
    BACKEND = "lightgbm"
except ImportError:
    from sklearn.ensemble import HistGradientBoostingRegressor
    BACKEND = "sklearn-HistGradientBoosting (FALLBACK: lightgbm not installed)"

GRID = [dict(leaves=l, mcs=m) for l, m in itertools.product([4, 8, 16], [50, 200])]


def fit_one(p, tr, va):
    X, y, w = tr[FEATURES], tr.avg_severity, tr.n_claims
    Xv, yv, wv = va[FEATURES], va.avg_severity, va.n_claims
    if BACKEND == "lightgbm":
        m = lgb.LGBMRegressor(objective="gamma", n_estimators=4000, learning_rate=0.02,
                              num_leaves=p["leaves"], min_child_samples=p["mcs"],
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                              reg_lambda=5.0, random_state=SEED, verbose=-1)
        m.fit(X, y, sample_weight=w, eval_set=[(Xv, yv)], eval_sample_weight=[wv],
              eval_metric="gamma_deviance", callbacks=[lgb.early_stopping(150, verbose=False)])
        return m, int(m.best_iteration_)
    kw = dict(loss="gamma", learning_rate=0.03, max_leaf_nodes=p["leaves"],
              min_samples_leaf=p["mcs"], l2_regularization=5.0,
              categorical_features="from_dtype", random_state=SEED)
    probe = HistGradientBoostingRegressor(max_iter=1500, **kw).fit(X, y, sample_weight=w)
    curve = [gamma_deviance(yv, mu, wv) for mu in probe.staged_predict(Xv)]
    best = int(np.argmin(curve)) + 1
    return HistGradientBoostingRegressor(max_iter=best, **kw).fit(X, y, sample_weight=w), best


if __name__ == "__main__":
    df = load_model_frame()
    tr, va, te = (df[df.split == s] for s in ("train", "valid", "test"))
    print(f"backend: {BACKEND}")
    best = None
    for p in GRID:
        m, n_iter = fit_one(p, tr, va)
        dev = gamma_deviance(va.avg_severity, m.predict(va[FEATURES]), va.n_claims)
        print(f"  leaves={p['leaves']:>2} min_leaf={p['mcs']:>3} trees={n_iter:>4}  valid deviance={dev:.4f}")
        if best is None or dev < best[0]:
            best = (dev, p, n_iter, m)
    dev_v, p, n_iter, model = best
    pred = model.predict(te[FEATURES])
    print(f"chosen {p}, {n_iter} trees | GBM test weighted mean Gamma deviance: "
          f"{gamma_deviance(te.avg_severity, pred, te.n_claims):.4f}")
    OUT.mkdir(exist_ok=True)
    pd.DataFrame(dict(idpol=te.idpol, y=te.avg_severity, w=te.n_claims, pred=pred)
                 ).to_csv(OUT / "gbm_test_preds.csv", index=False)
    joblib.dump(model, OUT / "gbm_model.joblib")
    (OUT / "gbm_meta.json").write_text(json.dumps(dict(backend=BACKEND, params=p, n_trees=n_iter,
                                                       valid_deviance=dev_v)))
