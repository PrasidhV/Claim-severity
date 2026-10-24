"""Frequency: 5-fold cross-validation with model selection INSIDE each fold (mirror of 06_cv.py).
Models: constant rate, full Poisson GLM, F-test-selected GLM (Pearson dispersion, p < 0.10), GBM (Poisson).
Folds = idpol % 5 over all policies. Scores = mean Poisson deviance per policy on the held-out fold."""
import json
import warnings

import numpy as np
import pandas as pd

import poisson_glm as pg
from common import (CAT, FEATURES, GLM_CATS, GLM_NUM, OUT, RES, SEED, load_frequency_frame,
                    poisson_dev_terms, poisson_deviance)

warnings.filterwarnings("ignore", message=".*eval_set.*deprecated.*")
K, P_KEEP = 5, 0.10
try:
    import lightgbm as lgb
    BACKEND = "lightgbm"
except ImportError:
    from sklearn.ensemble import HistGradientBoostingRegressor
    BACKEND = "sklearn-HistGradientBoosting (FALLBACK: lightgbm not installed)"


def fit_gbm(tr, params, seed):
    rng = np.random.default_rng(seed)
    is_val = rng.random(len(tr)) < 0.20                  # validation slice from TRAINING folds only
    a, v = tr[~is_val], tr[is_val]
    ra, rv = a.claim_nb / a.exposure, v.claim_nb / v.exposure
    if BACKEND == "lightgbm":
        m = lgb.LGBMRegressor(objective="poisson", n_estimators=3000, learning_rate=0.03,
                              num_leaves=params["leaves"], min_child_samples=params["mcs"],
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                              reg_lambda=5.0, random_state=SEED, verbose=-1)
        m.fit(a[FEATURES], ra, sample_weight=a.exposure, eval_set=[(v[FEATURES], rv)],
              eval_sample_weight=[v.exposure], eval_metric="poisson",
              callbacks=[lgb.early_stopping(100, verbose=False)])
        return m
    kw = dict(loss="poisson", learning_rate=0.05, max_leaf_nodes=params["leaves"],
              min_samples_leaf=params["mcs"], l2_regularization=5.0,
              categorical_features="from_dtype", random_state=SEED)
    probe = HistGradientBoostingRegressor(max_iter=800, **kw).fit(a[FEATURES], ra, sample_weight=a.exposure)
    curve = [poisson_deviance(v.claim_nb, r * v.exposure) for r in probe.staged_predict(v[FEATURES])]
    return HistGradientBoostingRegressor(max_iter=int(np.argmin(curve)) + 1, **kw
                                         ).fit(a[FEATURES], ra, sample_weight=a.exposure)


if __name__ == "__main__":
    df = load_frequency_frame()
    for c in GLM_CATS:
        if c not in CAT:
            df[c] = df[c].astype(str)
    params = json.loads((OUT / "freq_gbm_meta.json").read_text())["params"]
    fold = (df.idpol % K).values
    ALL = GLM_CATS + GLM_NUM
    models = ["constant", "glm_full", "glm_selected", "gbm"]
    D = {m: [] for m in models}
    N, kept_terms = [], []
    print(f"backend: {BACKEND} | GBM params {params}")
    for k in range(K):
        tr, te = df[fold != k], df[fold == k]
        y, e = te.claim_nb.values.astype(float), te.exposure.values
        N.append(len(te))
        preds = {"constant": tr.claim_nb.sum() / tr.exposure.sum() * e}
        m, s = pg.fit_glm(tr, ALL)
        preds["glm_full"] = pg.predict_glm(m, s, ALL, te)
        pv = pg.term_tests(tr, ALL)
        keep = [t for t in ALL if pv[t] < P_KEEP]
        kept_terms.append(keep)
        if keep:
            m, s = pg.fit_glm(tr, keep)
            preds["glm_selected"] = pg.predict_glm(m, s, keep, te)
        else:
            preds["glm_selected"] = preds["constant"]
        preds["gbm"] = fit_gbm(tr, params, SEED + k).predict(te[FEATURES]) * e
        for n_ in models:
            D[n_].append(poisson_dev_terms(y, preds[n_]).sum())
        print(f"fold {k}: " + "  ".join(f"{n_}={D[n_][-1] / N[-1]:.5f}" for n_ in models) +
              f"  | kept: {keep}")

    N = np.array(N)
    dev = {n_: np.array(D[n_]) / N for n_ in models}
    pooled = {n_: np.sum(D[n_]) / N.sum() for n_ in models}
    rows = []
    for n_ in models:
        diff = dev["constant"] - dev[n_]
        rows.append(dict(model=n_, mean_deviance=dev[n_].mean(), sd_across_folds=dev[n_].std(ddof=1),
                         deviance_explained_pooled=1 - pooled[n_] / pooled["constant"],
                         mean_gain_vs_constant=diff.mean() if n_ != "constant" else np.nan,
                         sd_of_gain=diff.std(ddof=1) if n_ != "constant" else np.nan,
                         folds_beating_constant=int((diff > 0).sum()) if n_ != "constant" else np.nan))
    res = pd.DataFrame(rows)
    res["clearly_better_than_constant"] = ["n/a" if r.model == "constant" else
                                           ("yes" if r.mean_gain_vs_constant > r.sd_of_gain else "no")
                                           for r in res.itertuples()]
    res["folds_beating_constant"] = res["folds_beating_constant"].astype("Int64")
    RES.mkdir(exist_ok=True)
    res.to_csv(RES / "freq_cv_results.csv", index=False)
    kept_md = "\n".join(f"- fold {k}: {', '.join(t) if t else '(none: constant used)'}" for k, t in enumerate(kept_terms))
    md = ["# Frequency: 5-fold cross-validation (selection nested inside each fold)\n",
          f"GBM backend: `{BACKEND}`; folds = `idpol % {K}` over all {len(df):,} policies "
          f"({int(df.claim_nb.sum()):,} claims). Metric: mean Poisson deviance per policy, lower is better.\n",
          res.to_markdown(index=False, floatfmt=".5f"), "",
          "`mean_gain_vs_constant` = constant deviance minus model deviance, per fold (positive = better than a "
          f"single claims-per-year rate for everyone); `sd_of_gain` is its standard deviation over the {K} folds.\n",
          "**Reading rule:** a model whose mean gain over the constant is smaller than the fold-to-fold SD of that "
          "gain should not be considered better.\n",
          "## Terms kept by `glm_selected` (quasi-Poisson F-test, p < 0.10)\n", kept_md, "",
          "## Known mild optimism (read gains as upper bounds)\n",
          "- The GBM's `num_leaves` / `min_child_samples` were chosen earlier on a validation split that overlaps these folds.",
          "- `glm_selected` is the only model whose selection is fully nested inside each fold.\n"]
    (RES / "freq_cv_results.md").write_text("\n".join(md))
    print("\n" + "\n".join(md))
