"""5-fold cross-validation with model selection done INSIDE each fold.

Question answered: does anything beat "predict the average severity for everyone" out of sample?

Models (all scored on the held-out fold, weighted mean Gamma deviance, weights = n_claims):
  constant      claim-weighted mean severity of the training folds
  glm_age_only  Gamma/log GLM on driv_age_band
  glm_full      the full banded GLM (same factors as r/02_glm.R)
  glm_selected  full GLM -> per-term F-tests (PEARSON dispersion) on the training folds ->
                keep terms with p < 0.10 (fall back to driv_age_band) -> refit.  Fully nested.
  gbm           LightGBM (Gamma objective), early-stopped on a 20% slice of the TRAINING folds.

The GLM is fitted by iteratively reweighted least squares in plain numpy (for Gamma + log link the
working weights are just the prior weights), so no statsmodels is needed. It matches R's glm().
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import f as fdist

from common import (CAT, FEATURES, GLM_CATS, GLM_NUM, OUT, RES, SEED, gamma_dev_terms,
                    load_model_frame)

K = 5
P_KEEP = 0.10

try:
    import lightgbm as lgb
    BACKEND = "lightgbm"
except ImportError:
    from sklearn.ensemble import HistGradientBoostingRegressor
    BACKEND = "sklearn-HistGradientBoosting (FALLBACK: lightgbm not installed)"


# ------------------------------------------------------------------ Gamma / log-link GLM (IRLS)
def make_spec(tr, terms):
    """Reference level = level with most claims (like r/02_glm.R); numerics are standardised."""
    spec = {}
    for t in terms:
        if t in GLM_CATS:
            claims = tr.groupby(tr[t].astype(str))["n_claims"].sum().sort_values(ascending=False)
            ref = claims.index[0]
            spec[t] = [ref] + sorted(l for l in claims.index if l != ref)
        else:
            spec[t] = (tr[t].mean(), tr[t].std() or 1.0)
    return spec


def design(df, terms, spec):
    """Intercept + one-hot (reference dropped) + standardised numerics. Unseen levels -> reference."""
    cols, names = [np.ones(len(df))], ["(Intercept)"]
    for t in terms:
        if t in GLM_CATS:
            v = df[t].astype(str).values
            for lvl in spec[t][1:]:
                cols.append((v == lvl).astype(float)); names.append(f"{t}{lvl}")
        else:
            m, s = spec[t]
            cols.append(((df[t].values - m) / s).astype(float)); names.append(t)
    return np.column_stack(cols), names


def fit_gamma_log(X, y, w, max_iter=100, tol=1e-10):
    beta = np.zeros(X.shape[1])
    beta[0] = np.log(np.average(y, weights=w))
    eta = X @ beta
    dev_old = np.inf
    sw = np.sqrt(w)
    for _ in range(max_iter):
        mu = np.exp(np.clip(eta, -30, 30))
        z = eta + (y - mu) / mu                      # working response; working weights = w
        beta = np.linalg.lstsq(X * sw[:, None], z * sw, rcond=None)[0]
        eta = X @ beta
        mu = np.exp(np.clip(eta, -30, 30))
        dev = gamma_dev_terms(y, mu, w).sum()
        if abs(dev_old - dev) < tol * (abs(dev) + 1e-12):
            break
        dev_old = dev
    rank = np.linalg.matrix_rank(X)
    df_resid = len(y) - rank
    pearson = (w * ((y - mu) / mu) ** 2).sum() / df_resid      # what R's summary() reports
    return dict(beta=beta, dev=dev, rank=rank, df_resid=df_resid, phi=pearson)


def fit_glm(tr, terms):
    spec = make_spec(tr, terms)
    X, _ = design(tr, terms, spec)
    return fit_gamma_log(X, tr.avg_severity.values, tr.n_claims.values), spec


def predict_glm(model, spec, terms, df):
    X, _ = design(df, terms, spec)
    return np.exp(np.clip(X @ model["beta"], -30, 30))


def term_tests(tr, terms):
    """Per-term F-test using the Pearson dispersion of the full model (see README, methodological note)."""
    full, _ = fit_glm(tr, terms)
    out = {}
    for t in terms:
        small, _ = fit_glm(tr, [u for u in terms if u != t])
        ddf = small["df_resid"] - full["df_resid"]
        F = ((small["dev"] - full["dev"]) / ddf) / full["phi"]
        out[t] = fdist.sf(F, ddf, full["df_resid"])
    return out


# ------------------------------------------------------------------ GBM
def fit_gbm(tr, params, seed):
    rng = np.random.default_rng(seed)
    is_val = rng.random(len(tr)) < 0.20                 # validation slice from TRAINING folds only
    a, v = tr[~is_val], tr[is_val]
    Xa, ya, wa = a[FEATURES], a.avg_severity, a.n_claims
    Xv, yv, wv = v[FEATURES], v.avg_severity, v.n_claims
    if BACKEND == "lightgbm":
        m = lgb.LGBMRegressor(objective="gamma", n_estimators=4000, learning_rate=0.02,
                              num_leaves=params["leaves"], min_child_samples=params["mcs"],
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                              reg_lambda=5.0, random_state=SEED, verbose=-1)
        m.fit(Xa, ya, sample_weight=wa, eval_set=[(Xv, yv)], eval_sample_weight=[wv],
              eval_metric="gamma_deviance", callbacks=[lgb.early_stopping(150, verbose=False)])
        return m
    kw = dict(loss="gamma", learning_rate=0.03, max_leaf_nodes=params["leaves"],
              min_samples_leaf=params["mcs"], l2_regularization=5.0,
              categorical_features="from_dtype", random_state=SEED)
    probe = HistGradientBoostingRegressor(max_iter=1500, **kw).fit(Xa, ya, sample_weight=wa)
    curve = [gamma_dev_terms(yv, mu, wv).sum() / wv.sum() for mu in probe.staged_predict(Xv)]
    return HistGradientBoostingRegressor(max_iter=int(np.argmin(curve)) + 1, **kw
                                         ).fit(Xa, ya, sample_weight=wa)


# ------------------------------------------------------------------ cross-validation
if __name__ == "__main__":
    df = load_model_frame()
    for c in CAT:                                       # one consistent categorical dtype everywhere
        df[c] = df[c].astype("category")
    for c in GLM_CATS:
        df[c] = df[c].astype(str) if c not in CAT else df[c]
    params = json.loads((OUT / "gbm_meta.json").read_text())["params"]
    fold = (df.idpol % K).values
    ALL_TERMS = GLM_CATS + GLM_NUM
    models = ["constant", "glm_age_only", "glm_full", "glm_selected", "gbm"]
    D = {m: [] for m in models}         # summed deviance contributions per fold
    kept_terms, W = [], []

    print(f"backend: {BACKEND} | GBM params {params}")
    for k in range(K):
        tr, te = df[fold != k], df[fold == k]
        y, w = te.avg_severity.values, te.n_claims.values
        W.append(w.sum())

        mu_c = np.average(tr.avg_severity, weights=tr.n_claims)
        preds = {"constant": np.full(len(te), mu_c)}

        m, s = fit_glm(tr, ["driv_age_band"])
        preds["glm_age_only"] = predict_glm(m, s, ["driv_age_band"], te)

        m, s = fit_glm(tr, ALL_TERMS)
        preds["glm_full"] = predict_glm(m, s, ALL_TERMS, te)

        pv = term_tests(tr, ALL_TERMS)
        keep = [t for t in ALL_TERMS if pv[t] < P_KEEP] or ["driv_age_band"]
        kept_terms.append(keep)
        m, s = fit_glm(tr, keep)
        preds["glm_selected"] = predict_glm(m, s, keep, te)

        preds["gbm"] = fit_gbm(tr, params, SEED + k).predict(te[FEATURES])

        for name in models:
            D[name].append(gamma_dev_terms(y, preds[name], w).sum())
        print(f"fold {k}: " + "  ".join(f"{n}={D[n][-1] / W[-1]:.4f}" for n in models) +
              f"  | kept: {keep}")

    W = np.array(W)
    dev = {n: np.array(D[n]) / W for n in models}                  # per-fold weighted mean deviance
    pooled = {n: np.sum(D[n]) / W.sum() for n in models}
    rows = []
    for n in models:
        diff = dev["constant"] - dev[n]                            # >0: model beats the constant
        rows.append(dict(
            model=n, mean_deviance=dev[n].mean(), sd_across_folds=dev[n].std(ddof=1),
            deviance_explained_pooled=1 - pooled[n] / pooled["constant"],
            mean_gain_vs_constant=diff.mean() if n != "constant" else np.nan,
            sd_of_gain=diff.std(ddof=1) if n != "constant" else np.nan,
            folds_beating_constant=int((diff > 0).sum()) if n != "constant" else np.nan))
    res = pd.DataFrame(rows)
    res["clearly_better_than_constant"] = [
        "n/a" if r.model == "constant" else
        ("yes" if r.mean_gain_vs_constant > r.sd_of_gain else "no") for r in res.itertuples()]
    RES.mkdir(exist_ok=True)
    res["folds_beating_constant"] = res["folds_beating_constant"].astype("Int64")
    res.to_csv(RES / "cv_results.csv", index=False)

    if abs(pooled["glm_full"] - 1.33) > 0.05:
        print(f"!! warning: pooled out-of-fold deviance of full GLM is {pooled['glm_full']:.4f}, "
              f"expected about 1.33")

    kept_md = "\n".join(f"- fold {k}: {', '.join(t) if t else '(none)'}" for k, t in enumerate(kept_terms))
    md = ["# 5-fold cross-validation (selection nested inside each fold)\n",
          f"GBM backend: `{BACKEND}`; folds = `idpol % {K}` over all {len(df):,} policies with claims "
          f"({int(df.n_claims.sum()):,} claims).  Lower deviance is better.\n",
          res.to_markdown(index=False, floatfmt=".4f"), "",
          "`mean_gain_vs_constant` = constant deviance minus model deviance, per fold (positive = better "
          "than predicting the average for everyone); `sd_of_gain` is its standard deviation across the "
          f"{K} folds.\n",
          "**Reading rule:** if a model's mean gain over the constant is smaller than the fold-to-fold SD of "
          "that gain, it should not be considered better. (`clearly_better_than_constant` applies exactly "
          "this rule; with 5 folds it is roughly a p < 0.1 bar.)\n",
          "## Terms kept by `glm_selected` (F-test, Pearson dispersion, p < 0.10)\n", kept_md, "",
          "## Known mild optimism (so read gains as upper bounds)\n",
          "- The GBM's `num_leaves` / `min_child_samples` were chosen earlier using the original validation "
          "split, which overlaps these folds.",
          "- `glm_age_only` was chosen after looking at full-sample results.",
          "- `glm_selected` is the only model whose selection is fully nested inside each fold.\n"]
    (RES / "cv_results.md").write_text("\n".join(md))
    print("\n" + "\n".join(md))
