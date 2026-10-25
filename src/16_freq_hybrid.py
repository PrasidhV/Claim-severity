"""Hybrid frequency model: GLM backbone + GBM correction ("boosting from a GLM").

Backbone  : GLM* = GLM+ (splines, 2 Rao-screened numeric interactions) + brand x vehicle-age band.
Correction: a GBM on the ratio  claims / GLM-expected-claims  with the GLM's expected count as the weight. That is the
            same Poisson likelihood as using the GLM prediction as an offset, so the GBM only learns what the GLM
            misses. Final prediction = GLM count x GBM correction factor (the factor is directly inspectable).
Compared in the nested 5-fold CV (same folds as 10/12): constant | banded GLM | GLM* | HYBRID | pure GBM.
Questions: does the hybrid match the pure GBM?  How hard does the correction push the GLM (governance)?
CAVEAT: brand x vehicle-age was chosen using GBM behaviour on the validation split (see 15), so GLM* and the hybrid get
a small, documented advantage over a fully blind pipeline.
Writes results/freq_hybrid.md and results/freq_hybrid_cv.csv. Slow on real data (~20 min)."""
import importlib.util
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import glm_plus as gp
import poisson_glm as pg
from common import (FEATURES, GLM_CATS, GLM_NUM, OUT, RES, SEED, is_synthetic, load_frequency_frame,
                    poisson_dev_terms, poisson_deviance)

warnings.filterwarnings("ignore", message=".*eval_set.*deprecated.*")


def _load(name, fname):
    s = importlib.util.spec_from_file_location(name, Path(__file__).with_name(fname))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


freq_cv, gfg = _load("freq_cv", "10_freq_cv.py"), _load("gfg", "15_freq_glm_from_gbm.py")
BACKEND = freq_cv.BACKEND
K = 5
TERM = ["brand x vehicle-age band"]


def fit_correction(a, v, ca, cv, params):
    """GBM on ratio = claims / GLM count, weighted by GLM count; early-stopped on the validation slice."""
    ra, rv = a.claim_nb.values / ca, v.claim_nb.values / cv
    if BACKEND == "lightgbm":
        lgb = freq_cv.lgb
        m = lgb.LGBMRegressor(objective="poisson", n_estimators=3000, learning_rate=0.03,
                              num_leaves=params["leaves"], min_child_samples=params["mcs"],
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0,
                              random_state=SEED, verbose=-1)
        m.fit(a[FEATURES], ra, sample_weight=ca, eval_set=[(v[FEATURES], rv)], eval_sample_weight=[cv],
              eval_metric="poisson", callbacks=[lgb.early_stopping(100, verbose=False)])
        return m
    H = freq_cv.HistGradientBoostingRegressor
    kw = dict(loss="poisson", learning_rate=0.05, max_leaf_nodes=params["leaves"], min_samples_leaf=params["mcs"],
              l2_regularization=5.0, categorical_features="from_dtype", random_state=SEED)
    probe = H(max_iter=400, **kw).fit(a[FEATURES], ra, sample_weight=ca)
    curve = [poisson_deviance(v.claim_nb, cv * r) for r in probe.staged_predict(v[FEATURES])]
    return H(max_iter=int(np.argmin(curve)) + 1, **kw).fit(a[FEATURES], ra, sample_weight=ca)


if __name__ == "__main__":
    raw = load_frequency_frame()                    # category dtypes for the GBM
    df = gp.prep(raw)                               # string dtypes for the GLM
    for c in ["veh_age_band", "driv_age_band", "bm_band"] + GLM_CATS:
        df[c] = df[c].astype(str)
    params = json.loads((OUT / "freq_gbm_meta.json").read_text())["params"]
    fold = (df.idpol % K).values
    ALL = GLM_CATS + GLM_NUM
    models = ["constant", "glm_banded", "glm_star", "hybrid", "gbm"]
    D, N, corr_stats = {m: [] for m in models}, [], []
    oof = []
    print(f"backend: {BACKEND} | GBM params {params}")
    for k in range(K):
        tr, te = df[fold != k], df[fold == k]
        rtr, rte = raw[fold != k], raw[fold == k]
        y, e = te.claim_nb.values.astype(float), te.exposure.values
        N.append(len(te))

        m, s = pg.fit_glm(tr, ALL)
        banded = pg.predict_glm(m, s, ALL, te)
        plus = gp.fit_plus(tr, k=2)
        mod, X_of = gfg.fit_with(tr, plus["pairs"], plus["spec"], TERM)     # GLM*
        glm_tr = np.exp(np.clip(X_of(tr) @ mod["beta"], -30, 30)) * tr.exposure.values   # in-sample counts
        glm_te = np.exp(np.clip(X_of(te) @ mod["beta"], -30, 30)) * e

        rng = np.random.default_rng(SEED + k)
        is_val = rng.random(len(tr)) < 0.20                                  # inner validation slice (training folds only)
        corr_model = fit_correction(rtr[~is_val], rtr[is_val], glm_tr[~is_val], glm_tr[is_val], params)
        c = corr_model.predict(rte[FEATURES])
        hybrid = glm_te * c

        gbm_pure = freq_cv.fit_gbm(rtr, params, SEED + k).predict(rte[FEATURES]) * e
        preds = {"constant": tr.claim_nb.sum() / tr.exposure.sum() * e, "glm_banded": banded,
                 "glm_star": glm_te, "hybrid": hybrid, "gbm": gbm_pure}
        oof.append(pd.DataFrame(dict(idpol=te.idpol.values, fold=k, y=y, exposure=e, glm_banded=banded,
                                     glm_star=glm_te, correction=c, gbm=gbm_pure)))
        for n_ in models:
            D[n_].append(poisson_dev_terms(y, preds[n_]).sum())
        lc = np.log(c)
        corr_stats.append(dict(sd_log_correction=lc.std(), mean_abs_log_correction=np.abs(lc).mean(),
                               share_moved_over_10pct=(np.abs(lc) > np.log(1.10)).mean(),
                               p05=np.quantile(c, .05), p50=np.quantile(c, .5), p95=np.quantile(c, .95)))
        print(f"fold {k}: " + "  ".join(f"{n_}={D[n_][-1] / N[-1]:.5f}" for n_ in models))

    N = np.array(N)
    dev = {n_: np.array(D[n_]) / N for n_ in models}
    pooled = {n_: np.sum(D[n_]) / N.sum() for n_ in models}
    gap = pooled["glm_banded"] - pooled["gbm"]
    rows = []
    for n_ in models:
        d = dev["glm_banded"] - dev[n_]
        rows.append(dict(model=n_, mean_deviance=dev[n_].mean(), deviance_explained_pooled=1 - pooled[n_] / pooled["constant"],
                         share_of_gbm_gap_closed=(pooled["glm_banded"] - pooled[n_]) / gap if (gap > 0.001 and n_ != "constant") else np.nan,
                         gain_vs_banded=d.mean() if n_ not in ("constant", "glm_banded") else np.nan,
                         sd_of_gain=d.std(ddof=1) if n_ not in ("constant", "glm_banded") else np.nan))
    res = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    pd.concat(oof).to_csv(OUT / "freq_hybrid_oof.csv.gz", index=False, compression="gzip")   # for 17_freq_hybrid_analysis.py
    RES.mkdir(exist_ok=True)
    res.to_csv(RES / "freq_hybrid_cv.csv", index=False)

    h_vs_g = dev["gbm"] - dev["hybrid"]                         # positive = hybrid better than the pure GBM
    h_vs_s = dev["glm_star"] - dev["hybrid"]                    # positive = correction helps the GLM
    cs = pd.DataFrame(corr_stats).mean()
    verdict = ("hybrid is better than the pure GBM" if h_vs_g.mean() > h_vs_g.std(ddof=1) else
               "pure GBM is better than the hybrid" if -h_vs_g.mean() > h_vs_g.std(ddof=1) else
               "hybrid and pure GBM are not distinguishable")
    md = ["# Frequency: hybrid (GLM backbone + GBM correction) vs pure GBM\n",
          f"Data: **{'SYNTHETIC - pipeline test only, not a finding' if is_synthetic() else 'real freMTPL2 (public)'}**; "
          f"GBM backend `{BACKEND}`; nested {K}-fold CV, mean Poisson deviance per policy (lower is better).\n",
          res.to_markdown(index=False, floatfmt=".5f"), "",
          f"- GBM correction on top of GLM*: gain {h_vs_s.mean():+.5f} per fold (SD {h_vs_s.std(ddof=1):.5f}, "
          f"better in {int((h_vs_s > 0).sum())}/{K} folds).",
          f"- Hybrid minus pure GBM: hybrid better by {h_vs_g.mean():+.5f} (SD {h_vs_g.std(ddof=1):.5f}, "
          f"better in {int((h_vs_g > 0).sum())}/{K} folds) -> **{verdict}** (rule: mean difference larger than its fold-to-fold SD).\n",
          "## How hard does the GBM push the GLM? (held-out folds, averaged)\n",
          f"- typical size of the correction: mean |log factor| = {cs.mean_abs_log_correction:.3f} (~{100 * cs.mean_abs_log_correction:.1f}%)",
          f"- share of policies whose GLM prediction is moved by more than 10%: {100 * cs.share_moved_over_10pct:.1f}%",
          f"- correction factor quantiles: 5% = {cs.p05:.3f}, median = {cs.p50:.3f}, 95% = {cs.p95:.3f}\n",
          "A small, tightly-centred correction means the GLM already carries the structure and the GBM is a monitored top-up; "
          "a wide one means the GBM is effectively the model.\n",
          "Caveat: brand x vehicle-age band was chosen using GBM behaviour on the validation split (see `freq_glm_from_gbm.md`), "
          "so GLM* and the hybrid have a small, documented advantage over a fully blind pipeline.\n"]
    (RES / "freq_hybrid.md").write_text("\n".join(md))
    print("\n" + "\n".join(md))
