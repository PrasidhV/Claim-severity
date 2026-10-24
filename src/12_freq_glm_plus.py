"""Does a better-specified GLM close the GLM-GBM gap?  (frequency)

Compares, on identical folds: constant | banded GLM (the baseline) | GLM+ splines | GLM+ splines + screened
interactions | GBM.  Interaction screening (Rao score test) is redone INSIDE every fold from the training folds only.
Also does the single train/test split (R GLM+ if available) with a paired bootstrap vs the GBM.
Writes results/freq_glm_plus_results.md and results/freq_glm_plus_cv.csv."""
import importlib.util
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import glm_plus as gp
import poisson_glm as pg
from common import (CAT, GLM_CATS, GLM_NUM, FEATURES, OUT, RES, SEED, is_synthetic, load_frequency_frame,
                    poisson_dev_terms)

warnings.filterwarnings("ignore", message=".*eval_set.*deprecated.*")
K = 5
spec_ = importlib.util.spec_from_file_location("freq_cv", Path(__file__).with_name("10_freq_cv.py"))
freq_cv = importlib.util.module_from_spec(spec_); spec_.loader.exec_module(freq_cv)   # reuse the exact GBM fitter

raw = load_frequency_frame()
df = gp.prep(raw)
for c in GLM_CATS:
    df[c] = df[c].astype(str)
params = json.loads((OUT / "freq_gbm_meta.json").read_text())["params"]
ALL = GLM_CATS + GLM_NUM
models = ["constant", "glm_banded", "glm_splines", "glm_splines_ix", "gbm"]
fold = (df.idpol % K).values
D, N, chosen = {m: [] for m in models}, [], []

print(f"GBM backend: {freq_cv.BACKEND} | params {params}")
for k in range(K):
    tr, te = df[fold != k], df[fold == k]
    y, e = te.claim_nb.values.astype(float), te.exposure.values
    N.append(len(te))
    m, s = pg.fit_glm(tr, ALL)
    fit = gp.fit_plus(tr, k=2)
    gbm_tr = raw[fold != k]
    gbm_pred = freq_cv.fit_gbm(gbm_tr, params, SEED + k).predict(raw[fold == k][FEATURES]) * e
    preds = {"constant": tr.claim_nb.sum() / tr.exposure.sum() * e,
             "glm_banded": pg.predict_glm(m, s, ALL, te),
             "glm_splines": gp.predict(fit["base"], fit["spec"], te),
             "glm_splines_ix": gp.predict(fit["model"], fit["spec"], te, fit["pairs"]),
             "gbm": gbm_pred}
    chosen.append(fit["pairs"])
    for n_ in models:
        D[n_].append(poisson_dev_terms(y, preds[n_]).sum())
    print(f"fold {k}: " + "  ".join(f"{n_}={D[n_][-1] / N[-1]:.5f}" for n_ in models) +
          f" | interactions: {[' x '.join(p) for p in fit['pairs']]}")

N = np.array(N)
dev = {n_: np.array(D[n_]) / N for n_ in models}
pooled = {n_: np.sum(D[n_]) / N.sum() for n_ in models}
gap = pooled["glm_banded"] - pooled["gbm"]
rows = []
for n_ in models:
    d_vs_banded = dev["glm_banded"] - dev[n_]
    rows.append(dict(model=n_, mean_deviance=dev[n_].mean(), deviance_explained_pooled=1 - pooled[n_] / pooled["constant"],
                     gain_vs_banded_glm=d_vs_banded.mean(), sd_of_gain=d_vs_banded.std(ddof=1),
                     folds_beating_banded=int((d_vs_banded > 0).sum()),
                     share_of_gbm_gap_closed=(pooled["glm_banded"] - pooled[n_]) / gap))
res = pd.DataFrame(rows)
res.loc[res.model.isin(["constant", "glm_banded"]),
        ["gain_vs_banded_glm", "sd_of_gain", "folds_beating_banded", "share_of_gbm_gap_closed"]] = np.nan
if gap <= 2 * (dev["glm_banded"] - dev["gbm"]).std(ddof=1) / np.sqrt(K):    # GBM edge not clearly > 0
    res["share_of_gbm_gap_closed"] = np.nan                                  # ratio would be unstable/meaningless
    print("note: GBM does not clearly beat the banded GLM here, so 'share of gap closed' is not reported")
res["folds_beating_banded"] = res["folds_beating_banded"].astype("Int64")
RES.mkdir(exist_ok=True)
res.to_csv(RES / "freq_glm_plus_cv.csv", index=False)

# ---------------- single split (matches the earlier test-set comparison) ----------------
fit_df, test_df = df[df.split.isin(["train", "valid"])], df[df.split == "test"]
sp = gp.fit_plus(fit_df, k=2)
py_pred = gp.predict(sp["model"], sp["spec"], test_df, sp["pairs"])
pd.DataFrame(dict(idpol=test_df.idpol, y=test_df.claim_nb, exposure=test_df.exposure, pred=py_pred)
             ).to_csv(OUT / "freq_glm_plus_pycheck_test_preds.csv", index=False)
r_path = OUT / "freq_glm_plus_r_test_preds.csv"
plus_src = "GLM+ (R)" if r_path.exists() else "GLM+ (python check, R not run)"
plus = pd.read_csv(r_path if r_path.exists() else OUT / "freq_glm_plus_pycheck_test_preds.csv")
base_p = pd.read_csv(OUT / ("freq_glm_r_test_preds.csv" if (OUT / "freq_glm_r_test_preds.csv").exists()
                            else "freq_glm_pycheck_test_preds.csv"))
gbm_p = pd.read_csv(OUT / "freq_gbm_test_preds.csv")
t = plus.merge(base_p[["idpol", "pred"]], on="idpol", suffixes=("_plus", "_base")).merge(
    gbm_p[["idpol", "pred"]].rename(columns={"pred": "pred_gbm"}), on="idpol")
yy = t.y.values
tm = {"banded GLM": poisson_dev_terms(yy, t.pred_base.values), plus_src: poisson_dev_terms(yy, t.pred_plus.values),
      "GBM": poisson_dev_terms(yy, t.pred_gbm.values)}
rng = np.random.default_rng(SEED)
idx = [rng.integers(0, len(yy), len(yy)) for _ in range(1000)]


def boot(a, b, la, lb):
    d = np.array([(a[i] - b[i]).mean() for i in idx])
    lo, hi = np.percentile(d, [2.5, 97.5])
    v = (f"{lb} is better by more than sampling noise" if lo > 0 else
         f"{la} is better by more than sampling noise" if hi < 0 else "within sampling noise")
    return f"- {la} minus {lb}: **{(a - b).mean():+.5f}** (95% CI {lo:+.5f} to {hi:+.5f}) -> {v}"


split_tab = pd.DataFrame([dict(model=k_, test_deviance=v_.mean()) for k_, v_ in tm.items()])
ix_counts = pd.Series([" x ".join(p) for ps in chosen for p in ps]).value_counts()
md = ["# Frequency: does a better-specified GLM close the gap to the GBM?\n",
      f"Data: **{'SYNTHETIC - pipeline test only, not a finding' if is_synthetic() else 'real freMTPL2 (public)'}**  ",
      f"GBM backend: `{freq_cv.BACKEND}`. Metric: mean Poisson deviance per policy (lower is better).\n",
      "## 5-fold cross-validation (interaction screening nested inside each fold)\n",
      res.to_markdown(index=False, floatfmt=".5f"), "",
      "`gain_vs_banded_glm` = banded-GLM deviance minus model deviance per fold (positive = better); `sd_of_gain` is its SD over "
      f"the {K} folds. `share_of_gbm_gap_closed` = (banded GLM - model) / (banded GLM - GBM), using pooled deviance: 0 = no better than "
      "the banded GLM, 1 = as good as the GBM (left blank if the GBM does not clearly beat the banded GLM).\n",
      "Interactions chosen per fold (top 2 by Rao score, training folds only):\n",
      *[f"- fold {k}: {', '.join(' x '.join(p) for p in ps)}" for k, ps in enumerate(chosen)], "",
      "## Single train/valid -> test split\n", split_tab.to_markdown(index=False, floatfmt=".5f"), "",
      boot(tm["banded GLM"], tm[plus_src], "banded GLM", plus_src),
      boot(tm[plus_src], tm["GBM"], plus_src, "GBM"), ""]
(RES / "freq_glm_plus_results.md").write_text("\n".join(md))
print("\n" + "\n".join(md))
