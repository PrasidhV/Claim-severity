"""Where does the GBM's advantage come from?  A complexity ladder on the single train/valid/test split.

Trees with 2 leaves are stumps, so the model is purely ADDITIVE (no interactions at all, but fully flexible
one-variable shapes). More leaves allow progressively higher-order interactions. Reading the ladder:
  additive GBM ~ GLM+            -> the gap is interactions
  additive GBM ~ full GBM        -> the gap is main-effect SHAPE that the GLM's splines don't capture
Each rung is early-stopped on 'valid' and scored once on 'test'. Writes results/freq_gbm_ladder.md"""
import warnings

import numpy as np
import pandas as pd

from common import FEATURES, OUT, RES, SEED, is_synthetic, load_frequency_frame, poisson_deviance

warnings.filterwarnings("ignore", message=".*eval_set.*deprecated.*")
try:
    import lightgbm as lgb
    BACKEND = "lightgbm"
except ImportError:
    from sklearn.ensemble import HistGradientBoostingRegressor
    BACKEND = "sklearn-HistGradientBoosting (FALLBACK: lightgbm not installed)"

LADDER = [(2, "additive (stumps, no interactions)"), (3, "up to 1 interaction per tree"),
          (4, "small trees"), (8, "medium trees"), (16, "larger trees"), (32, "full GBM (as chosen earlier)")]
MCS = 200


def fit(leaves, tr, va):
    r = lambda d: d.claim_nb / d.exposure
    if BACKEND == "lightgbm":
        m = lgb.LGBMRegressor(objective="poisson", n_estimators=8000, learning_rate=0.05, num_leaves=leaves,
                              min_child_samples=MCS, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                              reg_lambda=5.0, random_state=SEED, verbose=-1)
        m.fit(tr[FEATURES], r(tr), sample_weight=tr.exposure, eval_set=[(va[FEATURES], r(va))],
              eval_sample_weight=[va.exposure], eval_metric="poisson",
              callbacks=[lgb.early_stopping(150, verbose=False)])
        return m, int(m.best_iteration_)
    kw = dict(loss="poisson", learning_rate=0.1, max_leaf_nodes=leaves, min_samples_leaf=MCS,
              l2_regularization=5.0, categorical_features="from_dtype", random_state=SEED)
    probe = HistGradientBoostingRegressor(max_iter=600, **kw).fit(tr[FEATURES], r(tr), sample_weight=tr.exposure)
    curve = [poisson_deviance(va.claim_nb, p * va.exposure) for p in probe.staged_predict(va[FEATURES])]
    best = int(np.argmin(curve)) + 1
    return HistGradientBoostingRegressor(max_iter=best, **kw).fit(tr[FEATURES], r(tr), sample_weight=tr.exposure), best


df = load_frequency_frame()
tr, va, te = (df[df.split == s] for s in ("train", "valid", "test"))
rows = []
for leaves, label in LADDER:
    m, n_iter = fit(leaves, tr, va)
    dev = poisson_deviance(te.claim_nb, m.predict(te[FEATURES]) * te.exposure.values)
    rows.append(dict(model=f"GBM, {leaves} leaves: {label}", trees=n_iter, test_deviance=dev))
    print(f"leaves={leaves:>2} trees={n_iter:>5} test deviance={dev:.5f}")

ref = []
for name, f in [("banded GLM", "freq_glm_r_test_preds.csv"), ("GLM+ (python check)", "freq_glm_plus_pycheck_test_preds.csv"),
                ("GLM+ (R)", "freq_glm_plus_r_test_preds.csv")]:
    if (OUT / f).exists():
        p = pd.read_csv(OUT / f)
        ref.append(dict(model=name, trees=np.nan, test_deviance=poisson_deviance(p.y, p.pred)))
tab = pd.DataFrame(ref + rows)
md = ["# Frequency: GBM complexity ladder (test set, single split)\n",
      f"Data: **{'SYNTHETIC - pipeline test only, not a finding' if is_synthetic() else 'real freMTPL2 (public)'}**; "
      f"backend `{BACKEND}`; mean Poisson deviance per policy, lower is better.\n",
      tab.to_markdown(index=False, floatfmt=".5f"), "",
      "Reading it: the first GBM row is purely additive. If it is close to GLM+, the GBM's advantage is interactions; "
      "if it is already close to the full GBM, the advantage is main-effect shape that the GLM's splines miss. "
      "(Single split: differences of a few thousandths between neighbouring rows are within noise; see freq_results.md for bootstrap CIs.)\n"]
RES.mkdir(exist_ok=True)
(RES / "freq_gbm_ladder.md").write_text("\n".join(md))
print("\n" + "\n".join(md))
