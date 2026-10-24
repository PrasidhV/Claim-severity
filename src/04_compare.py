"""Compare GLM vs GBM on the untouched test set.

Metrics : weighted mean Gamma deviance, deviance explained vs. a no-feature model, portfolio bias.
Evidence: paired bootstrap CI on the deviance difference (is the gap bigger than noise?),
          decile calibration, and a double-lift chart (the standard actuarial head-to-head).
Writes  : results/results.md, results/calibration_and_lift.png
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import OUT, RES, SEED, gamma_dev_terms, gamma_deviance, is_synthetic, load_model_frame

glm_path, glm_label = OUT / "glm_r_test_preds.csv", "GLM (R)"
if not glm_path.exists():
    glm_path, glm_label = OUT / "glm_pycheck_test_preds.csv", "GLM (python cross-check, R not run)"
    print(f"!! R predictions not found, using {glm_path.name}")
glm = pd.read_csv(glm_path)
gbm = pd.read_csv(OUT / "gbm_test_preds.csv")
meta = json.loads((OUT / "gbm_meta.json").read_text())
d = glm.merge(gbm[["idpol", "pred"]], on="idpol", suffixes=("_glm", "_gbm"))
assert len(d) == len(glm) == len(gbm), "test sets differ between models"
y, w, p_glm, p_gbm = d.y.values, d.w.values, d.pred_glm.values, d.pred_gbm.values

# no-feature benchmark: weighted mean severity of the fitting data
df = load_model_frame()
fit = df[df.split.isin(["train", "valid"])]
null_mu = np.average(fit.avg_severity, weights=fit.n_claims)
dev_null = gamma_deviance(y, np.full_like(y, null_mu), w)

rows = []
for name, p in [(glm_label, p_glm), (f"GBM ({meta['backend'].split(' ')[0]})", p_gbm)]:
    dev = gamma_deviance(y, p, w)
    rows.append(dict(model=name, test_deviance=dev, deviance_explained=1 - dev / dev_null,
                     bias=np.sum(w * p) / np.sum(w * y) - 1))
# constant prediction baseline: claim-weighted mean severity of train+valid
rows.append(dict(model="Constant (null_mu)", test_deviance=dev_null, deviance_explained=0.0,
                 bias=np.sum(w * null_mu) / np.sum(w * y) - 1))
res = pd.DataFrame(rows)

# paired bootstrap over test policies: A deviance minus B deviance (>0 => B better)
rng = np.random.default_rng(SEED)
t_glm, t_gbm = gamma_dev_terms(y, p_glm, w), gamma_dev_terms(y, p_gbm, w)
t_const = gamma_dev_terms(y, np.full_like(y, null_mu), w)
n = len(y)
diffs = np.empty(2000)
diffs_gc = np.empty(2000)   # GLM minus constant
diffs_bc = np.empty(2000)   # GBM minus constant
for b in range(2000):
    i = rng.integers(0, n, n)
    diffs[b] = (t_glm[i].sum() - t_gbm[i].sum()) / w[i].sum()
    diffs_gc[b] = (t_glm[i].sum() - t_const[i].sum()) / w[i].sum()
    diffs_bc[b] = (t_gbm[i].sum() - t_const[i].sum()) / w[i].sum()
lo, hi = np.percentile(diffs, [2.5, 97.5])
lo_gc, hi_gc = np.percentile(diffs_gc, [2.5, 97.5])
lo_bc, hi_bc = np.percentile(diffs_bc, [2.5, 97.5])
point = gamma_deviance(y, p_glm, w) - gamma_deviance(y, p_gbm, w)
point_gc = gamma_deviance(y, p_glm, w) - dev_null
point_bc = gamma_deviance(y, p_gbm, w) - dev_null

# ---------- plots ----------
def binned(pred, key, q):
    g = pd.qcut(pd.Series(key).rank(method="first"), q, labels=False)
    t = pd.DataFrame(dict(g=g, y=y * w, p=pred * w, w=w)).groupby("g").sum()
    return t.y / t.w, t.p / t.w

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for name, p, c in [("GLM", p_glm, "tab:blue"), ("GBM", p_gbm, "tab:orange")]:
    a, e = binned(p, p, 10)
    ax[0].plot(a.index + 1, e.values / a.values, "o-", color=c, label=name)
ax[0].axhline(1, color="k", lw=.8)
ax[0].set(title="Decile calibration (predicted / actual, ranked by own prediction)",
          xlabel="prediction decile", ylabel="predicted / actual")
ax[0].legend()

ratio = p_gbm / p_glm
q = 5
g = pd.qcut(pd.Series(ratio).rank(method="first"), q, labels=False)
t = pd.DataFrame(dict(g=g, y=y * w, glm=p_glm * w, gbm=p_gbm * w, w=w)).groupby("g").sum()
x = np.arange(1, q + 1)
ax[1].plot(x, t.y / t.w, "k-o", label="actual")
ax[1].plot(x, t.glm / t.w, "--s", color="tab:blue", label="GLM")
ax[1].plot(x, t.gbm / t.w, "--^", color="tab:orange", label="GBM")
ax[1].set(title="Double lift: bins of GBM/GLM prediction ratio", ylabel="mean severity")
ax[1].set_xlabel("bin of GBM/GLM ratio (1 = GBM much lower than GLM, 5 = GBM much higher)")
ax[1].legend()
plt.tight_layout()
RES.mkdir(exist_ok=True)
plt.savefig(RES / "calibration_and_lift.png", dpi=130)

# ---------- write report ----------
verdict = ("GBM is better by more than sampling noise" if lo > 0 else
           "GLM is better by more than sampling noise" if hi < 0 else
           "the difference is within sampling noise (95% CI includes 0)")
verdict_gc = ("constant is better than GLM by more than sampling noise" if lo_gc > 0 else
              "GLM is better than constant by more than sampling noise" if hi_gc < 0 else
              "the difference is within sampling noise (95% CI includes 0)")
verdict_bc = ("constant is better than GBM by more than sampling noise" if lo_bc > 0 else
              "GBM is better than constant by more than sampling noise" if hi_bc < 0 else
              "the difference is within sampling noise (95% CI includes 0)")
md = [f"# Test-set results\n",
      f"Data: **{'SYNTHETIC - pipeline test only, not a finding' if is_synthetic() else 'real freMTPL2 (public)'}**  ",
      f"Test policies with claims: {n:,}; claims: {int(w.sum()):,}  ",
      f"GBM backend: `{meta['backend']}`, {meta['n_trees']} trees, params {meta['params']}\n",
      res.to_markdown(index=False, floatfmt=".4f"), "",
      f"Paired bootstrap, GLM minus GBM deviance: **{point:+.4f}** (95% CI {lo:+.4f} to {hi:+.4f}) "
      f"-> {verdict}.\n",
      f"Paired bootstrap, GLM minus constant deviance: **{point_gc:+.4f}** (95% CI {lo_gc:+.4f} to {hi_gc:+.4f}) "
      f"-> {verdict_gc}.\n",
      f"Paired bootstrap, GBM minus constant deviance: **{point_bc:+.4f}** (95% CI {lo_bc:+.4f} to {hi_bc:+.4f}) "
      f"-> {verdict_bc}.\n",
      "![calibration and lift](calibration_and_lift.png)\n"]
(RES / "results.md").write_text("\n".join(md))
print("\n".join(md))
