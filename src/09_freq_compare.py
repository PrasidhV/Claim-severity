"""Frequency: GLM vs GBM vs a constant-rate benchmark on the untouched test set.
Metric: mean Poisson deviance per policy. Also: deviance explained vs the constant rate, bias,
paired-bootstrap CIs, decile calibration and a double-lift chart.
Writes results/freq_results.md and results/freq_calibration_and_lift.png"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import OUT, RES, SEED, is_synthetic, load_frequency_frame, poisson_dev_terms, poisson_deviance

glm_path, glm_label = OUT / "freq_glm_r_test_preds.csv", "GLM (R)"
if not glm_path.exists():
    glm_path, glm_label = OUT / "freq_glm_pycheck_test_preds.csv", "GLM (python cross-check, R not run)"
    print(f"!! R predictions not found, using {glm_path.name}")
glm, gbm = pd.read_csv(glm_path), pd.read_csv(OUT / "freq_gbm_test_preds.csv")
meta = json.loads((OUT / "freq_gbm_meta.json").read_text())
d = glm.merge(gbm[["idpol", "pred"]], on="idpol", suffixes=("_glm", "_gbm"))
assert len(d) == len(glm) == len(gbm), "test sets differ between models"
y, e, p_glm, p_gbm = d.y.values, d.exposure.values, d.pred_glm.values, d.pred_gbm.values

df = load_frequency_frame()
fit = df[df.split.isin(["train", "valid"])]
lam = fit.claim_nb.sum() / fit.exposure.sum()          # constant claims per exposure-year
p_const = lam * e

terms = {"GLM": poisson_dev_terms(y, p_glm), "GBM": poisson_dev_terms(y, p_gbm),
         "Constant": poisson_dev_terms(y, p_const)}
dev_c = terms["Constant"].mean()
rows = []
for name, label, p in [("GLM", glm_label, p_glm), ("GBM", f"GBM ({meta['backend'].split(' ')[0]})", p_gbm),
                       ("Constant", "Constant rate (train+valid)", p_const)]:
    dev = terms[name].mean()
    rows.append(dict(model=label, test_deviance=dev, deviance_explained=1 - dev / dev_c,
                     bias=p.sum() / y.sum() - 1))
res = pd.DataFrame(rows)

rng = np.random.default_rng(SEED)
n = len(y)
idx = [rng.integers(0, n, n) for _ in range(1000)]


def boot(a, b, la, lb):
    """Paired bootstrap of mean(dev_a) - mean(dev_b); positive means b is better."""
    diffs = np.array([(a[i] - b[i]).mean() for i in idx])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    pt = (a - b).mean()
    v = (f"{lb} is better than {la} by more than sampling noise" if lo > 0 else
         f"{la} is better than {lb} by more than sampling noise" if hi < 0 else
         "the difference is within sampling noise (95% CI includes 0)")
    return f"Paired bootstrap, {la} minus {lb} deviance: **{pt:+.5f}** (95% CI {lo:+.5f} to {hi:+.5f}) -> {v}.\n"


lines = [boot(terms["GLM"], terms["GBM"], "GLM", "GBM"),
         boot(terms["Constant"], terms["GLM"], "constant", "GLM"),
         boot(terms["Constant"], terms["GBM"], "constant", "GBM")]

# ---- plots: calibration by predicted-rate decile, and double lift
def rate_bins(pred_rate, q):
    return pd.qcut(pd.Series(pred_rate).rank(method="first"), q, labels=False)


fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for name, p, c in [("GLM", p_glm, "tab:blue"), ("GBM", p_gbm, "tab:orange")]:
    t = pd.DataFrame(dict(g=rate_bins(p / e, 10), y=y, p=p)).groupby("g").sum()
    ax[0].plot(t.index + 1, t.p / t.y, "o-", color=c, label=name)
ax[0].axhline(1, color="k", lw=.8)
ax[0].set(title="Decile calibration (predicted / actual claims, ranked by own prediction)",
          xlabel="prediction decile", ylabel="predicted / actual")
ax[0].legend()
q = 5
t = pd.DataFrame(dict(g=rate_bins(p_gbm / p_glm, q), y=y, glm=p_glm, gbm=p_gbm, e=e)).groupby("g").sum()
x = np.arange(1, q + 1)
ax[1].plot(x, t.y / t.e, "k-o", label="actual")
ax[1].plot(x, t.glm / t.e, "--s", color="tab:blue", label="GLM")
ax[1].plot(x, t.gbm / t.e, "--^", color="tab:orange", label="GBM")
ax[1].set(title="Double lift: bins of GBM/GLM prediction ratio", ylabel="claims per year",
          xlabel="bin (1 = GBM much lower than GLM, 5 = GBM much higher)")
ax[1].legend()
plt.tight_layout()
RES.mkdir(exist_ok=True)
plt.savefig(RES / "freq_calibration_and_lift.png", dpi=130)

md = ["# Frequency: test-set results\n",
      f"Data: **{'SYNTHETIC - pipeline test only, not a finding' if is_synthetic() else 'real freMTPL2 (public)'}**  ",
      f"Test policies: {n:,}; claims: {int(y.sum()):,}; exposure-years: {e.sum():,.0f}  ",
      f"GBM backend: `{meta['backend']}`, {meta['n_trees']} trees, params {meta['params']}  ",
      f"Metric: mean Poisson deviance per policy (lower is better); constant rate = {lam:.4f} claims per year.\n",
      res.to_markdown(index=False, floatfmt=".5f"), "", *lines,
      "![calibration and lift](freq_calibration_and_lift.png)\n"]
(RES / "freq_results.md").write_text("\n".join(md))
print("\n".join(md))
