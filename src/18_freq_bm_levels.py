"""Is there a bonus-malus saw-tooth that the GLM's smooth splines miss?  (uses held-out predictions saved by 16)

A. DIAGNOSTIC: for each bonus-malus level with enough policies, actual claims / GLM* expected claims (with Poisson CI) and the
   mean GBM correction. A smooth risk gradient would show a smooth curve; a saw-tooth means level-specific effects.
B. TEST: GLM* + one indicator per well-populated BM level (levels chosen on the TRAINING folds only), scored on the same
   nested 5-fold CV folds, against GLM*, the hybrid and the pure GBM (taken from the saved held-out predictions).
CAVEAT: the idea came from looking at held-out predictions from these same folds (see freq_hybrid_analysis.md), so B is
optimistic; the French bonus-malus mechanism (x0.95 per claim-free year, x1.25 per at-fault claim) is what makes it plausible
rather than a data-dredged artefact, but there is no fresh data to confirm it on.
Writes results/freq_bm_levels.md and results/freq_bm_levels.png. Needs outputs/freq_hybrid_oof.csv.gz from 16."""
import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import glm_plus as gp
import poisson_glm as pg
from common import OUT, RES, is_synthetic, load_frequency_frame, poisson_dev_terms


def _load(name, fname):
    s = importlib.util.spec_from_file_location(name, Path(__file__).with_name(fname))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


gfg = _load("gfg", "15_freq_glm_from_gbm.py")
K = 5
oof = pd.read_csv(OUT / "freq_hybrid_oof.csv.gz")
raw = load_frequency_frame()
df = gp.prep(raw)
for c in ["veh_age_band", "driv_age_band", "bm_band"]:
    df[c] = df[c].astype(str)
oof = oof.merge(df[["idpol", "bonus_malus_c"]], on="idpol")
oof["hybrid"] = oof.glm_star * oof.correction
min_n = max(200, int(0.003 * len(df) * (K - 1) / K))                     # a level needs >= ~0.3% of training policies

# ---------------- A. level-by-level diagnostic ----------------
g = oof.groupby("bonus_malus_c").agg(policies=("y", "size"), claims=("y", "sum"), glm_star=("glm_star", "sum"),
                                     hybrid=("hybrid", "sum"))
g["mean_correction"] = oof.assign(l=np.log(oof.correction) * oof.glm_star).groupby("bonus_malus_c").l.sum() / g.glm_star
g["mean_correction"] = np.exp(g.mean_correction)
g["actual_vs_glm_star"] = g.claims / g.glm_star
g["actual_vs_hybrid"] = g.claims / g.hybrid
g["ci_halfwidth"] = 1.96 / np.sqrt(g.claims.clip(lower=1))
lv = g[g.policies >= min_n].reset_index()
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
ax[0].errorbar(lv.bonus_malus_c, lv.actual_vs_glm_star, yerr=lv.actual_vs_glm_star * lv.ci_halfwidth, fmt="o", capsize=2,
               color="tab:blue", label="actual / GLM* expected (95% CI)")
ax[0].plot(lv.bonus_malus_c, lv.actual_vs_hybrid, "^--", color="tab:orange", label="actual / hybrid expected")
ax[0].axhline(1, color="k", lw=.8)
ax[0].set(title="Calibration of GLM* by bonus-malus level", xlabel="bonus-malus", ylabel="actual / predicted claims"); ax[0].legend()
ax[1].plot(lv.bonus_malus_c, lv.mean_correction, "o-", color="tab:green"); ax[1].axhline(1, color="k", lw=.8)
ax[1].set(title="Mean GBM correction by bonus-malus level", xlabel="bonus-malus", ylabel="correction factor")
plt.tight_layout(); RES.mkdir(exist_ok=True); plt.savefig(RES / "freq_bm_levels.png", dpi=130); plt.close()

# ---------------- B. nested CV: GLM* + fine BM levels ----------------
fold = (df.idpol % K).values
D = {"glm_star": [], "glm_star_bm": [], "hybrid": [], "gbm": []}
N, nlev = [], []
o = oof.set_index("idpol")
for k in range(K):
    tr, te = df[fold != k], df[fold == k]
    base, spec, X0 = gp.fit(tr)
    pairs, _ = gp.screen(tr, spec, X0, base, 2)
    levels = [float(l) for l, n in tr.bonus_malus_c.value_counts().items() if n >= min_n and l != 50]
    nlev.append(len(levels))

    def X_of(d):
        return np.hstack([gp.build(d, spec, pairs),
                          np.column_stack(gfg.extra_cols(d, tr, spec, *gfg.TERMS["brand x vehicle-age band"])),
                          np.column_stack([(d.bonus_malus_c.values == l).astype(float) for l in levels])])
    m = pg.fit_poisson_log(X_of(tr), tr.claim_nb.values.astype(float), tr.exposure.values)
    pred = np.exp(np.clip(X_of(te) @ m["beta"], -30, 30)) * te.exposure.values
    ref = o.loc[te.idpol]
    y = te.claim_nb.values.astype(float)
    for n_, p in [("glm_star", ref.glm_star.values), ("glm_star_bm", pred), ("hybrid", ref.glm_star.values * ref.correction.values),
                  ("gbm", ref.gbm.values)]:
        D[n_].append(poisson_dev_terms(y, p).sum())
    N.append(len(te))
    print(f"fold {k}: " + "  ".join(f"{n_}={D[n_][-1] / N[-1]:.5f}" for n_ in D) + f" | {len(levels)} BM level indicators")

N = np.array(N)
dev = {n_: np.array(v) / N for n_, v in D.items()}
pooled = {n_: np.sum(v) / N.sum() for n_, v in D.items()}
d_band = poisson_dev_terms(oof.y.values, oof.glm_banded.values).mean()
gap = d_band - pooled["gbm"]
gain = dev["glm_star"] - dev["glm_star_bm"]
res = pd.DataFrame([dict(model=n_, mean_deviance=dev[n_].mean(),
                         share_of_gbm_gap_closed=(d_band - pooled[n_]) / gap if gap > 0.001 else np.nan) for n_ in D])
verdict = ("clearly helps" if gain.mean() > gain.std(ddof=1) else "does not clearly help")
md = ["# Frequency: is there a bonus-malus saw-tooth the GLM's splines miss?\n",
      f"Data: **{'SYNTHETIC - pipeline test only, not a finding' if is_synthetic() else 'real freMTPL2 (public)'}**. Levels shown have at least {min_n} policies.\n",
      "## A. Calibration and GBM correction by bonus-malus level (held-out predictions, all folds)\n",
      lv[["bonus_malus_c", "policies", "claims", "actual_vs_glm_star", "actual_vs_hybrid", "mean_correction"]].to_markdown(index=False, floatfmt=".3f"),
      "\n![bm levels](freq_bm_levels.png)\n",
      "A smooth risk gradient gives a smooth curve in the left panel; isolated levels far from 1 (with intervals excluding 1) indicate level-specific effects. "
      "Poisson CI = 1.96/sqrt(claims).\n",
      f"## B. Nested 5-fold CV: GLM* + one indicator per BM level (about {int(np.mean(nlev))} indicators per fold)\n",
      res.to_markdown(index=False, floatfmt=".5f"), "",
      f"Adding the BM-level indicators to GLM*: gain {gain.mean():+.5f} per fold (SD {gain.std(ddof=1):.5f}, better in {int((gain > 0).sum())}/{K} folds) -> **{verdict}** "
      "(rule: mean gain larger than its fold-to-fold SD).\n",
      "Caveat: the saw-tooth was spotted in held-out predictions from these same folds, so B is optimistic; treat it as an upper bound.\n"]
(RES / "freq_bm_levels.md").write_text("\n".join(md))
print("\n" + "\n".join(md[:1] + md[-5:]))
