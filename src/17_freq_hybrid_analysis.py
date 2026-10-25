"""What does the GBM correct, and how much of it can be capped away?  (uses held-out predictions saved by 16)

1. Correction-cap curve: limit the GBM's multiplicative correction to +/-cap and see how much accuracy survives.
   Each fold's total predicted count is rescaled to equal the uncapped hybrid's, so portfolio calibration is unchanged.
2. Segment profile: a shallow decision tree fitted to log(correction) on held-out policies (weighted by GLM expected claims)
   describes WHERE the GLM is wrong, and how much of the correction such simple segments explain.
Writes results/freq_hybrid_analysis.md"""
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from common import OUT, RES, is_synthetic, load_frequency_frame, poisson_dev_terms

oof = pd.read_csv(OUT / "freq_hybrid_oof.csv.gz")
y, w, corr, g = oof.y.values, oof.glm_star.values, oof.correction.values, oof.glm_star.values
dev = lambda pred: poisson_dev_terms(y, pred).mean()
d_band, d_gbm, d_glm = dev(oof.glm_banded.values), dev(oof.gbm.values), dev(g)
full = g * corr

# ---------- 1. cap curve ----------
rows = []
for cap in [0.0, 0.05, 0.10, 0.20, 0.30, 0.50, None]:
    c = corr if cap is None else np.clip(corr, np.exp(-np.log1p(cap)), np.exp(np.log1p(cap)))
    pred = g * c
    for k in oof.fold.unique():                       # keep each fold's total expected count equal to the uncapped hybrid
        m = (oof.fold == k).values
        pred[m] *= full[m].sum() / pred[m].sum()
    rows.append(dict(max_correction="none (full hybrid)" if cap is None else ("0% (GLM* only)" if cap == 0 else f"+/-{int(cap * 100)}%"),
                     deviance=dev(pred), share_of_gap_closed=(d_band - dev(pred)) / (d_band - d_gbm) if d_band - d_gbm > 0.001 else np.nan,
                     policies_hitting_cap=float((c != corr).mean())))
cap_tab = pd.DataFrame(rows)

# ---------- 2. segment profile ----------
fr = load_frequency_frame().set_index("idpol").loc[oof.idpol]
Xf = pd.DataFrame({"driv_age": fr.driv_age_c.values, "bonus_malus": fr.bonus_malus_c.values, "veh_age": fr.veh_age_c.values,
                   "log_density": fr.log_density.values, "veh_power": fr.veh_power.values, "area_ord": fr.area_ord.values,
                   "diesel": (fr.veh_gas.astype(str).values == "Diesel").astype(float)})
Xf = pd.concat([Xf, pd.get_dummies(fr.veh_brand.astype(str).reset_index(drop=True), prefix="brand").astype(float),
                pd.get_dummies(fr.region.astype(str).reset_index(drop=True), prefix="region").astype(float)], axis=1)
lc = np.log(corr)
tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=max(50, len(lc) // 150), random_state=0).fit(Xf, lc, sample_weight=w)
r2 = 1 - (w * (lc - tree.predict(Xf)) ** 2).sum() / (w * (lc - np.average(lc, weights=w)) ** 2).sum()


def leaf_rules(t, names):
    """Rule text per leaf, keeping only the tightest lower/upper bound for each variable."""
    tt, rules = t.tree_, {}

    def rec(n, bounds):
        if tt.children_left[n] == -1:
            parts = []
            for f, (lo, hi) in bounds.items():
                if lo > -np.inf and hi < np.inf:
                    parts.append(f"{lo:.3g} < {f} <= {hi:.3g}")
                elif hi < np.inf:
                    parts.append(f"{f} <= {hi:.3g}")
                else:
                    parts.append(f"{f} > {lo:.3g}")
            rules[n] = " AND ".join(parts) or "all policies"
            return
        f, th = names[tt.feature[n]], tt.threshold[n]
        lo, hi = bounds.get(f, (-np.inf, np.inf))
        rec(tt.children_left[n], {**bounds, f: (lo, min(hi, th))})
        rec(tt.children_right[n], {**bounds, f: (max(lo, th), hi)})
    rec(0, {})
    return rules


rules, leaf = leaf_rules(tree, list(Xf.columns)), tree.apply(Xf)
seg = []
for n, r in rules.items():
    m = leaf == n
    seg.append(dict(segment=r, share_of_policies=m.mean(), share_of_expected_claims=w[m].sum() / w.sum(),
                    mean_correction=float(np.exp(np.average(lc[m], weights=w[m]))),
                    actual_vs_GLM_star=float(y[m].sum() / g[m].sum()), actual_vs_hybrid=float(y[m].sum() / full[m].sum())))
seg = pd.DataFrame(seg).sort_values("mean_correction", ascending=False)

q = np.quantile(corr, [.05, .25, .5, .75, .95])
md = ["# Frequency hybrid: what the GBM corrects\n",
      f"Data: **{'SYNTHETIC - pipeline test only, not a finding' if is_synthetic() else 'real freMTPL2 (public)'}**; held-out predictions from the 5 CV folds "
      f"({len(oof):,} policies). Mean Poisson deviance per policy.\n",
      "## 1. How much accuracy survives a cap on the GBM's correction?\n", cap_tab.to_markdown(index=False, floatfmt=".5f"), "",
      f"Reference: banded GLM {d_band:.5f}, GLM* {d_glm:.5f}, full hybrid {dev(full):.5f}, pure GBM {d_gbm:.5f}. "
      "`share_of_gap_closed` is relative to banded GLM -> pure GBM. Totals are rescaled per fold so capping does not change portfolio-level calibration.\n",
      "## 2. Where is the GLM* wrong?\n",
      f"Correction factor quantiles (5/25/50/75/95%): {', '.join(f'{v:.3f}' for v in q)}. "
      f"A depth-3 tree on simple policy attributes explains **{100 * r2:.1f}%** of the (weighted) variance in log-correction.\n",
      seg.to_markdown(index=False, floatfmt=".3f"), "",
      "`mean_correction` = average multiplicative factor the GBM applies in that segment; `actual_vs_GLM_star` and `actual_vs_hybrid` are "
      "actual claims / predicted claims (1.0 = calibrated). A segment with `actual_vs_GLM_star` well above 1 is one the GLM under-prices. "
      "A low tree R-squared means the correction is a diffuse web rather than a few nameable segments.\n"]
RES.mkdir(exist_ok=True)
(RES / "freq_hybrid_analysis.md").write_text("\n".join(md))
print("\n".join(md))
