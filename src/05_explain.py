"""Explain the GBM and check it against the GLM.

With `shap` installed: TreeSHAP. For a Gamma/log-link model SHAP values live on the log scale, so
exp(SHAP) is a *multiplicative* effect - directly comparable with GLM relativities.
Without `shap` (fallback): permutation importance + partial-dependence effects instead.

Writes results/shap_importance.png, shap_beeswarm.png (shap only), glm_vs_gbm_effects.png
"""
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import FEATURES, OUT, RES, SEED, gamma_deviance, load_model_frame

try:
    import shap
    HAVE_SHAP = True
except ImportError:
    HAVE_SHAP = False
    print("!! `shap` not installed - using permutation importance / partial dependence fallback")

model = joblib.load(OUT / "gbm_model.joblib")
df = load_model_frame()
te = df[df.split == "test"].reset_index(drop=True)
X = te[FEATURES]
RES.mkdir(exist_ok=True)

# ---------------- global importance ----------------
if HAVE_SHAP:
    sv = shap.TreeExplainer(model).shap_values(X)          # (n, p) on the log-mean scale
    imp = pd.Series(np.abs(sv).mean(0), index=FEATURES).sort_values()
    xlabel = "mean |SHAP|  (log-severity units)"
else:
    from sklearn.inspection import permutation_importance
    r = permutation_importance(model, X, te.avg_severity, sample_weight=te.n_claims, n_repeats=10,
                               random_state=SEED,
                               scoring=lambda m, X_, y_, sample_weight=None:
                                   -gamma_deviance(y_, m.predict(X_), sample_weight))
    imp = pd.Series(r.importances_mean, index=FEATURES).sort_values()
    xlabel = "increase in test Gamma deviance when shuffled"
imp.plot.barh(figsize=(6, 4), title="GBM feature importance")
plt.xlabel(xlabel); plt.tight_layout(); plt.savefig(RES / "shap_importance.png", dpi=130); plt.close()
imp.sort_values(ascending=False).to_csv(OUT / "gbm_importance.csv", header=["importance"])
if HAVE_SHAP:
    shap.summary_plot(sv, X, show=False)
    plt.tight_layout(); plt.savefig(RES / "shap_beeswarm.png", dpi=130); plt.close()

# ---------------- GBM effect vs GLM relativity, by rating band ----------------
BANDS = {"driv_age_band": "driv_age_c", "veh_age_band": "veh_age_c", "bm_band": "bonus_malus_c"}
glm_rel_path = OUT / "glm_relativities.csv"
glm_tab = pd.read_csv(glm_rel_path).set_index("term") if glm_rel_path.exists() else None
glm_rel = glm_tab["relativity"] if glm_tab is not None else None
if glm_rel is None:
    print("!! outputs/glm_relativities.csv not found (R not run): plotting GBM effects only")


def gbm_effect(band, feat):
    """Log-scale effect of each band: mean SHAP of `feat` if available, else partial dependence."""
    levels = sorted(te[band].unique())
    if HAVE_SHAP:
        return pd.Series({l: sv[te[band] == l, FEATURES.index(feat)].mean() for l in levels})
    eff = {}
    for l in levels:
        Xl = X.copy()
        Xl[feat] = te.loc[te[band] == l, feat].median()
        eff[l] = np.log(model.predict(Xl)).mean()
    return pd.Series(eff)


fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (band, feat) in zip(axes, BANDS.items()):
    e = gbm_effect(band, feat)
    base = te.loc[te[band].map(te[band].value_counts()).idxmax(), band]   # most common level = GLM base
    gbm_rel = np.exp(e - e[base])
    ax.plot(range(len(e)), gbm_rel.values, "^--", color="tab:orange", label="GBM (" + ("SHAP" if HAVE_SHAP else "PDP") + ")")
    if glm_rel is not None:
        def pick(col):   # base level has relativity 1 and no interval
            return np.array([1.0 if l == base else glm_tab[col].get(f"{band}{l}", np.nan) for l in e.index])
        g, lo_, hi_ = pick("relativity"), pick("lo"), pick("hi")
        ax.errorbar(range(len(e)), g, yerr=[g - lo_, hi_ - g], fmt="s-", color="tab:blue",
                    capsize=3, label="GLM relativity (95% CI)")
    ax.set_xticks(range(len(e))); ax.set_xticklabels([l.split(": ")[1] for l in e.index], rotation=30)
    ax.axhline(1, color="grey", lw=.7); ax.set_title(f"{band} (base = {base.split(': ')[1]})")
    ax.set_ylabel("relativity vs base level"); ax.legend()
fig.suptitle("GBM SHAP effects are regularised (shrunk): compare direction, not size", fontsize=10)
plt.tight_layout(rect=[0, 0, 1, 0.95]); plt.savefig(RES / "glm_vs_gbm_effects.png", dpi=130)
print("importance:\n", imp.sort_values(ascending=False).round(4).to_string())
