"""Explain the frequency GBM and check it against the frequency GLM.
SHAP (TreeSHAP) if installed, else permutation importance / partial dependence. For a Poisson model SHAP values
are on the log-rate scale, so exp(SHAP) is a multiplicative effect comparable with GLM relativities.
Writes results/freq_shap_importance.png, freq_shap_beeswarm.png (shap only), freq_glm_vs_gbm_effects.png"""
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import FEATURES, OUT, RES, SEED, load_frequency_frame, poisson_deviance

warnings.filterwarnings("ignore", category=UserWarning)
try:
    import shap
    HAVE_SHAP = True
except ImportError:
    HAVE_SHAP = False
    print("!! `shap` not installed - using permutation importance / partial dependence fallback")

model = joblib.load(OUT / "freq_gbm_model.joblib")
df = load_frequency_frame()
te = df[df.split == "test"].reset_index(drop=True)
sample = te.sample(min(len(te), 20_000), random_state=SEED).reset_index(drop=True)   # SHAP on a sample
X = sample[FEATURES]
RES.mkdir(exist_ok=True)

if HAVE_SHAP:
    sv = shap.TreeExplainer(model).shap_values(X)
    imp = pd.Series(np.abs(sv).mean(0), index=FEATURES).sort_values()
    xlabel = "mean |SHAP|  (log claim-rate units)"
else:
    from sklearn.inspection import permutation_importance
    r = permutation_importance(
        model, X, sample.claim_nb / sample.exposure, sample_weight=sample.exposure, n_repeats=5,
        random_state=SEED,
        scoring=lambda m, X_, y_, sample_weight=None:
            -poisson_deviance(sample.claim_nb, m.predict(X_) * sample.exposure))
    imp = pd.Series(r.importances_mean, index=FEATURES).sort_values()
    xlabel = "increase in Poisson deviance when shuffled"
imp.plot.barh(figsize=(6, 4), title="Frequency GBM feature importance")
plt.xlabel(xlabel); plt.tight_layout(); plt.savefig(RES / "freq_shap_importance.png", dpi=130); plt.close()
imp.sort_values(ascending=False).to_csv(OUT / "freq_gbm_importance.csv", header=["importance"])
if HAVE_SHAP:
    shap.summary_plot(sv, X, show=False)
    plt.tight_layout(); plt.savefig(RES / "freq_shap_beeswarm.png", dpi=130); plt.close()

BANDS = {"driv_age_band": "driv_age_c", "veh_age_band": "veh_age_c", "bm_band": "bonus_malus_c"}
rel_path = OUT / "freq_glm_relativities.csv"
glm_tab = pd.read_csv(rel_path).set_index("term") if rel_path.exists() else None
if glm_tab is None:
    print("!! outputs/freq_glm_relativities.csv not found (R not run): plotting GBM effects only")


def gbm_effect(band, feat):
    levels = sorted(sample[band].unique())
    if HAVE_SHAP:
        return pd.Series({l: sv[sample[band] == l, FEATURES.index(feat)].mean() for l in levels})
    eff = {}
    for l in levels:
        Xl = X.copy(); Xl[feat] = sample.loc[sample[band] == l, feat].median()
        eff[l] = np.log(model.predict(Xl)).mean()
    return pd.Series(eff)


fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (band, feat) in zip(axes, BANDS.items()):
    e = gbm_effect(band, feat)
    base = sample.loc[sample[band].map(sample.groupby(band).exposure.sum()).idxmax(), band]   # most-exposed = GLM base
    ax.plot(range(len(e)), np.exp(e - e[base]).values, "^--", color="tab:orange",
            label="GBM (" + ("SHAP" if HAVE_SHAP else "PDP") + ")")
    if glm_tab is not None:
        def pick(col):
            return np.array([1.0 if l == base else glm_tab[col].get(f"{band}{l}", np.nan) for l in e.index])
        g, lo_, hi_ = pick("relativity"), pick("lo"), pick("hi")
        ax.errorbar(range(len(e)), g, yerr=[g - lo_, hi_ - g], fmt="s-", color="tab:blue",
                    capsize=3, label="GLM relativity (95% CI)")
    ax.set_xticks(range(len(e))); ax.set_xticklabels([l.split(": ")[1] for l in e.index], rotation=30)
    ax.axhline(1, color="grey", lw=.7); ax.set_title(f"{band} (base = {base.split(': ')[1]})")
    ax.set_ylabel("relativity in claim frequency"); ax.legend()
fig.suptitle("GBM effects are regularised (shrunk): compare direction, not size", fontsize=10)
plt.tight_layout(rect=[0, 0, 1, 0.95]); plt.savefig(RES / "freq_glm_vs_gbm_effects.png", dpi=130)
print("importance:\n", imp.sort_values(ascending=False).round(5).to_string())
