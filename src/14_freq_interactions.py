"""Which interactions does the frequency GBM rely on?  (validation split only, no test data)

Two views, both on the LOG-RATE scale, because a log-link GLM is additive there: an 'interaction' is exactly
what a GLM with main effects only cannot express.
  1. Friedman H-statistic from partial dependence (model-agnostic, always runs):
       resid_jk(i) = PD_jk(i) - PD_j(i) - PD_k(i)        [all centred]
       strength    = sqrt(mean(resid^2))                  typical log-rate departure from additivity (0.05 ~ 5%)
       H2          = sum(resid^2) / sum(PD_jk^2)          share of the pair's joint effect that is non-additive
  2. SHAP interaction values (only if `shap` can process this model; otherwise skipped with a note).
Also reports each variable's main-effect strength for scale, and plots where the non-additivity sits.
Usage: python src/14_freq_interactions.py [--n 800] [--n-shap 1500]
Writes results/freq_interactions.md, freq_interactions.png, freq_interactions_patterns.png, outputs/freq_interactions.csv"""
import argparse
import itertools
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import FEATURES, OUT, RES, SEED, is_synthetic, load_frequency_frame

warnings.filterwarnings("ignore")


def center(v):
    return v - v.mean()


def pd_vec(model, S, cols):
    """Partial dependence (log predicted rate) evaluated at each row of S, averaged over S as the reference set."""
    n = len(S)
    A, B = np.tile(np.arange(n), n), np.repeat(np.arange(n), n)      # reference rows / evaluation points
    X = S.iloc[A].reset_index(drop=True)
    for c in cols:
        X[c] = S[c].iloc[B].reset_index(drop=True)
    return np.log(model.predict(X)).reshape(n, n).mean(axis=1)


def interaction_table(model, S):
    cols = list(S.columns)
    pd1 = {c: center(pd_vec(model, S, [c])) for c in cols}
    main = {c: float(np.sqrt((pd1[c] ** 2).mean())) for c in cols}
    rows, resid = [], {}
    for j, k in itertools.combinations(cols, 2):
        pjk = center(pd_vec(model, S, [j, k]))
        r = pjk - pd1[j] - pd1[k]
        resid[(j, k)] = r
        rows.append(dict(pair=f"{j} x {k}", var_a=j, var_b=k, strength=float(np.sqrt((r ** 2).mean())),
                         h2=float((r ** 2).sum() / (pjk ** 2).sum())))
    return pd.DataFrame(rows).sort_values("strength", ascending=False).reset_index(drop=True), main, resid


def bins(S, c, max_bins=6, max_levels=8):
    s = S[c]
    if str(s.dtype) == "category":
        top = s.astype(str).value_counts().index[:max_levels]
        return s.astype(str).where(s.astype(str).isin(top), "other")
    q = pd.qcut(s, max_bins, duplicates="drop")
    return q.astype(str)


def pattern_panel(ax, S, resid, j, k):
    bj, bk = bins(S, j), bins(S, k)
    t = pd.DataFrame(dict(a=bj, b=bk, r=resid)).groupby(["a", "b"]).r.mean().unstack()
    im = ax.imshow(t.values, cmap="RdBu_r", vmin=-np.nanmax(np.abs(t.values)), vmax=np.nanmax(np.abs(t.values)), aspect="auto")
    ax.set_xticks(range(t.shape[1])); ax.set_xticklabels(t.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(t.shape[0])); ax.set_yticklabels(t.index, fontsize=7)
    for (a, b), v in np.ndenumerate(t.values):
        if not np.isnan(v):
            ax.text(b, a, f"{v:+.2f}", ha="center", va="center", fontsize=7)
    ax.set_xlabel(k); ax.set_ylabel(j); ax.set_title(f"{j} x {k}\nlog-rate departure from additivity", fontsize=9)
    return im


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--n-shap", type=int, default=1500)
    a = ap.parse_args()

    model = joblib.load(OUT / "freq_gbm_model.joblib")
    df = load_frequency_frame()
    valid = df[df.split == "valid"]
    sample = valid.sample(min(a.n, len(valid)), random_state=SEED).reset_index(drop=True)
    S = sample[FEATURES]
    print(f"{len(S)} validation policies -> {len(S) ** 2:,} rows per pair; 45 predictions, please wait...")
    tab, main, resid = interaction_table(model, S)

    shap_note = "SHAP interaction values: not run."
    try:
        import shap
        Xs = valid.sample(min(a.n_shap, len(valid)), random_state=SEED)[FEATURES]
        iv = shap.TreeExplainer(model).shap_interaction_values(Xs)
        M = np.abs(iv).mean(0) * 2
        sh = {(FEATURES[i], FEATURES[j]): M[i, j] for i in range(len(FEATURES)) for j in range(i + 1, len(FEATURES))}
        tab["shap_interaction"] = [sh[(r.var_a, r.var_b)] for r in tab.itertuples()]
        shap_note = f"SHAP interaction values computed on {len(Xs):,} validation policies (mean |phi_jk| x 2, log-rate units)."
    except ImportError:
        shap_note = "SHAP interaction values skipped: `shap` not installed."
    except Exception as e:                                   # e.g. categorical splits unsupported by shap's interaction code
        shap_note = f"SHAP interaction values failed ({type(e).__name__}: {str(e)[:120]}); H-statistic results only."
    print(shap_note)

    OUT.mkdir(exist_ok=True); RES.mkdir(exist_ok=True)
    tab.drop(columns=["var_a", "var_b"]).to_csv(OUT / "freq_interactions.csv", index=False)

    # ---- figure 1: strength matrix; figure 2: where the top pairs depart from additivity
    M1 = pd.DataFrame(0.0, index=FEATURES, columns=FEATURES)
    for r in tab.itertuples():
        M1.loc[r.var_a, r.var_b] = M1.loc[r.var_b, r.var_a] = r.strength
    for c in FEATURES:
        M1.loc[c, c] = np.nan
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(M1.values, cmap="viridis")
    ax.set_xticks(range(len(FEATURES))); ax.set_xticklabels(FEATURES, rotation=45, ha="right")
    ax.set_yticks(range(len(FEATURES))); ax.set_yticklabels(FEATURES)
    for (i, j), v in np.ndenumerate(M1.values):
        if not np.isnan(v):
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", color="w", fontsize=7)
    plt.colorbar(im, label="interaction strength (log-rate units)")
    ax.set_title("Non-additive structure in the frequency GBM")
    plt.tight_layout(); plt.savefig(RES / "freq_interactions.png", dpi=130); plt.close()

    top = tab.head(3)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax_, r in zip(axes, top.itertuples()):
        pattern_panel(ax_, S, resid[(r.var_a, r.var_b)], r.var_a, r.var_b)
    plt.tight_layout(); plt.savefig(RES / "freq_interactions_patterns.png", dpi=130); plt.close()

    mt = pd.DataFrame(dict(variable=list(main), main_effect_strength=list(main.values()))
                      ).sort_values("main_effect_strength", ascending=False)
    md = ["# Frequency GBM: which interactions does it use?\n",
          f"Data: **{'SYNTHETIC - pipeline test only, not a finding' if is_synthetic() else 'real freMTPL2 (public)'}**; "
          f"{len(S)} validation policies (no test data used). All quantities are on the log-rate scale: 0.05 means roughly a 5% "
          "departure in claim frequency.\n",
          "## Pairs ranked by non-additive strength\n",
          tab.drop(columns=["var_a", "var_b"]).head(15).to_markdown(index=False, floatfmt=".4f"), "",
          "`strength` = typical departure from log-additivity; `h2` = share of the pair's joint effect that is non-additive "
          "(Friedman H-squared; can be large for weak pairs, so rank by `strength`). " + shap_note + "\n",
          "## For scale: main-effect strength per variable (same units)\n", mt.to_markdown(index=False, floatfmt=".4f"), "",
          "![strength matrix](freq_interactions.png)\n", "![patterns](freq_interactions_patterns.png)\n",
          "Caveats: partial dependence assumes features can vary independently (bonus-malus, driver age and vehicle age are correlated, "
          "so some cells in the pattern plots are sparsely supported); cell means come from a small sample and are noisy.\n"]
    (RES / "freq_interactions.md").write_text("\n".join(md))
    print("\n".join(md[:8]))
