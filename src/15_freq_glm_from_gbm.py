"""Can a few GBM-identified interactions close the GLM-GBM gap?  (single split, DIAGNOSTIC)

Starts from GLM+ (splines + its two Rao-screened numeric interactions, fitted on train+valid) and adds
categorical interactions suggested by src/14_freq_interactions.py, one at a time and all together:
    brand x vehicle-age band | region x log-density | region x driver-age band | driver age x bonus-malus
Scored once on the untouched test set, with paired-bootstrap CIs vs GLM+ and vs the GBM.
CAVEAT: the terms were picked using the GBM's behaviour on the VALIDATION split (no test data), so this is an
upper-bound diagnostic on how far a hand-built GLM can get, not a nested/honest CV estimate of a final model."""
import numpy as np
import pandas as pd

import glm_plus as gp
import poisson_glm as pg
from common import OUT, RES, SEED, is_synthetic, load_frequency_frame, poisson_dev_terms

TERMS = {   # name -> (kind, a, b)
    "brand x vehicle-age band": ("cat_cat", "veh_brand", "veh_age_band"),
    "region x log-density": ("cat_num", "region", "log_density"),
    "region x driver-age band": ("cat_cat", "region", "driv_age_band"),
    "driver age x bonus-malus": ("num_num", "driv_age_c", "bonus_malus_c"),
}


def levels(tr, c):
    return sorted(tr[c].astype(str).unique())[1:]            # all but the first (reference) level


def extra_cols(df, tr, spec, kind, a, b):
    if kind == "num_num":
        return [gp.z(df, spec, a) * gp.z(df, spec, b)]
    da = [(df[a].astype(str).values == l).astype(float) for l in levels(tr, a)]
    if kind == "cat_num":
        zb = gp.z(df, spec, b)
        return [d * zb for d in da]
    db = [(df[b].astype(str).values == l).astype(float) for l in levels(tr, b)]
    return [x * y for x in da for y in db]


def fit_with(tr, base_pairs, spec, names):
    def X_of(d):
        cols = [gp.build(d, spec, base_pairs)]
        for n in names:
            cols.append(np.column_stack(extra_cols(d, tr, spec, *TERMS[n])))
        return np.hstack(cols)
    m = pg.fit_poisson_log(X_of(tr), tr.claim_nb.values.astype(float), tr.exposure.values)
    return m, X_of


if __name__ == "__main__":
    df = gp.prep(load_frequency_frame())
    for c in ["veh_age_band", "driv_age_band", "bm_band"]:
        df[c] = df[c].astype(str)
    fit_df, test_df = df[df.split.isin(["train", "valid"])], df[df.split == "test"]
    y = test_df.claim_nb.values.astype(float)
    plus = gp.fit_plus(fit_df, k=2)
    base_pairs, spec = plus["pairs"], plus["spec"]
    print("GLM+ interactions:", [" x ".join(p) for p in base_pairs])

    preds = {"GLM+ (screened numeric pairs)": gp.predict(plus["model"], spec, test_df, base_pairs)}
    info = {"GLM+ (screened numeric pairs)": (plus["model"]["rank"], plus["model"]["dev"])}
    configs = [(n, [n]) for n in TERMS] + [("ALL four added", list(TERMS))]
    for label, names in configs:
        m, X_of = fit_with(fit_df, base_pairs, spec, names)
        preds["GLM+ + " + label] = np.exp(np.clip(X_of(test_df) @ m["beta"], -30, 30)) * test_df.exposure.values
        info["GLM+ + " + label] = (m["rank"], m["dev"])
        print(f"fitted: {label} (rank {m['rank']})")

    refs = {}
    band_file = "freq_glm_r_test_preds.csv" if (OUT / "freq_glm_r_test_preds.csv").exists() else "freq_glm_pycheck_test_preds.csv"
    for name, f in [("banded GLM", band_file), ("GBM", "freq_gbm_test_preds.csv")]:
        p = pd.read_csv(OUT / f).set_index("idpol").loc[test_df.idpol]
        refs[name] = p.pred.values
    terms = {k: poisson_dev_terms(y, v) for k, v in {**refs, **preds}.items()}
    d_band, d_gbm = terms["banded GLM"].mean(), terms["GBM"].mean()
    d_plus = terms["GLM+ (screened numeric pairs)"].mean()
    rng = np.random.default_rng(SEED)
    idx = [rng.integers(0, len(y), len(y)) for _ in range(1000)]

    rows = []
    for k, t in terms.items():
        d = t.mean()
        row = dict(model=k, params=info.get(k, (np.nan,))[0], test_deviance=d,
                   share_of_gap_closed=(d_band - d) / (d_band - d_gbm))
        if k.startswith("GLM+ +"):
            diff = np.array([(terms["GLM+ (screened numeric pairs)"][i] - t[i]).mean() for i in idx])
            lo, hi = np.percentile(diff, [2.5, 97.5])
            row["gain_vs_GLM+ (95% CI)"] = f"{d_plus - d:+.5f} ({lo:+.5f}, {hi:+.5f})"
        rows.append(row)
    res = pd.DataFrame(rows)
    if d_band - d_gbm < 0.001:                       # GBM has no clear edge: the ratio would be meaningless
        res["share_of_gap_closed"] = np.nan
    md = ["# Frequency: can GBM-identified interactions close the gap?  (single split, diagnostic)\n",
          f"Data: **{'SYNTHETIC - pipeline test only, not a finding' if is_synthetic() else 'real freMTPL2 (public)'}**. "
          "Mean Poisson deviance per policy on the test set (lower is better). "
          "`share_of_gap_closed` = (banded GLM - model) / (banded GLM - GBM).\n",
          res.to_markdown(index=False, floatfmt=".5f"), "",
          "Positive `gain_vs_GLM+` with a CI above zero means the added interaction helps on unseen data by more than sampling noise. "
          "The terms were chosen from the GBM's behaviour on the validation split, so this is an upper-bound diagnostic, "
          "not a nested-CV estimate. Parameter counts show the price paid: a hand-built GLM needs many extra terms.\n"]
    RES.mkdir(exist_ok=True)
    (RES / "freq_glm_from_gbm.md").write_text("\n".join(md))
    print("\n" + "\n".join(md))
