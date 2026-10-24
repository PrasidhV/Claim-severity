"""Settle which interaction screening (R add1 vs Python Rao) is right, using ground truth.
For all 15 candidate pairs: Python Rao score, R Rao score (from outputs/freq_glm_plus_interaction_scores.csv),
and the REAL in-sample deviance drop from refitting with that pair. Then fits the top-2 picked by each side and
reports in-sample and held-out deviance. Takes ~5-10 minutes on the real data (16 GLM fits)."""
import numpy as np
import pandas as pd

import glm_plus as gp
from common import OUT, load_frequency_frame, poisson_deviance

df = gp.prep(load_frequency_frame())
fit_df, test_df = df[df.split.isin(["train", "valid"])], df[df.split == "test"]
base, spec, X = gp.fit(fit_df)
py = gp.rao_scores(fit_df, spec, X, base)

rows = []
for pr in gp.PAIRS:
    m, _, _ = gp.fit(fit_df, [pr], spec)
    rows.append(dict(pair=" x ".join(pr), python_rao=py[pr], true_dev_drop=base["dev"] - m["dev"]))
t = pd.DataFrame(rows)

r_path = OUT / "freq_glm_plus_interaction_scores.csv"
r_top = None
if r_path.exists():
    r = pd.read_csv(r_path)
    r["pair"] = r.interaction.str.replace("ix_", "", n=1, regex=False).str.replace("__", " x ", regex=False)
    t = t.merge(r[["pair", "rao_score"]].rename(columns={"rao_score": "r_rao"}), on="pair", how="left")
    r_top = [tuple(p.split(" x ")) for p in t.sort_values("r_rao", ascending=False).pair[:2]]
for c in ["python_rao", "r_rao", "true_dev_drop"]:
    if c in t:
        t[c + "_rank"] = t[c].rank(ascending=False).astype(int)
t = t.sort_values("true_dev_drop", ascending=False)
pd.set_option("display.width", 200)
print("\nCandidate interactions, sorted by REAL in-sample deviance drop (ground truth):\n")
print(t.round(2).to_string(index=False))
print(f"\ncorr(Python Rao, truth) = {t.python_rao.corr(t.true_dev_drop):.4f}")
if "r_rao" in t:
    print(f"corr(R Rao,      truth) = {t.r_rao.corr(t.true_dev_drop):.4f}")

py_top = sorted(py, key=py.get, reverse=True)[:2]
print("\nTop-2 fitted for real:")
for name, pairs in [("Python picks", py_top), ("R picks", r_top)]:
    if pairs is None:
        continue
    m, s, _ = gp.fit(fit_df, pairs, spec)
    pred = gp.predict(m, s, test_df, pairs)
    print(f"  {name:13s} {[' x '.join(p) for p in pairs]}  in-sample dev={m['dev']:.1f}  "
          f"test mean Poisson deviance={poisson_deviance(test_df.claim_nb, pred):.5f}")
