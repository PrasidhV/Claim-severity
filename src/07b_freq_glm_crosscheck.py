"""Python re-implementation of r/07_freq_glm.R (Poisson, log link, exposure offset, same banded factors).
Test deviance should match R to ~4 decimals; if not, one of them has a bug. Also lets 09_freq_compare.py
run on a machine without R."""
import pandas as pd

import poisson_glm as pg
from common import GLM_CATS, GLM_NUM, OUT, load_frequency_frame, poisson_deviance

df = load_frequency_frame()
for c in GLM_CATS:
    df[c] = df[c].astype(str)
fit, test = df[df.split.isin(["train", "valid"])], df[df.split == "test"]
terms = GLM_CATS + GLM_NUM
model, spec = pg.fit_glm(fit, terms)
pred = pg.predict_glm(model, spec, terms, test)
print(f"FREQ GLM (python cross-check) test mean Poisson deviance: "
      f"{poisson_deviance(test.claim_nb, pred):.5f} | Pearson dispersion {model['phi']:.3f}")
OUT.mkdir(exist_ok=True)
pd.DataFrame(dict(idpol=test.idpol, y=test.claim_nb, exposure=test.exposure, pred=pred)
             ).to_csv(OUT / "freq_glm_pycheck_test_preds.csv", index=False)
