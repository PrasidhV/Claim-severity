"""Python re-implementation of the R GLM (unpenalised Gamma, log link, same banded features).

Purpose: (1) sanity-check that the R GLM is correct (test deviance should match to ~3 d.p.);
         (2) lets 04_compare.py run end-to-end on a machine without R.
The R model remains the 'official' baseline; this file is only labelled as a cross-check.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import GammaRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import GLM_CATS, GLM_NUM, OUT, gamma_deviance, load_model_frame

df = load_model_frame()
for c in GLM_CATS:
    df[c] = df[c].astype(str)
fit, test = df[df.split.isin(["train", "valid"])], df[df.split == "test"]

pre = ColumnTransformer([
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), GLM_CATS),
    ("num", StandardScaler(), GLM_NUM)])
glm = make_pipeline(pre, GammaRegressor(alpha=0.0, max_iter=2000, tol=1e-8))
glm.fit(fit[GLM_CATS + GLM_NUM], fit.avg_severity, gammaregressor__sample_weight=fit.n_claims)

pred = glm.predict(test[GLM_CATS + GLM_NUM])
print(f"GLM (python cross-check) test weighted mean Gamma deviance: "
      f"{gamma_deviance(test.avg_severity, pred, test.n_claims):.4f}")
OUT.mkdir(exist_ok=True)
pd.DataFrame(dict(idpol=test.idpol, y=test.avg_severity, w=test.n_claims, pred=pred)
             ).to_csv(OUT / "glm_pycheck_test_preds.csv", index=False)
