"""Shared paths, feature lists and metrics used by every Python step."""
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "claims.db"
RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs"
RES = ROOT / "results"
SEED = 42

# Features given to the gradient boosting model: raw-ish, no manual banding needed.
NUM = ["veh_power", "veh_age_c", "driv_age_c", "bonus_malus_c", "log_density", "area_ord"]
CAT = ["veh_brand", "veh_gas", "region"]
FEATURES = NUM + CAT

# Features given to the GLM: the banded / grouped versions an actuary would use.
GLM_CATS = ["driv_age_band", "veh_age_band", "bm_band", "veh_gas",
            "area", "veh_power_g", "veh_brand", "region"]
GLM_NUM = ["log_density"]


def is_synthetic() -> bool:
    return (RAW / "SYNTHETIC").exists()


def load_model_frame() -> pd.DataFrame:
    """Read the SQL-built modelling table (policies with >= 1 claim)."""
    with sqlite3.connect(DB) as con:
        df = pd.read_sql("SELECT * FROM model_severity ORDER BY idpol", con)
    df["log_density"] = np.log(df["density"])
    df["area_ord"] = df["area"].map({c: i for i, c in enumerate("ABCDEF")})
    for c in CAT:
        df[c] = df[c].astype("category")
    return df


def gamma_dev_terms(y, mu, w):
    """Per-row Gamma deviance contributions (weighted)."""
    y, mu, w = (np.asarray(a, float) for a in (y, mu, w))
    return 2.0 * w * ((y - mu) / mu - np.log(y / mu))


def gamma_deviance(y, mu, w):
    """Weighted mean Gamma deviance (lower is better)."""
    w = np.asarray(w, float)
    return gamma_dev_terms(y, mu, w).sum() / w.sum()
