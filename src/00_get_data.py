"""Get the data into data/raw/.

Real data  : freMTPL2freq (OpenML 41214) + freMTPL2sev (OpenML 41215), public, ~678k policies.
             python src/00_get_data.py
Offline    : python src/00_get_data.py --synthetic
             Writes a *synthetic* dataset with the same schema (including deliberately dirty
             rows) so the pipeline can be tested without internet. Results from it are NOT
             findings about real insurance data; a data/raw/SYNTHETIC marker file flags this.
"""
import argparse

import numpy as np
import pandas as pd

from common import RAW, SEED


def get_real():
    from sklearn.datasets import fetch_openml
    freq = fetch_openml(data_id=41214, as_frame=True, parser="auto").frame
    sev = fetch_openml(data_id=41215, as_frame=True, parser="auto").frame
    return freq, sev


def get_synthetic(n=150_000):
    rng = np.random.default_rng(SEED)
    idpol = np.arange(1, n + 1)
    exposure = np.clip(rng.beta(2, 1.5, n), 0.01, 1.0)
    area = rng.choice(list("ABCDEF"), n, p=[.15, .2, .2, .2, .15, .1])
    veh_power = rng.integers(4, 16, n)
    veh_age = np.clip(rng.gamma(2.5, 3.0, n).astype(int), 0, 25)
    driv_age = np.clip(rng.normal(45, 14, n).astype(int), 18, 95)
    bm = np.clip(50 + rng.exponential(12, n) * (rng.random(n) < 0.45), 50, 230).astype(int)
    brand = rng.choice([f"B{i}" for i in (1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14)], n)
    gas = rng.choice(["Diesel", "Regular"], n)
    density = np.exp(rng.normal(6.0, 1.7, n)).round(0) + 1
    region = rng.choice([f"R{i}" for i in (11, 24, 25, 31, 52, 53, 54, 72, 82, 93)], n)

    # Frequency (so we get enough claims) and a *known* severity structure incl. an
    # interaction (young driver x powerful car) and a non-linear bonus-malus effect.
    lam = exposure * 0.09 * np.exp(0.015 * (bm - 50) + 0.4 * (driv_age < 25))
    nclaims = rng.poisson(lam)
    log_mu = (7.2
              + 0.30 * (driv_age < 25) - 0.10 * (driv_age > 65)
              + 0.025 * np.minimum(veh_age, 12)
              + 0.04 * (veh_power - 8)
              + 0.35 * ((driv_age < 28) & (veh_power >= 10))
              + 0.012 * np.maximum(bm - 90, 0)
              + 0.05 * (np.log(density) - 6) + 0.10 * (gas == "Diesel"))
    amounts, owners = [], []
    for i in np.flatnonzero(nclaims):
        k = nclaims[i]
        a = rng.gamma(1.2, np.exp(log_mu[i]) / 1.2, k)
        a *= np.where(rng.random(k) < 0.01, rng.uniform(15, 60, k), 1.0)   # heavy-tail contamination
        amounts.extend(a.round(2)); owners.extend([idpol[i]] * k)

    freq = pd.DataFrame(dict(IDpol=idpol, ClaimNb=nclaims, Exposure=exposure.round(4),
                             Area=area, VehPower=veh_power, VehAge=veh_age, DrivAge=driv_age,
                             BonusMalus=bm, VehBrand=brand, VehGas=gas, Density=density,
                             Region=region))
    sev = pd.DataFrame(dict(IDpol=owners, ClaimAmount=amounts))

    # ---- inject realistic mess for the SQL cleaning step to deal with ----
    r = rng.random
    freq.loc[r(n) < 0.004, "Exposure"] = 2.0                    # exposure > 1 year
    freq.loc[r(n) < 0.001, "Exposure"] = 0.0                    # zero exposure
    freq.loc[r(n) < 0.002, "VehAge"] = 100                      # absurd vehicle age
    freq.loc[r(n) < 0.001, "DrivAge"] = 17                      # underage driver
    for c in ("Area", "VehBrand", "VehGas", "Region"):          # OpenML-style quoted strings
        freq[c] = "'" + freq[c] + "'"
    freq = pd.concat([freq, freq.sample(300, random_state=1)])  # duplicate policy rows
    drop_idx = rng.choice(sev.index, 40, replace=False)         # claims whose policy row is lost
    sev = pd.concat([sev, pd.DataFrame(dict(IDpol=freq.IDpol.sample(60, random_state=2).values,
                                            ClaimAmount=-1.0))])   # non-positive amounts
    sev = sev.drop(index=drop_idx)
    return freq, sev


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    marker = RAW / "SYNTHETIC"
    freq, sev = get_synthetic() if a.synthetic else get_real()
    freq.to_csv(RAW / "freMTPL2freq.csv", index=False)
    sev.to_csv(RAW / "freMTPL2sev.csv", index=False)
    if a.synthetic:
        marker.write_text("synthetic data - pipeline test only\n")
    elif marker.exists():
        marker.unlink()
    print(f"freq {freq.shape}, sev {sev.shape}  ({'SYNTHETIC' if a.synthetic else 'real freMTPL2'})")
