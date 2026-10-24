"""Better-specified frequency GLM ("GLM+"): piecewise-linear splines + screened interactions.

Main effects: for driver age, bonus-malus, vehicle age and log-density a linear term plus hinge terms
max(x - knot, 0) at fixed knots (a continuous piecewise-linear spline), a bonus-malus == 50 indicator, and
the usual factors for gas / area / vehicle power / brand / region.
Interactions: the top-K of all pairwise products of six standardised variables, ranked by the Rao SCORE test
on the fitting data only (so the GBM never influences the GLM). Mirrors r/12_freq_glm_plus.R.
"""
import itertools

import numpy as np

import poisson_glm as pg

SPLINE_KNOTS = {"driv_age_c": [21, 25, 30, 40, 50, 60, 70],
                "bonus_malus_c": [60, 70, 80, 90, 100, 120],
                "veh_age_c": [1, 2, 4, 7, 11, 15],
                "log_density": [3, 4.5, 6, 7.5, 9]}
CATS = ["veh_gas", "area", "veh_power_g", "veh_brand", "region"]
IX_VARS = ["driv_age_c", "bonus_malus_c", "veh_age_c", "log_density", "veh_power", "gas_diesel"]
PAIRS = list(itertools.combinations(IX_VARS, 2))


def prep(df):
    d = df.copy()
    d["gas_diesel"] = (d["veh_gas"].astype(str) == "Diesel").astype(float)
    d["bm50"] = (d["bonus_malus_c"] == 50).astype(float)
    for c in CATS:
        d[c] = d[c].astype(str)
    return d


def make_spec(tr):
    spec = {"num": {v: (tr[v].mean(), tr[v].std()) for v in set(SPLINE_KNOTS) | set(IX_VARS)}}
    spec["cat"] = pg.make_spec(tr, CATS)
    return spec


def z(df, spec, v):
    m, s = spec["num"][v]
    return ((df[v].values - m) / s).astype(float)


def build(df, spec, pairs=()):
    cols = [np.ones(len(df))]
    for v, knots in SPLINE_KNOTS.items():
        m, s = spec["num"][v]
        x = df[v].values
        cols.append((x - m) / s)
        cols += [np.maximum(x - k, 0) / s for k in knots]
    cols.append(df["bm50"].values)
    for c in CATS:
        vals = df[c].values
        cols += [(vals == lvl).astype(float) for lvl in spec["cat"][c][1:]]   # unseen level -> reference
    cols += [z(df, spec, a) * z(df, spec, b) for a, b in pairs]
    return np.column_stack(cols)


def fit(tr, pairs=(), spec=None):
    spec = spec or make_spec(tr)
    X = build(tr, spec, pairs)
    return pg.fit_poisson_log(X, tr.claim_nb.values.astype(float), tr.exposure.values), spec, X


def predict(model, spec, df, pairs=()):
    return np.exp(np.clip(build(df, spec, pairs) @ model["beta"], -30, 30)) * df.exposure.values


def rao_scores(tr, spec, X, model):
    """Rao score statistic for adding each candidate pair-product to the fitted base model."""
    y, e = tr.claim_nb.values.astype(float), tr.exposure.values
    mu = np.exp(np.clip(X @ model["beta"], -30, 30)) * e
    C = np.column_stack([z(tr, spec, a) * z(tr, spec, b) for a, b in PAIRS])
    A, _ = pg._normal_eq(X, mu, np.zeros(len(y)))
    Ainv = np.linalg.pinv(A)
    u = C.T @ (y - mu)
    icc = (C ** 2 * mu[:, None]).sum(0)
    icx = X.T @ (C * mu[:, None])
    resid = icc - np.einsum("ij,ik,kj->j", icx, Ainv, icx)
    return dict(zip(PAIRS, u ** 2 / resid))


def screen(tr, spec, X, model, k=2):
    sc = rao_scores(tr, spec, X, model)
    return sorted(sc, key=sc.get, reverse=True)[:k], sc


def fit_plus(tr, k=2):
    """Base splines fit -> screen interactions on `tr` -> refit with the top k. Returns everything."""
    base, spec, X = fit(tr)
    pairs, scores = screen(tr, spec, X, base, k)
    model, _, _ = fit(tr, pairs, spec)
    return dict(base=base, model=model, spec=spec, pairs=pairs, scores=scores)
