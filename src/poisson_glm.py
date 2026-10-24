"""Poisson / log-link GLM with an exposure offset, fitted by IRLS in plain numpy.

Used by the Python cross-check of the R GLM and by the cross-validation. Same design as R:
reference level = most-exposed level, one-hot for banded factors, standardised numerics.
Normal equations are accumulated in chunks, so 700k policies x ~60 columns stays light on memory.
"""
import numpy as np
from scipy.stats import f as fdist

from common import GLM_CATS, poisson_dev_terms


def make_spec(tr, terms):
    spec = {}
    for t in terms:
        if t in GLM_CATS:
            tot = tr.groupby(tr[t].astype(str))["exposure"].sum().sort_values(ascending=False)
            ref = tot.index[0]
            spec[t] = [ref] + sorted(l for l in tot.index if l != ref)
        else:
            spec[t] = (tr[t].mean(), tr[t].std() or 1.0)
    return spec


def design(df, terms, spec):
    cols = [np.ones(len(df))]
    for t in terms:
        if t in GLM_CATS:
            v = df[t].astype(str).values
            cols += [(v == lvl).astype(float) for lvl in spec[t][1:]]   # unseen level -> reference
        else:
            m, s = spec[t]
            cols.append(((df[t].values - m) / s).astype(float))
    return np.column_stack(cols)


def _normal_eq(X, w, b_vec, chunk=100_000):
    A = np.zeros((X.shape[1], X.shape[1])); b = np.zeros(X.shape[1])
    for i in range(0, len(X), chunk):
        Xi, wi = X[i:i + chunk], w[i:i + chunk]
        A += (Xi * wi[:, None]).T @ Xi
        b += Xi.T @ b_vec[i:i + chunk]
    return A, b


def fit_poisson_log(X, y, expo, max_iter=50, tol=1e-9):
    off = np.log(expo)
    beta = np.zeros(X.shape[1]); beta[0] = np.log(y.sum() / expo.sum())
    eta = X @ beta + off
    dev_old = np.inf
    for _ in range(max_iter):
        mu = np.exp(np.clip(eta, -30, 30))
        z = (eta - off) + (y - mu) / mu                  # working response; working weights = mu
        A, b = _normal_eq(X, mu, mu * z)
        beta = np.linalg.lstsq(A, b, rcond=None)[0]
        eta = X @ beta + off
        mu = np.exp(np.clip(eta, -30, 30))
        dev = poisson_dev_terms(y, mu).sum()
        if abs(dev_old - dev) < tol * (abs(dev) + 1e-12):
            break
        dev_old = dev
    A, _ = _normal_eq(X, mu, np.zeros(len(y)))
    rank = np.linalg.matrix_rank(A)
    df_resid = len(y) - rank
    pearson = ((y - mu) ** 2 / mu).sum() / df_resid      # R's poisson summary() fixes this at 1; we estimate it
    return dict(beta=beta, dev=dev, rank=rank, df_resid=df_resid, phi=pearson)


def fit_glm(tr, terms):
    spec = make_spec(tr, terms)
    return fit_poisson_log(design(tr, terms, spec), tr.claim_nb.values.astype(float),
                           tr.exposure.values), spec


def predict_glm(model, spec, terms, df):
    """Predicted claim COUNTS (rate x exposure)."""
    eta = design(df, terms, spec) @ model["beta"]
    return np.exp(np.clip(eta, -30, 30)) * df.exposure.values


def term_tests(tr, terms):
    """Per-term quasi-Poisson F-tests: deviance change / df, divided by the Pearson dispersion."""
    full, _ = fit_glm(tr, terms)
    out = {}
    for t in terms:
        small, _ = fit_glm(tr, [u for u in terms if u != t])
        ddf = small["df_resid"] - full["df_resid"]
        F = ((small["dev"] - full["dev"]) / ddf) / full["phi"]
        out[t] = fdist.sf(F, ddf, full["df_resid"])
    return out
