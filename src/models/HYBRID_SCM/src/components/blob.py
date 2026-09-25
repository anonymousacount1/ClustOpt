from __future__ import annotations

import numpy as np

from ..priors import sample_param_dict
from .base import Component, register
from ._helpers import _resolve_refs


def _is_numeric_scalar(x):
    return isinstance(x, (int, float, np.integer, np.floating))


def _coerce_blob_sigma(rng, sigma_raw, n_features):
    """
    Supports:
      - scalar: 0.25
      - scalar interval: [0.1, 0.3]  -> sampled once
      - per-dim vector: [0.1, 0.3] when len == n_features
      - per-dim intervals: [[0.05, 0.10], [0.20, 0.40]]
      - mixed per-dim: [0.1, [0.2, 0.4]]
    Returns either:
      - float
      - list[float] of length n_features
    """
    if _is_numeric_scalar(sigma_raw):
        return float(abs(sigma_raw))

    if isinstance(sigma_raw, np.ndarray):
        sigma_raw = sigma_raw.tolist()

    if not isinstance(sigma_raw, (list, tuple)):
        raise ValueError("blob.sigma must be a number, list, or interval specification.")

    sigma_list = list(sigma_raw)
    if len(sigma_list) == 0:
        raise ValueError("blob.sigma cannot be empty.")

    if all(_is_numeric_scalar(v) for v in sigma_list):
        vals = [float(abs(v)) for v in sigma_list]

        if len(vals) == n_features:
            return vals

        if len(vals) == 2:
            lo, hi = vals
            if hi < lo:
                lo, hi = hi, lo
            return float(rng.uniform(lo, hi))

        return float(np.mean(vals))

    parsed = []
    for v in sigma_list:
        if _is_numeric_scalar(v):
            parsed.append(float(abs(v)))
        elif isinstance(v, np.ndarray):
            v = v.tolist()
            if len(v) == 2 and all(_is_numeric_scalar(t) for t in v):
                lo, hi = float(abs(v[0])), float(abs(v[1]))
                if hi < lo:
                    lo, hi = hi, lo
                parsed.append(float(rng.uniform(lo, hi)))
            else:
                raise ValueError("blob.sigma ndarray entries must be scalars or [low, high] intervals.")
        elif isinstance(v, (list, tuple)):
            if len(v) == 2 and all(_is_numeric_scalar(t) for t in v):
                lo, hi = float(abs(v[0])), float(abs(v[1]))
                if hi < lo:
                    lo, hi = hi, lo
                parsed.append(float(rng.uniform(lo, hi)))
            else:
                raise ValueError("blob.sigma nested entries must be scalars or [low, high] intervals.")
        else:
            raise ValueError("blob.sigma contains unsupported values.")

    if len(parsed) == n_features:
        return parsed

    if len(parsed) == 1:
        return float(parsed[0])

    raise ValueError(
        f"blob.sigma resolved to {len(parsed)} values, but expected either 1 or n_features={n_features}.")


class Blob(Component):
    name = "blob"

    def sample_params(self, rng, n_features, param_spec, parent_params):
        spec = _resolve_refs(param_spec, parent_params)
        s = sample_param_dict(rng, spec)

        mu = s.get("mu", None)
        if mu is None:
            mu = rng.normal(0, 1, size=n_features).tolist()
        if len(mu) != n_features:
            raise ValueError("blob.mu must have length n_features.")

        sigma = _coerce_blob_sigma(rng, s.get("sigma", 0.25), n_features)

        return {
            "mu": [float(x) for x in mu],
            "sigma": sigma,
        }

    def sample_points(self, rng, n_points, n_features, bounds_low, bounds_high, params_used, parent_params=None):
        mu = np.array(params_used["mu"], dtype=float)
        sigma_raw = params_used["sigma"]

        if isinstance(sigma_raw, (list, tuple, np.ndarray)):
            sigma = np.array(sigma_raw, dtype=float)
            if sigma.shape != (n_features,):
                raise ValueError(f"blob.sigma vector must have shape ({n_features},), got {sigma.shape}.")
            sigma = np.maximum(sigma, 1e-12)
        else:
            sigma = max(float(sigma_raw), 1e-12)

        X = rng.normal(mu, sigma, size=(n_points, n_features))
        np.clip(X, bounds_low, bounds_high, out=X)
        return X


register(Blob())
