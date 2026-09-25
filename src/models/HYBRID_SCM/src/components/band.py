from __future__ import annotations

import numpy as np

from ..priors import sample_param_dict
from .base import Component, register
from ._helpers import (
    _check_dims,
    _fill_irrelevant_dims,
    _generate_intervals_from_spec,
    _quantize,
    _resolve_refs,
    _sample_x,
)


class Band(Component):
    name = "band"

    def sample_params(self, rng, n_features, param_spec, parent_params):
        spec = _resolve_refs(param_spec, parent_params)
        s = sample_param_dict(rng, spec)
        i, j = _check_dims(s.get("dims", [0, 1]), n_features, "band")
        x_low = float(s.get("x_low", -1.0))
        x_high = float(s.get("x_high", 1.0))
        intervals = s.get("intervals", None)
        if intervals is None and s.get("interval_generation", None) is not None:
            intervals = _generate_intervals_from_spec(rng, s["interval_generation"], x_low=x_low, x_high=x_high)
        return {
            "dims": [i, j],
            "slope": float(s.get("slope", 0.0)),
            "intercept": float(s.get("intercept", 0.0)),
            "x_low": x_low,
            "x_high": x_high,
            "intervals": intervals,
            "thickness": float(s.get("thickness", 0.05)),
            "x_sampling": str(s.get("x_sampling", "uniform")).lower(),
            "x_bins_per_interval": int(s.get("x_bins_per_interval", 40)),
            "x_jitter_sigma": float(s.get("x_jitter_sigma", 0.0)),
            "y_quantization_step": s.get("y_quantization_step", None),
            "y_quantization_origin": float(s.get("y_quantization_origin", 0.0)),
            "irrelevant": s.get("irrelevant", {"mode": "gaussian_mid", "sigma": float(s.get("other_sigma", 0.2))}),
        }

    def sample_points(self, rng, n_points, n_features, bounds_low, bounds_high, params_used, parent_params=None):
        p = params_used
        i, j = _check_dims(p["dims"], n_features, "band")
        x = _sample_x(
            rng, n_points,
            float(p["x_low"]), float(p["x_high"]),
            p.get("intervals", None),
            p.get("x_sampling", "uniform"),
            int(p.get("x_bins_per_interval", 40)),
            float(p.get("x_jitter_sigma", 0.0)),
        )
        y = float(p["slope"]) * x + float(p["intercept"])
        y = y + rng.normal(0.0, float(p["thickness"]), size=n_points)
        y = _quantize(y, p.get("y_quantization_step", None), float(p.get("y_quantization_origin", 0.0)))
        X = np.zeros((n_points, n_features), dtype=float)
        X[:, i] = x
        X[:, j] = y
        _fill_irrelevant_dims(rng, X, bounds_low, bounds_high, (i, j), p.get("irrelevant", None))
        np.clip(X, bounds_low, bounds_high, out=X)
        return X


register(Band())
