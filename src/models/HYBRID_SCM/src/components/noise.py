from __future__ import annotations

import numpy as np

from ..priors import sample_param_dict
from .base import Component, register
from ._helpers import (
    _check_dims,
    _eps_gaussian_or_donut,
    _resolve_refs,
    _roi_or_global,
    _sample_x,
)


class Noise(Component):
    name = "noise"

    def sample_params(self, rng, n_features, param_spec, parent_params):
        spec = _resolve_refs(param_spec, parent_params)
        s = sample_param_dict(rng, spec)
        return {
            "mode": str(s.get("mode", "uniform")).lower(),
            "low": s.get("low", None),
            "high": s.get("high", None),
            "roi": s.get("roi", None),
            "strength": float(s.get("strength", 0.25)),
            "min_abs_offset": None if s.get("min_abs_offset", None) is None else float(s.get("min_abs_offset")),
            "max_abs_offset": None if s.get("max_abs_offset", None) is None else float(s.get("max_abs_offset")),
            "x_low": s.get("x_low", None),
            "x_high": s.get("x_high", None),
            "side": str(s.get("side", "below")).lower(),
            "offset_min": float(s.get("offset_min", 1.0)),
            "offset_max": float(s.get("offset_max", 6.0)),
            "y_jitter_sigma": float(s.get("y_jitter_sigma", 1.0)),
            "x_source": str(s.get("x_source", "parent")).lower(),
            "x_expand": float(s.get("x_expand", 0.0)),
            "intervals": s.get("intervals", None),
        }

    def sample_points(self, rng, n_points, n_features, bounds_low, bounds_high, params_used, parent_params=None):
        p = params_used
        parent_params = parent_params or {}
        roi_low, roi_high = _roi_or_global(bounds_low, bounds_high, p.get("roi", None))
        mode = p["mode"]

        if mode == "uniform":
            low = p.get("low", None)
            high = p.get("high", None)
            low_v = roi_low if low is None else np.array(low, dtype=float)
            high_v = roi_high if high is None else np.array(high, dtype=float)
            X = rng.uniform(low_v, high_v, size=(n_points, n_features))
            np.clip(X, bounds_low, bounds_high, out=X)
            return X

        if len(parent_params) != 1:
            raise ValueError(f"noise mode '{mode}' requires exactly one parent.")
        parent = list(parent_params.values())[0]
        i, j = _check_dims(parent["dims"], n_features, f"{mode}(parent)")
        m = float(parent.get("slope", 0.0))

        if "intercept" in parent:
            b_vec = np.full((n_points,), float(parent["intercept"]), dtype=float)
            intervals_parent = parent.get("intervals", None)
        elif "base_intercept" in parent and "n_rungs" in parent and "rung_spacing" in parent:
            n_rungs = max(1, int(parent["n_rungs"]))
            rung_idx = rng.integers(0, n_rungs, size=n_points)
            b_vec = float(parent["base_intercept"]) + rung_idx * float(parent["rung_spacing"])
            intervals_parent = None
        elif "intercept0" in parent and "n_bands" in parent and "spacing" in parent:
            n_bands = max(1, int(parent["n_bands"]))
            band_idx = rng.integers(0, n_bands, size=n_points)
            b_vec = float(parent["intercept0"]) + band_idx * float(parent["spacing"])
            intervals_parent = parent.get("intervals", None)
        else:
            raise ValueError("Parent for near_band/under_band_cloud must be band/ladder/parallel_bands-like.")

        x_low = float(p["x_low"] if p["x_low"] is not None else parent.get("x_low", roi_low[i]))
        x_high = float(p["x_high"] if p["x_high"] is not None else parent.get("x_high", roi_high[i]))
        intervals_use = p.get("intervals", None) if p.get("intervals", None) is not None else intervals_parent
        x = _sample_x(rng, n_points, x_low, x_high, intervals_use, "uniform", 20, 0.0)

        X = rng.uniform(roi_low, roi_high, size=(n_points, n_features))
        X[:, i] = x

        if mode == "near_band":
            eps = _eps_gaussian_or_donut(
                rng=rng,
                n=n_points,
                sigma=float(p["strength"]),
                min_abs_offset=p.get("min_abs_offset", None),
                max_abs_offset=p.get("max_abs_offset", None),
            )
            y = m * x + b_vec + eps
            X[:, j] = y
            np.clip(X, bounds_low, bounds_high, out=X)
            return X

        if mode == "under_band_cloud":
            sign = -1.0 if str(p.get("side", "below")).lower() in ("below", "under") else 1.0
            offset = rng.uniform(float(p.get("offset_min", 1.0)), float(p.get("offset_max", 6.0)), size=n_points)
            jitter = rng.normal(0.0, float(p.get("y_jitter_sigma", 1.0)), size=n_points)
            y = m * x + b_vec + sign * offset + jitter
            X[:, j] = y
            np.clip(X, bounds_low, bounds_high, out=X)
            return X

        raise ValueError(f"Unknown noise mode '{mode}'. Supported: uniform, near_band, under_band_cloud")


register(Noise())
