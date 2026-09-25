from __future__ import annotations

import numpy as np

from ..priors import sample_param_dict
from .base import Component, register
from ._helpers import _check_dims, _fill_irrelevant_dims, _resolve_refs


class FractalDust(Component):
    name = "fractal_dust"

    def sample_params(self, rng, n_features, param_spec, parent_params):
        spec = _resolve_refs(param_spec, parent_params)
        s = sample_param_dict(rng, spec)
        i, j = _check_dims(s.get("dims", [0, 1]), n_features, "fractal_dust")
        return {
            "dims": [i, j],
            "cx": float(s.get("cx", 0.0)),
            "cy": float(s.get("cy", 0.0)),
            "base_scale": float(s.get("base_scale", 1.8)),
            "depth": int(s.get("depth", 4)),
            "children_per_level": int(s.get("children_per_level", 3)),
            "decay": float(s.get("decay", 0.5)),
            "jitter": float(s.get("jitter", 0.08)),
            "thickness": float(s.get("jitter", 0.08)),
            "irrelevant": s.get("irrelevant", {"mode": "gaussian_mid", "sigma": float(s.get("other_sigma", 0.25))}),
        }

    def sample_points(self, rng, n_points, n_features, bounds_low, bounds_high, params_used, parent_params=None):
        p = params_used
        i, j = _check_dims(p["dims"], n_features, "fractal_dust")
        centers = [np.array([float(p["cx"]), float(p["cy"])], dtype=float)]
        scale = float(p["base_scale"])
        for _ in range(max(1, int(p["depth"]))):
            new_centers = []
            for c in centers:
                for _ in range(max(1, int(p["children_per_level"]))):
                    ang = rng.uniform(0.0, 2.0 * np.pi)
                    rad = rng.uniform(0.2, 1.0) * scale
                    new_centers.append(c + rad * np.array([np.cos(ang), np.sin(ang)]))
            centers = new_centers
            scale *= float(p["decay"])
        centers = np.asarray(centers, dtype=float)
        if len(centers) == 0:
            centers = np.asarray([[float(p["cx"]), float(p["cy"])]], dtype=float)
        idx = rng.integers(0, len(centers), size=n_points)
        pts = centers[idx] + rng.normal(0.0, max(1e-6, float(p["jitter"])), size=(n_points, 2))
        X = np.zeros((n_points, n_features), dtype=float)
        X[:, i] = pts[:, 0]
        X[:, j] = pts[:, 1]
        _fill_irrelevant_dims(rng, X, bounds_low, bounds_high, (i, j), p.get("irrelevant", None))
        np.clip(X, bounds_low, bounds_high, out=X)
        return X


register(FractalDust())
