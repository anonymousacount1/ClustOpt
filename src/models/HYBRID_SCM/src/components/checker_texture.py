from __future__ import annotations

import numpy as np

from ..priors import sample_param_dict
from .base import Component, register
from ._helpers import (
    _check_dims,
    _fill_irrelevant_dims,
    _resolve_refs,
    _rotation_matrix,
    _sample_rotated_rect,
)


class CheckerTexture(Component):
    name = "checker_texture"

    def sample_params(self, rng, n_features, param_spec, parent_params):
        spec = _resolve_refs(param_spec, parent_params)
        s = sample_param_dict(rng, spec)
        i, j = _check_dims(s.get("dims", [0, 1]), n_features, "checker_texture")
        return {
            "dims": [i, j], "cx": float(s.get("cx", 0.0)), "cy": float(s.get("cy", 0.0)),
            "width": float(s.get("width", 3.0)), "height": float(s.get("height", 3.0)),
            "cell_size_x": float(s.get("cell_size_x", 0.4)), "cell_size_y": float(s.get("cell_size_y", 0.4)),
            "orientation": float(s.get("orientation", 0.0)),
            "on_probability": float(s.get("on_probability", 1.0)),
            "jitter": float(s.get("jitter", 0.02)),
            "thickness": 0.04,
            "irrelevant": s.get("irrelevant", {"mode": "gaussian_mid", "sigma": float(s.get("other_sigma", 0.25))}),
        }

    def sample_points(self, rng, n_points, n_features, bounds_low, bounds_high, params_used, parent_params=None):
        p = params_used
        i, j = _check_dims(p["dims"], n_features, "checker_texture")
        pool = _sample_rotated_rect(rng, max(7000, n_points * 10), float(p["cx"]), float(p["cy"]), float(p["width"]), float(p["height"]), float(p["orientation"]))
        uv = (pool - np.array([[float(p["cx"]), float(p["cy"])]], dtype=float)) @ _rotation_matrix(-float(p["orientation"])).T
        cx = np.floor((uv[:, 0] + 0.5 * float(p["width"])) / max(1e-6, float(p["cell_size_x"]))).astype(int)
        cy = np.floor((uv[:, 1] + 0.5 * float(p["height"])) / max(1e-6, float(p["cell_size_y"]))).astype(int)
        on = ((cx + cy) % 2) == 0
        if float(p.get("on_probability", 1.0)) < 0.999:
            on &= rng.random(len(on)) < float(p.get("on_probability", 1.0))
        pts2 = pool[on]
        if len(pts2) == 0:
            pts2 = pool[: max(1, n_points)]
        idx = rng.choice(len(pts2), size=n_points, replace=True)
        pts2 = pts2[idx].copy()
        pts2 += rng.normal(0.0, float(p.get("jitter", 0.02)), size=pts2.shape)
        X = np.zeros((n_points, n_features), dtype=float)
        X[:, i] = pts2[:, 0]
        X[:, j] = pts2[:, 1]
        _fill_irrelevant_dims(rng, X, bounds_low, bounds_high, (i, j), p.get("irrelevant", None))
        np.clip(X, bounds_low, bounds_high, out=X)
        return X


register(CheckerTexture())
