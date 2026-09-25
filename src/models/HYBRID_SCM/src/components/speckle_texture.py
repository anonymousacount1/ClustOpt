from __future__ import annotations

import numpy as np

from ..priors import sample_param_dict
from .base import Component, register
from ._helpers import (
    _check_dims,
    _fill_irrelevant_dims,
    _resolve_refs,
    _sample_rotated_rect,
)


class SpeckleTexture(Component):
    name = "speckle_texture"

    def sample_params(self, rng, n_features, param_spec, parent_params):
        spec = _resolve_refs(param_spec, parent_params)
        s = sample_param_dict(rng, spec)
        i, j = _check_dims(s.get("dims", [0, 1]), n_features, "speckle_texture")
        return {
            "dims": [i, j], "cx": float(s.get("cx", 0.0)), "cy": float(s.get("cy", 0.0)),
            "width": float(s.get("width", 3.0)), "height": float(s.get("height", 2.0)),
            "speckle_rate": float(s.get("speckle_rate", 0.65)),
            "clustered_fraction": float(s.get("clustered_fraction", 0.35)),
            "cluster_sigma": float(s.get("cluster_sigma", 0.08)),
            "thickness": float(s.get("cluster_sigma", 0.08)),
            "irrelevant": s.get("irrelevant", {"mode": "gaussian_mid", "sigma": float(s.get("other_sigma", 0.25))}),
        }

    def sample_points(self, rng, n_points, n_features, bounds_low, bounds_high, params_used, parent_params=None):
        p = params_used
        i, j = _check_dims(p["dims"], n_features, "speckle_texture")
        n_clustered = int(round(float(p.get("clustered_fraction", 0.35)) * n_points))
        n_uniform = n_points - n_clustered
        pts_u = _sample_rotated_rect(rng, max(0, n_uniform), float(p["cx"]), float(p["cy"]), float(p["width"]), float(p["height"]), 0.0)
        anchors = _sample_rotated_rect(rng, max(3, n_clustered // 40 + 2), float(p["cx"]), float(p["cy"]), 0.7 * float(p["width"]), 0.7 * float(p["height"]), 0.0)
        ids = rng.integers(0, len(anchors), size=max(0, n_clustered))
        pts_c = anchors[ids] + rng.normal(0.0, float(p.get("cluster_sigma", 0.08)), size=(max(0, n_clustered), 2))
        pts2 = np.vstack([pts_u, pts_c]) if len(pts_u) or len(pts_c) else np.zeros((0, 2), dtype=float)
        rng.shuffle(pts2)
        X = np.zeros((n_points, n_features), dtype=float)
        X[:, i] = pts2[:, 0]
        X[:, j] = pts2[:, 1]
        _fill_irrelevant_dims(rng, X, bounds_low, bounds_high, (i, j), p.get("irrelevant", None))
        np.clip(X, bounds_low, bounds_high, out=X)
        return X


register(SpeckleTexture())
