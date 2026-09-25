from __future__ import annotations

import numpy as np

from ..priors import sample_param_dict
from .base import Component, register
from ._helpers import (
    _chaikin_smooth,
    _check_dims,
    _fill_irrelevant_dims,
    _polyline_vertices_from_spec,
    _resolve_refs,
    _sample_on_polyline,
)


class PiecewisePolyline(Component):
    name = "piecewise_polyline"

    def sample_params(self, rng, n_features, param_spec, parent_params):
        spec = _resolve_refs(param_spec, parent_params)
        s = sample_param_dict(rng, spec)
        i, j = _check_dims(s.get("dims", [0, 1]), n_features, "piecewise_polyline")
        verts = _polyline_vertices_from_spec(s, np.array([-1e9, -1e9]), np.array([1e9, 1e9]))
        rounding = float(s.get("corner_rounding", 0.0))
        if rounding > 0:
            verts = _chaikin_smooth(verts, weight=min(0.45, 0.2 + 0.25 * rounding), closed=False, n_iter=1 + int(rounding > 0.45))
        return {
            "dims": [i, j],
            "vertices": [[float(x), float(y)] for x, y in verts.tolist()],
            "corner_rounding": rounding,
            "base_thickness": float(s.get("base_thickness", s.get("thickness", 0.06))),
            "thickness": float(s.get("base_thickness", s.get("thickness", 0.06))),
            "irrelevant": s.get("irrelevant", {"mode": "gaussian_mid", "sigma": float(s.get("other_sigma", 0.25))}),
        }

    def sample_points(self, rng, n_points, n_features, bounds_low, bounds_high, params_used, parent_params=None):
        p = params_used
        i, j = _check_dims(p["dims"], n_features, "piecewise_polyline")
        verts = np.asarray(p["vertices"], dtype=float)
        pts2 = _sample_on_polyline(rng, verts, n_points, thickness=float(p.get("base_thickness", p.get("thickness", 0.06))), closed=False)
        X = np.zeros((n_points, n_features), dtype=float)
        X[:, i] = pts2[:, 0]
        X[:, j] = pts2[:, 1]
        _fill_irrelevant_dims(rng, X, bounds_low, bounds_high, (i, j), p.get("irrelevant", None))
        np.clip(X, bounds_low, bounds_high, out=X)
        return X


register(PiecewisePolyline())
