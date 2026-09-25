from __future__ import annotations

import numpy as np

from ..priors import sample_param_dict
from .base import Component, register
from ._helpers import (
    _build_meander_vertices,
    _check_dims,
    _fill_irrelevant_dims,
    _resolve_refs,
    _sample_on_polyline,
)


class MeanderCurve(Component):
    name = "meander_curve"

    def sample_params(self, rng, n_features, param_spec, parent_params):
        spec = _resolve_refs(param_spec, parent_params)
        s = sample_param_dict(rng, spec)
        i, j = _check_dims(s.get("dims", [0, 1]), n_features, "meander_curve")
        x_low = float(s.get("x_low", -3.0))
        x_high = float(s.get("x_high", 3.0))
        verts = _build_meander_vertices(
            rng=rng,
            x_low=x_low,
            x_high=x_high,
            y_center=float(s.get("y_center", 0.0)),
            amplitude=float(s.get("amplitude", 1.0)),
            frequency=float(s.get("frequency", 2.0)),
            phase=float(s.get("phase", 0.0)),
            amplitude_drift=float(s.get("amplitude_drift", 0.0)),
            frequency_drift=float(s.get("frequency_drift", 0.0)),
            n_anchors=int(s.get("n_anchors", 48)),
            anchor_jitter=float(s.get("anchor_jitter", 0.0)),
        )
        return {
            "dims": [i, j],
            "x_low": x_low,
            "x_high": x_high,
            "y_center": float(s.get("y_center", 0.0)),
            "amplitude": float(s.get("amplitude", 1.0)),
            "frequency": float(s.get("frequency", 2.0)),
            "phase": float(s.get("phase", 0.0)),
            "amplitude_drift": float(s.get("amplitude_drift", 0.0)),
            "frequency_drift": float(s.get("frequency_drift", 0.0)),
            "n_anchors": int(s.get("n_anchors", 48)),
            "anchor_jitter": float(s.get("anchor_jitter", 0.0)),
            "vertices": [[float(x), float(y)] for x, y in verts.tolist()],
            "base_thickness": float(s.get("base_thickness", s.get("thickness", 0.06))),
            "thickness": float(s.get("base_thickness", s.get("thickness", 0.06))),
            "irrelevant": s.get("irrelevant", {"mode": "gaussian_mid", "sigma": float(s.get("other_sigma", 0.25))}),
        }

    def sample_points(self, rng, n_points, n_features, bounds_low, bounds_high, params_used, parent_params=None):
        p = params_used
        i, j = _check_dims(p["dims"], n_features, "meander_curve")
        verts = np.asarray(p.get("vertices", []), dtype=float)
        pts2 = _sample_on_polyline(rng, verts, n_points, thickness=float(p.get("base_thickness", p.get("thickness", 0.06))), closed=False)
        X = np.zeros((n_points, n_features), dtype=float)
        X[:, i] = pts2[:, 0]
        X[:, j] = pts2[:, 1]
        _fill_irrelevant_dims(rng, X, bounds_low, bounds_high, (i, j), p.get("irrelevant", None))
        np.clip(X, bounds_low, bounds_high, out=X)
        return X


register(MeanderCurve())
