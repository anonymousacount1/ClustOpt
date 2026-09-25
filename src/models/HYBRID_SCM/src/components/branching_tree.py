from __future__ import annotations

import numpy as np

from ..priors import sample_param_dict
from .base import Component, register
from ._helpers import (
    _build_branching_tree_geometry,
    _check_dims,
    _coerce_xy_pair,
    _fill_irrelevant_dims,
    _resolve_refs,
    _sample_on_segment_set,
)


class BranchingTree(Component):
    name = "branching_tree"

    def sample_params(self, rng, n_features, param_spec, parent_params):
        spec = _resolve_refs(param_spec, parent_params)
        s = sample_param_dict(rng, spec)
        i, j = _check_dims(s.get("dims", [0, 1]), n_features, "branching_tree")
        root_xy = _coerce_xy_pair(s.get("root", [0.0, 0.0]))
        bf = max(1, int(s.get("branching_factor", 2)))
        depth = max(1, int(s.get("depth", 3)))
        angle_spread = float(s.get("angle_spread", 1.2))
        seg_len = float(s.get("segment_length", 1.0))
        length_decay = float(s.get("length_decay", 0.72))
        prune = float(s.get("pruning_probability", 0.15))
        angle_jitter = float(s.get("angle_jitter", 0.08))
        verts, edges = _build_branching_tree_geometry(rng, root_xy, bf, depth, angle_spread, seg_len, length_decay, prune, angle_jitter)
        return {
            "dims": [i, j],
            "root": [float(root_xy[0]), float(root_xy[1])],
            "branching_factor": bf,
            "depth": depth,
            "angle_spread": angle_spread,
            "segment_length": seg_len,
            "length_decay": length_decay,
            "segment_thickness": float(s.get("segment_thickness", 0.05)),
            "thickness": float(s.get("segment_thickness", 0.05)),
            "pruning_probability": prune,
            "angle_jitter": angle_jitter,
            "vertices": [[float(x), float(y)] for x, y in verts.tolist()],
            "edges": [[int(a), int(b)] for a, b in edges.tolist()],
            "irrelevant": s.get("irrelevant", {"mode": "gaussian_mid", "sigma": float(s.get("other_sigma", 0.25))}),
        }

    def sample_points(self, rng, n_points, n_features, bounds_low, bounds_high, params_used, parent_params=None):
        p = params_used
        i, j = _check_dims(p["dims"], n_features, "branching_tree")
        verts = np.asarray(p.get("vertices", []), dtype=float)
        edges = np.asarray(p.get("edges", []), dtype=int)
        pts2 = _sample_on_segment_set(rng, verts, edges, n_points, thickness=float(p.get("segment_thickness", p.get("thickness", 0.05))))
        X = np.zeros((n_points, n_features), dtype=float)
        X[:, i] = pts2[:, 0]
        X[:, j] = pts2[:, 1]
        _fill_irrelevant_dims(rng, X, bounds_low, bounds_high, (i, j), p.get("irrelevant", None))
        np.clip(X, bounds_low, bounds_high, out=X)
        return X


register(BranchingTree())
