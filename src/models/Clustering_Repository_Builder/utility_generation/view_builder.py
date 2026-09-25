"""Build the three ClustOpt input views (x_only / y_only / xy_2d).

For utility generation, ``X_decision`` is what ClustOpt clusters on, while
``X_full`` stays as the full 2-D point cloud so metrics that expect both axes
(image-based, geometry-based, etc.) keep working.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .record_loader import LoadedDatasetForUtility


@dataclass
class ClustOptView:
    view_id: str            # 'x_only' | 'y_only' | 'xy_2d'
    X_decision: np.ndarray  # what the clustering algorithm sees
    X_full: np.ndarray      # full 2-D coords (always, for image-based metrics)
    y_true: np.ndarray      # ground-truth labels


def _resolve_xy_columns(loaded: LoadedDatasetForUtility) -> tuple[int, int]:
    """Return ``(x_col_idx, y_col_idx)`` for the loaded dataset.

    Prefers exact 'x'/'y' names; otherwise falls back to columns 0 and 1.
    """
    cols = [c.lower() for c in loaded.feature_names]
    if "x" in cols and "y" in cols:
        return cols.index("x"), cols.index("y")

    if loaded.X.shape[1] < 2:
        raise ValueError(
            f"Dataset {loaded.dataset_id} has only {loaded.X.shape[1]} feature column(s); "
            f"need at least 2 to build xy_2d view."
        )
    return 0, 1


def build_view(loaded: LoadedDatasetForUtility, view_id: str) -> ClustOptView:
    if view_id not in {"x_only", "y_only", "xy_2d"}:
        raise ValueError(f"Unknown view_id '{view_id}'")

    x_idx, y_idx = _resolve_xy_columns(loaded)
    X_full = np.asarray(loaded.X[:, [x_idx, y_idx]], dtype=float)

    if view_id == "x_only":
        X_decision = X_full[:, [0]]
    elif view_id == "y_only":
        X_decision = X_full[:, [1]]
    else:  # xy_2d
        X_decision = X_full

    return ClustOptView(
        view_id=view_id,
        X_decision=np.ascontiguousarray(X_decision),
        X_full=np.ascontiguousarray(X_full),
        y_true=np.asarray(loaded.y_true),
    )
