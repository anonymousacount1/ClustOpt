"""The frozen three-view protocol, identical to the historical convention."""
from __future__ import annotations

from typing import Dict

import numpy as np

VIEW_IDS = ("1d_x", "1d_y", "2d")
VIEW_TO_HISTORICAL = {"1d_x": "x_only", "1d_y": "y_only", "2d": "xy_2d"}
HISTORICAL_TO_VIEW = {v: k for k, v in VIEW_TO_HISTORICAL.items()}


def build(X2: np.ndarray) -> Dict[str, np.ndarray]:
    """One frozen 2-D representation -> the three views every method receives."""
    if X2.ndim != 2 or X2.shape[1] != 2:
        raise ValueError("expected an (n, 2) frozen representation")
    return {"1d_x": X2[:, [0]], "1d_y": X2[:, [1]], "2d": X2}
