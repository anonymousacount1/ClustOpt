"""Light validation/audit helpers."""
from __future__ import annotations

from typing import Mapping

import numpy as np

from ..feature_schema import ALL_FEATURE_COLUMNS, BLOCK_COLUMNS


def count_nan_features(record: Mapping[str, object]) -> dict[str, int]:
    """Count NaN features per block and total."""
    out: dict[str, int] = {}
    total_nan = 0
    for block_name, cols in BLOCK_COLUMNS.items():
        n_nan = 0
        for c in cols:
            v = record.get(c, np.nan)
            try:
                if v is None or not np.isfinite(float(v)):
                    n_nan += 1
            except (TypeError, ValueError):
                n_nan += 1
        out[block_name] = n_nan
        total_nan += n_nan
    out["total"] = total_nan
    out["total_features"] = len(ALL_FEATURE_COLUMNS)
    return out
