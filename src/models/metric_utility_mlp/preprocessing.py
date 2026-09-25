"""
preprocessing.py
================

Feature cleaning, scaling, and target validation.

  * Invalid feature values (NaN, +/-Inf) are replaced with 0.0 and the number of
    replacements is logged/returned.
  * A ``StandardScaler`` is fit on the *training* features only and then applied
    to train / val / test.  The scaler is saved per fold as ``x_scaler.pkl``.
  * Targets are validated to be finite and within ``[0, 1]``; tiny numerical
    deviations are clipped (with a logged count).
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler


@dataclass
class CleaningReport:
    n_nan: int = 0
    n_posinf: int = 0
    n_neginf: int = 0
    n_target_clipped: int = 0
    extra: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "feature_nan_replaced": int(self.n_nan),
            "feature_posinf_replaced": int(self.n_posinf),
            "feature_neginf_replaced": int(self.n_neginf),
            "target_values_clipped": int(self.n_target_clipped),
            **self.extra,
        }


def clean_features(x: np.ndarray) -> Tuple[np.ndarray, CleaningReport]:
    """Replace NaN/+Inf/-Inf in features with 0.0. Returns cleaned copy + report."""
    x = np.asarray(x, dtype=np.float64)
    report = CleaningReport(
        n_nan=int(np.isnan(x).sum()),
        n_posinf=int(np.isposinf(x).sum()),
        n_neginf=int(np.isneginf(x).sum()),
    )
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x, report


def clean_and_clip_targets(y: np.ndarray) -> Tuple[np.ndarray, int]:
    """Replace non-finite targets with 0 and clip to [0, 1]. Returns (y, n_clipped)."""
    y = np.asarray(y, dtype=np.float64)
    y = np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=0.0)
    out_of_range = int(((y < 0.0) | (y > 1.0)).sum())
    y = np.clip(y, 0.0, 1.0)
    return y, out_of_range


class FeaturePipeline:
    """Fit-on-train feature cleaner + StandardScaler."""

    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.fitted = False

    def fit(self, x_train: np.ndarray) -> "FeaturePipeline":
        x_clean, _ = clean_features(x_train)
        self.scaler.fit(x_clean)
        self.fitted = True
        return self

    def transform(self, x: np.ndarray) -> Tuple[np.ndarray, CleaningReport]:
        if not self.fitted:
            raise RuntimeError("FeaturePipeline.transform called before fit().")
        x_clean, report = clean_features(x)
        x_scaled = self.scaler.transform(x_clean).astype(np.float32)
        # Guard against any residual non-finite values produced by scaling.
        x_scaled = np.nan_to_num(x_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        return x_scaled, report

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self.scaler, fh)

    @classmethod
    def load(cls, path: str | Path) -> "FeaturePipeline":
        obj = cls()
        with open(path, "rb") as fh:
            obj.scaler = pickle.load(fh)
        obj.fitted = True
        return obj
