"""
preprocessing.py
================

Leakage-safe, fold-local feature preprocessing for the KNN utility predictor.

Pipeline (fit on TRAIN features only, then applied to train and query alike):

    raw features
      -> replace +/-Inf with NaN
      -> median imputation (per-feature train median; all-NaN feature -> 0.0)
      -> feature-wise scaling (StandardScaler or RobustScaler)

Feature-wise scaling is mandatory because KNN distance is scale-sensitive; row-
wise L2 normalisation is deliberately NOT used as a substitute (plan §10).

Edge cases are handled deterministically and reported:
  * all-NaN feature in a training fold  -> imputed to 0.0
  * zero-variance feature               -> scaler leaves it unit-scaled (recorded)
  * infinite value                      -> treated as missing, then imputed

The imputer + scaler are serialisable so the frozen final model reproduces the
exact transform.
"""

from __future__ import annotations

import pickle
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler

from . import config as C


def _make_scaler(kind: str):
    if kind == "standard":
        return StandardScaler()
    if kind == "robust":
        return RobustScaler()
    raise ValueError(f"Unknown scaler '{kind}' (expected standard|robust).")


@dataclass
class PreprocReport:
    n_features: int = 0
    n_all_nan_features: int = 0
    n_zero_variance_features: int = 0
    n_inf_cells: int = 0
    n_nan_cells: int = 0
    all_nan_feature_idx: List[int] = field(default_factory=list)
    zero_variance_feature_idx: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "n_features": self.n_features,
            "n_all_nan_features": self.n_all_nan_features,
            "n_zero_variance_features": self.n_zero_variance_features,
            "n_inf_cells": self.n_inf_cells,
            "n_nan_cells": self.n_nan_cells,
            "all_nan_feature_idx": self.all_nan_feature_idx,
            "zero_variance_feature_idx": self.zero_variance_feature_idx,
        }


class FeaturePreprocessor:
    """Median imputer + feature-wise scaler, fit on training features only."""

    def __init__(self, scaler: str = "standard") -> None:
        self.scaler_kind = scaler
        self.scaler = _make_scaler(scaler)
        self.medians_: np.ndarray | None = None
        self.report_: PreprocReport | None = None
        self.fitted = False

    # ---- internals ------------------------------------------------------- #
    @staticmethod
    def _to_nan(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).copy()
        x[~np.isfinite(x)] = np.nan       # +/-Inf and existing NaN -> NaN
        return x

    def _impute(self, x_nan: np.ndarray) -> np.ndarray:
        out = x_nan.copy()
        inds = np.where(np.isnan(out))
        if inds[0].size:
            out[inds] = np.take(self.medians_, inds[1])
        return out

    # ---- API ------------------------------------------------------------- #
    def fit(self, x_train: np.ndarray) -> "FeaturePreprocessor":
        x_raw = np.asarray(x_train, dtype=np.float64)
        report = PreprocReport(
            n_features=int(x_raw.shape[1]),
            n_inf_cells=int(np.isinf(x_raw).sum()),
            n_nan_cells=int(np.isnan(x_raw).sum()),
        )
        x_nan = self._to_nan(x_raw)

        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            medians = np.nanmedian(x_nan, axis=0)
        all_nan = ~np.isfinite(medians)
        medians = np.where(all_nan, 0.0, medians)
        self.medians_ = medians
        report.all_nan_feature_idx = np.where(all_nan)[0].astype(int).tolist()
        report.n_all_nan_features = len(report.all_nan_feature_idx)

        x_imp = self._impute(x_nan)
        self.scaler.fit(x_imp)

        col_std = x_imp.std(axis=0)
        zv = np.where(col_std <= C.EPS)[0]
        report.zero_variance_feature_idx = zv.astype(int).tolist()
        report.n_zero_variance_features = int(zv.size)

        self.report_ = report
        self.fitted = True
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("FeaturePreprocessor.transform before fit().")
        x_nan = self._to_nan(x)
        x_imp = self._impute(x_nan)
        x_scaled = self.scaler.transform(x_imp)
        # Guard against any residual non-finite values from scaling.
        return np.nan_to_num(x_scaled, nan=0.0, posinf=0.0, neginf=0.0).astype(
            np.float64
        )

    def fit_transform(self, x_train: np.ndarray) -> np.ndarray:
        return self.fit(x_train).transform(x_train)

    # ---- persistence ----------------------------------------------------- #
    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(
                {
                    "scaler_kind": self.scaler_kind,
                    "scaler": self.scaler,
                    "medians_": self.medians_,
                    "report_": self.report_.to_dict() if self.report_ else None,
                },
                fh,
            )

    @classmethod
    def load(cls, path: str | Path) -> "FeaturePreprocessor":
        with open(path, "rb") as fh:
            state = pickle.load(fh)
        obj = cls(scaler=state["scaler_kind"])
        obj.scaler = state["scaler"]
        obj.medians_ = state["medians_"]
        obj.fitted = True
        return obj
