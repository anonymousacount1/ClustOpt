"""Feature sets F1/F2/F3 and the two view architectures.

Train-side utility features come from the Stage-2A-3B0 PAIR-exclusion source;
test-side utility features come from the Stage-2A-1 SINGLE-exclusion source for
the held-out split. They are never mixed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from models.Clustering_Repository_Builder.experiments.experiment_execution.output_writer import (
    _ext,
)

VIEW_IDS: Tuple[str, ...] = ("x_only", "y_only", "xy_2d")
VIEW_ONEHOT: Dict[str, Tuple[int, int, int]] = {
    "x_only": (1, 0, 0), "y_only": (0, 1, 0), "xy_2d": (0, 0, 1),
}

FEATURE_SETS: Tuple[str, ...] = ("META", "UTILITIES", "META_UTILITIES")
VIEW_ARCHS: Tuple[str, ...] = ("UNIFIED", "SEPARATE")


def load_fold(root: Path, s: int) -> Dict[str, pd.DataFrame]:
    """One outer fold's blocks. Train = nested-clean, test = single-exclusion."""
    d = root / "outer_folds" / f"outer_{s:02d}"

    def r(name: str) -> pd.DataFrame:
        return pd.read_csv(_ext(d / f"{name}.csv.gz"))
    return {
        "train_ident": r("train_identifiers"),
        "train_meta": r("train_meta_features"),
        "train_util": r("train_pair_utility_features"),
        "train_ari": r("train_policy_ari_targets"),
        "test_ident": r("test_identifiers"),
        "test_meta": r("test_meta_features"),
        "test_util": r("test_single_oof_utility_features"),
        "test_ari": r("test_policy_ari_targets"),
    }


def build_X(meta: pd.DataFrame, util: pd.DataFrame, feature_set: str,
            view_ids: pd.Series, view_arch: str
            ) -> Tuple[np.ndarray, int, List[str]]:
    """Feature matrix, count of CONTINUOUS columns, and column names.

    The one-hot view block is appended after the continuous columns so a scaler
    can be fitted on the continuous part alone.
    """
    if feature_set == "META":
        blocks, names = [meta.to_numpy(np.float32)], list(meta.columns)
    elif feature_set == "UTILITIES":
        blocks, names = [util.to_numpy(np.float32)], list(util.columns)
    elif feature_set == "META_UTILITIES":
        blocks = [meta.to_numpy(np.float32), util.to_numpy(np.float32)]
        names = list(meta.columns) + list(util.columns)
    else:
        raise ValueError(f"unknown feature set {feature_set!r}")
    X = np.hstack(blocks) if len(blocks) > 1 else blocks[0]
    n_cont = X.shape[1]
    if view_arch == "UNIFIED":
        oh = np.asarray([VIEW_ONEHOT[v] for v in view_ids], dtype=np.float32)
        X = np.hstack([X, oh])
        names = names + ["view__x_only", "view__y_only", "view__xy_2d"]
    return X, n_cont, names


def expected_dim(feature_set: str, view_arch: str) -> int:
    base = {"META": 250, "UTILITIES": 120, "META_UTILITIES": 370}[feature_set]
    return base + (3 if view_arch == "UNIFIED" else 0)
