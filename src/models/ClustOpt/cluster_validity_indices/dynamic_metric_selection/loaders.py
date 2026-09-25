"""On-demand loaders (with caches) for dataset-folder feature / utility files.

These are used only when the experiment runner does *not* preload the relevant
vectors. Both loaders cache their results process-wide so repeated records do
not re-read the same files.

Dataset-folder layout consumed::

    <dataset_dir>/
        features/features_records.csv        # one row per view (view_mode col)
        utility/<view_id>/utility_vector.json # {"utilities": {"utility__m": s}}
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# (dataset_dir, view_id, feature_source) -> ordered feature dict
_FEATURE_CACHE: Dict[tuple, Dict[str, float]] = {}
# (dataset_dir, view_id, utility_source) -> {metric_name: utility}
_ORACLE_CACHE: Dict[tuple, Dict[str, float]] = {}

_VIEW_COLUMN_CANDIDATES = ("view_mode", "view_id")


def _select_view_row(df: pd.DataFrame, view_id: str, view_aware: bool) -> pd.Series:
    """Pick the row matching ``view_id`` (when view-aware) or the first row."""
    if not view_aware:
        return df.iloc[0]

    for col in _VIEW_COLUMN_CANDIDATES:
        if col in df.columns:
            matches = df[df[col].astype(str) == str(view_id)]
            if len(matches) == 0:
                raise ValueError(
                    f"No row with {col}=='{view_id}' in features file "
                    f"(available: {sorted(df[col].astype(str).unique())})."
                )
            return matches.iloc[0]
    raise KeyError(
        f"Features file has no view column (looked for {_VIEW_COLUMN_CANDIDATES}); "
        f"cannot resolve view '{view_id}'."
    )


def load_feature_vector(
    dataset_dir: str | Path,
    view_id: str,
    feature_columns: List[str],
    feature_source: str = "features/features_records.csv",
    view_aware: bool = True,
) -> Dict[str, float]:
    """Load an ordered ``{feature_name: value}`` mapping for a dataset/view.

    Returned as a dict keyed by feature name so the predictor can align it to
    its own ``feature_columns`` order regardless of CSV column order.
    """
    dataset_dir = Path(dataset_dir)
    cache_key = (str(dataset_dir.resolve()), str(view_id), str(feature_source))
    cached = _FEATURE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    csv_path = dataset_dir / feature_source
    if not csv_path.is_file():
        raise FileNotFoundError(f"Features file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    row = _select_view_row(df, view_id, view_aware)

    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise KeyError(
            f"Features file {csv_path} is missing {len(missing)} expected "
            f"feature column(s), e.g. {missing[:5]}"
        )

    feature_dict = {c: float(row[c]) for c in feature_columns}
    _FEATURE_CACHE[cache_key] = feature_dict
    return feature_dict


def _utility_path(
    dataset_dir: Path, view_id: str, utility_source: Optional[str]
) -> Path:
    """Resolve the utility-vector path, honouring a ``<view_id>`` placeholder."""
    if utility_source:
        rel = utility_source.replace("<view_id>", str(view_id))
        return dataset_dir / rel
    return dataset_dir / "utility" / str(view_id) / "utility_vector.json"


def load_oracle_utilities(
    dataset_dir: str | Path,
    view_id: str,
    utility_source: Optional[str] = "utility/<view_id>/utility_vector.json",
    metric_name_prefix: str = "utility__",
) -> Dict[str, float]:
    """Load the *true* ``{metric_name: utility}`` vector for a dataset/view.

    Metric names have ``metric_name_prefix`` stripped so they match the metric
    ``.name`` used downstream.
    """
    dataset_dir = Path(dataset_dir)
    cache_key = (str(dataset_dir.resolve()), str(view_id), str(utility_source))
    cached = _ORACLE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    path = _utility_path(dataset_dir, view_id, utility_source)
    if not path.is_file():
        raise FileNotFoundError(f"Oracle utility vector not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    raw = payload.get("utilities", payload)
    if not isinstance(raw, dict):
        raise ValueError(
            f"Utility vector at {path} has no 'utilities' mapping."
        )

    prefix = metric_name_prefix or ""
    utilities: Dict[str, float] = {}
    for name, score in raw.items():
        bare = name[len(prefix):] if prefix and name.startswith(prefix) else name
        utilities[bare] = float(score)

    _ORACLE_CACHE[cache_key] = utilities
    return utilities
