"""
data_loader.py
==============

Load the unified training CSV and robustly detect the three column groups:

  * identifier / metadata columns  (default 19)
  * input feature columns          (default 250)
  * utility target columns         (default 60, prefixed ``utility__``)

Detection strategy
------------------
1. Target columns = every column whose name starts with ``target_prefix``.
2. Id columns = the configured ``id_columns`` that are actually present; if too
   few of those are found, fall back to the first ``n_id_columns`` columns.
3. Feature columns = the remaining columns, in CSV order, that are neither id
   nor target columns.

Counts are validated against the configured expectations.  Under
``strict_column_counts`` a mismatch raises ``ValueError``; otherwise it warns.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from .config import DataConfig


@dataclass
class LoadedData:
    df: pd.DataFrame
    id_columns: List[str]
    feature_columns: List[str]
    target_columns: List[str]
    group_column: str


def _resolve_id_columns(all_columns: List[str], cfg: DataConfig) -> List[str]:
    present = [c for c in cfg.id_columns if c in all_columns]
    # If the configured names mostly match, trust them (keep CSV order).
    if len(present) >= max(1, cfg.n_id_columns - 2):
        present_set = set(present)
        return [c for c in all_columns if c in present_set]
    # Fall back to positional detection.
    warnings.warn(
        f"Only {len(present)}/{cfg.n_id_columns} configured id columns found; "
        f"falling back to the first {cfg.n_id_columns} columns as ids.",
        stacklevel=2,
    )
    return list(all_columns[: cfg.n_id_columns])


def _resolve_target_columns(all_columns: List[str], cfg: DataConfig) -> List[str]:
    """Resolve the ordered target columns.

    ``cfg.target_include is None`` (default) reproduces the historical
    behaviour exactly: every column starting with ``cfg.target_prefix``, in CSV
    order.  When ``target_include`` is given it is authoritative: exactly those
    metrics, in exactly the supplied order, with loud failures on duplicates or
    missing columns.  Other ``utility__`` columns are never silently appended.
    """
    discovered = [c for c in all_columns if c.startswith(cfg.target_prefix)]
    if cfg.target_include is None:
        return discovered

    include = list(cfg.target_include)
    if not include:
        raise ValueError("target_include was provided but is empty.")

    duplicates = sorted({m for m in include if include.count(m) > 1})
    if duplicates:
        raise ValueError(
            f"target_include contains duplicate metrics: {duplicates}."
        )

    # Accept both bare metric names and already-prefixed column names.
    wanted = [
        m if m.startswith(cfg.target_prefix) else f"{cfg.target_prefix}{m}"
        for m in include
    ]
    available = set(all_columns)
    missing = [c for c in wanted if c not in available]
    if missing:
        raise ValueError(
            f"target_include requested {len(missing)} target column(s) that are "
            f"not present in the dataset: {missing}. "
            f"Available targets: {len(discovered)}."
        )
    return wanted


def detect_columns(df: pd.DataFrame, cfg: DataConfig):
    all_columns = list(df.columns)

    target_columns = _resolve_target_columns(all_columns, cfg)
    id_columns = _resolve_id_columns(all_columns, cfg)
    id_set, target_set = set(id_columns), set(target_columns)
    # Any utility__ column NOT selected as a target must not leak into the
    # feature matrix -- that would hand the model the answer.
    excluded_targets = {
        c for c in all_columns
        if c.startswith(cfg.target_prefix) and c not in target_set
    }
    feature_columns = [
        c for c in all_columns
        if c not in id_set and c not in target_set and c not in excluded_targets
    ]

    expected_targets = (
        len(cfg.target_include) if cfg.target_include is not None
        else cfg.n_target_columns
    )

    _validate_count("id", len(id_columns), cfg.n_id_columns, cfg)
    _validate_count("feature", len(feature_columns), cfg.n_feature_columns, cfg)
    _validate_count("target", len(target_columns), expected_targets, cfg)

    if cfg.split_group_column not in df.columns:
        raise ValueError(
            f"Split group column '{cfg.split_group_column}' not present in CSV."
        )
    return id_columns, feature_columns, target_columns


def _validate_count(kind: str, found: int, expected: int, cfg: DataConfig) -> None:
    if found == expected:
        return
    msg = f"Detected {found} {kind} columns but expected {expected}."
    if cfg.strict_column_counts:
        raise ValueError(msg + " Set strict_column_counts=False to override.")
    warnings.warn(msg, stacklevel=2)


def load_unified_dataset(cfg: DataConfig) -> LoadedData:
    """Load the CSV and return the dataframe plus detected column groups."""
    read_kwargs = {}
    if cfg.limit_rows is not None:
        read_kwargs["nrows"] = int(cfg.limit_rows)

    df = pd.read_csv(cfg.csv_path, **read_kwargs)
    id_columns, feature_columns, target_columns = detect_columns(df, cfg)

    # Ensure feature / target columns are numeric (coerce non-numeric to NaN so
    # downstream cleaning can handle them rather than crashing the scaler).
    for col in feature_columns + target_columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return LoadedData(
        df=df,
        id_columns=id_columns,
        feature_columns=feature_columns,
        target_columns=target_columns,
        group_column=cfg.split_group_column,
    )
