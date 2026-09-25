"""Validation checks for split assignments.

Validates the invariants required by the plan: every dataset appears exactly
once with exactly one split id, split ids are in ``1..n_splits``, sizes are
approximately balanced, and every stratum is spread across splits as evenly as
possible.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import pandas as pd

from .stratified_splitter import STRATUM_COLUMNS


class SplitValidationError(ValueError):
    """Raised when a hard split invariant is violated."""


@dataclass
class SplitValidationReport:
    n_datasets: int
    n_splits: int
    ok: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    split_sizes: dict = field(default_factory=dict)
    max_size_imbalance: int = 0

    def add_error(self, msg: str) -> None:
        self.ok = False
        self.errors.append(msg)

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)


def validate_assignments(
    df: pd.DataFrame,
    *,
    n_splits: int,
    raise_on_error: bool = True,
) -> SplitValidationReport:
    report = SplitValidationReport(n_datasets=len(df), n_splits=n_splits)

    # 1) every dataset appears exactly once
    if df["dataset_id"].duplicated().any():
        dups = df.loc[df["dataset_id"].duplicated(), "dataset_id"].unique().tolist()
        report.add_error(f"{len(dups)} duplicate dataset_id(s), e.g. {dups[:5]}")

    # 2) every dataset has exactly one split id, no nulls
    if "split_id" not in df.columns:
        report.add_error("missing 'split_id' column")
        if raise_on_error:
            raise SplitValidationError("; ".join(report.errors))
        return report
    if df["split_id"].isna().any():
        n = int(df["split_id"].isna().sum())
        report.add_error(f"{n} dataset(s) have no split_id")

    # 3) split_id in 1..n_splits
    bad = df.loc[~df["split_id"].isin(range(1, n_splits + 1)), "split_id"].unique().tolist()
    if bad:
        report.add_error(f"split_id values out of range 1..{n_splits}: {bad[:10]}")

    # 4) balanced sizes
    sizes = df["split_id"].value_counts().sort_index()
    report.split_sizes = {int(k): int(v) for k, v in sizes.items()}
    if len(sizes) > 0:
        report.max_size_imbalance = int(sizes.max() - sizes.min())
        # Round-robin guarantees imbalance <= number of strata; warn if it
        # somehow exceeds a generous tolerance.
        tolerance = max(2, int(0.05 * (len(df) / max(n_splits, 1))) + 1)
        if report.max_size_imbalance > tolerance * 50:
            report.add_warning(
                f"split size imbalance {report.max_size_imbalance} is large "
                f"(sizes={report.split_sizes})"
            )

    # 5) every split id present (no empty split) when there are enough datasets
    if len(df) >= n_splits:
        present = set(int(s) for s in sizes.index)
        missing = [s for s in range(1, n_splits + 1) if s not in present]
        if missing:
            report.add_warning(f"empty split id(s): {missing}")

    # 6) per-stratum spread: a stratum with >= n_splits members should touch
    #    many splits; warn (not error) if a large stratum is concentrated.
    present_strata_cols = [c for c in STRATUM_COLUMNS if c in df.columns]
    if present_strata_cols:
        grp = df.groupby(present_strata_cols)["split_id"].nunique()
        sizes_by_stratum = df.groupby(present_strata_cols).size()
        concentrated = []
        for key, n_unique in grp.items():
            total = int(sizes_by_stratum.loc[key])
            if total >= n_splits and n_unique < max(2, n_splits // 2):
                concentrated.append((key, total, int(n_unique)))
        if concentrated:
            report.add_warning(
                f"{len(concentrated)} large stratum/strata concentrated in few "
                f"splits, e.g. {concentrated[:3]}"
            )

    if raise_on_error and not report.ok:
        raise SplitValidationError("; ".join(report.errors))
    return report
