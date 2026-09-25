"""
splitters.py
============

Group-safe cross-validation splitting and leakage validation.

The unified dataset contributes exactly three rows (``x_only``, ``y_only``,
``xy_2d``) per raw dataset, all sharing the same ``dataset_id``.  These rows are
strongly correlated and must never be split across train / validation / test.

We therefore:
  * use ``GroupKFold(n_splits)`` over ``dataset_id`` for the outer test folds,
  * use ``GroupShuffleSplit`` over the train+val groups for the inner
    validation split (also group-safe),
  * validate that no ``dataset_id`` leaks between any two of train / val / test,
  * emit a per-fold split summary (row & group counts, group distributions).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit


@dataclass
class FoldSplit:
    fold: int
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    summary: Dict = field(default_factory=dict)


def _group_distribution(df: pd.DataFrame, idx: np.ndarray, columns: List[str]) -> Dict:
    out: Dict[str, Dict] = {}
    sub = df.iloc[idx]
    for col in columns:
        if col in sub.columns:
            vc = sub[col].value_counts(dropna=False)
            out[col] = {str(k): int(v) for k, v in vc.items()}
    return out


def validate_no_leakage(
    groups: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> Dict[str, int]:
    """Raise if any group id appears in more than one split. Returns overlap sizes."""
    g_train = set(groups[train_idx])
    g_val = set(groups[val_idx])
    g_test = set(groups[test_idx])

    tv = g_train & g_val
    tt = g_train & g_test
    vt = g_val & g_test
    if tv or tt or vt:
        raise AssertionError(
            "Group leakage detected: "
            f"train∩val={len(tv)}, train∩test={len(tt)}, val∩test={len(vt)}"
        )
    return {"train_val": 0, "train_test": 0, "val_test": 0}


def build_folds(
    df: pd.DataFrame,
    group_column: str,
    n_splits: int,
    inner_val_ratio: float,
    seed: int,
    group_summary_columns: List[str] | None = None,
) -> List[FoldSplit]:
    """Build ``n_splits`` group-safe train/val/test folds with leakage checks."""
    groups = df[group_column].to_numpy()
    n_rows = len(df)
    indices = np.arange(n_rows)
    group_summary_columns = group_summary_columns or []

    gkf = GroupKFold(n_splits=n_splits)
    folds: List[FoldSplit] = []

    for fold_id, (trainval_idx, test_idx) in enumerate(
        gkf.split(indices, groups=groups)
    ):
        trainval_groups = groups[trainval_idx]
        # Inner group-safe validation split over the train+val portion.
        gss = GroupShuffleSplit(
            n_splits=1, test_size=inner_val_ratio, random_state=seed
        )
        inner_train_pos, inner_val_pos = next(
            gss.split(trainval_idx, groups=trainval_groups)
        )
        train_idx = trainval_idx[inner_train_pos]
        val_idx = trainval_idx[inner_val_pos]

        validate_no_leakage(groups, train_idx, val_idx, test_idx)

        summary = {
            "fold": fold_id,
            "n_rows": {
                "train": int(len(train_idx)),
                "val": int(len(val_idx)),
                "test": int(len(test_idx)),
                "total": int(n_rows),
            },
            "n_groups": {
                "train": int(len(set(groups[train_idx]))),
                "val": int(len(set(groups[val_idx]))),
                "test": int(len(set(groups[test_idx]))),
                "total": int(len(set(groups))),
            },
            "leakage": {"train_val": 0, "train_test": 0, "val_test": 0},
            "group_distributions": {
                "train": _group_distribution(df, train_idx, group_summary_columns),
                "val": _group_distribution(df, val_idx, group_summary_columns),
                "test": _group_distribution(df, test_idx, group_summary_columns),
            },
        }
        folds.append(
            FoldSplit(
                fold=fold_id,
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
                summary=summary,
            )
        )
    return folds


def build_holdout_split_folds(
    df: pd.DataFrame,
    group_column: str,
    split_assignments_csv: str,
    test_split_id: int,
    train_split_ids: List[int] | None,
    inner_val_ratio: float,
    seed: int,
    group_summary_columns: List[str] | None = None,
    split_id_column: str = "split_id",
    assignment_key_column: str = "dataset_id",
) -> List[FoldSplit]:
    """Build a single train/val/test fold from *predefined* dataset splits.

    Unlike :func:`build_folds` (which derives the outer test fold with
    ``GroupKFold``), the outer test fold here is fixed by the experiment split
    assignments: every row whose ``dataset_id`` belongs to ``test_split_id`` is
    the test set, and rows whose ``dataset_id`` belongs to ``train_split_ids``
    form the train+val pool.  The inner train/validation division is then made
    **exactly as in** :func:`build_folds` — a group-safe ``GroupShuffleSplit``
    with the same ``inner_val_ratio`` and ``seed`` — so this run differs from the
    standard cross-validation only in how the outer test fold is chosen.

    Returns a one-element list (``fold`` index ``0``) so the rest of the
    pipeline (training, evaluation, aggregation) is reused untouched.
    """
    groups = df[group_column].to_numpy()
    n_rows = len(df)
    indices = np.arange(n_rows)
    group_summary_columns = group_summary_columns or []
    test_split_id = int(test_split_id)

    # dataset_id -> split_id mapping from the experiment split assignments.
    assign = pd.read_csv(
        split_assignments_csv, usecols=[assignment_key_column, split_id_column]
    )
    id_to_split = {
        str(k): int(v)
        for k, v in zip(assign[assignment_key_column], assign[split_id_column])
    }

    row_keys = df[group_column].astype(str).to_numpy()
    unknown = sorted({k for k in row_keys if k not in id_to_split})
    if unknown:
        raise ValueError(
            f"{len(unknown)} dataset_id(s) in the data have no split assignment "
            f"in {split_assignments_csv}; e.g. {unknown[:5]}"
        )
    row_split = np.array([id_to_split[k] for k in row_keys], dtype=int)

    all_split_ids = sorted(set(int(s) for s in id_to_split.values()))
    if train_split_ids is None:
        train_split_ids = [s for s in all_split_ids if s != test_split_id]
    train_set = sorted(set(int(s) for s in train_split_ids))
    if test_split_id in train_set:
        raise ValueError(
            f"test_split_id={test_split_id} must not appear in train_split_ids={train_set}"
        )

    test_idx = indices[row_split == test_split_id]
    trainval_idx = indices[np.isin(row_split, train_set)]
    if len(test_idx) == 0:
        raise ValueError(f"No rows found for test split {test_split_id}.")
    if len(trainval_idx) == 0:
        raise ValueError(f"No rows found for train splits {train_set}.")

    # Inner group-safe validation split over the train+val portion — identical
    # settings to build_folds (GroupShuffleSplit, same ratio and seed).
    trainval_groups = groups[trainval_idx]
    gss = GroupShuffleSplit(n_splits=1, test_size=inner_val_ratio, random_state=seed)
    inner_train_pos, inner_val_pos = next(gss.split(trainval_idx, groups=trainval_groups))
    train_idx = trainval_idx[inner_train_pos]
    val_idx = trainval_idx[inner_val_pos]

    validate_no_leakage(groups, train_idx, val_idx, test_idx)

    summary = {
        "fold": 0,
        "split_holdout": {
            "test_split_id": test_split_id,
            "train_split_ids": train_set,
            "all_split_ids": all_split_ids,
            "split_assignments_csv": str(split_assignments_csv),
            "inner_val_ratio": inner_val_ratio,
            "seed": seed,
        },
        "n_rows": {
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
            "total": int(n_rows),
        },
        "n_groups": {
            "train": int(len(set(groups[train_idx]))),
            "val": int(len(set(groups[val_idx]))),
            "test": int(len(set(groups[test_idx]))),
            "total": int(len(set(groups))),
        },
        "leakage": {"train_val": 0, "train_test": 0, "val_test": 0},
        "group_distributions": {
            "train": _group_distribution(df, train_idx, group_summary_columns),
            "val": _group_distribution(df, val_idx, group_summary_columns),
            "test": _group_distribution(df, test_idx, group_summary_columns),
        },
    }
    return [
        FoldSplit(
            fold=0,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            summary=summary,
        )
    ]


def cv_split_summary_frame(folds: List[FoldSplit]) -> pd.DataFrame:
    """Flat per-fold summary frame for ``cv_split_summary.csv``."""
    rows = []
    for f in folds:
        s = f.summary
        rows.append(
            {
                "fold": f.fold,
                "train_rows": s["n_rows"]["train"],
                "val_rows": s["n_rows"]["val"],
                "test_rows": s["n_rows"]["test"],
                "train_groups": s["n_groups"]["train"],
                "val_groups": s["n_groups"]["val"],
                "test_groups": s["n_groups"]["test"],
                "leak_train_val": s["leakage"]["train_val"],
                "leak_train_test": s["leakage"]["train_test"],
                "leak_val_test": s["leakage"]["val_test"],
            }
        )
    return pd.DataFrame(rows)
