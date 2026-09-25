r"""
cross_validation.py
===================

Leave-One-Split-Out (LOSO) cross-validation over development splits 2-16, plus
per-fold leakage auditing (plan §8).

For validation split ``s``::

    training_splits  = {2..16} \ {s}
    validation_split = s

All three views of a dataset are kept together by construction (they share a
``dataset_id`` hence a ``split_id``).  Split 1 (the locked holdout) is never part
of any development fold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from . import config as C
from .data_loader import UnifiedData


@dataclass
class LosoFold:
    validation_split: int
    training_splits: List[int]
    audit: Dict = field(default_factory=dict)


def build_loso_folds(data: UnifiedData) -> List[LosoFold]:
    present = sorted(set(int(s) for s in data.split_id) & set(C.DEV_SPLITS))
    folds: List[LosoFold] = []
    for s in present:
        train_splits = [x for x in present if x != s]
        folds.append(
            LosoFold(
                validation_split=s,
                training_splits=train_splits,
                audit=fold_leakage_audit(data, s, train_splits),
            )
        )
    return folds


def fold_leakage_audit(data: UnifiedData, val_split: int,
                       train_splits: List[int]) -> Dict:
    """Verify the three mandatory LOSO invariants for one fold."""
    train_mask = data.split_mask(train_splits)
    val_mask = data.split_id == val_split

    train_splits_present = sorted(set(int(s) for s in data.split_id[train_mask]))
    val_ds = set(data.dataset_id[val_mask].tolist())
    train_ds = set(data.dataset_id[train_mask].tolist())
    dataset_overlap = val_ds & train_ds

    split1_in_dev = bool(np.any(data.split_id[train_mask | val_mask]
                                == C.LOCKED_HOLDOUT_SPLIT))

    audit = {
        "validation_split": int(val_split),
        "n_train_records": int(train_mask.sum()),
        "n_val_records": int(val_mask.sum()),
        "validation_split_absent_from_train": val_split not in train_splits_present,
        "n_dataset_id_overlap_train_val": int(len(dataset_overlap)),
        "dataset_ids_disjoint": len(dataset_overlap) == 0,
        "split1_absent_from_fold": not split1_in_dev,
    }
    # Hard-fail on any leakage — the search must never run on a leaky fold.
    if not audit["validation_split_absent_from_train"]:
        raise AssertionError(f"Fold {val_split}: val split present in train.")
    if not audit["dataset_ids_disjoint"]:
        raise AssertionError(
            f"Fold {val_split}: {len(dataset_overlap)} dataset_id(s) leak train<->val."
        )
    if not audit["split1_absent_from_fold"]:
        raise AssertionError(f"Fold {val_split}: locked split 1 present in fold data.")
    return audit


def assert_split1_absent(data: UnifiedData) -> Dict:
    """Confirm the loaded *development* data excludes the locked split 1."""
    n_split1 = int(np.sum(data.split_id == C.LOCKED_HOLDOUT_SPLIT))
    audit = {
        "n_split1_records_in_development_data": n_split1,
        "split1_absent": n_split1 == 0,
        "development_splits_present": sorted(set(int(s) for s in data.split_id)),
    }
    if n_split1 != 0:
        raise AssertionError(
            "Split 1 present in development data used for configuration search."
        )
    return audit
