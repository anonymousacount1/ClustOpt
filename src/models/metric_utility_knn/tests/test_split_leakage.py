"""Leakage policy: LOSO folds are group-safe and split 1 is never in dev data."""
import numpy as np
import pytest

from models.metric_utility_knn import config as C
from models.metric_utility_knn.cross_validation import (assert_split1_absent,
                                                        build_loso_folds)
from models.metric_utility_knn.data_loader import load_unified_data, UnifiedData

pytestmark = pytest.mark.skipif(
    not C.UNIFIED_CSV.exists(), reason="unified dataset CSV not available")


def _dev_subset(d):
    m = d.split_mask(C.DEV_SPLITS)
    return UnifiedData(
        feature_columns=d.feature_columns, target_columns=d.target_columns,
        metric_names=d.metric_names, X=d.X[m], Y=d.Y[m],
        dataset_id=d.dataset_id[m], record_id=d.record_id[m], view=d.view[m],
        split_id=d.split_id[m], family=d.family[m], subfamily=d.subfamily[m],
        difficulty=d.difficulty[m], report=d.report)


def test_split1_absent_from_development():
    d = load_unified_data()
    dev = _dev_subset(d)
    audit = assert_split1_absent(dev)
    assert audit["split1_absent"] is True
    assert C.LOCKED_HOLDOUT_SPLIT not in audit["development_splits_present"]


def test_loso_folds_group_safe():
    d = load_unified_data()
    folds = build_loso_folds(d)
    assert len(folds) == 15
    for f in folds:
        a = f.audit
        assert a["validation_split_absent_from_train"]
        assert a["dataset_ids_disjoint"]
        assert a["split1_absent_from_fold"]
        assert f.validation_split not in f.training_splits


def test_all_three_views_share_a_split():
    d = load_unified_data(limit_rows=9000)
    # every dataset_id maps to exactly one split across its views
    import pandas as pd
    df = pd.DataFrame({"dataset_id": d.dataset_id, "split_id": d.split_id})
    per_ds = df.groupby("dataset_id")["split_id"].nunique()
    assert int(per_ds.max()) == 1
