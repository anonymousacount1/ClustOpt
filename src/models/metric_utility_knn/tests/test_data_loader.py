"""Data loader: schema alignment, split attachment, record uniqueness."""
import numpy as np
import pytest

from models.metric_utility_knn import config as C
from models.metric_utility_knn.data_loader import load_unified_data

pytestmark = pytest.mark.skipif(
    not C.UNIFIED_CSV.exists(), reason="unified dataset CSV not available")


def test_schema_alignment_and_counts():
    d = load_unified_data(limit_rows=6000)
    assert len(d.feature_columns) == C.N_FEATURE_COLUMNS
    assert len(d.target_columns) == C.N_TARGET_COLUMNS
    assert all(c.startswith(C.TARGET_PREFIX) for c in d.target_columns)
    assert d.X.shape[1] == C.N_FEATURE_COLUMNS
    assert d.Y.shape[1] == C.N_TARGET_COLUMNS
    # feature/target columns disjoint from id columns
    assert not (set(d.feature_columns) & set(C.ID_COLUMNS))


def test_views_and_splits_present():
    d = load_unified_data(limit_rows=6000)
    assert set(d.view) <= set(C.VIEWS)
    assert d.split_id.min() >= 1
    assert d.report["record_id_unique"] is True


def test_metric_names_strip_prefix():
    d = load_unified_data(limit_rows=3000)
    assert all(not m.startswith(C.TARGET_PREFIX) for m in d.metric_names)
    assert len(d.metric_names) == len(d.target_columns)
