"""Phase-E2 KNN integration + backward-compatibility tests (plan §21).

Fast tests (no clustering): frozen-model loading, schema alignment, prediction
behaviour, fixed/dynamic metric selection, raw/softmax weighting, and proof that
the existing (default / phaseD) method resolution is unchanged.

Data-dependent tests skip cleanly when the frozen KNN model or a split-1 dataset
folder is unavailable.
"""
from __future__ import annotations

import functools
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[4]
KNN_MODEL_DIR = REPO / "results_analysis" / "metric_utility_knn" / "phase_e1" / "models"

pytestmark = pytest.mark.skipif(
    not KNN_MODEL_DIR.is_dir(), reason="frozen Phase-E1 KNN model not available")


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=1)
def _provider():
    from models.ClustOpt.cluster_validity_indices.dynamic_metric_selection import (
        get_knn_provider,
    )
    return get_knn_provider(str(KNN_MODEL_DIR))


@functools.lru_cache(maxsize=1)
def _split1_dataset():
    """First split-1 dataset dir with features + x_only utility, or None."""
    from .dataset_discovery import discover_datasets
    from .run_phaseD_all_subfamilies import discover_subfamilies
    from .split_filter import filter_dataset_dirs_by_split, load_split_assignments

    analyzed = REPO / "results_analysis" / "clustering_repository" / "analyzed_data"
    split_csv = analyzed / "experiment_splits" / "dataset_split_assignments.csv"
    if not split_csv.is_file():
        return None
    assign = load_split_assignments(split_csv)
    for sub in discover_subfamilies(analyzed):
        valid, _ = discover_datasets(sub, limit=None)
        if not valid:
            continue
        sel = filter_dataset_dirs_by_split([d.dataset_dir for d in valid], assign, 1)
        for d in sel:
            d = Path(d)
            if ((d / "features" / "features_records.csv").is_file()
                    and (d / "utility" / "x_only" / "utility_vector.json").is_file()):
                return d
    return None


def _resolve(params, dataset_dir, view="x_only"):
    from models.ClustOpt.cluster_validity_indices.dynamic_metric_selection import (
        resolve_knn_weights,
    )
    base = dict(knn_model_dir=str(KNN_MODEL_DIR),
                metric_name_prefix="utility__",
                feature_source="features/features_records.csv", view_aware=True)
    return resolve_knn_weights(params={**base, **params},
                               dataset_dir=str(dataset_dir), view_id=view)


# --------------------------------------------------------------------------- #
# Frozen model loading
# --------------------------------------------------------------------------- #
def test_frozen_config_is_the_selected_one():
    p = _provider()
    assert p.config_id == "n015__standard__euclidean__inverse_distance"
    assert p.n_neighbors == 15
    assert p.scaler == "standard"
    assert p.distance == "euclidean"
    assert p.neighbor_weighting == "inverse_distance"


def test_train_splits_exclude_holdout():
    p = _provider()
    assert p.train_splits == list(range(2, 17))
    assert 1 not in p.train_splits
    assert p.heldout_split == 1


def test_three_view_models_loaded():
    p = _provider()
    for view in ("x_only", "y_only", "xy_2d"):
        assert view in p._predictor._views


# --------------------------------------------------------------------------- #
# Schema alignment
# --------------------------------------------------------------------------- #
def test_feature_and_metric_schema():
    p = _provider()
    assert len(p.feature_columns) == 250
    assert len(p.metric_names) == 60
    assert all(t.startswith("utility__") for t in p.target_columns)


def test_metric_names_resolve_to_registry_keys():
    from models.ClustOpt.cluster_validity_indices.dynamic_metric_selection import (
        resolve_metric_key,
    )
    p = _provider()
    resolvable = sum(resolve_metric_key(m) is not None for m in p.metric_names)
    # The vast majority of utility metrics must map to CVI registry keys.
    assert resolvable >= 55


# --------------------------------------------------------------------------- #
# Prediction behaviour
# --------------------------------------------------------------------------- #
def test_prediction_finite_shape_range_and_deterministic():
    d = _split1_dataset()
    if d is None:
        pytest.skip("no split-1 dataset available")
    from .dataset_discovery import discover_datasets  # noqa: F401
    from models.ClustOpt.cluster_validity_indices.dynamic_metric_selection import (
        load_feature_vector,
    )
    p = _provider()
    fv = load_feature_vector(d, "x_only", p.feature_columns)
    u1, _ = p.predict(fv, "x_only")
    u2, _ = p.predict(fv, "x_only")
    v = np.array(list(u1.values()))
    assert len(u1) == 60
    assert np.all(np.isfinite(v))
    assert v.min() >= 0.0 and v.max() <= 1.0
    assert u1 == u2  # deterministic


def test_view_specific_predictions_differ():
    d = _split1_dataset()
    if d is None:
        pytest.skip("no split-1 dataset available")
    from models.ClustOpt.cluster_validity_indices.dynamic_metric_selection import (
        load_feature_vector,
    )
    p = _provider()
    ux, _ = p.predict(load_feature_vector(d, "x_only", p.feature_columns), "x_only")
    uy, _ = p.predict(load_feature_vector(d, "y_only", p.feature_columns), "y_only")
    assert list(ux.values()) != list(uy.values())


# --------------------------------------------------------------------------- #
# Fixed Top-K selection
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("k", [1, 3, 5, 10])
def test_fixed_top_k_selects_k_metrics(k):
    d = _split1_dataset()
    if d is None:
        pytest.skip("no split-1 dataset available")
    r = _resolve({"top_k": k, "weighting": "normalized_positive"}, d)
    assert len(r.selected_metrics) == k
    assert r.debug["selected_k_metrics_count"] == k


# --------------------------------------------------------------------------- #
# Dynamic Top-K
# --------------------------------------------------------------------------- #
def test_dynamic_predictK_in_bounds_and_uses_knn():
    d = _split1_dataset()
    if d is None:
        pytest.skip("no split-1 dataset available")
    r = _resolve({"top_k": "dynamic", "weighting": "normalized_positive",
                  "k_source": "predicted"}, d)
    k = r.debug["selected_k_metrics_count"]
    assert 1 <= k <= 10
    assert r.debug["dynamic_k_source"] == "predicted"
    assert len(r.selected_metrics) == k


def test_dynamic_realK_uses_real_only_for_count():
    d = _split1_dataset()
    if d is None:
        pytest.skip("no split-1 dataset available")
    rp = _resolve({"top_k": "dynamic", "weighting": "normalized_positive",
                   "k_source": "predicted"}, d)
    rr = _resolve({"top_k": "dynamic", "weighting": "normalized_positive",
                   "k_source": "real",
                   "real_utility_source": "utility/<view_id>/utility_vector.json"}, d)
    assert 1 <= rr.debug["selected_k_metrics_count"] <= 10
    assert rr.debug["dynamic_k_source"] == "real"
    assert rr.debug["real_utility_path"] == "utility/x_only/utility_vector.json"
    # Ranking still comes from the KNN utility -> same #1 metric as predictK.
    assert rr.selected_metrics[0] == rp.selected_metrics[0]


# --------------------------------------------------------------------------- #
# Weighting
# --------------------------------------------------------------------------- #
def test_raw_and_softmax_select_same_subset_and_sum_to_one():
    d = _split1_dataset()
    if d is None:
        pytest.skip("no split-1 dataset available")
    raw = _resolve({"top_k": 5, "weighting": "normalized_positive"}, d)
    sm = _resolve({"top_k": 5, "weighting": "softmax", "softmax_temperature": 0.5}, d)
    assert raw.selected_metrics == sm.selected_metrics
    assert abs(sum(raw.weights.values()) - 1.0) < 1e-9
    assert abs(sum(sm.weights.values()) - 1.0) < 1e-9


def test_top1_raw_and_softmax_both_weight_one():
    d = _split1_dataset()
    if d is None:
        pytest.skip("no split-1 dataset available")
    raw = _resolve({"top_k": 1, "weighting": "normalized_positive"}, d)
    sm = _resolve({"top_k": 1, "weighting": "softmax", "softmax_temperature": 0.5}, d)
    assert raw.selected_metrics == sm.selected_metrics
    assert list(raw.weights.values()) == [1.0]
    assert abs(list(sm.weights.values())[0] - 1.0) < 1e-9


# --------------------------------------------------------------------------- #
# Backward compatibility
# --------------------------------------------------------------------------- #
def test_default_method_set_unchanged():
    from .config import METHOD_SETS, METHODS
    assert METHOD_SETS["default"] == METHODS
    assert len(METHOD_SETS["default"]) == 10


def test_phaseD_method_set_unchanged():
    from .config import METHOD_SETS
    assert len(METHOD_SETS["phaseD"]) == 28


def test_phaseE2_knn_method_set_has_12():
    from .config import METHOD_SETS
    names = [m.name for m in METHOD_SETS["phaseE2_knn"]]
    assert len(names) == 12
    assert all(n.startswith("knn_") for n in names)
    assert all(m.group == "knn" for m in METHOD_SETS["phaseE2_knn"])


def test_all_method_sets_resolve_to_valid_configs():
    from .config import METHOD_SETS
    from .experiment_config_resolver import resolve_configs, validate_or_raise
    root = REPO / "models" / "ClustOpt" / "configs" / "experiments"
    for name, methods in METHOD_SETS.items():
        r = resolve_configs(root, methods=methods, validate=True)
        validate_or_raise(r)
        assert len(r.paths) == len(methods) * 3


def test_cvi_registry_additive():
    from models.ClustOpt.cluster_validity_indices.cvi_registry import CVI_REGISTRY
    for k in ("generic", "regressor_dynamic", "oracle_dynamic", "knn_dynamic"):
        assert k in CVI_REGISTRY
