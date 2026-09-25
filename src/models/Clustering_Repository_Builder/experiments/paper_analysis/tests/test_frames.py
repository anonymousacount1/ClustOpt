"""Best-view and dataset-level frame tests (§29.2)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.Clustering_Repository_Builder.experiments.paper_analysis import frames
from models.Clustering_Repository_Builder.experiments.paper_analysis.method_registry import (
    VIEW_IDS,
)


def _row(best_view: pd.DataFrame, dataset_id: str, method: str) -> pd.Series:
    sub = best_view[(best_view["dataset_id"] == dataset_id)
                    & (best_view["method_name"] == method)]
    assert len(sub) == 1, f"expected exactly one row for {dataset_id}/{method}"
    return sub.iloc[0]


def test_selects_highest_ari_successful_view(tiny_best_view):
    r = _row(tiny_best_view, "d2", "knn_top5_raw")
    assert r["best_view_id"] == "x_only"
    assert r["best_view_ari"] == pytest.approx(0.95)


def test_exact_tie_broken_by_lower_runtime(tiny_best_view):
    """d1/regressor_top5_raw ties at 0.80 on x_only (10s) and y_only (4s)."""
    r = _row(tiny_best_view, "d1", "regressor_top5_raw")
    assert r["best_view_ari"] == pytest.approx(0.80)
    assert r["view_tie_count"] == 2
    assert r["best_view_id"] == "y_only"
    assert r["best_view_runtime_sec"] == pytest.approx(4.0)
    assert r["view_tie_broken_by"] == "runtime"


def test_failure_rows_are_preserved(tiny_best_view):
    r = _row(tiny_best_view, "d4", "knn_top5_raw")
    assert bool(r["has_success"]) is False
    assert pd.isna(r["best_view_ari"])
    assert r["n_views_success"] == 0
    assert r["n_views_attempted"] == 3
    # Runtime of the failed attempts still counts towards the cost.
    assert r["total_runtime_all_views_sec"] == pytest.approx(9.0)


def test_success_rate_accounting_includes_failures(tiny_best_view):
    summary = frames.summarize_methods(tiny_best_view, ("method_name",))
    knn = summary[summary["method_name"] == "knn_top5_raw"].iloc[0]
    assert knn["n_datasets"] == 4
    assert knn["n_success"] == 3
    assert knn["success_rate"] == pytest.approx(0.75)
    assert knn["failure_rate"] == pytest.approx(0.25)


def test_total_all_views_runtime_is_the_sum(tiny_best_view, tiny_run_frame):
    for (dataset_id, method), grp in tiny_run_frame.groupby(
            ["dataset_id", "method_name"]):
        want = pd.to_numeric(grp["runtime_sec"], errors="coerce").sum()
        got = _row(tiny_best_view, dataset_id, method)["total_runtime_all_views_sec"]
        assert got == pytest.approx(float(want))


def test_view_gap_is_best_minus_mean(tiny_best_view, tiny_run_frame):
    r = _row(tiny_best_view, "d2", "regressor_top5_raw")
    aris = [0.90, 0.85, 0.88]
    assert r["view_ari_mean"] == pytest.approx(np.mean(aris))
    assert r["view_gap"] == pytest.approx(0.90 - np.mean(aris))


def test_best_view_never_below_any_view(tiny_best_view, tiny_run_frame):
    ok = tiny_run_frame[tiny_run_frame["status"] == "success"].copy()
    ok["_ari"] = pd.to_numeric(ok["ari"], errors="coerce")
    per_pair = ok.groupby(["dataset_id", "method_name"])["_ari"].max()
    for (dataset_id, method), want in per_pair.items():
        got = _row(tiny_best_view, dataset_id, method)["best_view_ari"]
        assert got >= want - 1e-12
        assert got == pytest.approx(float(want))


def test_uses_view_oracle_flag_is_always_true(tiny_best_view):
    assert bool(tiny_best_view["uses_view_oracle"].all())
    assert set(tiny_best_view["selection_type"].unique()) == {"fixed_variant"}
    assert set(tiny_best_view["paper_claim_level"].unique()) == {"fixed_observed"}


def test_one_row_per_dataset_method(tiny_best_view, tiny_run_frame):
    n_pairs = tiny_run_frame.groupby(["dataset_id", "method_name"]).ngroups
    assert len(tiny_best_view) == n_pairs
    assert not tiny_best_view.duplicated(
        subset=["dataset_id", "method_name"]).any()


def test_frame_validation_passes(tiny_run_frame, tiny_best_view):
    assert frames.validate_frames(tiny_run_frame, tiny_best_view) == []


def test_frame_validation_catches_a_wrong_best_view(tiny_run_frame, tiny_best_view):
    broken = tiny_best_view.copy()
    idx = broken.index[0]
    broken.loc[idx, "best_view_ari"] = 0.0
    problems = frames.validate_frames(tiny_run_frame, broken)
    assert any("differ from the max successful view ARI" in p for p in problems)


def test_capability_flags_follow_the_registry(tiny_best_view):
    reg = _row(tiny_best_view, "d1", "regressor_top5_raw")
    assert bool(reg["has_metric_selection"]) and bool(reg["has_trace"])
    assert bool(reg["has_total_all_views_runtime"])
    assert bool(reg["has_end_to_end_runtime"]) is False


def test_dataset_level_adds_rank_and_delta(tiny_best_view):
    dl = frames.build_dataset_level_frame(tiny_best_view)
    assert len(dl) == len(tiny_best_view)
    d2 = dl[dl["dataset_id"] == "d2"]
    assert d2["dataset_best_ari"].nunique() == 1
    assert d2["dataset_best_ari"].iloc[0] == pytest.approx(0.95)
    winner = d2[d2["method_name"] == "knn_top5_raw"].iloc[0]
    assert winner["rank_within_pool"] == 1
    assert bool(winner["is_dataset_best"])
    assert winner["delta_to_dataset_best"] == pytest.approx(0.0)


def test_dataset_level_pool_restriction(tiny_best_view):
    dl = frames.build_dataset_level_frame(
        tiny_best_view, methods=["regressor_top5_raw", "knn_top5_raw"])
    assert int(dl["comparison_pool_size"].iloc[0]) == 2
    # silhouette_single is outside the pool, so it has no rank.
    outside = dl[dl["method_name"] == "silhouette_single"]
    assert outside["rank_within_pool"].isna().all()


def test_summary_column_names_are_explicit(tiny_best_view):
    summary = frames.summarize_methods(tiny_best_view, ("method_name",))
    assert "best_view_mean_ari" in summary.columns
    assert "mean_ari" not in summary.columns, \
        "a collapsed frame must not expose an ambiguous bare 'mean_ari'"
    for col in ("best_view_median_ari", "best_view_std_ari",
                "best_view_k_accuracy", "total_runtime_all_views_median_sec",
                "all_views_mean_ari", "n_datasets", "success_rate"):
        assert col in summary.columns


def test_empty_input_is_handled():
    empty = pd.DataFrame(columns=["dataset_id", "method_name", "view_id", "ari",
                                  "status", "runtime_sec"])
    assert frames.build_best_view_frame(empty, verbose=False).empty
    assert frames.summarize_methods(
        pd.DataFrame(columns=["method_name"]), ("method_name",)).empty
