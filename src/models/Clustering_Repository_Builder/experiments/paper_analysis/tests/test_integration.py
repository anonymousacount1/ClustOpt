"""Integration test: collect real Split-1 datasets end to end.

Marked ``slow`` because it reads the real per-dataset output tree. Skips itself
cleanly when the analyzed-data root is unavailable, so the unit suite still runs
on a machine without the experiment outputs.

Run just this file with::

    python -m pytest models/Clustering_Repository_Builder/experiments/paper_analysis/tests/test_integration.py -q
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from models.Clustering_Repository_Builder.experiments.paper_analysis import (
    capability, collector, frames, metric_analysis, metric_registry, paths,
    portfolio_registry, selection, vbs,
)
from models.Clustering_Repository_Builder.experiments.paper_analysis.method_registry import (
    METHOD_NAMES, VIEW_IDS,
)

pytestmark = pytest.mark.slow

N_DATASETS = 8


def _split_csv() -> Path:
    return (paths.repo_root() / "results_analysis" / "clustering_repository"
            / "analyzed_data" / "experiment_splits"
            / "dataset_split_assignments.csv")


@pytest.fixture(scope="module")
def real_frames():
    csv = _split_csv()
    if not paths.exists(csv):
        pytest.skip(f"analyzed-data root unavailable ({csv})")
    assignments = paths.read_csv(csv)
    run_frame, stats = collector.collect_run_frame(
        assignments=assignments, split_id=1, limit_datasets=N_DATASETS,
        max_workers=8, verbose=False)
    if run_frame.empty:
        pytest.skip("no runs collected for split 1")
    best_view = frames.build_best_view_frame(run_frame, verbose=False)
    dataset_level = frames.build_dataset_level_frame(best_view)
    return run_frame, best_view, dataset_level, stats


def test_collects_every_registered_method(real_frames):
    run_frame, _bv, _dl, stats = real_frames
    assert run_frame["method_name"].nunique() == len(METHOD_NAMES) == 51
    assert run_frame["dataset_id"].nunique() == N_DATASETS
    assert set(run_frame["view_id"]) == set(VIEW_IDS)
    assert len(run_frame) == N_DATASETS * 51 * 3


def test_no_missing_runs_and_no_provenance_mismatch(real_frames):
    _rf, _bv, _dl, stats = real_frames
    assert stats.n_missing == 0, stats.warnings
    assert stats.n_provenance_mismatch == 0, \
        "the registry taxonomy disagrees with what a run recorded about itself"


def test_frames_validate_on_real_data(real_frames):
    run_frame, best_view, _dl, _stats = real_frames
    assert frames.validate_frames(run_frame, best_view) == []


def test_every_selected_metric_is_in_the_taxonomy(real_frames):
    run_frame, _bv, _dl, _stats = real_frames
    observed = collector.observed_selected_metrics(run_frame)
    assert observed, "no metric selections were recorded"
    assert metric_registry.validate_metric_registry(observed) == []


def test_metric_usage_validates_on_real_data(real_frames):
    _rf, best_view, _dl, _stats = real_frames
    long_frame = metric_analysis.explode_selections(best_view)
    assert not long_frame.empty
    assert metric_analysis.validate_usage(long_frame) == []
    audit = metric_analysis.usage_audit(long_frame)
    assert audit["n_unknown_metric_rows"] == 0
    # Two recorded spellings alias onto canonical registry names.
    assert audit["n_distinct_recorded_spellings"] >= audit["n_distinct_metrics"]


def test_capability_matrix_matches_the_known_schema_split(real_frames):
    run_frame, _bv, _dl, _stats = real_frames
    matrix = capability.build_capability_matrix(run_frame, verbose=False)
    assert len(matrix) == 51
    clustopt = matrix[matrix["schema"] == "clustopt"]
    external = matrix[matrix["schema"] == "external"]
    assert len(clustopt) == 40 and len(external) == 11
    # ClustOpt runs expose metric selection and a candidate trace; externals do not.
    assert bool(clustopt["selected_metric_weights"].all())
    assert bool(clustopt["per_trial_trace"].all())
    assert bool(clustopt["trace_best_gap_inputs"].all())
    assert not bool(external["selected_metric_weights"].any())
    assert not bool(external["per_trial_trace"].any())
    # Both schemas record ARI and all-view runtime.
    assert bool(matrix["external_ari"].all())
    assert bool(matrix["all_view_runtime"].all())
    # No adapter records end-to-end runtime components.
    assert not bool(matrix["end_to_end_runtime"].any())


def test_preflight_skips_only_the_unsupported_pairs(real_frames):
    run_frame, _bv, _dl, _stats = real_frames
    matrix = capability.build_capability_matrix(run_frame, verbose=False)
    supported, warns = capability.preflight(
        matrix, comparison_id="integration", methods=list(METHOD_NAMES),
        analyses=["primary_ari", "best_view", "k_analysis",
                  "runtime_total_all_views", "metric_selection",
                  "trace_best_gap", "end_to_end_runtime"], verbose=False)
    assert len(supported["primary_ari"]) == 51
    assert len(supported["runtime_total_all_views"]) == 51
    assert len(supported["metric_selection"]) == 40
    assert len(supported["trace_best_gap"]) == 40
    assert len(supported["end_to_end_runtime"]) == 0
    frame = capability.warnings_frame(warns)
    assert frame["reason"].str.len().gt(0).all()


def test_selection_and_vbs_run_on_real_data(real_frames):
    _rf, best_view, _dl, _stats = real_frames
    sel, table = selection.select_representative(
        best_view, comparison_id="integration",
        representative_id="c4_best_observed_mlp_fixed",
        candidate_pool_id="mlp_fixed_all", split_id=1)
    assert sel is not None
    assert sel.selected_method in table["method_name"].tolist()
    assert sel.to_record()["is_independently_preselected"] is False

    pids = list(portfolio_registry.COMPARISON_05_PORTFOLIOS)
    vbs_frames = vbs.build_all_vbs_frames(best_view, pids, verbose=False)
    assert vbs.validate_vbs(vbs_frames, best_view,
                            subset_pairs=portfolio_registry.SUBSET_INVARIANTS) == []
    summary = vbs.vbs_summary(vbs_frames)
    assert len(summary) == len(pids)
    # Every VBS must be at least as good as the best single member.
    for pid in pids:
        members = portfolio_registry.portfolio(pid).methods
        best_single = (best_view[best_view["method_name"].isin(members)]
                       .groupby("method_name")["best_view_ari"].mean().max())
        vbs_mean = summary[summary["portfolio_id"] == pid
                          ]["best_view_mean_ari"].iloc[0]
        assert vbs_mean >= best_single - 1e-12, pid


def test_runtime_semantics_on_real_data(real_frames):
    _rf, best_view, _dl, _stats = real_frames
    total = pd.to_numeric(best_view["total_runtime_all_views_sec"], errors="coerce")
    single = pd.to_numeric(best_view["best_view_runtime_sec"], errors="coerce")
    ok = total.notna() & single.notna()
    # The all-views cost must exceed any single view's cost.
    assert (total[ok] >= single[ok] - 1e-9).all()
    assert not bool(best_view["has_end_to_end_runtime"].any())


def test_dry_run_stage_writes_nothing(tmp_path):
    from models.Clustering_Repository_Builder.experiments.paper_analysis.run_analysis import (
        main,
    )
    csv = _split_csv()
    if not paths.exists(csv):
        pytest.skip("analyzed-data root unavailable")
    rc = main(["--stage", "preliminary-selection", "--split-id", "1",
               "--limit-datasets", "3", "--collect-workers", "4",
               "--output-root", str(tmp_path / "out"), "--dry-run",
               "--no-excel", "--no-plots", "--no-stats", "--quiet"])
    assert rc == 0
    assert not paths.exists(tmp_path / "out" / "analysis_manifest.csv")
