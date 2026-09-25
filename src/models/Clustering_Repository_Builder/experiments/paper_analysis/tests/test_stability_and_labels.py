"""Selection-stability and label-manifest tests (amendment §3.5, §3.7)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.Clustering_Repository_Builder.experiments.paper_analysis import (
    labels, selection, stability,
)
from models.Clustering_Repository_Builder.experiments.paper_analysis.method_registry import (
    CANDIDATE_POOL_BY_ID, METHODS,
)


# --------------------------------------------------------------------------- #
# Selection stability
# --------------------------------------------------------------------------- #
def test_bootstrap_repeats_selection_inside_each_resample(pool_best_view):
    """The whole point: re-select per resample, don't bootstrap a fixed winner.

    A bootstrap of the already-selected method would always report a stability of
    1.0 by construction. With genuine re-selection, a pool of 8 near-equal
    candidates must show a stability below 1.
    """
    result = stability.bootstrap_pool_stability(
        pool_best_view, representative_id="rep",
        candidate_pool_id="mlp_fixed_all", n_samples=300, seed=3)
    assert result.error == ""
    assert result.n_samples == 300
    assert 0.0 < result.point_winner_frequency < 1.0, (
        "a pool of near-equal candidates cannot have a stability of exactly 1; "
        "that would mean the selection step is not being repeated")
    assert result.n_distinct_bootstrap_winners_check() > 1


def test_frequencies_form_a_distribution(pool_best_view):
    result = stability.bootstrap_pool_stability(
        pool_best_view, representative_id="rep",
        candidate_pool_id="knn_fixed_all", n_samples=200, seed=11)
    assert pytest.approx(sum(result.frequencies.values()), abs=1e-9) == 1.0
    assert all(0.0 <= v <= 1.0 for v in result.frequencies.values())
    assert set(result.frequencies) <= set(
        CANDIDATE_POOL_BY_ID["knn_fixed_all"].methods)


def test_entropy_bounds(pool_best_view):
    result = stability.bootstrap_pool_stability(
        pool_best_view, representative_id="rep",
        candidate_pool_id="mlp_fixed_all", n_samples=200, seed=7)
    assert 0.0 <= result.entropy <= result.max_entropy + 1e-9
    assert result.max_entropy == pytest.approx(np.log2(8))


def test_deterministic_under_the_same_seed(pool_best_view):
    a = stability.bootstrap_pool_stability(
        pool_best_view, representative_id="rep",
        candidate_pool_id="mlp_fixed_all", n_samples=150, seed=42)
    b = stability.bootstrap_pool_stability(
        pool_best_view, representative_id="rep",
        candidate_pool_id="mlp_fixed_all", n_samples=150, seed=42)
    assert a.frequencies == b.frequencies
    assert a.point_selected_method == b.point_selected_method


def test_a_dominant_candidate_is_perfectly_stable():
    """Sanity check in the other direction: an obvious winner must score 1.0."""
    rows = []
    for i, method in enumerate(CANDIDATE_POOL_BY_ID["mlp_fixed_all"].methods):
        base = 0.95 if i == 0 else 0.20      # candidate 0 dominates everywhere
        for j in range(30):
            rows.append({"dataset_id": f"d{j}", "method_name": method,
                         "best_view_ari": base, "family": "F",
                         "has_success": True})
    result = stability.bootstrap_pool_stability(
        pd.DataFrame(rows), representative_id="rep",
        candidate_pool_id="mlp_fixed_all", n_samples=100, seed=1)
    assert result.point_winner_frequency == pytest.approx(1.0)
    assert result.entropy == pytest.approx(0.0)


def test_summary_and_frequency_frames(pool_best_view):
    sels = []
    for rep_id, pool_id in (("c4_best_observed_mlp_fixed", "mlp_fixed_all"),
                            ("c4_best_observed_knn_fixed", "knn_fixed_all")):
        sel, _ = selection.select_representative(
            pool_best_view, comparison_id="test", representative_id=rep_id,
            candidate_pool_id=pool_id, split_id=1)
        sels.append(sel)
    summary, freq, results = stability.stability_for_selections(
        pool_best_view, sels, n_samples=150, seed=5, verbose=False)
    assert len(summary) == 2
    assert stability.validate_stability(summary) == []
    for col in ("selection_frequency", "runner_up_method",
                "runner_up_frequency",
                "entropy_of_selected_method_distribution",
                "probability_point_winner_remains_winner",
                "mean_delta_to_bootstrap_winner"):
        assert col in summary.columns
    # One frequency row per candidate per pool.
    assert len(freq) == 16
    assert (summary["claim_level"] == "post_selection_exploratory").all()

    # attach_stability writes the result back onto the frozen selections.
    stability.attach_stability(sels, results)
    for sel in sels:
        assert sel.selection_stability is not None
        rec = sel.to_record()
        assert rec["bootstrap_selection_frequency"] is not None
        assert rec["selection_stability"] == sel.selection_stability


def test_family_scope_selections_are_skipped(pool_best_view):
    fam_sels, _ = selection.select_family_representatives(
        pool_best_view, comparison_id="test", representative_id="rep",
        candidate_pool_id="mlp_fixed_all", split_id=1)
    summary, freq, _ = stability.stability_for_selections(
        pool_best_view, fam_sels, n_samples=50, seed=1, verbose=False)
    assert summary.empty and freq.empty


# --------------------------------------------------------------------------- #
# Representative reporting
# --------------------------------------------------------------------------- #
def test_selection_record_names_the_concrete_method(pool_best_view):
    sel, _ = selection.select_representative(
        pool_best_view, comparison_id="test",
        representative_id="c4_best_observed_mlp_fixed",
        candidate_pool_id="mlp_fixed_all", split_id=1)
    rec = sel.to_record()
    for col in ("representative_id", "short_label", "display_name",
                "selected_method_name", "selected_method_folder",
                "candidate_pool_id", "candidate_pool_size", "candidate_methods",
                "selection_type", "selection_scope", "selection_split",
                "selection_metric", "is_independently_preselected",
                "paper_claim_level", "selected_best_view_mean_ari",
                "selected_best_view_median_ari", "selected_best_view_std_ari",
                "selected_k_accuracy",
                "selected_median_total_all_views_runtime_sec",
                "runner_up_method", "runner_up_short_label",
                "runner_up_mean_best_view_ari", "delta_to_runner_up",
                "n_tied_on_primary", "tie_break_reason"):
        assert col in rec, col
    assert rec["selected_method_name"] == sel.selected_method
    assert rec["selected_method_folder"] == sel.selected_method
    assert rec["short_label"]


def test_runner_up_is_the_second_ranked_candidate(pool_best_view):
    sel, table = selection.select_representative(
        pool_best_view, comparison_id="test", representative_id="rep",
        candidate_pool_id="mlp_fixed_all", split_id=1)
    assert sel.runner_up_method == table.iloc[1]["method_name"]
    assert sel.delta_to_runner_up == pytest.approx(
        table.iloc[0]["best_view_mean_ari"] - table.iloc[1]["best_view_mean_ari"])
    assert sel.delta_to_runner_up >= 0.0


def test_tie_break_reason_is_human_readable(pool_best_view):
    sel, _ = selection.select_representative(
        pool_best_view, comparison_id="test", representative_id="rep",
        candidate_pool_id="mlp_fixed_all", split_id=1)
    assert sel.tie_break_reason == "unique winner on the primary metric"


# --------------------------------------------------------------------------- #
# Label manifest
# --------------------------------------------------------------------------- #
def test_label_manifest_covers_methods_and_portfolios():
    manifest = labels.build_label_manifest()
    assert len(manifest) == len(METHODS) + 17
    assert set(manifest["candidate_type"]) == {"method", "portfolio"}
    assert labels.validate_labels(manifest) == []


def test_label_manifest_includes_representatives():
    frozen = pd.DataFrame([{
        "representative_id": "c4_best_observed_mlp_fixed",
        "selected_method": "regressor_top5_raw", "selection_scope": "global",
        "candidate_pool_id": "mlp_fixed_all", "candidate_pool_size": 8}])
    manifest = labels.build_label_manifest(frozen)
    reps = manifest[manifest["candidate_type"] == "representative"]
    assert len(reps) == 1
    row = reps.iloc[0]
    assert row["short_label"] == "MLP T5-R"
    assert "regressor_top5_raw" in row["notes"]
    assert "retrospective" in row["oracle_status"]
    assert labels.validate_labels(manifest) == []


def test_labels_are_unique_within_type():
    manifest = labels.build_label_manifest()
    for kind in ("method", "portfolio"):
        sub = manifest[manifest["candidate_type"] == kind]
        assert not sub["short_label"].duplicated().any(), kind


def test_labels_fit_the_length_budget():
    manifest = labels.build_label_manifest()
    assert (manifest["label_length"] <= labels.MAX_LABEL_LENGTH).all()


def test_oracle_assisted_variants_are_marked():
    manifest = labels.build_label_manifest()
    methods = manifest[manifest["candidate_type"] == "method"]
    for m in METHODS:
        if not m.is_oracle_assisted:
            continue
        row = methods[methods["identifier"] == m.method_name].iloc[0]
        assert "DynRK" in row["short_label"], m.method_name
        assert "oracle-assisted" in row["oracle_status"]


def test_unmatched_search_spaces_are_marked():
    manifest = labels.build_label_manifest()
    methods = manifest[manifest["candidate_type"] == "method"]
    for m in METHODS:
        if not m.is_external:
            continue
        row = methods[methods["identifier"] == m.method_name].iloc[0]
        if m.same_search_space:
            assert row["search_space_status"].startswith("matched")
        else:
            assert row["search_space_status"].startswith("UNMATCHED")
            assert (row["short_label"].endswith("~")
                    or "Paper" in row["short_label"]
                    or row["short_label"] == "AML4C")


def test_portfolio_labels_state_vbs():
    manifest = labels.build_label_manifest()
    for _, r in manifest[manifest["candidate_type"] == "portfolio"].iterrows():
        assert "VBS" in r["short_label"] or "Strict" in r["short_label"]
        assert "oracle" in r["oracle_status"]


def test_validate_catches_a_duplicate_label():
    manifest = labels.build_label_manifest()
    broken = pd.concat([manifest, manifest.iloc[[0]]], ignore_index=True)
    problems = labels.validate_labels(broken)
    assert any("duplicate" in p for p in problems)


def test_manifest_markdown_documents_the_conventions():
    text = labels.manifest_markdown(labels.build_label_manifest())
    assert "# Label Manifest" in text
    assert "DynRK" in text
    assert "oracle-assisted" in text
    assert "non-deployable" in text
