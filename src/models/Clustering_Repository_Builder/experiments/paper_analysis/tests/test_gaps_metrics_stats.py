"""Gap, metric-usage and statistics tests (§29.5-29.7)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.Clustering_Repository_Builder.experiments.paper_analysis import (
    frames, gaps, metric_analysis, stats,
)


# --------------------------------------------------------------------------- #
# Gaps (§29.6): every formula and sign
# --------------------------------------------------------------------------- #
def test_gap_definitions_are_complete():
    frame = gaps.gap_definitions_frame()
    assert set(frame["gap_type"]) == set(gaps.GAP_TYPES)
    assert len(frame) == 7
    assert (frame["sign_convention"] == "positive = something better was available").all()
    assert frame["availability"].str.len().gt(0).all()


def test_view_gap_formula_and_sign(tiny_best_view):
    summary = gaps.view_gap_summary(tiny_best_view, ("method_name",))
    row = summary[summary["method_name"] == "regressor_top5_raw"].iloc[0]
    assert row["gap_type"] == "view_gap"
    # best_view_ari >= mean view ari, so the gap is never negative.
    assert row["view_gap_mean"] >= -1e-12
    assert row["best_view_mean_ari"] >= row["all_views_mean_ari"] - 1e-12
    assert row["paper_claim_level"] == "diagnostic_oracle"
    shares = (row["best_view_x_only_share"] + row["best_view_y_only_share"]
              + row["best_view_xy_2d_share"])
    assert shares == pytest.approx(1.0)


def test_paired_gap_row_sign_is_reference_minus_comparison(pool_best_view):
    row = gaps.paired_gap_row(
        pool_best_view, gap_type="weighting_gap",
        reference="regressor_top1_raw", comparison="regressor_top1_softmax_t05")
    a = pool_best_view[pool_best_view["method_name"] == "regressor_top1_raw"]
    b = pool_best_view[pool_best_view["method_name"] == "regressor_top1_softmax_t05"]
    merged = a[["dataset_id", "best_view_ari"]].merge(
        b[["dataset_id", "best_view_ari"]], on="dataset_id", suffixes=("_a", "_b"))
    want = (merged["best_view_ari_a"] - merged["best_view_ari_b"]).mean()
    assert row["gap_mean"] == pytest.approx(want)
    assert row["n_paired"] == len(merged)
    shares = (row["share_reference_better"] + row["share_equal"]
              + row["share_comparison_better"])
    assert shares == pytest.approx(1.0)


def test_utility_prediction_gap_uses_real_as_reference(pool_best_view):
    out = gaps.utility_prediction_gaps(pool_best_view)
    assert not out.empty
    assert (out["gap_type"] == "utility_prediction_gap").all()
    assert out["reference_uses_real_utility"].all(), \
        "the reference arm of a prediction gap must be the real-utility method"
    assert set(out["utility_source"]) == {"mlp", "knn"}
    # Every row is a diagnostic because the reference is an oracle.
    assert out["paper_claim_level"].unique().tolist() == ["diagnostic_oracle"]


def test_weighting_gap_covers_all_seventeen_matched_pairs(pool_best_view):
    out = gaps.weighting_gaps(pool_best_view)
    assert len(out) == 17
    assert (out["gap_type"] == "weighting_gap").all()
    assert out["variant_family"].nunique() == 17


def test_metric_count_gap_reference_is_oracle_assisted(pool_best_view):
    out = gaps.metric_count_gaps(pool_best_view)
    assert len(out) == 4
    assert (out["gap_type"] == "metric_count_gap").all()
    assert out["reference_is_oracle_assisted"].all()
    assert out["paper_claim_level"].unique().tolist() == ["diagnostic_oracle"]
    assert set(out["utility_source"]) == {"mlp", "knn"}


def test_external_leader_gap_labels_the_asymmetry(pool_best_view):
    out = gaps.external_leader_gaps(
        pool_best_view,
        external_methods=["AutoClust_Extended_InDomain_same_search_space"],
        clustopt_representatives={"best_mlp": "regressor_top5_raw"})
    assert len(out) == 1
    row = out.iloc[0]
    assert row["gap_type"] == "external_leader_gap"
    assert row["comparison_selection_type"] == "retrospective_global"
    assert row["paper_claim_level"] == "retrospective_summary"
    assert "RETROSPECTIVE" in row["notes"]


def test_trace_gap_handles_a_missing_trace():
    fields, warn = gaps.trace_gap_for_run(None, selected_ari=0.5)
    assert warn == "missing_trace"
    assert np.isnan(fields["search_selection_gap"])
    fields, warn = gaps.trace_gap_for_run("/nope/trace.csv", selected_ari=0.5)
    assert warn == "missing_trace"


def test_trace_gap_computes_from_a_real_csv(tmp_path):
    csv = tmp_path / "clustopt_results.csv"
    pd.DataFrame({
        "algorithm": ["kmeans", "dbscan", "gmm"],
        "ARI": [0.40, 0.90, 0.60],
        "aggregate_score": [0.90, 0.20, 0.50],   # objective prefers the 0.40 run
        "valid": [True, True, True],
    }).to_csv(csv, index=False)
    fields, warn = gaps.trace_gap_for_run(str(csv), selected_ari=0.40)
    assert warn is None
    assert fields["trace_best_ari"] == pytest.approx(0.90)
    assert fields["selected_ari"] == pytest.approx(0.40)
    assert fields["search_selection_gap"] == pytest.approx(0.50)
    assert fields["selected_is_trace_best"] == 0.0
    assert fields["n_trace_candidates"] == 3
    assert fields["trace_best_selected_algorithm"] == "dbscan"
    # The trace best is ranked 3rd by the internal objective -> objective failure.
    assert fields["trace_best_rank_by_internal_objective"] == pytest.approx(3.0)


def test_trace_gap_zero_when_objective_picks_the_best(tmp_path):
    csv = tmp_path / "clustopt_results.csv"
    pd.DataFrame({"ARI": [0.4, 0.9], "aggregate_score": [0.2, 0.9]}).to_csv(
        csv, index=False)
    fields, warn = gaps.trace_gap_for_run(str(csv), selected_ari=0.9)
    assert warn is None
    assert fields["search_selection_gap"] == pytest.approx(0.0)
    assert fields["selected_is_trace_best"] == 1.0


def test_trace_gap_reports_a_missing_ari_column(tmp_path):
    csv = tmp_path / "clustopt_results.csv"
    pd.DataFrame({"aggregate_score": [0.2, 0.9]}).to_csv(csv, index=False)
    _fields, warn = gaps.trace_gap_for_run(str(csv), selected_ari=0.9)
    assert warn == "trace_has_no_ari_column"


# --------------------------------------------------------------------------- #
# Metric usage (§29.5): normalisation correctness
# --------------------------------------------------------------------------- #
def test_explode_selections_skips_methods_without_a_record(tiny_best_view):
    long_frame = metric_analysis.explode_selections(tiny_best_view)
    assert not long_frame.empty
    assert long_frame["is_known_metric"].all()
    assert set(long_frame["metric_category"]) <= {"core_cvi", "structural",
                                                  "pattern", "image"}


def test_shares_sum_to_one_within_each_method(pool_best_view):
    long_frame = metric_analysis.explode_selections(pool_best_view)
    usage = metric_analysis.metric_usage(long_frame, ("method_name",))
    sums = usage.groupby("method_name")["share_of_selected_slots"].sum()
    assert np.allclose(sums, 1.0)
    cats = metric_analysis.category_usage(long_frame, ("method_name",))
    sums = cats.groupby("method_name")["selected_slot_share"].sum()
    assert np.allclose(sums, 1.0)
    sums = cats.groupby("method_name")["weight_mass_share"].sum()
    assert np.allclose(sums, 1.0)


def test_selection_rate_per_run_is_comparable_across_topk():
    """A Top-1 and a Top-10 method that always pick silhouette must both show a
    selection rate of 1.0, even though their raw counts differ 10-fold."""
    rows = []
    for method, metrics in (("regressor_top1_raw", ["silhouette"]),
                            ("regressor_top10_raw",
                             ["silhouette", "dbcv", "s_dbw", "avg_pca_isotropy",
                              "neighborhood_purity", "hollow_score",
                              "gridness_2d", "arc_circle_fit", "ladderness",
                              "parallel_bands"])):
        for i in range(10):
            for m in metrics:
                rows.append({
                    "dataset_id": f"d{i}", "family": "F", "family_id": "f",
                    "subfamily": "s", "difficulty": "easy", "cluster_count": 3,
                    "method_name": method, "method_short_label": method,
                    "method_family": "clustopt_mlp", "utility_source": "mlp",
                    "k_policy": "fixed", "weighting_mode": "raw",
                    "metric_name": m, "metric_name_as_recorded": m,
                    "metric_category": "core_cvi", "metric_subcategory": "classic_cvi",
                    "is_new_metric": False, "is_known_metric": True,
                    "weight": 1.0 / len(metrics), "utility": 0.8,
                    "is_top_metric": m == metrics[0],
                    "n_selected_in_run": len(metrics),
                })
    long_frame = pd.DataFrame(rows)
    usage = metric_analysis.metric_usage(long_frame, ("method_name",))
    sil = usage[usage["metric_name"] == "silhouette"].set_index("method_name")
    assert sil.loc["regressor_top1_raw", "selection_rate_per_run"] == pytest.approx(1.0)
    assert sil.loc["regressor_top10_raw", "selection_rate_per_run"] == pytest.approx(1.0)
    # Raw counts are equal here (10 each) but the SLOT shares differ, as they must.
    assert sil.loc["regressor_top1_raw", "share_of_selected_slots"] == pytest.approx(1.0)
    assert sil.loc["regressor_top10_raw", "share_of_selected_slots"] == pytest.approx(0.1)
    # Unconditional weight mass is comparable: each run's weights sum to 1.
    assert sil.loc["regressor_top1_raw", "mean_unconditional_weight_mass"] \
        == pytest.approx(1.0)
    assert sil.loc["regressor_top10_raw", "mean_unconditional_weight_mass"] \
        == pytest.approx(0.1)


def test_new_metric_usage_fields(pool_best_view):
    long_frame = metric_analysis.explode_selections(pool_best_view)
    per_run = metric_analysis.per_run_new_metric_usage(long_frame)
    assert (per_run["new_metric_count"] <= per_run["n_selected"]).all()
    assert per_run["new_metric_fraction"].between(0.0, 1.0).all()
    assert (per_run["new_metric_weight_mass"] >= -1e-12).all()
    agg = metric_analysis.new_metric_usage(long_frame, ("method_name",))
    assert agg["share_runs_with_new_metric"].between(0.0, 1.0).all()


def test_new_metric_vs_gain_is_labelled_association_only(pool_best_view):
    long_frame = metric_analysis.explode_selections(pool_best_view)
    out = metric_analysis.new_metric_vs_ari_gain(
        pool_best_view, long_frame, reference_method="uniform_classic_cvi")
    assert not out.empty
    assert out["interpretation_note"].str.contains("ASSOCIATION ONLY").all()
    assert "pearson_r" in out.columns and "spearman_rho" in out.columns


def test_usage_validation_catches_an_unknown_metric():
    long_frame = pd.DataFrame([{
        "dataset_id": "d0", "method_name": "regressor_top1_raw",
        "metric_name": "made_up_metric", "metric_name_as_recorded": "made_up_metric",
        "metric_category": "unknown", "metric_subcategory": "unknown",
        "is_new_metric": False, "is_known_metric": False, "weight": 1.0,
        "utility": 0.5, "is_top_metric": True, "n_selected_in_run": 1,
        "family": "F", "family_id": "f", "subfamily": "s", "difficulty": "easy",
        "cluster_count": 3, "method_short_label": "MLP T1-R",
        "method_family": "clustopt_mlp", "utility_source": "mlp",
        "k_policy": "fixed", "weighting_mode": "raw",
    }])
    problems = metric_analysis.validate_usage(long_frame)
    assert any("missing from the taxonomy" in p for p in problems)
    audit = metric_analysis.usage_audit(long_frame)
    assert audit["unknown_metrics"] == ["made_up_metric"]


# --------------------------------------------------------------------------- #
# Statistics (§29.7)
# --------------------------------------------------------------------------- #
def _dl(pool_best_view):
    return frames.build_dataset_level_frame(pool_best_view)


def test_pairing_is_by_dataset_id(pool_best_view):
    dl = _dl(pool_best_view)
    a, b, ids = stats.paired_values(dl, "regressor_top1_raw", "knn_top1_raw")
    assert a.size == b.size == ids.size
    assert len(set(ids)) == ids.size
    # Values must correspond to the same dataset on both sides.
    for dataset_id, va, vb in zip(ids, a, b):
        want_a = dl[(dl["dataset_id"] == dataset_id)
                    & (dl["method_name"] == "regressor_top1_raw")
                    ]["best_view_ari"].iloc[0]
        want_b = dl[(dl["dataset_id"] == dataset_id)
                    & (dl["method_name"] == "knn_top1_raw")
                    ]["best_view_ari"].iloc[0]
        assert va == pytest.approx(want_a)
        assert vb == pytest.approx(want_b)


def test_representative_arms_are_marked_post_selection():
    hyp = stats.Hypothesis(
        hypothesis_id="h", family="f", method_a="regressor_top5_raw",
        method_b="AutoClust_Extended_InDomain_same_search_space",
        claim_level=stats.CLAIM_CONFIRMATORY, a_is_representative=True)
    assert hyp.effective_claim_level == stats.CLAIM_POST_SELECTION
    clean = stats.Hypothesis(hypothesis_id="h2", family="f",
                             method_a="all_metrics_uniform",
                             method_b="uniform_classic_cvi",
                             claim_level=stats.CLAIM_CONFIRMATORY)
    assert clean.effective_claim_level == stats.CLAIM_CONFIRMATORY


def test_paired_test_row_is_complete(pool_best_view):
    dl = _dl(pool_best_view)
    hyp = stats.Hypothesis("h", "fam", "regressor_top10_raw", "regressor_top1_raw",
                           direction="a_greater", is_primary=True,
                           claim_level=stats.CLAIM_CONFIRMATORY)
    row = stats.paired_test(dl, hyp, n_bootstrap=200)
    assert row["n_paired"] == pool_best_view["dataset_id"].nunique()
    assert row["mean_delta"] == pytest.approx(row["mean_a"] - row["mean_b"])
    assert row["mean_delta_ci_low"] <= row["mean_delta_boot"] <= row["mean_delta_ci_high"]
    shares = row["share_a_better"] + row["share_tied"] + row["share_b_better"]
    assert shares == pytest.approx(1.0)
    assert -1.0 <= row["rank_biserial"] <= 1.0


def test_rank_biserial_extremes():
    a = np.array([0.5, 0.6, 0.7])
    b = np.array([0.1, 0.2, 0.3])
    assert stats.rank_biserial(a, b) == pytest.approx(1.0)
    assert stats.rank_biserial(b, a) == pytest.approx(-1.0)
    assert stats.rank_biserial(a, a) == pytest.approx(0.0)


def test_holm_correction_within_family():
    frame = pd.DataFrame({
        "hypothesis_family": ["A", "A", "A", "B"],
        "wilcoxon_p": [0.01, 0.02, 0.04, 0.01],
    })
    out = stats.holm_correct(frame, alpha=0.05)
    fam_a = out[out["hypothesis_family"] == "A"].sort_values("wilcoxon_p")
    # Holm: p*(m), p*(m-1), p*(m-2), then enforce monotonicity.
    assert fam_a["holm_p"].tolist() == pytest.approx([0.03, 0.04, 0.04])
    assert (out["holm_family_size"] == [3, 3, 3, 1]).all()
    # Family B is corrected alone, so its single p-value is unchanged.
    assert out[out["hypothesis_family"] == "B"]["holm_p"].iloc[0] == pytest.approx(0.01)


def test_holm_is_monotone_non_decreasing():
    frame = pd.DataFrame({"hypothesis_family": ["A"] * 5,
                          "wilcoxon_p": [0.001, 0.03, 0.031, 0.2, 0.9]})
    out = stats.holm_correct(frame).sort_values("wilcoxon_p")
    assert (out["holm_p"].diff().dropna() >= -1e-12).all()


def test_selection_aware_bootstrap_is_wider_and_reports_stability(pool_best_view):
    from models.Clustering_Repository_Builder.experiments.paper_analysis.method_registry import (
        CANDIDATE_POOL_BY_ID,
    )
    pool = CANDIDATE_POOL_BY_ID["mlp_fixed_all"]
    rec = stats.selection_aware_bootstrap(
        pool_best_view, candidate_methods=pool.methods,
        competitor_method="AutoClust_Extended_InDomain_same_search_space",
        n_samples=200, seed=3)
    assert rec["n_samples"] == 200
    assert rec["candidate_pool_size"] == 8
    assert rec["ci_low"] <= rec["bootstrap_mean_delta"] <= rec["ci_high"]
    assert 0.0 < rec["selection_stability"] <= 1.0
    assert rec["claim_level"] == stats.CLAIM_POST_SELECTION
    assert rec["n_distinct_winners"] >= 1
    assert "re-selection" in rec["notes"]


def test_selection_aware_bootstrap_handles_a_missing_competitor(pool_best_view):
    rec = stats.selection_aware_bootstrap(
        pool_best_view, candidate_methods=["regressor_top1_raw"],
        competitor_method="not_a_method", n_samples=10)
    assert rec["n_samples"] == 0
    assert "error" in rec


def test_winrates_sum_to_one(pool_best_view):
    dl = _dl(pool_best_view)
    wr = stats.winrates(dl, methods=["regressor_top5_raw"],
                        references=["knn_top5_raw", "all_metrics_uniform"])
    assert len(wr) == 2
    assert np.allclose(wr["win_rate"] + wr["tie_rate"] + wr["loss_rate"], 1.0)


def test_run_hypotheses_applies_holm_and_labels(pool_best_view):
    dl = _dl(pool_best_view)
    hyps = [
        stats.Hypothesis("h1", "famA", "regressor_top10_raw", "regressor_top1_raw",
                         claim_level=stats.CLAIM_CONFIRMATORY, is_primary=True),
        stats.Hypothesis("h2", "famA", "knn_top10_raw", "knn_top1_raw",
                         claim_level=stats.CLAIM_CONFIRMATORY, is_primary=True),
        stats.Hypothesis("h3", "famB", "regressor_top5_raw", "knn_top5_raw",
                         claim_level=stats.CLAIM_EXPLORATORY,
                         a_is_representative=True),
    ]
    out = stats.run_hypotheses(dl, hyps, n_bootstrap=100, verbose=False)
    assert len(out) == 3
    assert set(out["hypothesis_family"]) == {"famA", "famB"}
    assert out[out["hypothesis_id"] == "h3"]["claim_level"].iloc[0] \
        == stats.CLAIM_POST_SELECTION
    assert out["holm_p"].notna().all()
