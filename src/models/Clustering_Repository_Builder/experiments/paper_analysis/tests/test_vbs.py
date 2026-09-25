"""VBS engine tests (§29.4): winners, ties, invariants, runtime separation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.Clustering_Repository_Builder.experiments.paper_analysis import vbs
from models.Clustering_Repository_Builder.experiments.paper_analysis import (
    portfolio_registry as P,
)

PIDS = ("mlp_all_variants_vbs", "mlp_strict_deployable_vbs",
        "knn_all_variants_vbs", "knn_strict_deployable_vbs",
        "real_utility_vbs", "learned_source_all_variants_vbs",
        "all_variant_clustopt_vbs", "raw_only_vbs", "softmax_only_vbs",
        "raw_softmax_union_vbs", "autoclust_vbs", "ml2dac_vbs",
        "strict_deployable_clustopt_vbs")


@pytest.fixture
def vbs_frames(pool_best_view):
    return vbs.build_all_vbs_frames(pool_best_view, PIDS, verbose=False)


def test_one_winner_per_dataset(vbs_frames, pool_best_view):
    n_ds = pool_best_view["dataset_id"].nunique()
    for pid, frame in vbs_frames.items():
        assert len(frame) == n_ds, pid
        assert not frame.duplicated(subset=["dataset_id"]).any(), pid


def test_winner_is_the_per_dataset_max(vbs_frames, pool_best_view):
    for pid, frame in vbs_frames.items():
        members = P.portfolio(pid).methods
        want = (pool_best_view[pool_best_view["method_name"].isin(members)]
                .groupby("dataset_id")["best_view_ari"].max())
        got = frame.set_index("dataset_id")["winning_best_view_ari"]
        pd.testing.assert_series_equal(
            got.sort_index().astype(float), want.sort_index().astype(float),
            check_names=False)


def test_vbs_never_below_best_single_candidate(vbs_frames, pool_best_view):
    for pid, frame in vbs_frames.items():
        members = P.portfolio(pid).methods
        per_member = (pool_best_view[pool_best_view["method_name"].isin(members)]
                      .groupby("method_name")["best_view_ari"].mean())
        vbs_mean = frame["winning_best_view_ari"].mean()
        assert vbs_mean >= per_member.max() - 1e-12, pid


def test_superset_vbs_dominates_subset_per_dataset(vbs_frames):
    for sub, sup in P.SUBSET_INVARIANTS:
        if sub not in vbs_frames or sup not in vbs_frames:
            continue
        merged = vbs_frames[sub][["dataset_id", "winning_best_view_ari"]].merge(
            vbs_frames[sup][["dataset_id", "winning_best_view_ari"]],
            on="dataset_id", suffixes=("_sub", "_sup"))
        assert (merged["winning_best_view_ari_sup"]
                >= merged["winning_best_view_ari_sub"] - 1e-12).all(), \
            f"{sup} below its subset {sub}"


def test_winners_are_portfolio_members(vbs_frames):
    for pid, frame in vbs_frames.items():
        members = set(P.portfolio(pid).methods)
        assert set(frame["winning_method_name"].dropna()) <= members, pid


def test_full_portfolio_runtime_is_the_sum(vbs_frames, pool_best_view):
    for pid, frame in vbs_frames.items():
        members = P.portfolio(pid).methods
        want = (pool_best_view[pool_best_view["method_name"].isin(members)]
                .groupby("dataset_id")["total_runtime_all_views_sec"].sum())
        got = frame.set_index("dataset_id")["portfolio_total_execution_runtime_sec"]
        assert np.allclose(got.sort_index(), want.sort_index()), pid


def test_oracle_and_full_runtime_are_separate_columns(vbs_frames):
    frame = vbs_frames["all_variant_clustopt_vbs"]
    assert "selected_member_total_all_views_runtime_sec" in frame.columns
    assert "portfolio_total_execution_runtime_sec" in frame.columns
    # The oracle-selected cost is one member's; the full cost is all 34 members'.
    assert (frame["portfolio_total_execution_runtime_sec"]
            > frame["selected_member_total_all_views_runtime_sec"]).all()


def test_vbs_rows_declare_their_oracle_status(vbs_frames):
    for pid, frame in vbs_frames.items():
        assert frame["selection_type"].unique().tolist() == ["dataset_vbs"], pid
        assert bool(frame["uses_variant_oracle"].all()), pid
        assert bool(frame["uses_view_oracle"].all()), pid
        assert frame["paper_claim_level"].unique().tolist() == ["upper_bound"], pid


def test_contamination_flags_propagate_to_rows(vbs_frames):
    assert bool(vbs_frames["real_utility_vbs"]["contains_real_utility"].all())
    assert not bool(vbs_frames["mlp_all_variants_vbs"]["contains_real_utility"].any())
    assert bool(vbs_frames["mlp_all_variants_vbs"]["contains_oracle_dynamic_k"].all())
    assert not bool(vbs_frames["mlp_all_variants_vbs"]["fully_operational"].any())
    assert bool(vbs_frames["mlp_strict_deployable_vbs"]["fully_operational"].all())
    strict = vbs_frames["strict_deployable_clustopt_vbs"]
    assert not bool(strict["contains_real_utility"].any())
    assert not bool(strict["contains_oracle_dynamic_k"].any())
    assert bool(strict["fully_operational"].all())
    assert (strict["oracle_contamination_reason"].fillna("") == "").all()


def test_full_portfolio_cost_never_below_selected_member(vbs_frames):
    """Realising a VBS costs the whole portfolio, not the picked member."""
    for pid, frame in vbs_frames.items():
        rt = frame[["portfolio_total_execution_runtime_sec",
                    "selected_member_total_all_views_runtime_sec"]].apply(
            pd.to_numeric, errors="coerce").dropna()
        assert len(rt), pid
        assert (rt["portfolio_total_execution_runtime_sec"]
                >= rt["selected_member_total_all_views_runtime_sec"] - 1e-6).all(), pid
        if P.portfolio(pid).size > 1:
            assert (rt["portfolio_total_execution_runtime_sec"]
                    > rt["selected_member_total_all_views_runtime_sec"] + 1e-9).any(), pid


def test_matched_size_table_controls_for_portfolio_size(pool_best_view):
    curves = []
    for pid in ("strict_deployable_clustopt_vbs", "autoclust_vbs", "ml2dac_vbs"):
        c = vbs.random_subset_curve(pool_best_view, pid, sizes=(1, 2, 3, 5, 10),
                                    n_samples=40, seed=5)
        curves.append(c)
    matched = vbs.matched_size_table(pd.concat(curves, ignore_index=True),
                                     sizes=(1, 2))
    assert not matched.empty
    at_two = matched[matched["size"] == 2]
    assert set(at_two["portfolio_id"]) == {"strict_deployable_clustopt_vbs",
                                           "autoclust_vbs", "ml2dac_vbs"}
    # The two-member external portfolios are exhaustive at size 2.
    assert bool(at_two[at_two["portfolio_id"] == "autoclust_vbs"]
                ["is_exhaustive"].iloc[0])
    assert "fully_operational" in matched.columns


def test_random_subset_curve_returns_samples(pool_best_view):
    curve, samples = vbs.random_subset_curve(
        pool_best_view, "mlp_all_variants_vbs", sizes=(2, 3), n_samples=25,
        seed=9, return_samples=True)
    assert not curve.empty and not samples.empty
    assert set(samples["size"]) >= {2, 3}
    assert samples["members"].str.len().gt(0).all()
    # Unique-subset accounting is reported, not assumed.
    for _, r in curve[~curve["skipped"].astype(bool)].iterrows():
        assert r["n_unique_subsets_evaluated"] <= r["n_samples"]
        assert r["n_unique_subsets_evaluated"] <= r["n_possible_subsets"]


def test_greedy_membership_is_flat_and_ordered(pool_best_view):
    greedy = vbs.greedy_portfolio(pool_best_view, "mlp_all_variants_vbs")
    membership = vbs.greedy_membership(greedy)
    assert len(membership) == len(greedy)
    assert list(membership["step"]) == list(range(1, len(greedy) + 1))
    assert (membership["portfolio_size_after_step"]
            == membership["step"]).all()
    assert membership["utility_source"].eq("mlp").all()


def test_tie_handling_is_deterministic_and_recorded():
    """Two members with identical ARI must resolve by median, then runtime,
    then name -- and the reason must be recorded."""
    rows = []
    for method, runtime in (("regressor_top1_raw", 20.0),
                            ("regressor_top3_raw", 10.0)):
        rows.append({
            "dataset_id": "d0", "method_name": method, "family": "F",
            "family_id": "f", "subfamily": "s", "difficulty": "easy",
            "cluster_count": 3, "best_view_ari": 0.7, "has_success": True,
            "view_ari_median": 0.7, "view_ari_mean": 0.7,
            "total_runtime_all_views_sec": runtime,
            "best_view_runtime_sec": runtime / 3, "best_view_id": "x_only",
            "k_correct": True, "selected_k": 3, "true_k": 3,
            "method_short_label": method,
        })
    frame = vbs.build_vbs_frame(pd.DataFrame(rows), "mlp_all_variants_vbs")
    assert len(frame) == 1
    r = frame.iloc[0]
    assert r["tie_count"] == 2
    assert r["tie_broken_by"] == "total_runtime_all_views_sec"
    assert r["winning_method_name"] == "regressor_top3_raw"


def test_validate_vbs_passes(vbs_frames, pool_best_view):
    problems = vbs.validate_vbs(vbs_frames, pool_best_view,
                                subset_pairs=P.SUBSET_INVARIANTS)
    assert problems == []


def test_validate_vbs_catches_a_broken_frame(vbs_frames, pool_best_view):
    broken = {k: v.copy() for k, v in vbs_frames.items()}
    broken["mlp_all_variants_vbs"].loc[broken["mlp_all_variants_vbs"].index[0], "winning_best_view_ari"] = 0.0
    problems = vbs.validate_vbs(broken, pool_best_view)
    assert any("max member ARI" in p for p in problems)


def test_vbs_summary_reports_both_runtimes(vbs_frames):
    summary = vbs.vbs_summary(vbs_frames)
    assert len(summary) == len(PIDS)
    for col in ("selected_member_total_all_views_runtime_median_sec",
                "portfolio_total_execution_runtime_median_sec",
                "portfolio_total_execution_runtime_total_sec"):
        assert col in summary.columns
    assert summary["paper_claim_level"].unique().tolist() == ["upper_bound"]


def test_headroom_reports_paired_and_unpaired(pool_best_view, vbs_frames):
    hr = vbs.headroom(vbs_frames, pool_best_view,
                      reference_methods={"ref": "regressor_top5_raw"})
    assert not hr.empty
    for col in ("headroom_unpaired", "headroom_paired_mean",
                "headroom_paired_median", "n_paired"):
        assert col in hr.columns
    # A VBS containing the reference can never have negative paired headroom.
    containing = hr[hr["portfolio_id"] == "mlp_all_variants_vbs"].iloc[0]
    assert containing["headroom_paired_mean"] >= -1e-12
    assert containing["paper_claim_level"] == "upper_bound"


def test_unique_coverage_is_at_most_win_count(pool_best_view, vbs_frames):
    cov = vbs.unique_coverage(vbs_frames, pool_best_view)
    assert not cov.empty
    assert (cov["n_unique_coverage"] <= cov["n_at_portfolio_best"]).all()
    assert (cov["unique_coverage_share"] <= 1.0).all()


def test_greedy_is_monotone_and_reaches_full_vbs(pool_best_view):
    greedy = vbs.greedy_portfolio(pool_best_view, "mlp_all_variants_vbs")
    assert len(greedy) == 12
    vals = greedy["vbs_best_view_mean_ari"].to_numpy(float)
    assert np.all(np.diff(vals) >= -1e-12), "greedy VBS must be non-decreasing"
    assert vals[-1] == pytest.approx(greedy["full_vbs_best_view_mean_ari"].iloc[0])
    assert greedy["share_of_achievable_gain"].iloc[-1] == pytest.approx(1.0)
    # Marginal gains after the first step must be non-negative.
    assert (greedy["marginal_gain"].iloc[1:] >= -1e-12).all()


def test_greedy_first_pick_is_the_best_single_variant(pool_best_view):
    greedy = vbs.greedy_portfolio(pool_best_view, "mlp_all_variants_vbs")
    members = P.portfolio("mlp_all_variants_vbs").methods
    means = (pool_best_view[pool_best_view["method_name"].isin(members)]
             .groupby("method_name")["best_view_ari"].mean())
    assert greedy.iloc[0]["added_method"] == means.idxmax()


def test_min_size_for_gain_share(pool_best_view):
    greedy = vbs.greedy_portfolio(pool_best_view, "all_variant_clustopt_vbs")
    out = vbs.min_size_for_gain_share(greedy, 0.90)
    assert len(out) == 1
    row = out.iloc[0]
    assert 1 <= int(row["min_size"]) <= int(row["portfolio_size"])
    assert row["achievable_gain"] >= 0.0


def test_random_subset_curve_skips_oversized_requests(pool_best_view):
    curve = vbs.random_subset_curve(pool_best_view, "autoclust_vbs",
                                    sizes=(1, 2, 5, 10), n_samples=30)
    skipped = curve[curve["skipped"]]
    assert set(skipped["size"]) == {5, 10}
    assert (skipped["skip_reason"].str.contains("only 2 members")).all(), \
        "dropped sizes must be logged with a reason, never silently clipped"
    kept = curve[~curve["skipped"]]
    assert set(kept["size"]) == {1, 2}
    full = kept[kept["size"] == 2].iloc[0]
    assert bool(full["is_exhaustive"])
    assert full["vbs_mean"] == pytest.approx(full["full_vbs_mean"])


def test_random_subset_curve_is_seeded(pool_best_view):
    a = vbs.random_subset_curve(pool_best_view, "mlp_all_variants_vbs", sizes=(3,),
                                n_samples=50, seed=11)
    b = vbs.random_subset_curve(pool_best_view, "mlp_all_variants_vbs", sizes=(3,),
                                n_samples=50, seed=11)
    pd.testing.assert_frame_equal(a, b)
    assert (a["ci_low"] <= a["vbs_mean"]).all()
    assert (a["vbs_mean"] <= a["ci_high"]).all()


def test_winner_composition_shares_sum_to_one(vbs_frames):
    comp = vbs.winner_composition(vbs_frames, ("winning_method_name",))
    for pid, grp in comp.groupby("portfolio_id"):
        assert grp["win_share"].sum() == pytest.approx(1.0), pid
