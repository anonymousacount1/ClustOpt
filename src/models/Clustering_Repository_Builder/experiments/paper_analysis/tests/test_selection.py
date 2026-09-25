"""Retrospective selection engine tests (§29.3)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.Clustering_Repository_Builder.experiments.paper_analysis import (
    frames, selection,
)
from models.Clustering_Repository_Builder.experiments.paper_analysis.method_registry import (
    CANDIDATE_POOL_BY_ID,
)


def _select(best_view, pool_id="mlp_fixed_all", **kwargs):
    return selection.select_representative(
        best_view, comparison_id="test", representative_id="rep",
        candidate_pool_id=pool_id, split_id=1, **kwargs)


def test_selects_highest_mean_best_view_ari(pool_best_view):
    sel, table = _select(pool_best_view)
    assert sel is not None
    pool = CANDIDATE_POOL_BY_ID["mlp_fixed_all"]
    means = (pool_best_view[pool_best_view["method_name"].isin(pool.methods)]
             .groupby("method_name")["best_view_ari"].mean())
    assert sel.selected_method == means.idxmax()
    assert sel.stats["best_view_mean_ari"] == pytest.approx(means.max())
    assert table.iloc[0]["is_selected"]
    assert list(table["selection_rank"]) == list(range(1, len(table) + 1))


def test_selection_metadata_is_complete_and_honest(pool_best_view):
    sel, _ = _select(pool_best_view)
    rec = sel.to_record()
    assert rec["selection_type"] == "retrospective_global"
    assert rec["selection_scope"] == "global"
    assert rec["selection_metric"] == "best_view_mean_ari"
    assert rec["selection_split"] == 1
    assert rec["is_independently_preselected"] is False
    assert rec["paper_claim_level"] == "retrospective_summary"
    assert rec["uses_view_oracle"] is True
    assert rec["uses_variant_oracle"] is True
    assert rec["candidate_pool_size"] == 8
    assert rec["tie_break_rules"] == selection.TIE_BREAK_DESCRIPTION


def test_deterministic_under_row_shuffling(pool_best_view):
    assert selection.selection_is_deterministic(
        pool_best_view, comparison_id="test", representative_id="rep",
        candidate_pool_id="mlp_fixed_all", split_id=1, n_repeats=5)


def test_tie_break_chain_order():
    """Two candidates with an identical mean must be split by the median, then
    the std, then K accuracy, then runtime, then the name."""
    rows = []
    # a and b have the same mean (0.5) but b has the higher median.
    plan = {
        "regressor_top1_raw": [0.4, 0.6, 0.5, 0.5],      # median 0.50
        "regressor_top3_raw": [0.3, 0.7, 0.55, 0.45],    # median 0.50
        "regressor_top5_raw": [0.45, 0.55, 0.6, 0.4],    # median 0.50
    }
    for method, aris in plan.items():
        for i, ari in enumerate(aris):
            rows.append({
                "dataset_id": f"d{i}", "method_name": method, "family": "F",
                "best_view_ari": ari, "has_success": True,
                "total_runtime_all_views_sec": 10.0 + len(method),
                "k_correct": True, "k_signed_error": 0, "k_abs_error": 0,
                "method_short_label": method, "view_ari_mean": ari,
                "view_ari_median": ari, "view_ari_std": 0.0, "view_gap": 0.0,
                "best_view_runtime_sec": 3.0, "selected_metric_count": 3,
                "metric_weight_entropy": 0.9,
            })
    frame = pd.DataFrame(rows)
    sel, table = _select(frame, pool_id="mlp_fixed_all")
    # All three tie on the mean; the chain must be recorded as invoked.
    assert sel.n_tied_on_primary == 3
    assert sel.tie_breaks_invoked[0] == "best_view_median_ari"
    # Medians also tie -> std must be consulted next.
    assert "best_view_std_ari" in sel.tie_breaks_invoked


def test_exact_tie_falls_through_to_method_name():
    """Fully identical candidates must resolve lexicographically, not randomly."""
    rows = []
    for method in ("regressor_top1_raw", "regressor_top3_raw"):
        for i in range(6):
            rows.append({
                "dataset_id": f"d{i}", "method_name": method, "family": "F",
                "best_view_ari": 0.5, "has_success": True,
                "total_runtime_all_views_sec": 10.0, "k_correct": True,
                "k_signed_error": 0, "k_abs_error": 0,
                "method_short_label": method, "view_ari_mean": 0.5,
                "view_ari_median": 0.5, "view_ari_std": 0.0, "view_gap": 0.0,
                "best_view_runtime_sec": 3.0, "selected_metric_count": 3,
                "metric_weight_entropy": 0.9,
            })
    frame = pd.DataFrame(rows)
    sel, _ = _select(frame)
    assert sel.selected_method == "regressor_top1_raw"
    assert "method_name" in sel.tie_breaks_invoked
    assert sel.n_tied_on_primary == 2


def test_practical_equivalence_never_overrides_the_raw_winner(pool_best_view):
    policy = selection.SelectionPolicy(practical_ari_tolerance=0.20,
                                       report_practical_equivalence=True)
    sel, _ = _select(pool_best_view, policy=policy)
    rec = sel.to_record()
    # With a wide tolerance many candidates are "practically equivalent", but the
    # selected method must still be the raw ARI winner.
    assert rec["raw_metric_winner"] == sel.selected_method
    assert rec["n_practical_equivalents"] > 1
    assert rec["practical_equivalence_representative"] is not None
    # Both are recorded separately, so a reader can see the divergence.
    assert "practical_equivalence_representative" in rec


def test_global_selection_is_the_same_method_for_every_dataset(pool_best_view):
    sel, _ = _select(pool_best_view)
    rep = frames.build_representative_frame(
        pool_best_view, representative_id="rep",
        selection=sel.to_record() | {"selected_method": sel.selected_method},
        selection_scope="global")
    assert rep["selected_method_name"].nunique() == 1
    assert rep["selection_type"].unique().tolist() == ["retrospective_global"]
    assert bool((~rep["is_independently_preselected"]).all())
    assert rep["dataset_id"].nunique() == pool_best_view["dataset_id"].nunique()


def test_family_selection_varies_only_by_family(pool_best_view):
    sels, table = selection.select_family_representatives(
        pool_best_view, comparison_id="test", representative_id="rep",
        candidate_pool_id="mlp_fixed_all", split_id=1)
    families = sorted(pool_best_view["family"].dropna().unique())
    assert len(sels) == len(families)
    assert {s.scope_key for s in sels} == set(families)
    for s in sels:
        assert s.selection_type == "retrospective_family"
        assert s.to_record()["paper_claim_level"] == "retrospective_summary"
    rep = frames.build_representative_frame(
        pool_best_view, representative_id="rep",
        selection=[s.to_record() | {"selected_method": s.selected_method,
                                    "family": s.scope_key} for s in sels],
        selection_scope="family")
    # Within one family exactly one method is used.
    for fam, grp in rep.groupby("family"):
        assert grp["selected_method_name"].nunique() == 1, fam


def test_family_divergence_reports_rates(pool_best_view):
    gsel, _ = _select(pool_best_view)
    fsels, _ = selection.select_family_representatives(
        pool_best_view, comparison_id="test", representative_id="rep",
        candidate_pool_id="mlp_fixed_all", split_id=1)
    div = selection.family_divergence([gsel], fsels)
    assert len(div) == len(fsels)
    assert set(["differs_from_global", "divergence_rate", "n_families"]) \
        <= set(div.columns)
    assert div["divergence_rate"].between(0.0, 1.0).all()
    assert div["paper_claim_level"].unique().tolist() == ["diagnostic_oracle"]


def test_manifest_is_reproducible(pool_best_view):
    def build():
        sels = []
        for pool_id in ("mlp_fixed_all", "knn_fixed_all", "real_fixed_topk_raw"):
            sel, _ = selection.select_representative(
                pool_best_view, comparison_id="test",
                representative_id=f"rep_{pool_id}", candidate_pool_id=pool_id,
                split_id=1)
            sels.append(sel)
        return selection.selection_manifest(sels)
    a, b = build(), build()
    pd.testing.assert_frame_equal(a, b)


def test_missing_pool_returns_none_not_a_fabricated_winner():
    empty = pd.DataFrame(columns=["dataset_id", "method_name", "best_view_ari",
                                  "has_success", "family"])
    sel, table = _select(empty)
    assert sel is None
    assert table.empty


def test_apply_global_representatives_by_family(pool_best_view):
    gsel, _ = _select(pool_best_view)
    out = selection.apply_global_representatives_by_family(pool_best_view, [gsel])
    assert out["selected_method"].nunique() == 1
    assert out["selection_scope"].unique().tolist() == ["global"]
    assert set(out["family"]) == set(pool_best_view["family"].dropna().unique())


def test_observed_representative_map_names_the_concrete_method(pool_best_view):
    gsel, _ = _select(pool_best_view)
    rep_map = selection.observed_representative_map([gsel])
    assert len(rep_map) == 1
    row = rep_map.iloc[0]
    assert row["selected_method"] == gsel.selected_method
    assert row["selected_short_label"]
    assert row["selected_full_name"]
    assert bool(row["is_independently_preselected"]) is False
