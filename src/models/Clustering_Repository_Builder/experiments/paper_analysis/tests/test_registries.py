"""Registry tests (§29.1, §29.5): counts, uniqueness, taxonomy coverage."""
from __future__ import annotations

import pytest

from models.Clustering_Repository_Builder.experiments.paper_analysis import (
    metric_registry as MR, method_registry as R, portfolio_registry as P,
)


# --------------------------------------------------------------------------- #
# Method registry
# --------------------------------------------------------------------------- #
def test_method_registry_self_validates():
    assert R.validate_registry() == []


def test_expected_method_counts():
    counts = {}
    for m in R.METHODS:
        counts[m.method_family] = counts.get(m.method_family, 0) + 1
    assert counts["clustopt_mlp"] == 12
    assert counts["clustopt_knn"] == 12
    assert counts["clustopt_real"] == 10
    assert counts["clustopt_fixed"] == 6
    assert counts["external_same_search_space"] == 4


def test_thirty_four_clustopt_variants():
    variants = R.names_where(method_family=("clustopt_mlp", "clustopt_knn",
                                           "clustopt_real"))
    assert len(variants) == R.EXPECTED_CLUSTOPT_VARIANTS == 34


def test_no_duplicate_names_or_labels():
    for attr in ("method_name", "on_disk_name", "short_label"):
        values = [getattr(m, attr) for m in R.METHODS]
        assert len(values) == len(set(values)), f"duplicate {attr}"


def test_short_labels_are_plot_sized():
    for m in R.METHODS:
        assert 0 < len(m.short_label) <= 16, m.method_name


@pytest.mark.parametrize("name", [
    "AutoClust_Original_InDomain_same_search_space",
    "AutoClust_Extended_InDomain_same_search_space",
    "ML2DAC_Original_InDomain_same_search_space",
    "ML2DAC_Extended_InDomain_same_search_space",
])
def test_same_search_space_externals_mapped(name):
    spec = R.spec(name)
    assert spec.is_external and spec.same_search_space and spec.is_paper_method
    assert spec.schema == "external"
    assert spec.view_dirs == R.EXTERNAL_VIEW_DIRS
    assert spec.search_budget == R.DEFAULT_SEARCH_BUDGET


def test_supplementary_externals_are_not_paper_methods():
    for name in ("AutoClust_Original_InDomain", "ML2DAC_Extended_InDomain",
                 "AutoClust_phase_B", "AutoML4Clust_phase_B"):
        assert R.spec(name).is_paper_method is False


def test_candidate_pool_sizes():
    expected = {
        "real_fixed_topk_raw": 4,
        "mlp_fixed_all": 8,
        "knn_fixed_all": 8,
        "mlp_fixed_plus_dynamic_predict": 10,
        "knn_fixed_plus_dynamic_predict": 10,
    }
    for pool_id, size in expected.items():
        assert R.CANDIDATE_POOL_BY_ID[pool_id].size == size, pool_id
    for fam in R.VARIANT_FAMILIES:
        assert R.CANDIDATE_POOL_BY_ID[f"variant_family__{fam.family_id}"].size == 2


def test_real_dynamic_is_not_in_the_fixed_pool():
    """RQ1-D compares dynamic against the best FIXED count, so the dynamic
    variant must not be able to win its own comparison."""
    pool = R.CANDIDATE_POOL_BY_ID["real_fixed_topk_raw"]
    assert "real_top_dynamic_raw" not in pool.methods
    assert all(R.spec(m).k_policy == "fixed" for m in pool.methods)


def test_primary_dynamic_cells_exclude_oracle_assisted():
    for fam in R.PRIMARY_VARIANT_FAMILIES:
        for member in fam.members:
            assert R.spec(member).is_oracle_assisted is False, member


def test_diagnostic_cells_are_oracle_assisted():
    assert len(R.DIAGNOSTIC_VARIANT_FAMILIES) == 2
    for fam in R.DIAGNOSTIC_VARIANT_FAMILIES:
        assert fam.oracle_assisted
        for member in fam.members:
            assert R.spec(member).is_oracle_assisted


def test_fifteen_primary_cells():
    assert len(R.PRIMARY_VARIANT_FAMILIES) == 15
    sources = {f.utility_source for f in R.PRIMARY_VARIANT_FAMILIES}
    assert sources == {"real", "mlp", "knn"}
    for source in sources:
        cells = [f for f in R.PRIMARY_VARIANT_FAMILIES if f.utility_source == source]
        assert len(cells) == 5, source


def test_oracle_levels_are_consistent():
    for m in R.METHODS:
        if m.uses_real_utility:
            assert m.oracle_level == "utility_oracle", m.method_name
        elif m.is_oracle_assisted:
            assert m.oracle_level == "dynamic_k_oracle_assisted", m.method_name
        else:
            assert m.oracle_level == "none", m.method_name


def test_label_map_covers_every_method():
    records = R.label_map_records()
    assert len(records) == len(R.METHODS)
    assert {r["method_name"] for r in records} == set(R.METHOD_NAMES)


def test_registry_snapshot_is_serialisable():
    import json
    snapshot = R.registry_snapshot()
    json.dumps(snapshot)                        # must not raise
    assert snapshot["n_methods"] == len(R.METHODS)
    assert len(snapshot["variant_families"]) == len(R.VARIANT_FAMILIES)


# --------------------------------------------------------------------------- #
# Portfolio registry
# --------------------------------------------------------------------------- #
def test_portfolio_registry_self_validates():
    assert P.validate_portfolios() == []


def test_portfolio_sizes():
    for pid, size in P.EXPECTED_SIZES.items():
        assert P.portfolio(pid).size == size, pid


def test_matched_weighting_portfolios_are_one_to_one():
    raw, sm = P.portfolio("raw_only_vbs"), P.portfolio("softmax_only_vbs")
    assert raw.size == sm.size == 17
    for a, b in zip(raw.methods, sm.methods):
        sa, sb = R.spec(a), R.spec(b)
        assert sa.utility_source == sb.utility_source
        assert sa.k_policy == sb.k_policy
        assert sa.fixed_metric_count == sb.fixed_metric_count
        assert sa.dynamic_k_source == sb.dynamic_k_source
        assert (sa.weighting_mode, sb.weighting_mode) == ("raw", "softmax_t05")


def test_strict_portfolios_are_fully_operational():
    """A portfolio named strict/deployable must contain no oracle members."""
    for pid in P.STRICT_PORTFOLIOS:
        pf = P.portfolio(pid)
        assert pf.fully_operational, f"{pid}: {pf.oracle_contamination_reason}"
        assert not pf.contains_real_utility
        assert not pf.contains_oracle_dynamic_k
        assert pf.oracle_contamination_reason == ""
    assert P.portfolio("strict_deployable_clustopt_vbs").size == 20
    assert P.portfolio("mlp_strict_deployable_vbs").size == 10
    assert P.portfolio("knn_strict_deployable_vbs").size == 10


def test_all_variant_portfolios_declare_oracle_contamination():
    """The 12-variant MLP/KNN sets include Dynamic Real-K, so they must say so."""
    for pid in P.ORACLE_DYNAMIC_K_PORTFOLIOS:
        pf = P.portfolio(pid)
        assert pf.contains_oracle_dynamic_k, pid
        assert not pf.fully_operational, pid
        assert "dynamic_realK" in pf.oracle_contamination_reason, pid
    for pid in P.REAL_UTILITY_PORTFOLIOS:
        pf = P.portfolio(pid)
        assert pf.contains_real_utility, pid
        assert "real_utility" in pf.oracle_contamination_reason, pid


def test_no_portfolio_name_overclaims_deployability():
    for pf in P.PORTFOLIOS:
        if "deployable" in pf.portfolio_id or "strict" in pf.portfolio_id:
            assert pf.fully_operational, (
                f"{pf.portfolio_id} implies deployability but contains "
                f"{pf.oracle_contamination_reason}")


def test_strict_subsets_drop_exactly_the_dynamic_realk_pair():
    for all_id, strict_id in (("mlp_all_variants_vbs", "mlp_strict_deployable_vbs"),
                              ("knn_all_variants_vbs", "knn_strict_deployable_vbs")):
        dropped = (set(P.portfolio(all_id).methods)
                   - set(P.portfolio(strict_id).methods))
        assert len(dropped) == 2
        assert all(R.spec(m).is_oracle_assisted for m in dropped), dropped


def test_deprecated_portfolio_ids_resolve():
    """Old ids still resolve, but the live registry no longer offers them."""
    for old_id, new_id in P.DEPRECATED_PORTFOLIO_IDS.items():
        assert old_id not in P.PORTFOLIO_BY_ID
        assert P.portfolio(old_id).portfolio_id == new_id
        assert P.canonical_portfolio_id(old_id) == new_id
    records = P.manifest_records()
    aliases = [r for r in records
               if str(r["portfolio_label"]).startswith("DEPRECATED")]
    assert len(aliases) == len(P.DEPRECATED_PORTFOLIO_IDS)


def test_portfolio_labels_identify_themselves_as_vbs():
    for pf in P.PORTFOLIOS:
        assert "VBS" in pf.short_label or "Strict" in pf.short_label, pf.portfolio_id


def test_subset_invariants_hold_structurally():
    for sub, sup in P.SUBSET_INVARIANTS:
        assert set(P.portfolio(sub).methods) <= set(P.portfolio(sup).methods), \
            f"{sub} not a subset of {sup}"


def test_union_equals_all_variant_membership():
    assert (set(P.portfolio("raw_softmax_union_vbs").methods)
            == set(P.portfolio("all_variant_clustopt_vbs").methods))


# --------------------------------------------------------------------------- #
# Metric taxonomy
# --------------------------------------------------------------------------- #
def test_metric_taxonomy_self_validates():
    assert MR.validate_metric_registry() == []


def test_sixty_metrics_all_categorised():
    assert MR.METRIC_COUNT == MR.EXPECTED_METRIC_COUNT == 60
    assert MR.METRIC_AUDIT["unmapped_category"] == []
    assert MR.METRIC_AUDIT["unmapped_subcategory"] == []
    for m in MR.METRICS:
        assert m.metric_category in MR.METRIC_CATEGORIES


def test_no_unknown_or_duplicate_category():
    names = [m.metric_name for m in MR.METRICS]
    assert len(names) == len(set(names))
    assert "unknown" not in {m.metric_category for m in MR.METRICS}


def test_classic_and_new_flags_are_complementary():
    for m in MR.METRICS:
        assert m.is_new is not m.is_classic_head_group
    assert MR.METRIC_AUDIT["n_classic_head_group"] == 14
    assert MR.METRIC_AUDIT["n_new"] == 46
    assert MR.METRIC_AUDIT["n_classic_head_group"] + MR.METRIC_AUDIT["n_new"] == 60


def test_classic_baseline_metrics_are_registered():
    for name in MR.CLASSIC_BASELINE_METRICS:
        assert name in MR.METRIC_BY_NAME
        assert MR.METRIC_BY_NAME[name].is_classic_baseline_metric


def test_utility_vector_aliases_resolve():
    """The recorded selected_metrics use the head-group spelling for two metrics;
    both spellings must land on the same taxonomy row."""
    for alias, canonical in MR.HEAD_GROUP_ALIASES.items():
        assert MR.canonical_metric_name(alias) == canonical
        assert MR.category_of(alias) == MR.category_of(canonical) != "unknown"
        assert MR.is_new_metric(alias) == MR.is_new_metric(canonical)
        assert MR.is_known_metric(alias)


def test_unknown_metric_is_reported_not_silently_categorised():
    assert MR.category_of("not_a_real_metric") == "unknown"
    assert not MR.is_known_metric("not_a_real_metric")
    problems = MR.validate_metric_registry(["silhouette", "not_a_real_metric"])
    assert any("not_a_real_metric" in p for p in problems)


def test_category_counts_match_implementation_packages():
    counts = MR.METRIC_AUDIT["counts_by_category"]
    assert counts == {"core_cvi": 6, "image": 24, "pattern": 22, "structural": 8}


def test_audit_markdown_renders():
    text = MR.audit_markdown()
    assert "# Metric Registry Audit" in text
    assert "is_classic_baseline_metric" in text
    for name in MR.CLASSIC_BASELINE_METRICS:
        assert name in text
