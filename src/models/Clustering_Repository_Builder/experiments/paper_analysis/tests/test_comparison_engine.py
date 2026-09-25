"""Comparison-engine tests: all five suites execute end to end on synthetic data.

These do not validate scientific conclusions -- the synthetic ARIs are random.
They validate that each declared suite *runs*, produces the tables and figures its
config declares, resolves its representatives from the frozen manifest only, and
records every skipped sub-analysis. That is exactly the readiness question.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from models.Clustering_Repository_Builder.experiments.paper_analysis import (
    capability, collector, config as config_mod, frames, paths, selection,
)
from models.Clustering_Repository_Builder.experiments.paper_analysis.artifact_manifest import (
    ArtifactManifest,
)
from models.Clustering_Repository_Builder.experiments.paper_analysis.comparison_engine import (
    ComparisonContext, run_comparison,
)
from models.Clustering_Repository_Builder.experiments.paper_analysis.config import (
    RepresentativeIndex, RunConfig,
)
from models.Clustering_Repository_Builder.experiments.paper_analysis.method_registry import (
    VARIANT_FAMILIES,
)

SUITES = ("comparison-01", "comparison-02", "comparison-03", "comparison-04",
          "comparison-05")


@pytest.fixture
def suite_ctx(pool_best_view, tmp_path):
    """A full ComparisonContext over synthetic data, with frozen selections."""
    best_view = pool_best_view
    dataset_level = frames.build_dataset_level_frame(best_view)

    # Freeze the selections the suites depend on, exactly as the preliminary
    # stage does.
    sels = []
    plan = [("c1_best_observed_real_fixed_raw", "real_fixed_topk_raw"),
            ("c4_best_observed_mlp_fixed", "mlp_fixed_all"),
            ("c4_best_observed_knn_fixed", "knn_fixed_all"),
            ("c4_best_observed_mlp_incl_dynamic", "mlp_fixed_plus_dynamic_predict"),
            ("c4_best_observed_knn_incl_dynamic", "knn_fixed_plus_dynamic_predict")]
    plan += [(f"c2_rep__{f.family_id}", f"variant_family__{f.family_id}")
             for f in VARIANT_FAMILIES]
    for rep_id, pool_id in plan:
        sel, _ = selection.select_representative(
            best_view, comparison_id="preliminary", representative_id=rep_id,
            candidate_pool_id=pool_id, split_id=1)
        assert sel is not None, rep_id
        sels.append(sel)
    manifest_frame = selection.selection_manifest(sels)

    # A synthetic run frame is needed only for the capability matrix.
    run_frame = _synthetic_run_frame(best_view)
    matrix = capability.build_capability_matrix(run_frame, verbose=False)

    run = RunConfig(
        repo_root=paths.repo_root(), analyzed_data_root=tmp_path,
        split_assignments=tmp_path / "x.csv", output_root=tmp_path / "out",
        split_id=1, make_plots=True, make_excel=False, make_stats=True,
        bootstrap_samples=50, min_plot_importance="paper_main", verbose=False)
    run.ensure_dirs()
    return ComparisonContext(
        run=run, manifest=ArtifactManifest(run.output_root), best_view=best_view,
        dataset_level=dataset_level, capability_matrix=matrix,
        representatives=RepresentativeIndex.from_manifest(manifest_frame),
        selection_manifest=manifest_frame, run_frame=run_frame)


def _synthetic_run_frame(best_view: pd.DataFrame) -> pd.DataFrame:
    """Expand a best-view frame back into a minimal 3-view run frame."""
    rows = []
    for _, r in best_view.iterrows():
        for view_id in ("x_only", "y_only", "xy_2d"):
            row = {c: None for c in collector.RUN_FRAME_COLUMNS}
            for col in r.index:
                if col in row:
                    row[col] = r[col]
            row.update({
                "view_id": view_id, "status": "success",
                "ari": r["best_view_ari"], "runtime_sec": 5.0,
                "fit_predict_sec": 3.0, "cvi_eval_sec": 2.0,
                "provenance_mismatch": "", "failure_reason": "",
                "trace_path": None,
            })
            rows.append(row)
    return collector._coerce_types(
        pd.DataFrame(rows, columns=list(collector.RUN_FRAME_COLUMNS)))


@pytest.mark.parametrize("stage", SUITES)
def test_suite_runs_end_to_end(stage, suite_ctx):
    result = run_comparison(stage, suite_ctx)
    cfg = config_mod.load_config(stage)
    assert result.comparison_id == cfg["comparison_id"]
    assert result.section == cfg["section"]
    # Something substantive was produced.
    non_empty = [k for k, v in result.tables.items()
                 if isinstance(v, pd.DataFrame) and not v.empty]
    assert len(non_empty) >= 4, f"{stage} produced almost nothing: {non_empty}"
    # Every declared report exists.
    reports_dir = suite_ctx.run.output_root / result.section / "reports"
    for report in cfg["reports"]:
        assert paths.exists(reports_dir / report), f"{stage} missing {report}"
    # Every suite also owes its own named report and a validation record.
    num = cfg["comparison_id"].split("_")[-1]
    assert paths.exists(reports_dir / f"COMPARISON_{num}_REPORT.md")
    validation_dir = suite_ctx.run.output_root / result.section / "validation"
    assert paths.exists(validation_dir / f"COMPARISON_{num}_VALIDATION.md")
    # The eight standard subdirectories exist for every suite.
    for sub in ("config_snapshot", "manifests", "tables", "plots", "reports",
                "formatted_excel", "validation", "package"):
        assert paths.isdir(suite_ctx.run.output_root / result.section / sub), sub


@pytest.mark.parametrize("stage", SUITES)
def test_suite_registers_every_artifact(stage, suite_ctx):
    result = run_comparison(stage, suite_ctx)
    frame = suite_ctx.manifest.to_frame()
    mine = frame[frame["comparison_id"] == result.comparison_id]
    assert len(mine) > 0
    # Registered-and-created artifacts must exist on disk.
    suite_ctx.manifest.refresh_existence()
    frame = suite_ctx.manifest.to_frame()
    mine = frame[(frame["comparison_id"] == result.comparison_id)
                 & frame["created"]]
    assert bool(mine["exists"].all()), \
        mine.loc[~mine["exists"], "relative_path"].tolist()
    # Every registered artifact carries a description and an importance level.
    assert mine["description"].str.len().gt(0).all()
    assert mine["paper_importance"].isin(
        ["paper_main", "paper_supplement", "appendix", "exploratory"]).all()


@pytest.mark.parametrize("stage", SUITES)
def test_suite_writes_the_research_question_table(stage, suite_ctx):
    result = run_comparison(stage, suite_ctx)
    rqs = result.tables.get("research_questions")
    assert rqs is not None and not rqs.empty
    cfg = config_mod.load_config(stage)
    assert len(rqs) == len(cfg["research_questions"])
    assert rqs["question"].str.len().gt(0).all()


def test_comparison_04_carries_the_disclosure(suite_ctx):
    result = run_comparison("comparison-04", suite_ctx)
    out_dir = suite_ctx.run.output_root / result.section / "reports"
    readme = paths.read_text(out_dir / "README.md")
    assert "retrospective" in readme.lower()
    assert "benchmark summary" in readme.lower()
    cmap = result.tables["candidate_map"]
    for col in ("selection_type", "candidate_pool_size",
                "selected_concrete_method"):
        assert col in cmap.columns
    # The two ClustOpt rows are labelled retrospective; externals are fixed.
    assert (cmap["selection_type"] == "retrospective_global").sum() == 2
    assert (cmap["selection_type"] == "fixed_variant").sum() == 4
    # The retrospective rows disclose the pool they were drawn from.
    retro = cmap[cmap["selection_type"] == "retrospective_global"]
    assert (retro["candidate_pool_size"].astype(int) == 8).all()


def test_comparison_01_carries_the_causal_caveat(suite_ctx):
    result = run_comparison("comparison-01", suite_ctx)
    out_dir = suite_ctx.run.output_root / result.section / "reports"
    limitations = paths.read_text(out_dir / "LIMITATIONS.md")
    assert "cannot be identified" in limitations
    readme = paths.read_text(out_dir / "README.md")
    assert "Causal caveat" in readme


def test_vbs_suites_record_both_runtimes(suite_ctx):
    for stage in ("comparison-03", "comparison-05"):
        result = run_comparison(stage, suite_ctx)
        diag = result.tables.get("runtime_diagnostics")
        assert diag is not None and not diag.empty, stage
        assert not bool(diag["operational_runtime_claim_allowed"].any())
        assert bool(diag["excluded_from_operational_pareto"].all())
        for col in ("selected_member_total_all_views_runtime_median_sec",
                    "portfolio_total_execution_runtime_median_sec",
                    "portfolio_total_execution_runtime_total_sec"):
            assert col in diag.columns, f"{stage}/{col}"
        # Realising a VBS costs the whole portfolio, never just the member
        # the oracle happened to pick.
        ok = diag.dropna(subset=[
            "portfolio_total_execution_runtime_median_sec",
            "selected_member_total_all_views_runtime_median_sec"])
        assert (ok["portfolio_total_execution_runtime_median_sec"]
                >= ok["selected_member_total_all_views_runtime_median_sec"]
                - 1e-6).all(), stage


def test_comparison_05_computes_size_sensitivity(suite_ctx):
    result = run_comparison("comparison-05", suite_ctx)
    greedy = result.tables.get("greedy_portfolio")
    curve = result.tables.get("random_subset_summary")
    membership = result.tables.get("greedy_portfolio_membership")
    matched = result.tables.get("matched_size_vbs")
    minimum = result.tables.get("minimum_portfolio_size")
    assert greedy is not None and not greedy.empty
    assert curve is not None and not curve.empty
    assert membership is not None and not membership.empty
    assert matched is not None and not matched.empty
    assert minimum is not None and not minimum.empty
    # Both reporting thresholds must be present.
    assert set(minimum["target_share_of_achievable_gain"]) == {0.90, 0.95}
    # Every requested matched size must actually be evaluated somewhere.
    evaluated = set(curve.loc[~curve["skipped"].astype(bool), "size"])
    assert {1, 2, 3, 5, 10} <= evaluated
    # The headline matched-size comparison must be computable at size 2.
    at_two = matched[matched["size"] == 2]
    assert {"strict_deployable_clustopt_vbs", "autoclust_vbs",
            "ml2dac_vbs"} <= set(at_two["portfolio_id"])
    # Bulk sampled subsets are captured for reproducibility.
    samples = result.large_frames.get("random_subset_vbs_samples")
    assert samples is not None and not samples.empty
    # Oversized requests for the 2-member external portfolios are logged.
    skipped = curve[curve["skipped"].astype(bool)]
    assert len(skipped) > 0
    assert skipped["skip_reason"].str.len().gt(0).all()
    assert any("skipped because the requested size" in n for n in result.notes)


def test_comparison_05_reports_portfolio_operational_status(suite_ctx):
    """A portfolio's name must never outrun its membership."""
    result = run_comparison("comparison-05", suite_ctx)
    by_id = result.tables["vbs_summary"].set_index("portfolio_id")
    assert bool(by_id.loc["strict_deployable_clustopt_vbs", "fully_operational"])
    assert bool(by_id.loc["mlp_strict_deployable_vbs", "fully_operational"])
    assert not bool(by_id.loc["mlp_all_variants_vbs", "fully_operational"])
    assert (by_id.loc["mlp_all_variants_vbs", "oracle_contamination_reason"]
            == "dynamic_realK")
    assert not bool(by_id.loc["real_utility_vbs", "fully_operational"])
    assert "real_utility" in by_id.loc["all_variant_clustopt_vbs",
                                       "oracle_contamination_reason"]


def test_suites_emit_selection_transparency(suite_ctx):
    result = run_comparison("comparison-04", suite_ctx)
    t = result.tables.get("candidate_selection_transparency")
    assert t is not None and not t.empty
    for col in ("selection_type", "candidate_pool_size",
                "selected_concrete_method", "short_label", "runner_up",
                "delta_to_runner_up", "disclosure"):
        assert col in t.columns
    assert (t["selection_type"] == "retrospective_global").sum() == 2
    assert (t["selection_type"] == "fixed_method").sum() == 4
    externals = t[t["selection_type"] == "fixed_method"]
    assert (externals["candidate_pool_size"].astype(int) == 1).all()
    assert not bool(t["is_independently_preselected"].any())


def test_comparison_02_computes_the_matched_deltas(suite_ctx):
    result = run_comparison("comparison-02", suite_ctx)
    matched = result.tables.get("matched_source_deltas")
    assert matched is not None and not matched.empty
    assert set(matched["delta_kind"]) >= {"mlp_vs_knn", "real_vs_mlp", "real_vs_knn"}
    # The real-vs-learned rows must be flagged as oracle diagnostics.
    oracle_rows = matched[matched["delta_kind"].str.startswith("real_vs")]
    assert oracle_rows["reference_uses_real_utility"].all()
    dyn = result.tables.get("dynamic_predict_vs_realK")
    assert dyn is not None and not dyn.empty
    assert dyn["reference_is_oracle_assisted"].all()


def test_external_only_analyses_are_skipped_with_a_reason(suite_ctx):
    """Unsupported sub-analyses must be skipped per method, with a stated reason.

    The four externals always lack a metric-selection record and a candidate
    trace. (On this synthetic fixture the ClustOpt arms have no trace file either,
    so the trace warning legitimately covers more than four methods -- what
    matters is that every external is named and every warning has a reason.)
    """
    result = run_comparison("comparison-04", suite_ctx)
    warns = result.warnings
    assert not warns.empty
    externals = {"AutoClust_Original_InDomain_same_search_space",
                 "AutoClust_Extended_InDomain_same_search_space",
                 "ML2DAC_Original_InDomain_same_search_space",
                 "ML2DAC_Extended_InDomain_same_search_space"}
    for analysis in ("trace_best_gap", "metric_selection"):
        rows = warns[warns["analysis"] == analysis]
        assert externals <= set(rows["method_name"]), analysis
        assert rows["reason"].str.len().gt(0).all()
        assert rows["missing_capabilities"].str.len().gt(0).all()
        assert (rows["action"] == "skipped").all()
    # metric_selection is a pure schema property: exactly the four externals.
    sel_rows = warns[warns["analysis"] == "metric_selection"]
    assert set(sel_rows["method_name"]) == externals


def test_suites_never_reselect(suite_ctx):
    """A suite must resolve representatives from the frozen manifest only.

    Removing the frozen manifest must make the suite fail loudly rather than pick
    a winner of its own.
    """
    empty_index = RepresentativeIndex.from_manifest(pd.DataFrame())
    ctx = ComparisonContext(
        run=suite_ctx.run, manifest=ArtifactManifest(suite_ctx.run.output_root),
        best_view=suite_ctx.best_view, dataset_level=suite_ctx.dataset_level,
        capability_matrix=suite_ctx.capability_matrix,
        representatives=empty_index, selection_manifest=pd.DataFrame())
    with pytest.raises(config_mod.ConfigError, match="frozen selection manifest"):
        run_comparison("comparison-04", ctx)


def test_plots_are_written_with_their_data(suite_ctx):
    result = run_comparison("comparison-04", suite_ctx)
    created = [r for r in result.plot_results if r.created]
    assert created, "no figures were produced"
    for r in created:
        assert paths.exists(r.png)
        if r.data_csv is not None:
            assert paths.exists(r.data_csv), \
                "a figure must ship the data it plotted"
