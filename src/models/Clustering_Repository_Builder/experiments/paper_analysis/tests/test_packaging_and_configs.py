"""Artifact-manifest, config, packaging and backward-compatibility tests.

Covers §29.8 (packaging uses the manifest and tolerates missing optional files)
and §29.9 (the old aggregation CLI still imports and runs).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from models.Clustering_Repository_Builder.experiments.paper_analysis import (
    config as config_mod, paths, plots,
)
from models.Clustering_Repository_Builder.experiments.paper_analysis.artifact_manifest import (
    ArtifactManifest,
)
from models.Clustering_Repository_Builder.experiments.paper_packaging.package_builder import (
    PackageBuilder, PackageConfig,
)
from models.Clustering_Repository_Builder.experiments.paper_packaging.package_layout import (
    SECTIONS, sections_for, subdir_for,
)


# --------------------------------------------------------------------------- #
# Artifact manifest
# --------------------------------------------------------------------------- #
def _seed_analysis_root(root: Path) -> ArtifactManifest:
    """A miniature analysis output root with one real and one missing artifact."""
    manifest = ArtifactManifest(root)
    table_dir = root / "01_preliminary_selection"
    paths.write_csv(table_dir / "selection_manifest.csv",
                    pd.DataFrame({"representative_id": ["r1"],
                                  "selected_method": ["regressor_top5_raw"]}),
                    index=False)
    manifest.add_table(table_dir / "selection_manifest.csv",
                       comparison_id="preliminary",
                       section="01_preliminary_selection",
                       description="Frozen selection manifest.",
                       paper_importance="paper_main",
                       selection_type="retrospective_global")
    paths.write_text(root / "00_protocol_and_definitions" / "terminology.md",
                     "# Terminology\n")
    manifest.add_report(root / "00_protocol_and_definitions" / "terminology.md",
                        comparison_id="protocol",
                        section="00_protocol_and_definitions",
                        description="Terminology contract.",
                        paper_importance="paper_main")
    # A registered-but-never-created artifact: must survive as a visible gap.
    manifest.add_table(None, comparison_id="comparison_04",
                       section="05_comparison_04_operational_benchmarks",
                       description="Trace-best gap (unsupported for externals).",
                       paper_importance="paper_supplement", created=False,
                       notes="skipped: external methods expose no candidate trace")
    return manifest


def test_manifest_records_created_and_skipped(tmp_path):
    manifest = _seed_analysis_root(tmp_path)
    frame = manifest.to_frame()
    assert len(frame) == 3
    assert int(frame["created"].sum()) == 2
    skipped = frame[~frame["created"]].iloc[0]
    assert skipped["relative_path"] == ""
    assert "no candidate trace" in skipped["notes"]


def test_manifest_paths_are_relative_to_the_root(tmp_path):
    manifest = _seed_analysis_root(tmp_path)
    frame = manifest.to_frame()
    for rel in frame.loc[frame["created"], "relative_path"]:
        assert not Path(rel).is_absolute()
        assert paths.exists(tmp_path / rel)


def test_manifest_validate_catches_a_vanished_file(tmp_path):
    manifest = _seed_analysis_root(tmp_path)
    assert manifest.validate() == []
    (tmp_path / "01_preliminary_selection" / "selection_manifest.csv").unlink()
    problems = manifest.validate()
    assert any("absent on disk" in p for p in problems)


def test_manifest_write_produces_csv_and_json(tmp_path):
    manifest = _seed_analysis_root(tmp_path)
    written = manifest.write()
    assert paths.exists(written["csv"]) and paths.exists(written["json"])
    frame = paths.read_csv(written["csv"])
    assert set(frame.columns) >= {"comparison_id", "section", "artifact_type",
                                 "relative_path", "description",
                                 "paper_importance", "selection_type", "exists"}


def test_manifest_requires_a_description(tmp_path):
    manifest = ArtifactManifest(tmp_path)
    paths.write_text(tmp_path / "x.md", "x")
    manifest.add_report(tmp_path / "x.md", comparison_id="c", section="s",
                        description="")
    assert any("no description" in p for p in manifest.validate())


def test_manifest_rejects_unknown_types(tmp_path):
    manifest = ArtifactManifest(tmp_path)
    with pytest.raises(ValueError):
        manifest.add(tmp_path / "x", comparison_id="c", section="s",
                     artifact_type="nonsense", description="d")
    with pytest.raises(ValueError):
        manifest.add(tmp_path / "x", comparison_id="c", section="s",
                     artifact_type="table", description="d",
                     paper_importance="super_important")


# --------------------------------------------------------------------------- #
# Package layout / builder
# --------------------------------------------------------------------------- #
def test_every_analysis_section_maps_to_a_package_section():
    for key, section in config_mod.SECTION_DIRS.items():
        if key == "packaged":
            continue
        assert sections_for(section, ""), f"{section} has no package section"


def test_section_dir_names_are_unique_and_ordered():
    names = [s.dir_name for s in SECTIONS]
    assert len(names) == len(set(names))
    assert names == sorted(names), "package sections must be numerically ordered"


def test_subdir_routing():
    assert subdir_for("table") == "tables"
    assert subdir_for("plot") == "plots"
    assert subdir_for("report") == ""


def test_builder_requires_a_manifest(tmp_path):
    cfg = PackageConfig(analysis_root=tmp_path, output_root=tmp_path / "out",
                        package_name="pkg")
    with pytest.raises(FileNotFoundError, match="manifest-driven"):
        PackageBuilder(cfg).build()


def test_builder_places_registered_artifacts(tmp_path):
    root = tmp_path / "analysis"
    manifest = _seed_analysis_root(root)
    manifest.write()
    cfg = PackageConfig(analysis_root=root, output_root=tmp_path / "packages",
                        package_name="pkg")
    builder = PackageBuilder(cfg)
    summary = builder.build()
    assert summary["n_placed"] >= 2
    assert paths.exists(cfg.package_dir / "README.md")
    assert paths.exists(cfg.package_dir / "package_manifest.csv")
    # The preliminary manifest lands in the reproducibility section's tables dir.
    assert paths.exists(cfg.package_dir / "14_RAW_TABLES_AND_REPRODUCIBILITY"
                        / "tables" / "selection_manifest.csv")
    assert paths.exists(cfg.package_dir / "01_PROTOCOL_AND_DEFINITIONS"
                        / "terminology.md")
    assert builder.validate() == []


def test_builder_tolerates_a_missing_optional_file(tmp_path):
    root = tmp_path / "analysis"
    manifest = _seed_analysis_root(root)
    manifest.write()
    # Delete a *registered and created* artifact to simulate a partial run.
    (root / "01_preliminary_selection" / "selection_manifest.csv").unlink()
    cfg = PackageConfig(analysis_root=root, output_root=tmp_path / "packages",
                        package_name="pkg")
    builder = PackageBuilder(cfg)
    summary = builder.build()                       # must not raise
    assert summary["n_skipped_missing"] >= 1
    pm = paths.read_csv(cfg.package_dir / "package_manifest.csv")
    skipped = pm[pm["action"] == "skipped_missing"]
    assert len(skipped) >= 1
    assert skipped["skip_reason"].str.len().gt(0).all(), \
        "a skipped artifact must carry a reason"


def test_builder_writes_a_readme_per_section(tmp_path):
    root = tmp_path / "analysis"
    _seed_analysis_root(root).write()
    cfg = PackageConfig(analysis_root=root, output_root=tmp_path / "packages",
                        package_name="pkg")
    builder = PackageBuilder(cfg)
    builder.build()
    for section in SECTIONS:
        readme = cfg.package_dir / section.dir_name / "README.md"
        assert paths.exists(readme), section.dir_name
        text = paths.read_text(readme)
        assert section.title in text
        assert "## How to read it" in text


def test_builder_dry_run_writes_nothing(tmp_path):
    root = tmp_path / "analysis"
    _seed_analysis_root(root).write()
    cfg = PackageConfig(analysis_root=root, output_root=tmp_path / "packages",
                        package_name="pkg", dry_run=True)
    builder = PackageBuilder(cfg)
    summary = builder.build()
    assert summary["n_entries"] > 0
    assert not paths.exists(cfg.package_dir / "README.md")


def test_builder_refuses_to_clobber_without_overwrite(tmp_path):
    root = tmp_path / "analysis"
    _seed_analysis_root(root).write()
    cfg = PackageConfig(analysis_root=root, output_root=tmp_path / "packages",
                        package_name="pkg")
    PackageBuilder(cfg).build()
    with pytest.raises(FileExistsError):
        PackageBuilder(cfg).build()
    cfg.overwrite = True
    PackageBuilder(cfg).build()                     # now allowed


def test_builder_is_not_basename_driven(tmp_path):
    """A file present on disk but NOT registered must not be packaged -- this is
    the behaviour the old basename-lookup packager could not guarantee."""
    root = tmp_path / "analysis"
    manifest = _seed_analysis_root(root)
    manifest.write()
    paths.write_csv(root / "01_preliminary_selection" / "stray_table.csv",
                    pd.DataFrame({"a": [1]}), index=False)
    cfg = PackageConfig(analysis_root=root, output_root=tmp_path / "packages",
                        package_name="pkg")
    builder = PackageBuilder(cfg)
    builder.build()
    pm = paths.read_csv(cfg.package_dir / "package_manifest.csv")
    assert not pm["relative_path"].str.contains("stray_table").any()


# --------------------------------------------------------------------------- #
# Configs
# --------------------------------------------------------------------------- #
def test_all_configs_load_and_validate():
    assert config_mod.validate_all_configs() == []


def test_every_stage_has_a_config():
    for stage, filename in config_mod.STAGE_CONFIGS.items():
        cfg = config_mod.load_config(stage)
        assert cfg["_config_name"] == filename


def test_comparison_configs_declare_research_questions():
    for stage in ("comparison-01", "comparison-02", "comparison-03",
                  "comparison-04", "comparison-05"):
        cfg = config_mod.load_config(stage)
        rqs = cfg["research_questions"]
        assert len(rqs) >= 4, stage
        for rq_id, body in rqs.items():
            assert body.get("question"), f"{stage}/{rq_id}"


def test_comparison_04_requires_the_asymmetry_disclosure():
    cfg = config_mod.load_config("comparison-04")
    text = cfg["required_disclosure"]
    assert "retrospective" in text.lower()
    assert "benchmark summary" in text.lower()
    assert set(cfg["required_columns"]) == {
        "selection_type", "candidate_pool_size", "selected_concrete_method"}


def test_comparison_01_carries_the_causal_caveat():
    cfg = config_mod.load_config("comparison-01")
    caveat = cfg["causal_caveat"].lower()
    assert "inventory" in caveat and "selection" in caveat
    assert "cannot be identified" in caveat


def test_vbs_configs_require_the_runtime_disclosure():
    for stage in ("comparison-03", "comparison-05"):
        cfg = config_mod.load_config(stage)
        assert cfg["runtime_disclosure_required"] is True, stage


def test_placeholder_resolution_uses_the_frozen_manifest():
    manifest = pd.DataFrame([{
        "representative_id": "c4_best_observed_mlp_fixed",
        "selected_method": "regressor_top5_raw", "selection_scope": "global",
        "candidate_pool_size": 8}])
    index = config_mod.RepresentativeIndex.from_manifest(manifest)
    node = {"a": "@representative:c4_best_observed_mlp_fixed",
            "b": ["literal", "@representative:c4_best_observed_mlp_fixed"]}
    resolved = config_mod.resolve_placeholders(node, index)
    assert resolved["a"] == "regressor_top5_raw"
    assert resolved["b"] == ["literal", "regressor_top5_raw"]


def test_unknown_representative_fails_loudly():
    index = config_mod.RepresentativeIndex.from_manifest(pd.DataFrame())
    with pytest.raises(config_mod.ConfigError, match="not in the frozen"):
        config_mod.resolve_placeholders("@representative:nope", index)


def test_missing_manifest_fails_with_actionable_guidance(tmp_path):
    with pytest.raises(config_mod.ConfigError, match="preliminary-selection"):
        config_mod.RepresentativeIndex.from_path(tmp_path / "nope.csv")


def test_configs_only_reference_declared_representatives():
    """Every @representative id a comparison config names must be produced by the
    preliminary config, so a suite can never depend on an unfrozen selection."""
    prelim = config_mod.load_config("preliminary-selection")
    produced = {
        str(prelim["comparison_01_helper"]["representative_id"]),
        *(str(i["representative_id"])
          for i in prelim["comparison_04_representatives"]["primary"]),
        *(str(i["representative_id"])
          for i in prelim["comparison_04_representatives"]["supplemental"]),
    }
    prefix = str(prelim["comparison_02_representatives"]
                 ["representative_id_prefix"])
    for stage in ("comparison-01", "comparison-02", "comparison-03",
                  "comparison-04", "comparison-05"):
        cfg = config_mod.load_config(stage)
        for rep_id in config_mod.referenced_representative_ids(cfg):
            assert rep_id in produced or rep_id.startswith(prefix), \
                f"{stage} references unfrozen representative {rep_id}"


# --------------------------------------------------------------------------- #
# Plot layer
# --------------------------------------------------------------------------- #
def test_plots_write_png_svg_and_data(tmp_path):
    ctx = plots.PlotContext(output_dir=tmp_path)
    summary = pd.DataFrame({
        "method_short_label": ["MLP T5-R", "KNN T3-R", "AC-Ext"],
        "utility_source": ["mlp", "knn", "external_internal"],
        "best_view_mean_ari": [0.62, 0.60, 0.58],
        "best_view_q25_ari": [0.5, 0.48, 0.45],
        "best_view_q75_ari": [0.72, 0.70, 0.69],
    })
    spec = plots.PlotSpec(name="test_bar", title="Mean Best-View ARI",
                          ylabel="Mean Best-View ARI", importance="paper_main",
                          caption="ClustOpt rows are retrospective representatives.")
    result = plots.bar_chart(summary, spec, ctx, value_col="best_view_mean_ari",
                             error_low_col="best_view_q25_ari",
                             error_high_col="best_view_q75_ari")
    assert result.created
    assert paths.exists(result.png) and paths.exists(result.svg)
    assert paths.exists(result.data_csv)
    assert len(ctx.created_frame()) == 1


def test_plots_respect_the_importance_gate(tmp_path):
    ctx = plots.PlotContext(output_dir=tmp_path, min_importance="paper_main")
    summary = pd.DataFrame({"method_short_label": ["a", "b"],
                            "best_view_mean_ari": [0.1, 0.2]})
    spec = plots.PlotSpec(name="skipped", title="t", importance="appendix")
    result = plots.bar_chart(summary, spec, ctx, value_col="best_view_mean_ari")
    assert not result.created
    assert "importance below" in result.skipped_reason


def test_plots_skip_missing_columns_without_raising(tmp_path):
    ctx = plots.PlotContext(output_dir=tmp_path)
    spec = plots.PlotSpec(name="nope", title="t")
    result = plots.bar_chart(pd.DataFrame({"x": [1]}), spec, ctx,
                             value_col="best_view_mean_ari")
    assert not result.created
    assert "missing column" in result.skipped_reason


def test_pareto_frontier_table_has_no_composite_score():
    summary = pd.DataFrame({
        "method_short_label": ["fast_ok", "slow_best", "dominated"],
        "total_runtime_all_views_median_sec": [10.0, 100.0, 50.0],
        "best_view_mean_ari": [0.60, 0.70, 0.55],
    })
    out = plots.pareto_frontier_table(
        summary, x_col="total_runtime_all_views_median_sec",
        y_col="best_view_mean_ari")
    front = out[out["is_pareto_optimal"]]["method_short_label"].tolist()
    assert set(front) == {"fast_ok", "slow_best"}
    dominated = out[~out["is_pareto_optimal"]].iloc[0]
    assert "fast_ok" in dominated["dominated_by"]
    assert not any("per_second" in c for c in out.columns)
    assert "no composite" in out["pareto_note"].iloc[0]


# --------------------------------------------------------------------------- #
# Backward compatibility (§29.9)
# --------------------------------------------------------------------------- #
def test_old_aggregation_modules_still_import():
    """The previous aggregation + packaging layers must remain usable."""
    from models.Clustering_Repository_Builder.experiments.result_aggregation import (
        aggregators, method_registry as old_registry, raw_result_collector,
        run_aggregate_results, trace_best_gap,
    )
    assert callable(aggregators.collapse_best_view)
    assert callable(raw_result_collector.collect_raw_results)
    assert callable(run_aggregate_results.run)
    assert callable(trace_best_gap.compute_trace_best_table)
    assert len(old_registry.METHODS) > 0


def test_old_packaging_modules_still_import():
    from models.Clustering_Repository_Builder.experiments.result_packaging import (
        package_builder as old_pkg, package_config,
    )
    assert hasattr(old_pkg, "PackageBuilder")
    assert len(package_config.SECTIONS) > 0
    assert callable(package_config.describe)


def test_old_aggregation_cli_parser_still_works():
    from models.Clustering_Repository_Builder.experiments.result_aggregation import (
        run_aggregate_results as old_cli,
    )
    parser = old_cli._build_arg_parser()
    args = parser.parse_args([
        "--analyzed-data-root", "x", "--split-assignments", "y",
        "--split-ids", "1"])
    assert args.split_ids == [1]


def test_new_and_old_best_view_agree_on_the_same_input(tiny_run_frame):
    """The new best-view collapse must pick the same view as the old one.

    The old implementation returned the whole run row; the new one returns a
    richer row. The selected view and its ARI must match exactly, so no previous
    result is silently reinterpreted.
    """
    from models.Clustering_Repository_Builder.experiments.result_aggregation import (
        aggregators as old_agg,
    )
    from models.Clustering_Repository_Builder.experiments.paper_analysis import frames

    old_input = tiny_run_frame.rename(columns={"method_family": "method_group"})
    old = old_agg.collapse_best_view(old_input)
    new = frames.build_best_view_frame(tiny_run_frame, verbose=False)
    old_ok = old[old["status"] == "success"]
    for _, r in old_ok.iterrows():
        match = new[(new["dataset_id"] == r["dataset_id"])
                    & (new["method_name"] == r["method_name"])].iloc[0]
        assert match["best_view_ari"] == pytest.approx(float(r["ari"]))
        # Ties may resolve to a different view id by design (the new engine has an
        # explicit runtime tie-break), but never to a different ARI.
        if match["view_tie_count"] == 1:
            assert match["best_view_id"] == r["view_id"]


# --------------------------------------------------------------------------- #
# Manifest merging across stages (amendment: stages run independently)
# --------------------------------------------------------------------------- #
def test_manifest_merges_across_stages(tmp_path):
    """A later stage must not erase an earlier stage's artifacts.

    Each stage builds its own ArtifactManifest. Without merging, writing a
    comparison suite's manifest would wipe the preliminary rows and the
    manifest-driven packager would ship a package missing the protocol documents
    and the frozen manifests.
    """
    first = _seed_analysis_root(tmp_path)
    first.write()
    n_first = len(paths.read_csv(tmp_path / "analysis_manifest.csv"))

    second = ArtifactManifest(tmp_path)
    paths.write_csv(tmp_path / "05_comparison_04" / "tables" / "summary.csv",
                    pd.DataFrame({"a": [1]}), index=False)
    second.add_table(tmp_path / "05_comparison_04" / "tables" / "summary.csv",
                     comparison_id="comparison_04",
                     section="05_comparison_04_operational_benchmarks",
                     description="A later stage's table.",
                     paper_importance="paper_main")
    second.write()

    merged = paths.read_csv(tmp_path / "analysis_manifest.csv")
    assert len(merged) == n_first + 1
    assert set(merged["comparison_id"]) >= {"preliminary", "protocol",
                                            "comparison_04"}


def test_manifest_rerun_replaces_only_its_own_rows(tmp_path):
    first = _seed_analysis_root(tmp_path)
    first.write()
    before = paths.read_csv(tmp_path / "analysis_manifest.csv")

    again = _seed_analysis_root(tmp_path)
    again.write()
    after = paths.read_csv(tmp_path / "analysis_manifest.csv")
    # Re-running the same stage upserts rather than duplicating -- including the
    # not-created rows, which have no path to key on.
    assert len(after) == len(before)
    with_path = after[after["relative_path"].fillna("") != ""]
    assert not with_path["relative_path"].duplicated().any()


def test_manifest_drops_rows_whose_file_vanished(tmp_path):
    manifest = _seed_analysis_root(tmp_path)
    manifest.write()
    (tmp_path / "01_preliminary_selection" / "selection_manifest.csv").unlink()

    later = ArtifactManifest(tmp_path)
    paths.write_text(tmp_path / "note.md", "x")
    later.add_report(tmp_path / "note.md", comparison_id="c", section="s",
                     description="A new artifact.")
    later.write()
    merged = paths.read_csv(tmp_path / "analysis_manifest.csv")
    # The vanished file must not be promised to the packager.
    assert not merged["relative_path"].fillna("").str.contains(
        "selection_manifest").any()
    # Every row claiming to have been created must still be on disk. Rows that
    # were deliberately never created (an expected capability skip) legitimately
    # have exists=False and are kept as a visible gap.
    created = merged[merged["created"].astype(str).str.lower().isin(
        ("true", "1", "yes"))]
    assert bool(created["exists"].all())
