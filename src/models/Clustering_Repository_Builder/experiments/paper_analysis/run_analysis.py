"""CLI for the paper-oriented analysis.

Run from the repo root::

    python -m models.Clustering_Repository_Builder.experiments.paper_analysis.run_analysis \
        --split-id 1 --stage preliminary-selection --collect-workers 8

Stages
------
``validate``              registries + configs only; touches no experiment output
``preliminary-selection`` collect Split 1, build frames, freeze the selection and
                          portfolio manifests, write the audits and the workbook
``comparison-01`` .. ``comparison-05``
                          one comparison suite (requires the frozen manifest)
``all-comparisons``       suites 1-5 in order
``package``               build the advisor-facing package from the artifact manifest

Safety
------
The default output root is a **new** directory
(``results_analysis/clustering_repository/paper_analysis_v2``); nothing under the
old ``experiments/`` or ``experiment_result_packages/`` trees is read or written.
Re-running a stage overwrites only that stage's own files, and only when
``--overwrite`` is passed or the files do not yet exist.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd

from . import capability, config as config_mod, gaps, paths, reports
from .artifact_manifest import ArtifactManifest
from .config import (
    ConfigError, RepresentativeIndex, RunConfig, SECTION_DIRS, STAGE_CONFIGS,
    validate_all_configs,
)
from .method_registry import METHOD_NAMES, validate_registry
from .metric_registry import validate_metric_registry
from .portfolio_registry import validate_portfolios

DEFAULT_OUTPUT_ROOT = ("results_analysis", "clustering_repository",
                       "paper_analysis_v2")
DEFAULT_ANALYZED_ROOT = ("results_analysis", "clustering_repository",
                         "analyzed_data")

COMPARISON_STAGES: Sequence[str] = (
    "comparison-01", "comparison-02", "comparison-03", "comparison-04",
    "comparison-05",
)
STAGES: Sequence[str] = (
    ("validate", "preliminary-selection") + tuple(COMPARISON_STAGES)
    + ("all-comparisons", "package")
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="paper_analysis.run_analysis",
        description="Paper-oriented ClustOpt result analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    p.add_argument("--stage", choices=list(STAGES), default="preliminary-selection")
    p.add_argument("--split-id", type=int, default=1,
                   help="Evaluation split (only 1 has downstream experiments).")
    p.add_argument("--repo-root", type=str, default=None)
    p.add_argument("--analyzed-data-root", type=str, default=None)
    p.add_argument("--split-assignments", type=str, default=None)
    p.add_argument("--output-root", type=str, default=None,
                   help="Defaults to results_analysis/clustering_repository/"
                        "paper_analysis_v2 (a NEW root; old outputs untouched).")
    p.add_argument("--collect-workers", type=int, default=None,
                   help="Thread-pool size for the I/O-bound per-run reads. "
                        "Output is identical to serial.")
    p.add_argument("--limit-datasets", type=int, default=None,
                   help="Cap datasets (smoke tests only).")
    p.add_argument("--methods", nargs="+", default=None,
                   help="Subset of registered method names (default: all).")
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.add_argument("--overwrite", action="store_true",
                   help="Allow a stage to overwrite its own previous outputs.")
    p.add_argument("--dry-run", action="store_true",
                   help="Do everything except write files.")
    p.add_argument("--reuse-frames", action="store_true",
                   help="Reuse the canonical frames already on disk instead of "
                        "re-reading every per-run output file. Use when only the "
                        "downstream analysis changed.")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--no-excel", action="store_true")
    p.add_argument("--no-stats", action="store_true")
    p.add_argument("--min-plot-importance", default="appendix",
                   choices=["paper_main", "paper_supplement", "appendix",
                            "exploratory"],
                   help="Skip plots below this importance level.")
    p.add_argument("--bootstrap-samples", type=int, default=2000)
    p.add_argument("--selection-bootstrap-samples", type=int, default=1000)
    p.add_argument("--seed", type=int, default=20260725)
    p.add_argument("--trace-gaps", action="store_true",
                   help="Compute the per-candidate search-selection gap "
                        "(reads every clustopt_results.csv; expensive).")
    p.add_argument("--quiet", action="store_true")
    return p


def resolve_run_config(args: argparse.Namespace) -> RunConfig:
    repo = Path(args.repo_root).resolve() if args.repo_root else paths.repo_root()
    analyzed = (Path(args.analyzed_data_root) if args.analyzed_data_root
                else repo.joinpath(*DEFAULT_ANALYZED_ROOT))
    split_csv = (Path(args.split_assignments) if args.split_assignments
                 else analyzed / "experiment_splits" / "dataset_split_assignments.csv")
    out_root = (Path(args.output_root) if args.output_root
                else repo.joinpath(*DEFAULT_OUTPUT_ROOT))
    return RunConfig(
        repo_root=repo, analyzed_data_root=analyzed, split_assignments=split_csv,
        output_root=out_root, split_id=int(args.split_id), stage=args.stage,
        collect_workers=args.collect_workers, limit_datasets=args.limit_datasets,
        methods=args.methods, resume=bool(args.resume),
        overwrite=bool(args.overwrite), dry_run=bool(args.dry_run),
        reuse_frames=bool(args.reuse_frames),
        trace_gaps=bool(args.trace_gaps),
        make_plots=not args.no_plots, make_excel=not args.no_excel,
        make_stats=not args.no_stats,
        min_plot_importance=args.min_plot_importance,
        bootstrap_samples=int(args.bootstrap_samples),
        selection_bootstrap_samples=int(args.selection_bootstrap_samples),
        seed=int(args.seed), verbose=not args.quiet)


# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #
def stage_validate(run: RunConfig) -> int:
    """Registry + config validation. Reads no experiment output."""
    problems: List[str] = []
    checks = {
        "method registry": validate_registry(),
        "portfolio registry": validate_portfolios(),
        "metric taxonomy": validate_metric_registry(),
        "comparison configs": validate_all_configs(),
    }
    print("=" * 72)
    print("VALIDATION")
    print("=" * 72)
    for name, found in checks.items():
        status = "OK" if not found else f"{len(found)} PROBLEM(S)"
        print(f"  {name:<24} {status}")
        for f in found:
            print(f"      - {f}")
        problems.extend(found)

    if not paths.exists(run.split_assignments):
        problems.append(f"split assignments not found: {run.split_assignments}")
        print(f"  split assignments        MISSING ({run.split_assignments})")
    else:
        assignments = paths.read_csv(run.split_assignments)
        n = int((assignments["split_id"].astype("Int64") == run.split_id).sum())
        print(f"  split assignments        OK ({n} datasets in split "
              f"{run.split_id})")
        if n == 0:
            problems.append(f"no datasets assigned to split {run.split_id}")

    print(f"\n  registered methods: {len(METHOD_NAMES)}")
    print(f"  output root:        {run.output_root}")
    print(f"\n{'VALIDATION PASSED' if not problems else 'VALIDATION FAILED'}")
    return 0 if not problems else 1


def stage_preliminary(run: RunConfig) -> int:
    from .preliminary import run_preliminary
    t0 = time.perf_counter()
    manifest = ArtifactManifest(run.output_root)
    out = run_preliminary(run, manifest)

    if run.dry_run:
        print("\n[dry-run] preliminary stage completed without writing files.")
        _print_preliminary_summary(out, run)
        return 0

    if run.make_stats and getattr(run, "_trace_gaps", False):
        pass  # trace gaps are computed in the comparison stages that need them

    _write_root_docs(run, manifest, out)
    written = manifest.write(dry_run=run.dry_run)
    problems = manifest.validate()
    print("\n" + "=" * 72)
    print("PRELIMINARY STAGE COMPLETE")
    print("=" * 72)
    _print_preliminary_summary(out, run)
    print(f"\n  artifact manifest: {written['csv']}")
    summary = manifest.summary()
    print(f"  artifacts: {summary['n_artifacts']} registered, "
          f"{summary['n_existing']} on disk, "
          f"{summary.get('n_skipped', 0)} skipped")
    if problems:
        print("  manifest problems:")
        for p in problems:
            print(f"    - {p}")
    print(f"  elapsed: {(time.perf_counter() - t0) / 60:.1f} min")
    hard = [p for group, found in out.validation.items() for p in found]
    return 0 if not (hard or problems) else 1


def _print_preliminary_summary(out, run: RunConfig) -> None:
    s = out.stats
    print(f"\n  split {s.get('split_id')}: {s.get('n_datasets')} datasets, "
          f"{s.get('n_families')} families, {s.get('n_subfamilies')} subfamilies")
    print(f"  methods: {s.get('n_methods')}/{s.get('n_methods_expected')} | "
          f"runs: {s.get('n_runs')} "
          f"(success={s.get('n_success')} failed={s.get('n_failed')} "
          f"missing={s.get('n_missing')})")
    print(f"  best-view rows: {s.get('n_best_view_rows')} "
          f"({s.get('n_best_view_with_success')} with >=1 successful view)")
    print(f"  selections: {s.get('n_global_selections')} global, "
          f"{s.get('n_family_selections')} family | "
          f"portfolios: {s.get('n_portfolios')}")
    for name, found in out.validation.items():
        print(f"  validation/{name}: {'OK' if not found else found}")
    if not out.selection_manifest.empty:
        print("\n  frozen global representatives:")
        cols = ["representative_id", "selected_short_label",
                "selected_best_view_mean_ari", "candidate_pool_size"]
        have = [c for c in cols if c in out.selection_manifest.columns]
        for _, r in out.selection_manifest[have].iterrows():
            print("    " + "  ".join(
                f"{r[c]:.4f}" if isinstance(r[c], float) else f"{r[c]}"
                for c in have))


def stage_comparison(stage: str, run: RunConfig) -> int:
    from .comparison_engine import ComparisonContext, run_comparison
    prelim_dir = run.output_root / SECTION_DIRS["preliminary"]
    raw_dir = run.output_root / SECTION_DIRS["raw_reproducibility"]
    index = RepresentativeIndex.from_path(prelim_dir / "selection_manifest.csv")
    selection_manifest = paths.read_csv(prelim_dir / "selection_manifest.csv")

    best_view = _load_frame(raw_dir, "canonical_best_view_frame")
    dataset_level = _load_frame(raw_dir, "canonical_dataset_level_frame")
    matrix = paths.read_csv(raw_dir / "method_capability_matrix.csv")
    if best_view is None or dataset_level is None:
        raise ConfigError(
            f"canonical frames not found under {raw_dir}; run "
            f"--stage preliminary-selection first")

    trace_table: Optional[pd.DataFrame] = None
    trace_path = raw_dir / "search_selection_gap_table.parquet"
    if paths.exists(trace_path):
        trace_table = paths.read_table(trace_path)

    manifest = ArtifactManifest(run.output_root)
    ctx = ComparisonContext(
        run=run, manifest=manifest, best_view=best_view,
        dataset_level=dataset_level, capability_matrix=matrix,
        representatives=index, selection_manifest=selection_manifest,
        trace_table=trace_table)
    result = run_comparison(stage, ctx)
    manifest.write(dry_run=run.dry_run)
    print(f"\n[{stage}] complete: {result.comparison_id} -> {result.section}")
    for note in result.notes:
        print(f"  note: {note}")
    return 0


def stage_all_comparisons(run: RunConfig) -> int:
    rc = 0
    for stage in COMPARISON_STAGES:
        print(f"\n{'=' * 72}\n{stage}\n{'=' * 72}")
        rc |= stage_comparison(stage, run)
    return rc


def stage_package(run: RunConfig) -> int:
    from ..paper_packaging.run_build_paper_package import build_from_run
    return build_from_run(run)


def _load_frame(directory: Path, stem: str) -> Optional[pd.DataFrame]:
    for suffix in (".parquet", ".csv"):
        path = directory / f"{stem}{suffix}"
        if paths.exists(path):
            return paths.read_table(path)
    return None


def _write_root_docs(run: RunConfig, manifest: ArtifactManifest, out) -> None:
    done = ["validate", "preliminary-selection"]
    pending = list(COMPARISON_STAGES) + ["package"]
    paths.write_text(run.output_root / "README_START_HERE.md",
                     reports.start_here_md(
                         split_id=run.split_id, output_root=str(run.output_root),
                         stages_done=done, stages_pending=pending))
    manifest.add_report(run.output_root / "README_START_HERE.md",
                        comparison_id="root", section="root",
                        description="Entry point: reading order, terminology "
                                    "caveat, directory map, stage status.",
                        paper_importance="paper_main")
    paths.write_text(run.output_root / "LIMITATIONS.md",
                     reports.limitations_md(
                         capability_warnings=out.capability_warnings))
    manifest.add_report(run.output_root / "LIMITATIONS.md",
                        comparison_id="root", section="root",
                        description="Run-level limitations.",
                        paper_importance="paper_main")
    for key, section in SECTION_DIRS.items():
        if key in ("protocol", "preliminary", "raw_reproducibility"):
            continue
        readme = run.output_root / section / "README.md"
        # Write the placeholder only when the section is still empty -- a stage
        # that has already produced real content must not be clobbered -- but
        # register it either way, so the manifest always knows the file is there.
        already = paths.exists(readme)
        if not already:
            paths.write_text(readme, "\n".join([
                f"# {section}",
                "",
                "_Not yet generated._ This section is produced by a comparison "
                "stage that has not been run. The preliminary selection study "
                "must be reviewed and approved first.",
                "",
                "Run it with:",
                "",
                "```",
                "python -m models.Clustering_Repository_Builder.experiments."
                "paper_analysis.run_analysis \\",
                f"    --split-id {run.split_id} --stage <stage>",
                "```",
                "",
            ]))
        manifest.add_report(
            readme, comparison_id="root", section=section,
            description=(f"Section README for {section}"
                         if already else
                         f"Placeholder for {section} (stage not yet run)"),
            paper_importance="appendix", appendix_only=True)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    run = resolve_run_config(args)
    if run.verbose:
        print(f"[run] stage={run.stage} split={run.split_id} "
              f"output_root={run.output_root} dry_run={run.dry_run}")
    try:
        if run.stage == "validate":
            return stage_validate(run)
        if run.stage == "preliminary-selection":
            rc = stage_validate(run)
            if rc:
                print("\nAborting: validation failed. Fix the problems above "
                      "before collecting results.")
                return rc
            return stage_preliminary(run)
        if run.stage in COMPARISON_STAGES:
            return stage_comparison(run.stage, run)
        if run.stage == "all-comparisons":
            return stage_all_comparisons(run)
        if run.stage == "package":
            return stage_package(run)
    except ConfigError as exc:
        print(f"\nCONFIG ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"unknown stage {run.stage!r}", file=sys.stderr)     # pragma: no cover
    return 2


if __name__ == "__main__":
    sys.exit(main())
