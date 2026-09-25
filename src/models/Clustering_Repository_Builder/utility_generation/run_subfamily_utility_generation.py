"""CLI entry point for utility generation on a single subfamily.

Run as a module::

    python -m models.Clustering_Repository_Builder.utility_generation.run_subfamily_utility_generation \
        --repo-root <PATH> \
        --subfamily-dir <PATH> \
        --metric-mode fast \
        --sample-size 3000 \
        --random-state 42 \
        --max-workers 8 \
        --resume
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import List

from .config import UtilityGenerationConfig, VIEW_IDS
from .dataset_discovery import detect_family_subfamily, discover_datasets
from .multiprocessing_runner import dry_run_plan, run_pool
from .progress_tracker import DatasetWorkerResult, ProgressTracker
from .summary_report import (
    write_checkpoint,
    write_failures_csv,
    write_summary_csv,
    write_summary_json,
    write_summary_md,
)
from .utility_output_writer import write_json_atomic
from .validation import ValidationError, run_full_validation
from .worker_setup import configure_process

# Apply UTF-8 stdio reconfiguration and warning filters as early as possible
# so banner / progress prints in the main process never hit cp1252 / cp1255.
configure_process()


SUBFAMILY_OUTPUT_FILES: List[str] = [
    "utility_generation_summary.csv",
    "utility_generation_summary.json",
    "utility_generation_report.md",
    "utility_generation_failures.csv",
    "utility_generation_checkpoint.json",
]


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Clustering Repository — utility generation for one subfamily.",
    )
    p.add_argument("--repo-root", required=True, type=str,
                   help="Path to the repository root.")
    p.add_argument("--subfamily-dir", required=True, type=str,
                   help="Path to a single subfamily directory under analyzed_data/.../subfamilies/")
    p.add_argument("--metric-mode", choices=("fast", "exact"), default="fast",
                   help="ClustOpt metric mode. 'fast' uses sampled metrics; 'exact' is the legacy full-data behaviour.")
    p.add_argument("--sample-size", type=int, default=3000,
                   help="Fast-metric sample size (only used when metric-mode=fast).")
    p.add_argument("--random-state", type=int, default=42,
                   help="Fast-metric random state (only used when metric-mode=fast).")
    p.add_argument("--max-workers", type=int, default=None,
                   help="Process-pool workers. Defaults to min(cpu_count-1, 8).")
    p.add_argument("--resume", action="store_true", default=True,
                   help="Skip datasets whose utility outputs are already complete (default: True).")
    p.add_argument("--no-resume", dest="resume", action="store_false",
                   help="Disable resume; recompute everything.")
    p.add_argument("--overwrite", action="store_true",
                   help="Force recomputation of existing utility outputs.")
    p.add_argument("--dry-run", action="store_true",
                   help="Discover datasets, validate maps, print planned work; do not run ClustOpt.")
    p.add_argument("--limit-datasets", type=int, default=None,
                   help="Process at most N datasets (after discovery).")
    p.add_argument("--dataset-id", type=str, default=None,
                   help="Only process datasets whose folder name contains this substring.")
    p.add_argument("--views", nargs="+", choices=list(VIEW_IDS), default=list(VIEW_IDS),
                   help=f"Which views to process. Default: {' '.join(VIEW_IDS)}.")
    p.add_argument("--progress-every", type=int, default=5,
                   help="Print verbose progress block every N datasets.")
    return p


def _build_config_from_args(args: argparse.Namespace) -> UtilityGenerationConfig:
    kwargs = dict(
        repo_root=Path(args.repo_root).resolve(),
        subfamily_dir=Path(args.subfamily_dir).resolve(),
        metric_mode=str(args.metric_mode),
        fast_metric_sample_size=int(args.sample_size),
        fast_metric_random_state=int(args.random_state),
        resume=bool(args.resume) and not bool(args.overwrite),
        overwrite=bool(args.overwrite),
        dry_run=bool(args.dry_run),
        limit_datasets=args.limit_datasets,
        dataset_id_filter=args.dataset_id,
        views=tuple(args.views),
        progress_every=int(args.progress_every),
    )
    if args.max_workers is not None:
        kwargs["max_workers"] = int(args.max_workers)
    return UtilityGenerationConfig(**kwargs)


def run(cfg: UtilityGenerationConfig) -> int:
    # 1) validate everything that doesn't require listing datasets
    try:
        run_full_validation(cfg)
    except ValidationError as exc:
        print(f"[validation error] {exc}", file=sys.stderr)
        return 2

    # 2) discover datasets
    valid, invalid = discover_datasets(
        cfg.subfamily_dir,
        limit=cfg.limit_datasets,
        dataset_id_filter=cfg.dataset_id_filter,
    )
    if not valid:
        print(
            f"[validation error] No valid dataset folders found under {cfg.subfamily_dir} "
            f"(invalid={len(invalid)}).",
            file=sys.stderr,
        )
        if invalid:
            print("First few invalid folders:", file=sys.stderr)
            for d in invalid[:5]:
                print(f"  - {d.dataset_id}: missing {d.missing_files}", file=sys.stderr)
        return 3

    if invalid:
        print(f"Warning: {len(invalid)} dataset folder(s) are incomplete and will be skipped:")
        for d in invalid[:10]:
            print(f"  - {d.dataset_id}: missing {d.missing_files}")
        if len(invalid) > 10:
            print(f"  ... and {len(invalid) - 10} more")

    family_id, subfamily_id, specific_subfamily = detect_family_subfamily(cfg.subfamily_dir)

    # 3) dry-run path: write nothing under datasets; just print and exit
    if cfg.dry_run:
        plan = dry_run_plan(cfg, [d.dataset_dir for d in valid])
        print()
        print("=" * 60)
        print("Dry run plan")
        print("=" * 60)
        print(f"Subfamily dir: {cfg.subfamily_dir}")
        print(f"Datasets valid: {len(valid)} | invalid: {len(invalid)}")
        print(f"Views per dataset: {len(cfg.views)} ({', '.join(cfg.views)})")
        print(f"Total planned records: {plan['total_planned_records']}")
        print(f"  -> would be skipped (already complete): {plan['would_be_skipped_records']}")
        print(f"  -> pending: {plan['pending_records']}")
        print()
        # Persist a dry-run plan JSON for inspection.
        try:
            plan_path = cfg.subfamily_dir / "utility_generation_dry_run_plan.json"
            write_json_atomic(plan_path, {"config": cfg.to_summary_dict(), **plan})
            print(f"Wrote dry-run plan: {plan_path}")
        except Exception as exc:
            print(f"(could not write dry-run plan: {exc})")
        return 0

    # 4) progress tracker + run
    tracker = ProgressTracker(
        total_datasets=len(valid),
        total_views=len(cfg.views),
        progress_every=cfg.progress_every,
    )
    tracker.start(
        subfamily_dir=cfg.subfamily_dir,
        metadata={
            "family_id": family_id,
            "subfamily_id": subfamily_id,
            "datasets_found": len(valid),
            "metric_mode": cfg.metric_mode,
            "sample_size": cfg.fast_metric_sample_size if cfg.metric_mode == "fast" else 0,
            "random_state": cfg.fast_metric_random_state if cfg.metric_mode == "fast" else 0,
            "max_workers": cfg.max_workers,
            "resume": cfg.resume,
            "overwrite": cfg.overwrite,
            "dry_run": cfg.dry_run,
        },
    )

    def _on_dataset_done(idx: int, result: DatasetWorkerResult) -> None:
        tracker.on_dataset_done(idx, result)
        # Incremental checkpoint write so a crash mid-run doesn't lose progress.
        try:
            write_checkpoint(
                path=cfg.subfamily_dir / "utility_generation_checkpoint.json",
                cfg=cfg,
                records=tracker.record_results,
            )
        except Exception:
            pass

    run_pool(
        cfg=cfg,
        dataset_dirs=[d.dataset_dir for d in valid],
        on_dataset_done=_on_dataset_done,
    )

    # 5) subfamily-level outputs
    summary_csv = cfg.subfamily_dir / "utility_generation_summary.csv"
    summary_json = cfg.subfamily_dir / "utility_generation_summary.json"
    summary_md = cfg.subfamily_dir / "utility_generation_report.md"
    failures_csv = cfg.subfamily_dir / "utility_generation_failures.csv"
    checkpoint = cfg.subfamily_dir / "utility_generation_checkpoint.json"

    tracker_summary = tracker.finish()
    write_summary_csv(tracker.record_results, summary_csv)
    write_failures_csv(tracker.failed_records, failures_csv)
    write_summary_json(
        path=summary_json,
        cfg=cfg,
        tracker_summary=tracker_summary,
        family_id=family_id,
        subfamily_id=subfamily_id,
        output_files=SUBFAMILY_OUTPUT_FILES,
    )
    write_summary_md(
        path=summary_md,
        cfg=cfg,
        tracker_summary=tracker_summary,
        family_id=family_id,
        subfamily_id=subfamily_id,
        output_files=SUBFAMILY_OUTPUT_FILES,
    )
    write_checkpoint(path=checkpoint, cfg=cfg, records=tracker.record_results)

    tracker.print_final_summary(
        subfamily_id=subfamily_id or cfg.subfamily_dir.name,
        output_files=SUBFAMILY_OUTPUT_FILES,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    try:
        cfg = _build_config_from_args(args)
        return run(cfg)
    except SystemExit:
        raise
    except ValidationError as exc:
        print(f"[validation error] {exc}", file=sys.stderr)
        return 2
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
