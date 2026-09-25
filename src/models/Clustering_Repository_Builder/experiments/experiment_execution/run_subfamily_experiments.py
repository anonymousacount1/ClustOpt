"""CLI: run AutoClustering experiments for one subfamily and one split id.

Run as a module::

    python -m models.Clustering_Repository_Builder.experiments.experiment_execution.run_subfamily_experiments \
        --repo-root <PATH> \
        --subfamily-dir <PATH> \
        --split-assignments <PATH>/dataset_split_assignments.csv \
        --split-id 1 \
        --max-workers 8 \
        --resume
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import List

from models.Clustering_Repository_Builder.utility_generation.worker_setup import (
    configure_process,
)

from .config import (
    ExperimentExecutionConfig, METHOD_NAMES, METHOD_SETS, VIEW_IDS,
)
from .dataset_discovery import detect_family_subfamily, discover_datasets
from .experiment_config_resolver import resolve_configs, validate_or_raise
from .multiprocessing_runner import dry_run_plan, run_pool, run_pool_records
from .record_worker import write_dataset_summaries
from .output_writer import ensure_dir, write_json_atomic
from .progress_tracker import ExperimentProgressTracker
from .split_filter import filter_dataset_dirs_by_split, load_split_assignments
from .summary_writer import (
    SUMMARY_FILES,
    summary_dir,
    write_checkpoint,
    write_execution_summaries,
)

configure_process()


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="AutoClustering experiments for one subfamily and split id.",
    )
    p.add_argument("--repo-root", required=True, type=str)
    p.add_argument("--subfamily-dir", required=True, type=str)
    p.add_argument("--split-assignments", required=True, type=str)
    p.add_argument("--split-id", required=True, type=int)
    p.add_argument("--split-dir-suffix", type=str, default="",
                   help="Append a suffix to the output split folder, i.e. write to "
                        "experiments/split_XX_<suffix>/ instead of split_XX/. "
                        "Dataset selection still uses --split-id; only the output "
                        "folder name changes (lets a re-run isolate its outputs).")
    p.add_argument("--metric-mode", choices=("fast", "exact"), default="fast")
    p.add_argument("--sample-size", type=int, default=3000)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--max-workers", type=int, default=None)
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit-datasets", type=int, default=None)
    p.add_argument("--method-set", choices=sorted(METHOD_SETS.keys()),
                   default="default",
                   help="Method registry to draw from: 'default' (10 methods) or "
                        "'phaseD' (28 ablation methods). Default: default.")
    p.add_argument("--methods", nargs="+", default=None,
                   help="Subset of methods. Default: all methods in --method-set.")
    p.add_argument("--views", nargs="+", choices=list(VIEW_IDS), default=list(VIEW_IDS))
    p.add_argument("--progress-every", type=int, default=1)
    p.add_argument("--parallelism", choices=("dataset", "record"), default="dataset",
                   help="Scheduling granularity: 'dataset' (legacy, one worker per "
                        "dataset) or 'record' (one task per (dataset, view, method) "
                        "run, maximising worker utilisation). Default: dataset.")
    return p


def _build_config(args: argparse.Namespace) -> ExperimentExecutionConfig:
    method_set = str(args.method_set)
    registry = METHOD_SETS[method_set]
    methods = tuple(args.methods) if args.methods else tuple(m.name for m in registry)
    kwargs = dict(
        repo_root=Path(args.repo_root).resolve(),
        subfamily_dir=Path(args.subfamily_dir).resolve(),
        split_assignments=Path(args.split_assignments).resolve(),
        split_id=int(args.split_id),
        split_dir_suffix=str(args.split_dir_suffix),
        metric_mode=str(args.metric_mode),
        fast_metric_sample_size=int(args.sample_size),
        fast_metric_random_state=int(args.random_state),
        resume=bool(args.resume) and not bool(args.overwrite),
        overwrite=bool(args.overwrite),
        dry_run=bool(args.dry_run),
        limit_datasets=args.limit_datasets,
        method_set=method_set,
        methods=methods,
        views=tuple(args.views),
        progress_every=int(args.progress_every),
        parallelism=str(args.parallelism),
    )
    if args.max_workers is not None:
        kwargs["max_workers"] = int(args.max_workers)
    return ExperimentExecutionConfig(**kwargs)


def _run_records(cfg, selected, methods, valid, family_id, subfamily_id) -> int:
    """Record-level execution: one pool task per (dataset, view, method)."""
    import time

    n_datasets = len(selected)
    total = n_datasets * len(cfg.views) * len(methods)
    print("=" * 70)
    print("AutoClustering Experiment Execution (record-level parallelism)")
    print("=" * 70)
    print(f"Subfamily: {family_id}/{subfamily_id} | split={cfg.split_id}")
    print(f"Datasets={n_datasets} | methods={len(methods)} | views={len(cfg.views)} "
          f"| tasks={total} | workers={cfg.max_workers}", flush=True)

    counts = {"success": 0, "failed": 0, "skipped": 0}
    t0 = time.perf_counter()

    def _on_record(done, tot, rec):
        counts[rec.status] = counts.get(rec.status, 0) + 1
        if done == 1 or done % 200 == 0 or done == tot:
            el = time.perf_counter() - t0
            rate = done / el if el > 0 else 0
            eta = (tot - done) / rate if rate > 0 else 0
            print(f"  [{done}/{tot}] ok={counts['success']} failed={counts['failed']} "
                  f"skipped={counts['skipped']} | elapsed={el/60:.1f}m "
                  f"eta={eta/60:.1f}m", flush=True)

    records = run_pool_records(cfg=cfg, dataset_dirs=selected, on_record_done=_on_record)

    # per-dataset summaries (main process) + subfamily summaries.
    write_dataset_summaries(cfg, records)
    elapsed = time.perf_counter() - t0
    tracker_summary = {
        "datasets_selected": n_datasets,
        "datasets_processed": len({r.dataset_id for r in records}),
        "datasets_fatal": 0,
        "expected_runs": total,
        "runs_success": counts["success"],
        "runs_failed": counts["failed"],
        "runs_skipped": counts["skipped"],
        "runs_success_rate": counts["success"] / total if total else 0.0,
        "runtime_total_sec": float(elapsed),
        "runtime_total_str": _fmt_hms(elapsed),
        "avg_dataset_time_sec": float(elapsed / n_datasets) if n_datasets else 0.0,
        "parallelism": "record",
    }
    write_execution_summaries(
        cfg=cfg, records=records, tracker_summary=tracker_summary,
        family_id=family_id, subfamily_id=subfamily_id)
    print("\n" + "=" * 70)
    print("Experiment execution completed (record-level)")
    print(f"Tasks: ok={counts['success']} failed={counts['failed']} "
          f"skipped={counts['skipped']} / {total}")
    print(f"Total runtime: {_fmt_hms(elapsed)}")
    for f in SUMMARY_FILES:
        print(f"  {summary_dir(cfg) / f}")
    return 0


def _fmt_hms(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def run(cfg: ExperimentExecutionConfig) -> int:
    # 1) validate all active method × view configs up front.
    resolved = resolve_configs(
        cfg.experiments_config_root(), methods=cfg.active_registry(), validate=True)
    try:
        validate_or_raise(resolved)
    except Exception as exc:
        print(f"[config error] {exc}", file=sys.stderr)
        return 2

    # 2) discover datasets in the subfamily.
    valid, invalid = discover_datasets(cfg.subfamily_dir, limit=None)
    if not valid:
        print(f"[error] no valid dataset folders under {cfg.subfamily_dir}", file=sys.stderr)
        return 3

    # 3) filter by split assignment.
    assignments = load_split_assignments(cfg.split_assignments)
    all_dirs = [d.dataset_dir for d in valid]
    selected = filter_dataset_dirs_by_split(all_dirs, assignments, cfg.split_id)
    if cfg.limit_datasets:
        selected = selected[: cfg.limit_datasets]

    if not selected:
        print(f"[info] no datasets in subfamily assigned to split {cfg.split_id}; nothing to do.")
        return 0

    family_id, subfamily_id, _ = detect_family_subfamily(cfg.subfamily_dir)
    methods = cfg.selected_methods()
    runs_per_dataset = len(methods) * len(cfg.views)

    # 4) dry-run.
    if cfg.dry_run:
        plan = dry_run_plan(cfg, selected)
        print()
        print("=" * 60)
        print("Dry run plan")
        print("=" * 60)
        print(f"Subfamily dir: {cfg.subfamily_dir}")
        print(f"Split id: {cfg.split_id}")
        print(f"Datasets discovered: {len(valid)} | selected for split: {len(selected)}")
        print(f"Methods: {len(methods)} | Views: {len(cfg.views)}")
        print(f"Expected records: {len(selected)} × 3 = {len(selected) * 3}")
        print(f"Total planned runs: {plan['total_planned_runs']}")
        print(f"  -> would be skipped (already complete): {plan['would_be_skipped_runs']}")
        print(f"  -> pending: {plan['pending_runs']}")
        try:
            out = summary_dir(cfg)
            ensure_dir(out)
            write_json_atomic(out / "dry_run_plan.json", {"config": cfg.to_summary_dict(), **plan})
            print(f"Wrote dry-run plan: {out / 'dry_run_plan.json'}")
        except Exception as exc:
            print(f"(could not write dry-run plan: {exc})")
        return 0

    # 5) run.
    if cfg.parallelism == "record":
        return _run_records(cfg, selected, methods, valid, family_id, subfamily_id)

    tracker = ExperimentProgressTracker(
        total_datasets=len(selected),
        runs_per_dataset=runs_per_dataset,
        progress_every=cfg.progress_every,
    )
    tracker.start(metadata={
        "subfamily_dir": str(cfg.subfamily_dir),
        "split_id": cfg.split_id,
        "datasets_discovered": len(valid),
        "methods": [m.name for m in methods],
        "views": list(cfg.views),
        "max_workers": cfg.max_workers,
        "resume": cfg.resume,
        "overwrite": cfg.overwrite,
        "dry_run": cfg.dry_run,
        "metric_mode": cfg.metric_mode,
        "sample_size": cfg.fast_metric_sample_size if cfg.metric_mode == "fast" else 0,
    })

    def _on_done(idx, result):
        tracker.on_dataset_done(idx, result)
        try:
            write_checkpoint(cfg=cfg, records=tracker.record_results,
                             tracker_summary=tracker.finish())
        except Exception:
            pass

    run_pool(cfg=cfg, dataset_dirs=selected, on_dataset_done=_on_done)

    tracker_summary = tracker.finish()
    write_execution_summaries(
        cfg=cfg, records=tracker.record_results, tracker_summary=tracker_summary,
        family_id=family_id, subfamily_id=subfamily_id,
    )
    out_files = [str(summary_dir(cfg) / f) for f in SUMMARY_FILES]
    tracker.print_final_summary(out_files)
    return 0


def main(argv: List[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    try:
        return run(_build_config(args))
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
