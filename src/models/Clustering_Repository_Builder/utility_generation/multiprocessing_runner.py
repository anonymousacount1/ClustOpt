"""Process-pool runner: one worker per dataset folder.

Each worker processes the configured views sequentially (``x_only`` →
``y_only`` → ``xy_2d``). Failures in one dataset never propagate to other
datasets; instead the worker returns a :class:`DatasetWorkerResult` carrying
the per-view records and (optionally) a ``fatal_error`` for the whole
dataset.
"""
from __future__ import annotations

import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

from .config import UtilityGenerationConfig
from .progress_tracker import DatasetWorkerResult
from .record_loader import load_dataset_for_utility
from .utility_output_writer import write_json_atomic
from .utility_record_runner import (
    RecordResult,
    is_view_complete,
    run_view,
    view_output_dir,
)
from .worker_setup import configure_process


def _process_dataset(args: Tuple[str, dict, list[str]]) -> DatasetWorkerResult:
    """Top-level worker function (must be picklable for ProcessPoolExecutor)."""
    # Workers inherit nothing from the main interpreter on Windows (spawn), so
    # we must install stdio + warning filters here.
    configure_process()

    dataset_dir_str, cfg_payload, views = args
    dataset_dir = Path(dataset_dir_str)

    cfg = UtilityGenerationConfig(
        repo_root=Path(cfg_payload["repo_root"]),
        subfamily_dir=Path(cfg_payload["subfamily_dir"]),
        metric_mode=str(cfg_payload["metric_mode"]),
        fast_metric_sample_size=int(cfg_payload["fast_metric_sample_size"]),
        fast_metric_random_state=int(cfg_payload["fast_metric_random_state"]),
        max_workers=int(cfg_payload["max_workers"]),
        resume=bool(cfg_payload["resume"]),
        overwrite=bool(cfg_payload["overwrite"]),
        dry_run=bool(cfg_payload["dry_run"]),
        limit_datasets=cfg_payload.get("limit_datasets"),
        dataset_id_filter=cfg_payload.get("dataset_id_filter"),
        views=tuple(views),
    )

    t0 = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()

    try:
        loaded = load_dataset_for_utility(dataset_dir)
    except Exception as exc:
        return DatasetWorkerResult(
            dataset_id=dataset_dir.name,
            dataset_dir=str(dataset_dir),
            records=[],
            fatal_error=f"{type(exc).__name__}: {exc}",
            total_runtime_sec=time.perf_counter() - t0,
        )

    records: List[RecordResult] = []
    for view_id in views:
        try:
            rec = run_view(loaded=loaded, view_id=view_id, cfg=cfg)
        except Exception as exc:
            # Defensive: ``run_view`` already handles per-view failures, but if
            # something escapes we record it here so the other views still run.
            tb = "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            )
            rec = RecordResult(
                dataset_id=loaded.dataset_id,
                dataset_dir=str(loaded.dataset_dir),
                family=loaded.family_id,
                subfamily=loaded.subfamily_id,
                specific_subfamily=loaded.specific_subfamily,
                view_id=view_id,
                status="failed",
                metric_mode=cfg.metric_mode,
                sample_size=cfg.fast_metric_sample_size if cfg.metric_mode == "fast" else 0,
                map_name=cfg.map_path_for_view(view_id).name,
                error_message=f"{type(exc).__name__}: {exc}\n{tb}",
            )
        records.append(rec)

    # Per-dataset summary file
    elapsed = time.perf_counter() - t0
    try:
        dataset_summary = {
            "dataset_id": loaded.dataset_id,
            "status": _dataset_status(records),
            "views": {
                r.view_id: {
                    "status": r.status,
                    "runtime_sec": float(r.runtime_sec),
                    "n_configs": int(r.n_configs),
                    "n_valid_configs": int(r.n_valid_configs),
                    "n_metrics": int(r.n_metrics),
                    "top1_metric": r.top1_metric,
                    "top1_utility": float(r.top1_utility),
                    "error_message": r.error_message,
                }
                for r in records
            },
            "total_runtime_sec": float(elapsed),
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "metric_mode": cfg.metric_mode,
            "sample_size": int(cfg.fast_metric_sample_size if cfg.metric_mode == "fast" else 0),
        }
        write_json_atomic(
            Path(loaded.dataset_dir) / "utility" / "utility_dataset_summary.json",
            dataset_summary,
        )
    except Exception:
        # Never fail the worker because of a summary write — records are already saved.
        pass

    return DatasetWorkerResult(
        dataset_id=loaded.dataset_id,
        dataset_dir=str(loaded.dataset_dir),
        records=records,
        fatal_error=None,
        total_runtime_sec=elapsed,
    )


def _dataset_status(records: List[RecordResult]) -> str:
    if not records:
        return "empty"
    statuses = {r.status for r in records}
    if statuses == {"success"}:
        return "success"
    if statuses == {"skipped"}:
        return "skipped"
    if statuses == {"failed"}:
        return "failed"
    if "failed" in statuses:
        return "partial_failure"
    return "partial"


def run_pool(
    *,
    cfg: UtilityGenerationConfig,
    dataset_dirs: List[Path],
    on_dataset_done,
) -> None:
    """Run the dataset workers and call ``on_dataset_done(idx, result)`` per completion."""
    cfg_payload = {
        "repo_root": str(cfg.repo_root),
        "subfamily_dir": str(cfg.subfamily_dir),
        "metric_mode": cfg.metric_mode,
        "fast_metric_sample_size": int(cfg.fast_metric_sample_size),
        "fast_metric_random_state": int(cfg.fast_metric_random_state),
        "max_workers": int(cfg.max_workers),
        "resume": bool(cfg.resume),
        "overwrite": bool(cfg.overwrite),
        "dry_run": bool(cfg.dry_run),
        "limit_datasets": cfg.limit_datasets,
        "dataset_id_filter": cfg.dataset_id_filter,
    }
    args_list = [(str(d), cfg_payload, list(cfg.views)) for d in dataset_dirs]

    if cfg.max_workers <= 1 or len(args_list) <= 1:
        # Sequential path: easier to debug and avoids spawning a pool for tiny runs.
        for completion_idx, args in enumerate(args_list, start=1):
            result = _process_dataset(args)
            on_dataset_done(completion_idx, result)
        return

    with ProcessPoolExecutor(max_workers=cfg.max_workers) as pool:
        future_to_submit_idx = {
            pool.submit(_process_dataset, args): i
            for i, args in enumerate(args_list, start=1)
        }
        completion_idx = 0
        for fut in as_completed(future_to_submit_idx):
            completion_idx += 1
            submit_idx = future_to_submit_idx[fut]
            try:
                result = fut.result()
            except Exception as exc:  # pragma: no cover - top-level safety net
                tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
                dataset_dir_str = args_list[submit_idx - 1][0]
                result = DatasetWorkerResult(
                    dataset_id=Path(dataset_dir_str).name,
                    dataset_dir=dataset_dir_str,
                    records=[],
                    fatal_error=f"{type(exc).__name__}: {exc}\n{tb}",
                    total_runtime_sec=0.0,
                )
            # idx passed to the tracker is the *completion* order, so the
            # progress display always reads as `N completed of TOTAL` rather
            # than reflecting submission order under parallel execution.
            on_dataset_done(completion_idx, result)


# --------------------------------------------------------------------- dry-run

def dry_run_plan(
    cfg: UtilityGenerationConfig,
    dataset_dirs: List[Path],
) -> dict[str, object]:
    """Return a dict describing what the run *would* do, without executing ClustOpt."""
    plan_rows: list[dict[str, object]] = []
    pending = 0
    skipped = 0
    for d in dataset_dirs:
        for v in cfg.views:
            completed = (not cfg.overwrite) and cfg.resume and is_view_complete(d, v)
            status = "skip (already complete)" if completed else "pending"
            if completed:
                skipped += 1
            else:
                pending += 1
            plan_rows.append(
                {
                    "dataset_id": d.name,
                    "view_id": v,
                    "view_output_dir": str(view_output_dir(d, v)),
                    "status": status,
                }
            )
    return {
        "n_datasets": len(dataset_dirs),
        "n_views_per_dataset": len(cfg.views),
        "total_planned_records": len(plan_rows),
        "pending_records": pending,
        "would_be_skipped_records": skipped,
        "plan": plan_rows,
    }
