"""Dataset-level ProcessPoolExecutor (one worker per raw dataset).

No nested multiprocessing: each worker processes its dataset's views × methods
sequentially. A worker crash is caught and surfaced as a fatal dataset result
so the overall run keeps going.
"""
from __future__ import annotations

import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, List

from models.Clustering_Repository_Builder.utility_generation.worker_setup import (
    configure_process,
)

from .config import ExperimentExecutionConfig
from .dataset_worker import DatasetWorkerResult, process_dataset
from .method_runner import is_run_complete, run_output_dir


def _worker_entry(args) -> DatasetWorkerResult:
    """Top-level (picklable) worker entry. Installs stdio/warning filters."""
    configure_process()
    dataset_dir_str, cfg = args
    return process_dataset(Path(dataset_dir_str), cfg)


def run_pool(
    *,
    cfg: ExperimentExecutionConfig,
    dataset_dirs: List[Path],
    on_dataset_done: Callable[[int, DatasetWorkerResult], None],
) -> None:
    args_list = [(str(d), cfg) for d in dataset_dirs]

    if cfg.max_workers <= 1 or len(args_list) <= 1:
        for completion_idx, args in enumerate(args_list, start=1):
            on_dataset_done(completion_idx, _worker_entry(args))
        return

    with ProcessPoolExecutor(max_workers=cfg.max_workers) as pool:
        future_to_idx = {
            pool.submit(_worker_entry, args): i
            for i, args in enumerate(args_list, start=1)
        }
        completion_idx = 0
        for fut in as_completed(future_to_idx):
            completion_idx += 1
            submit_idx = future_to_idx[fut]
            try:
                result = fut.result()
            except Exception as exc:  # pragma: no cover - top-level safety net
                tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
                dataset_dir_str = args_list[submit_idx - 1][0]
                result = DatasetWorkerResult(
                    dataset_id=Path(dataset_dir_str).name,
                    dataset_dir=dataset_dir_str,
                    fatal_error=f"{type(exc).__name__}: {exc}\n{tb}",
                )
            on_dataset_done(completion_idx, result)


def run_pool_records(
    *,
    cfg: ExperimentExecutionConfig,
    dataset_dirs: List[Path],
    on_record_done: Callable[[int, int, "RunRecord"], None],
) -> "List[RunRecord]":
    """Record-level pool: schedule each (dataset, view, method) run as its own
    task so free workers never idle on a busy dataset.

    ``on_record_done(completed, total, record)`` is called as each run finishes.
    Returns the full list of :class:`RunRecord`. Workers cache the loaded dataset
    + built views per process (see :mod:`record_worker`).
    """
    from .method_runner import RunRecord  # local import to avoid cycle at top
    from .record_worker import build_record_tasks, init_worker, process_one_record

    tasks = build_record_tasks(cfg, dataset_dirs)
    total = len(tasks)
    records: List[RunRecord] = []

    if cfg.max_workers <= 1 or total <= 1:
        init_worker(cfg)
        for i, task in enumerate(tasks, start=1):
            rec = process_one_record(task)
            records.append(rec)
            on_record_done(i, total, rec)
        return records

    with ProcessPoolExecutor(
        max_workers=cfg.max_workers, initializer=init_worker, initargs=(cfg,)
    ) as pool:
        futures = {pool.submit(process_one_record, t): t for t in tasks}
        completed = 0
        for fut in as_completed(futures):
            completed += 1
            try:
                rec = fut.result()
            except Exception as exc:  # pragma: no cover - safety net
                d, v, m = futures[fut]
                rec = RunRecord(
                    dataset_id=Path(d).name, dataset_dir=d, split_id=int(cfg.split_id),
                    family=None, subfamily=None, difficulty=None, cluster_count=None,
                    view_id=v, method_name=m, method_group="", status="failed",
                    failure_reason=f"{type(exc).__name__}: {exc}")
            records.append(rec)
            on_record_done(completed, total, rec)
    return records


def dry_run_plan(
    cfg: ExperimentExecutionConfig,
    dataset_dirs: List[Path],
) -> dict:
    """Describe planned work without running ClustOpt."""
    methods = cfg.selected_methods()
    pending = skipped = 0
    rows = []
    for d in dataset_dirs:
        for view_id in cfg.views:
            for method in methods:
                out_dir = run_output_dir(d, cfg.split_dir_name(), method.name, view_id)
                done = (not cfg.overwrite) and cfg.resume and is_run_complete(out_dir)
                if done:
                    skipped += 1
                else:
                    pending += 1
                rows.append({
                    "dataset_id": d.name,
                    "view_id": view_id,
                    "method_name": method.name,
                    "status": "skip (already complete)" if done else "pending",
                })
    return {
        "n_datasets": len(dataset_dirs),
        "n_methods": len(methods),
        "n_views": len(cfg.views),
        "total_planned_runs": len(rows),
        "pending_runs": pending,
        "would_be_skipped_runs": skipped,
        "plan": rows,
    }
