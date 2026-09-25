"""Run the Stage-1 offline 2x3 ablation for ONE subfamily and one split id.

Offline counterpart of :mod:`run_subfamily_experiments`. It owns exactly one
subfamily; the serial sweep across subfamilies lives in
:mod:`run_offline_2x3_all_subfamilies`, mirroring how
``run_phaseD_all_subfamilies`` drives ``run_subfamily_experiments``.

Parallelism model (Stage-1C contract):
  * this module creates ONE ``ProcessPoolExecutor`` whose task list contains
    only datasets of THIS subfamily, so a worker can never cross a subfamily
    boundary;
  * the pool is created, drained and closed inside :func:`run`, so the pool is
    gone before the caller advances -- there is no global cross-subfamily pool;
  * the worker unit is one dataset (see :mod:`offline_dataset_worker`).

Reuses the online aggregation layer unchanged: results are emitted as
:class:`~.method_runner.RunRecord` objects and handed to
:func:`~.summary_writer.write_execution_summaries`, so Best-View and per-view
summaries use the SAME implementation as the historical online runs.
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from models.Clustering_Repository_Builder.utility_generation.worker_setup import (
    configure_process,
)

from .config import ExperimentExecutionConfig, VIEW_IDS
from .dataset_discovery import discover_datasets
from .method_runner import RunRecord
from .offline_candidate_execution import ArmSpec
from .offline_dataset_worker import OfflineDatasetResult, process_dataset_offline
from .offline_method_runner import EXECUTION_MODE, offline_run_is_complete
from .output_writer import ensure_dir, path_exists, write_json_atomic
from .record_worker import write_dataset_summaries
from .split_filter import filter_dataset_dirs_by_split, load_split_assignments
from .summary_writer import summary_dir, write_execution_summaries

configure_process()

OFFLINE_SPLIT_DIR_SUFFIX = "offline_fixed32"

FULL60_MODEL = "results_analysis/mlp/20260615_231715__metric_utility_mlp_split1_holdout"


def build_arms(head14_model_run_dir: Optional[str],
               full60_model_run_dir: str = FULL60_MODEL) -> Tuple[ArmSpec, ...]:
    """The six factorial arms, in canonical order."""
    return (
        ArmSpec("stage1_head14_uniform_fixed50", "head14", "uniform", True),
        ArmSpec("stage1_head14_mlp_top5_raw_fixed50", "head14", "predicted", True,
                model_run_dir=head14_model_run_dir),
        ArmSpec("stage1_head14_oracle_top5_raw_fixed50", "head14", "oracle", False),
        ArmSpec("stage1_full60_uniform_fixed50", "full60", "uniform", True),
        ArmSpec("stage1_full60_mlp_top5_raw_fixed50", "full60", "predicted", True,
                model_run_dir=full60_model_run_dir),
        ArmSpec("stage1_full60_oracle_top5_raw_fixed50", "full60", "oracle", False),
    )


def metric_inventories() -> Tuple[List[str], List[str]]:
    """(head14, full60) metric names from the canonical registry."""
    from models.metric_utility_mlp.head_groups import (
        METRIC_HEAD_GROUPS, resolve_target_group,
    )
    head14 = resolve_target_group("head14")
    full60 = [m for ms in METRIC_HEAD_GROUPS.values() for m in ms]
    return head14, full60


def _worker_entry(payload):
    """Top-level picklable worker entry (one dataset)."""
    configure_process()
    dataset_dir_str, cfg, arms, head14, full60 = payload
    return process_dataset_offline(Path(dataset_dir_str), cfg, arms, head14, full60)


def subfamily_is_complete(cfg: ExperimentExecutionConfig,
                          dataset_dirs: Sequence[Path],
                          arms: Sequence[ArmSpec]) -> Tuple[bool, Dict[str, Any]]:
    """HARD completion barrier for one subfamily.

    Complete iff EVERY expected (dataset, view, arm) wrote a complete,
    non-partial output set AND the subfamily summary files exist.
    """
    split_dir = cfg.split_dir_name()
    missing: List[str] = []
    for d in dataset_dirs:
        for view in VIEW_IDS:
            for arm in arms:
                if not offline_run_is_complete(Path(d), split_dir, arm.method_id, view):
                    missing.append(f"{Path(d).name}/{view}/{arm.method_id}")
    sd = summary_dir(cfg)
    # long-path-safe (see offline_method_runner.offline_run_is_complete)
    summaries_ok = path_exists(sd / "execution_summary.csv")
    expected = len(dataset_dirs) * len(VIEW_IDS) * len(arms)
    return (not missing and summaries_ok), {
        "expected_runs": expected,
        "missing_runs": len(missing),
        "missing_examples": missing[:5],
        "summaries_present": summaries_ok,
        "datasets": len(dataset_dirs),
    }


def run(cfg: ExperimentExecutionConfig, *, arms: Sequence[ArmSpec],
        head14: Sequence[str], full60: Sequence[str],
        on_dataset_done=None) -> Dict[str, Any]:
    """Execute one subfamily. Returns a status dict; never raises for a dataset."""
    # Same discovery + split-filter path the online runner uses:
    # discover_datasets returns (valid, invalid) DiscoveredDataset lists.
    assignments = load_split_assignments(cfg.split_assignments)
    valid, _invalid = discover_datasets(cfg.subfamily_dir, limit=None)
    all_dirs = [d.dataset_dir for d in valid]
    dataset_dirs = filter_dataset_dirs_by_split(all_dirs, assignments, cfg.split_id)
    if cfg.limit_datasets:
        dataset_dirs = dataset_dirs[: int(cfg.limit_datasets)]

    if not dataset_dirs:
        return {"status": "empty", "datasets": 0, "records": 0,
                "subfamily": cfg.subfamily_dir.name}

    payloads = [(str(d), cfg, tuple(arms), list(head14), list(full60))
                for d in dataset_dirs]
    records: List[RunRecord] = []
    fatals: List[str] = []

    t0 = time.perf_counter()
    if cfg.max_workers <= 1 or len(payloads) <= 1:
        for i, p in enumerate(payloads, 1):
            res = _worker_entry(p)
            records.extend(res.records)
            if res.fatal_error:
                fatals.append(f"{res.dataset_id}: {res.fatal_error}")
            if on_dataset_done:
                on_dataset_done(i, len(payloads), res)
    else:
        # ONE pool, this subfamily only. Created here and closed before return,
        # so two subfamilies can never be in flight simultaneously.
        with ProcessPoolExecutor(max_workers=int(cfg.max_workers)) as pool:
            futures = {pool.submit(_worker_entry, p): i
                       for i, p in enumerate(payloads, 1)}
            done = 0
            for fut in as_completed(futures):
                done += 1
                try:
                    res = fut.result()
                except Exception as exc:                       # pool-level safety
                    idx = futures[fut]
                    tb = "".join(traceback.format_exception(
                        type(exc), exc, exc.__traceback__))
                    res = OfflineDatasetResult(
                        dataset_id=Path(payloads[idx - 1][0]).name,
                        dataset_dir=payloads[idx - 1][0],
                        fatal_error=f"{type(exc).__name__}: {exc}\n{tb}")
                records.extend(res.records)
                if res.fatal_error:
                    fatals.append(f"{res.dataset_id}: {res.fatal_error}")
                if on_dataset_done:
                    on_dataset_done(done, len(payloads), res)
    elapsed = time.perf_counter() - t0

    # ---- reuse the ONLINE aggregation layer, unchanged ---------------------
    # Same functions the online runner calls, so Best-View / per-view summaries
    # are produced by ONE implementation for both execution modes.
    tracker_summary: Dict[str, Any] = {
        "execution_mode": EXECUTION_MODE,
        "datasets_total": len(dataset_dirs),
        "runs_total": len(records),
        "success": sum(1 for r in records if r.status == "success"),
        "skipped": sum(1 for r in records if r.status == "skipped"),
        "failed": sum(1 for r in records if r.status == "failed"),
        "fatal_datasets": len(fatals),
        "runtime_sec": round(elapsed, 3),
        "max_workers": int(cfg.max_workers),
    }
    write_dataset_summaries(cfg, records)
    write_execution_summaries(
        cfg=cfg, records=records, tracker_summary=tracker_summary,
        family_id=cfg.subfamily_dir.parent.parent.name,
        subfamily_id=cfg.subfamily_dir.name,
    )

    complete, detail = subfamily_is_complete(cfg, dataset_dirs, arms)
    status = {
        "status": "complete" if complete else "incomplete",
        "execution_mode": EXECUTION_MODE,
        "subfamily": cfg.subfamily_dir.name,
        "family": cfg.subfamily_dir.parent.parent.name,
        "split_id": int(cfg.split_id),
        "split_dir": cfg.split_dir_name(),
        "datasets": len(dataset_dirs),
        "records": len(records),
        "n_success": sum(1 for r in records if r.status == "success"),
        "n_skipped": sum(1 for r in records if r.status == "skipped"),
        "n_failed": sum(1 for r in records if r.status == "failed"),
        "fatal_datasets": fatals,
        "runtime_sec": round(elapsed, 3),
        "max_workers": int(cfg.max_workers),
        "completion": detail,
    }
    sd = ensure_dir(summary_dir(cfg))
    write_json_atomic(sd / "offline_subfamily_status.json", status)
    return status


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", required=True)
    p.add_argument("--subfamily-dir", required=True)
    p.add_argument("--split-assignments", required=True)
    p.add_argument("--split-id", type=int, default=1)
    p.add_argument("--split-dir-suffix", default=OFFLINE_SPLIT_DIR_SUFFIX)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--head14-model-run-dir", default=None)
    p.add_argument("--full60-model-run-dir", default=FULL60_MODEL)
    p.add_argument("--limit-datasets", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)

    cfg = ExperimentExecutionConfig(
        repo_root=Path(args.repo_root).resolve(),
        subfamily_dir=Path(args.subfamily_dir).resolve(),
        split_assignments=Path(args.split_assignments).resolve(),
        split_id=int(args.split_id),
        split_dir_suffix=str(args.split_dir_suffix),
        method_set="stage1_2x3",
        max_workers=int(args.workers),
        resume=not args.overwrite, overwrite=bool(args.overwrite),
        limit_datasets=args.limit_datasets,
    )
    arms = build_arms(args.head14_model_run_dir, args.full60_model_run_dir)
    head14, full60 = metric_inventories()
    status = run(cfg, arms=arms, head14=head14, full60=full60)
    print(status)
    return 0 if status["status"] in ("complete", "empty") else 1


if __name__ == "__main__":
    sys.exit(main())
