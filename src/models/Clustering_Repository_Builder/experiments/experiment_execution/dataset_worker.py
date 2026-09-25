"""Process one raw dataset: all views × all methods, sequentially.

This is the multiprocessing unit. A failure in any single method/view is
captured by :func:`run_method_view` and never stops the remaining work.
"""
from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import ExperimentExecutionConfig
from .experiment_config_resolver import resolve_configs
from .method_runner import RunRecord, run_method_view
from .output_writer import write_json_atomic
from .view_builder import build_view, load_dataset_for_utility


@dataclass
class DatasetWorkerResult:
    dataset_id: str
    dataset_dir: str
    records: List[RunRecord] = field(default_factory=list)
    fatal_error: Optional[str] = None
    total_runtime_sec: float = 0.0


def _dataset_meta(loaded) -> Dict[str, Any]:
    tags = {}
    if isinstance(getattr(loaded, "metadata", None), dict):
        t = loaded.metadata.get("tags")
        if isinstance(t, dict):
            tags = t
    return {
        "family": loaded.family_name or loaded.family_id,
        "subfamily": loaded.subfamily_id,
        "difficulty": tags.get("difficulty"),
        "cluster_count": loaded.cluster_count_from_generator
        if loaded.cluster_count_from_generator is not None
        else tags.get("cluster_count"),
    }


def process_dataset(dataset_dir: Path, cfg: ExperimentExecutionConfig) -> DatasetWorkerResult:
    """Run all configured methods/views for one dataset folder."""
    t0 = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    dataset_dir = Path(dataset_dir)

    try:
        loaded = load_dataset_for_utility(dataset_dir)
    except Exception as exc:
        return DatasetWorkerResult(
            dataset_id=dataset_dir.name,
            dataset_dir=str(dataset_dir),
            fatal_error=f"{type(exc).__name__}: {exc}",
            total_runtime_sec=time.perf_counter() - t0,
        )

    meta = _dataset_meta(loaded)
    resolved = resolve_configs(
        cfg.experiments_config_root(), methods=cfg.active_registry(), validate=False)
    methods = cfg.selected_methods()

    records: List[RunRecord] = []
    for view_id in cfg.views:
        try:
            view = build_view(loaded, view_id)
        except Exception as exc:
            # If a view cannot be built, mark every method for it as failed.
            tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            for method in methods:
                records.append(RunRecord(
                    dataset_id=loaded.dataset_id, dataset_dir=str(loaded.dataset_dir),
                    split_id=int(cfg.split_id), family=meta.get("family"),
                    subfamily=meta.get("subfamily"), difficulty=meta.get("difficulty"),
                    cluster_count=meta.get("cluster_count"), view_id=view_id,
                    method_name=method.name, method_group=method.group, status="failed",
                    failure_reason=f"view_build: {type(exc).__name__}: {exc}\n{tb}",
                ))
            continue

        for method in methods:
            try:
                rec = run_method_view(
                    loaded=loaded, view=view, method=method,
                    config_path=resolved.config_path(method.name, view_id),
                    dataset_meta=meta, cfg=cfg,
                )
            except Exception as exc:
                tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
                rec = RunRecord(
                    dataset_id=loaded.dataset_id, dataset_dir=str(loaded.dataset_dir),
                    split_id=int(cfg.split_id), family=meta.get("family"),
                    subfamily=meta.get("subfamily"), difficulty=meta.get("difficulty"),
                    cluster_count=meta.get("cluster_count"), view_id=view_id,
                    method_name=method.name, method_group=method.group, status="failed",
                    failure_reason=f"{type(exc).__name__}: {exc}\n{tb}",
                )
            records.append(rec)

    elapsed = time.perf_counter() - t0

    # Per-dataset summary file under experiments/split_XX/.
    try:
        summary = _build_dataset_summary(loaded, meta, records, cfg, started_at, elapsed)
        write_json_atomic(
            Path(loaded.dataset_dir) / "experiments" / cfg.split_dir_name()
            / "dataset_experiment_summary.json",
            summary,
        )
    except Exception:
        pass  # never fail the worker on a summary write

    return DatasetWorkerResult(
        dataset_id=loaded.dataset_id,
        dataset_dir=str(loaded.dataset_dir),
        records=records,
        total_runtime_sec=elapsed,
    )


def _build_dataset_summary(loaded, meta, records, cfg, started_at, elapsed) -> Dict[str, Any]:
    by_status: Dict[str, int] = {"success": 0, "failed": 0, "skipped": 0}
    per_method: Dict[str, Dict[str, Any]] = {}
    for r in records:
        by_status[r.status] = by_status.get(r.status, 0) + 1
        per_method.setdefault(r.method_name, {})[r.view_id] = {
            "status": r.status,
            "ari": r.ari,
            "selected_k": r.selected_k,
            "true_k": r.true_k,
            "runtime_sec": r.runtime_sec,
        }
    return {
        "dataset_id": loaded.dataset_id,
        "split_id": int(cfg.split_id),
        "family": meta.get("family"),
        "subfamily": meta.get("subfamily"),
        "difficulty": meta.get("difficulty"),
        "cluster_count": meta.get("cluster_count"),
        "n_runs": len(records),
        "counts": by_status,
        "methods": per_method,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "total_runtime_sec": float(elapsed),
        "metric_mode": cfg.metric_mode,
    }
