"""Record-level worker: one pool task == one (dataset, view, method) run.

This is the fine-grained alternative to :mod:`dataset_worker` (which owns a whole
dataset per task). Scheduling each individual run as its own task means a free
worker immediately picks up the next run instead of idling while another worker
finishes a slow dataset -- maximising utilisation and shrinking the tail.

To keep that fine granularity cheap, each worker process **caches** the loaded
dataset and its built views (small, bounded LRU), so a dataset/view is loaded
once per process no matter how many of its method-runs land on that worker. The
resolved config map is also built once per process.

The main process assembles per-dataset and subfamily summaries from the returned
records (see :func:`write_dataset_summaries`); output files are identical to the
dataset-level path.
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from models.Clustering_Repository_Builder.utility_generation.worker_setup import (
    configure_process,
)

from .config import ExperimentExecutionConfig, MethodSpec
from .dataset_worker import _build_dataset_summary, _dataset_meta
from .experiment_config_resolver import resolve_configs
from .method_runner import RunRecord, run_method_view
from .output_writer import write_json_atomic
from .view_builder import build_view, load_dataset_for_utility

# Task = (dataset_dir, view_id, method_name). cfg travels via the pool initializer.
RecordTask = Tuple[str, str, str]

_MAX_CACHED_DATASETS = 3

# ---- process-global state (populated by _init) ------------------------------
_CFG: Optional[ExperimentExecutionConfig] = None
_RESOLVED = None
_METHOD_MAP: Dict[str, MethodSpec] = {}
_DATASET_CACHE: "OrderedDict[str, Any]" = OrderedDict()
_LOAD_ERRORS: Dict[str, str] = {}
_VIEW_CACHE: "OrderedDict[Tuple[str, str], Any]" = OrderedDict()
_VIEW_ERRORS: Dict[Tuple[str, str], str] = {}


def init_worker(cfg: ExperimentExecutionConfig) -> None:
    """ProcessPool initializer: install filters, resolve configs once."""
    global _CFG, _RESOLVED, _METHOD_MAP
    configure_process()
    _CFG = cfg
    _RESOLVED = resolve_configs(
        cfg.experiments_config_root(), methods=cfg.active_registry(), validate=False)
    _METHOD_MAP = {m.name: m for m in cfg.active_registry()}


def _get_loaded(dataset_dir: str):
    """Return the cached LoadedDatasetForUtility, loading on first use."""
    if dataset_dir in _LOAD_ERRORS:
        raise RuntimeError(_LOAD_ERRORS[dataset_dir])
    cached = _DATASET_CACHE.get(dataset_dir)
    if cached is not None:
        _DATASET_CACHE.move_to_end(dataset_dir)
        return cached
    try:
        loaded = load_dataset_for_utility(Path(dataset_dir))
    except Exception as exc:  # noqa: BLE001
        _LOAD_ERRORS[dataset_dir] = f"{type(exc).__name__}: {exc}"
        raise
    _DATASET_CACHE[dataset_dir] = loaded
    _DATASET_CACHE.move_to_end(dataset_dir)
    while len(_DATASET_CACHE) > _MAX_CACHED_DATASETS:
        old, _ = _DATASET_CACHE.popitem(last=False)
        for key in [k for k in _VIEW_CACHE if k[0] == old]:
            _VIEW_CACHE.pop(key, None)
    return loaded


def _get_view(loaded, dataset_dir: str, view_id: str):
    key = (dataset_dir, view_id)
    if key in _VIEW_ERRORS:
        raise RuntimeError(_VIEW_ERRORS[key])
    cached = _VIEW_CACHE.get(key)
    if cached is not None:
        _VIEW_CACHE.move_to_end(key)
        return cached
    try:
        view = build_view(loaded, view_id)
    except Exception as exc:  # noqa: BLE001
        _VIEW_ERRORS[key] = f"view_build: {type(exc).__name__}: {exc}"
        raise
    _VIEW_CACHE[key] = view
    _VIEW_CACHE.move_to_end(key)
    return view


def _failed_record(dataset_dir: str, view_id: str, method: Optional[MethodSpec],
                   method_name: str, reason: str,
                   meta: Optional[Dict[str, Any]] = None) -> RunRecord:
    meta = meta or {}
    return RunRecord(
        dataset_id=Path(dataset_dir).name, dataset_dir=dataset_dir,
        split_id=int(_CFG.split_id), family=meta.get("family"),
        subfamily=meta.get("subfamily"), difficulty=meta.get("difficulty"),
        cluster_count=meta.get("cluster_count"), view_id=view_id,
        method_name=method_name,
        method_group=(method.group if method else ""),
        status="failed", failure_reason=reason)


def process_one_record(task: RecordTask) -> RunRecord:
    """Run a single (dataset, view, method); never raises."""
    dataset_dir, view_id, method_name = task
    method = _METHOD_MAP.get(method_name)
    try:
        loaded = _get_loaded(dataset_dir)
    except Exception as exc:  # noqa: BLE001
        return _failed_record(dataset_dir, view_id, method, method_name,
                              f"dataset_load: {exc}")
    meta = _dataset_meta(loaded)
    try:
        view = _get_view(loaded, dataset_dir, view_id)
    except Exception as exc:  # noqa: BLE001
        return _failed_record(dataset_dir, view_id, method, method_name,
                              str(exc), meta)
    if method is None:
        return _failed_record(dataset_dir, view_id, None, method_name,
                              f"unknown method '{method_name}'", meta)
    try:
        return run_method_view(
            loaded=loaded, view=view, method=method,
            config_path=_RESOLVED.config_path(method_name, view_id),
            dataset_meta=meta, cfg=_CFG)
    except Exception as exc:  # noqa: BLE001 - safety net; run_method_view rarely raises
        import traceback
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        return _failed_record(dataset_dir, view_id, method, method_name,
                              f"{type(exc).__name__}: {exc}\n{tb}", meta)


def build_record_tasks(cfg: ExperimentExecutionConfig,
                       dataset_dirs: List[Path]) -> List[RecordTask]:
    """Dataset-major, view-major, method-minor task list (keeps a worker's
    consecutive tasks on the same dataset/view -> maximises cache hits)."""
    methods = [m.name for m in cfg.selected_methods()]
    tasks: List[RecordTask] = []
    for d in dataset_dirs:
        for view_id in cfg.views:
            for m in methods:
                tasks.append((str(d), view_id, m))
    return tasks


def write_dataset_summaries(cfg: ExperimentExecutionConfig,
                            records: List[RunRecord]) -> None:
    """Write each dataset's ``dataset_experiment_summary.json`` (main process).

    Mirrors the dataset-level worker's per-dataset summary. Best-effort: a write
    failure for one dataset never aborts the run.
    """
    from datetime import datetime, timezone

    by_dataset: Dict[str, List[RunRecord]] = {}
    dirs: Dict[str, str] = {}
    for r in records:
        by_dataset.setdefault(r.dataset_id, []).append(r)
        dirs[r.dataset_id] = r.dataset_dir

    for dataset_id, recs in by_dataset.items():
        try:
            r0 = recs[0]
            meta = {"family": r0.family, "subfamily": r0.subfamily,
                    "difficulty": r0.difficulty, "cluster_count": r0.cluster_count}

            class _L:  # minimal stand-in for _build_dataset_summary's ``loaded``
                dataset_id = r0.dataset_id
                dataset_dir = dirs[dataset_id]
            total = sum(float(r.runtime_sec or 0) for r in recs)
            summary = _build_dataset_summary(
                _L(), meta, recs, cfg,
                datetime.now(timezone.utc).isoformat(), total)
            write_json_atomic(
                Path(dirs[dataset_id]) / "experiments" / cfg.split_dir_name()
                / "dataset_experiment_summary.json", summary)
        except Exception:
            continue
