"""Collect per-run experiment outputs into a single raw_results DataFrame.

Walks the per-dataset output trees for the datasets in the requested split ids
(looked up from the split-assignments table) and reads each run's outputs into
one row per ``dataset x view x method`` (successes *and* failures).

Two on-disk schemas are supported transparently (see
:mod:`method_registry`):

* **ClustOpt** methods (regressor + fixed-CVI baselines):
  ``<dataset>/experiments/split_XX/<method>/<x_only|y_only|xy_2d>/`` with
  ``status.json`` / ``external_metrics.json`` / ``best_result.json`` /
  ``selected_metrics.json`` / ``timing_summary.json``.
* **External** AutoClustering baselines (AutoClust / AutoML4Clust / ML2DAC):
  ``<dataset>/experiments/split_XX/<method>/<1d_x|1d_y|2d>/`` with a single
  ``result.json`` (plus ``summary_metrics.json``). These carry no
  metric-selection / search-trace data, so those columns are left null.

Canonical view ids (``x_only/y_only/xy_2d``) are used in the output regardless
of the on-disk record-folder names, so downstream analyses are uniform.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from models.Clustering_Repository_Builder.experiments.experiment_execution.output_writer import (
    read_json,
)

from .method_registry import (
    METHODS, VIEW_IDS, AggMethod, display_name, select_methods,
)

RAW_COLUMNS: List[str] = [
    "dataset_id", "dataset_dir", "split_id", "family", "subfamily", "difficulty",
    "cluster_count", "view_id", "method_name", "method_group", "status",
    "failure_reason", "selected_algorithm", "selected_params", "selected_k",
    "true_k", "k_correct", "k_abs_error", "k_signed_error", "best_objective_score",
    "n_trials_completed", "runtime_sec", "ari", "nmi", "ami", "v_measure",
    "homogeneity", "completeness", "fowlkes_mallows", "purity",
    "predicted_noise_ratio", "true_noise_ratio", "noise_precision",
    "noise_recall", "noise_f1", "selected_metrics", "selected_metric_weights",
    "metric_weight_entropy", "top_metric", "config_path",
]

_EXTERNAL_METRIC_KEYS = (
    "ari", "nmi", "ami", "v_measure", "homogeneity", "completeness",
    "fowlkes_mallows", "purity", "predicted_noise_ratio", "true_noise_ratio",
    "noise_precision", "noise_recall", "noise_f1",
)


def _split_dir_name(split_id: int) -> str:
    return f"split_{int(split_id):02d}"


def _base_row(dataset_row: pd.Series, dataset_dir: Path, split_id: int,
              method: AggMethod, view_id: str, config_path: Path) -> Dict:
    return {
        "dataset_id": dataset_row.get("dataset_id"),
        "dataset_dir": str(dataset_dir),
        "split_id": int(split_id),
        "family": dataset_row.get("family"),
        "subfamily": dataset_row.get("subfamily"),
        "difficulty": dataset_row.get("difficulty"),
        "cluster_count": dataset_row.get("cluster_count"),
        "view_id": view_id,
        # Display (trimmed) name in outputs; the on-disk folder stays phase-tagged.
        "method_name": display_name(method.name),
        "method_group": method.group,
        "status": "missing",
        "failure_reason": "",
        "config_path": str(config_path),
    }


def _collect_clustopt(run_dir: Path, base: Dict) -> Dict:
    """Read one ClustOpt run directory into a raw row."""
    status_payload = read_json(run_dir / "status.json")
    if not status_payload:
        return {**{c: None for c in RAW_COLUMNS}, **base}
    status = str(status_payload.get("status", "")).lower()
    base["status"] = status
    if status != "success":
        base["failure_reason"] = status_payload.get("error_message", "")
        return {**{c: None for c in RAW_COLUMNS}, **base}

    ext = read_json(run_dir / "external_metrics.json") or {}
    best = read_json(run_dir / "best_result.json") or {}
    sel = read_json(run_dir / "selected_metrics.json") or {}
    timing = read_json(run_dir / "timing_summary.json") or {}

    row = {**{c: None for c in RAW_COLUMNS}, **base}
    row.update({
        "selected_algorithm": best.get("selected_algorithm"),
        "selected_params": best.get("selected_params"),
        "selected_k": best.get("selected_k"),
        "true_k": best.get("true_k"),
        "k_correct": best.get("k_correct"),
        "k_abs_error": best.get("k_abs_error"),
        "k_signed_error": best.get("k_signed_error"),
        "best_objective_score": best.get("best_objective_score"),
        "n_trials_completed": best.get("n_trials_completed"),
        "runtime_sec": timing.get("runtime_sec", status_payload.get("runtime_sec")),
        "selected_metrics": json.dumps(sel.get("selected_metrics", [])),
        "selected_metric_weights": json.dumps(sel.get("selected_metric_weights", {})),
        "metric_weight_entropy": sel.get("metric_weight_entropy"),
        "top_metric": sel.get("top_metric"),
    })
    for k in _EXTERNAL_METRIC_KEYS:
        row[k] = ext.get(k)
    return row


def _collect_external(run_dir: Path, base: Dict) -> Dict:
    """Read one external-baseline run directory (result.json) into a raw row.

    Externals share a final-result schema across AutoClust / AutoML4Clust /
    ML2DAC. They have **no** metric-selection or per-iteration search trace, so
    those columns stay null (the phase-B comparison drops the corresponding
    sections downstream).
    """
    result = read_json(run_dir / "result.json")
    if not result:
        # Fall back to summary_metrics.json if result.json is absent.
        result = read_json(run_dir / "summary_metrics.json")
    if not result:
        return {**{c: None for c in RAW_COLUMNS}, **base}

    status = str(result.get("status", "")).lower()
    base["status"] = status or "missing"
    if status and status != "success":
        base["failure_reason"] = (result.get("error_message")
                                  or result.get("failure_kind") or "")
        return {**{c: None for c in RAW_COLUMNS}, **base}

    metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else result
    row = {**{c: None for c in RAW_COLUMNS}, **base}
    row.update({
        "selected_algorithm": result.get("selected_algorithm"),
        "selected_params": json.dumps(result.get("best_configuration"))
        if result.get("best_configuration") is not None else None,
        "selected_k": result.get("selected_k", metrics.get("selected_k")),
        "true_k": metrics.get("true_k"),
        "k_correct": metrics.get("k_correct", metrics.get("exact_k_match")),
        "k_abs_error": metrics.get("k_abs_error", metrics.get("abs_k_error")),
        "k_signed_error": metrics.get("k_signed_error", metrics.get("k_error")),
        "best_objective_score": None,
        "n_trials_completed": result.get("optimizer_evaluations"),
        "runtime_sec": result.get("runtime_sec", metrics.get("runtime_sec")),
    })
    for k in _EXTERNAL_METRIC_KEYS:
        row[k] = metrics.get(k)
    return row


def _collect_one(*, dataset_row: pd.Series, split_id: int, method: AggMethod,
                 view_id: str) -> Dict:
    dataset_dir = Path(str(dataset_row["dataset_dir"]))
    record_dir = method.view_dirs[view_id]
    run_dir = (dataset_dir / "experiments" / _split_dir_name(split_id)
               / method.name / record_dir)
    config_name = "config_snapshot.json" if method.schema == "clustopt" else "result.json"
    base = _base_row(dataset_row, dataset_dir, split_id, method, view_id,
                     run_dir / config_name)
    if method.schema == "external":
        return _collect_external(run_dir, base)
    return _collect_clustopt(run_dir, base)


def _collect_dataset_rows(dataset_row: pd.Series, methods, views,
                          include_missing: bool) -> List[Dict]:
    """All raw rows for one dataset (serial over its methods x views)."""
    split_id = int(dataset_row["split_id"])
    rows: List[Dict] = []
    for method in methods:
        for view_id in views:
            row = _collect_one(
                dataset_row=dataset_row, split_id=split_id,
                method=method, view_id=view_id,
            )
            if row["status"] == "missing" and not include_missing:
                continue
            rows.append(row)
    return rows


def collect_raw_results(
    *,
    assignments: pd.DataFrame,
    split_ids: List[int],
    method_names: Optional[List[str]] = None,
    views=VIEW_IDS,
    include_missing: bool = False,
    verbose: bool = True,
    max_workers: Optional[int] = None,
) -> pd.DataFrame:
    """Return a raw_results DataFrame for the selected methods/datasets/splits.

    ``max_workers`` (default ``None`` -> serial, unchanged) enables a
    dataset-level thread pool. The per-run reads are I/O-bound (many small
    long-path JSON opens that release the GIL), so threads give a large speedup
    on Windows. ``ThreadPoolExecutor.map`` preserves input (dataset) order, so
    the resulting row order -- and therefore ``raw_results.csv`` -- is identical
    to the serial path.
    """
    methods = select_methods(method_names)
    wanted_splits = set(int(s) for s in split_ids)
    sub = assignments[assignments["split_id"].astype("Int64").isin(wanted_splits)]
    dataset_rows = [r for _, r in sub.iterrows()]
    n = len(dataset_rows)
    rows: List[Dict] = []

    if max_workers and max_workers > 1 and n:
        done = 0
        with ThreadPoolExecutor(max_workers=int(max_workers)) as pool:
            for per_dataset in pool.map(
                lambda dr: _collect_dataset_rows(dr, methods, views, include_missing),
                dataset_rows,
            ):
                rows.extend(per_dataset)
                done += 1
                if verbose and done % 200 == 0:
                    print(f"[collect] scanned {done}/{n} datasets "
                          f"({int(max_workers)} workers)...", flush=True)
    else:
        for i, dataset_row in enumerate(dataset_rows, start=1):
            rows.extend(_collect_dataset_rows(dataset_row, methods, views, include_missing))
            if verbose and i % 200 == 0:
                print(f"[collect] scanned {i}/{n} datasets...", flush=True)

    df = pd.DataFrame(rows, columns=RAW_COLUMNS)
    if verbose:
        print(f"[collect] {len(df)} raw rows from {n} datasets "
              f"({len(methods)} methods).", flush=True)
    return df
