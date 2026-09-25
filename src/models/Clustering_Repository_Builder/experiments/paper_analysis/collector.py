"""Build the canonical **run frame**: one row per ``dataset x method x view``.

Reads the per-run outputs straight off disk (nothing is recomputed) and attaches
the registry provenance for every row, so no downstream module has to re-derive a
method's taxonomy from its name.

Two on-disk schemas are supported:

**ClustOpt** (``schema='clustopt'``, record dirs ``x_only|y_only|xy_2d``)
    ``status.json``, ``external_metrics.json``, ``best_result.json``,
    ``selected_metrics.json``, ``timing_summary.json``, ``phaseD_metadata.json``
    (present for the Phase-D/E2 methods; carries the *executed* utility source,
    weighting mode and dynamic-K source, which the collector cross-checks against
    the registry and reports as a provenance mismatch if they disagree).

**External** (``schema='external'``, record dirs ``1d_x|1d_y|2d``)
    ``result.json`` (falling back to ``summary_metrics.json``). No metric
    selection, no candidate trace, no runtime components -- those columns stay
    null and the capability matrix records why.

Every read goes through the long-path helpers in :mod:`paths`: the run
directories are ~270 characters before a filename is appended.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import paths
from .method_registry import (
    METHODS, VIEW_IDS, MethodSpec, select_methods,
)

# --------------------------------------------------------------------------- #
# Column contract
# --------------------------------------------------------------------------- #
IDENTITY_COLUMNS: Tuple[str, ...] = (
    "dataset_id", "dataset_dir", "split_id", "family", "family_id",
    "subfamily", "difficulty", "cluster_count", "n_points", "view_id",
)

PROVENANCE_COLUMNS: Tuple[str, ...] = (
    "method_name", "method_full_name", "method_short_label", "paper_label",
    "on_disk_name", "schema", "framework", "method_family", "utility_source",
    "k_policy", "fixed_metric_count", "dynamic_k_source", "weighting_mode",
    "softmax_temperature", "metric_inventory", "metric_inventory_size",
    "selection_capability", "oracle_level", "uses_real_utility",
    "uses_real_dynamic_k", "is_oracle_assisted", "is_utility_oracle",
    "is_meta_learning", "is_external", "is_paper_method", "same_search_space",
    "search_budget",
)

STATUS_COLUMNS: Tuple[str, ...] = ("status", "failure_reason")

RESULT_COLUMNS: Tuple[str, ...] = (
    "selected_algorithm", "selected_params", "selected_k", "true_k",
    "k_correct", "k_abs_error", "k_signed_error", "best_objective_score",
    "n_trials_completed", "used_fallback",
)

EXTERNAL_METRIC_COLUMNS: Tuple[str, ...] = (
    "ari", "nmi", "ami", "v_measure", "homogeneity", "completeness",
    "fowlkes_mallows", "purity", "predicted_noise_ratio", "true_noise_ratio",
    "noise_precision", "noise_recall", "noise_f1",
)

RUNTIME_COLUMNS: Tuple[str, ...] = (
    "runtime_sec", "fit_predict_sec", "cvi_eval_sec", "cvi_normalize_sec",
    "cvi_aggregate_sec", "n_cvi_evaluations",
)

METRIC_SELECTION_COLUMNS: Tuple[str, ...] = (
    "selected_metrics", "selected_metric_weights", "selected_metric_utilities",
    "selected_metric_count", "metric_weight_entropy", "top_metric",
    "cvi_type", "dynamic_topk_policy", "executed_utility_source",
    "executed_weighting_mode", "executed_dynamic_k_source", "provenance_mismatch",
)

PATH_COLUMNS: Tuple[str, ...] = ("run_dir", "config_path", "trace_path")

RUN_FRAME_COLUMNS: Tuple[str, ...] = (
    IDENTITY_COLUMNS + PROVENANCE_COLUMNS + STATUS_COLUMNS + RESULT_COLUMNS
    + EXTERNAL_METRIC_COLUMNS + RUNTIME_COLUMNS + METRIC_SELECTION_COLUMNS
    + PATH_COLUMNS
)

# Status vocabulary. ``missing`` == the run directory has no readable outcome
# file at all (never silently dropped: it is a coverage fact).
STATUS_SUCCESS = "success"
STATUS_MISSING = "missing"


@dataclass
class CollectionStats:
    """Counters + warnings from one collection pass."""

    n_datasets: int = 0
    n_rows: int = 0
    n_success: int = 0
    n_failed: int = 0
    n_missing: int = 0
    n_provenance_mismatch: int = 0
    warnings: Dict[str, int] = None            # warning key -> count

    def __post_init__(self) -> None:
        if self.warnings is None:
            self.warnings = {}

    def warn(self, key: str) -> None:
        self.warnings[key] = self.warnings.get(key, 0) + 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_datasets": self.n_datasets, "n_rows": self.n_rows,
            "n_success": self.n_success, "n_failed": self.n_failed,
            "n_missing": self.n_missing,
            "n_provenance_mismatch": self.n_provenance_mismatch,
            "warnings": dict(sorted(self.warnings.items())),
        }


# --------------------------------------------------------------------------- #
# Provenance mapping between the on-disk metadata block and the registry
# --------------------------------------------------------------------------- #
_DISK_UTILITY_SOURCE = {
    "regressor": "mlp", "knn": "knn", "real": "real", "oracle": "real",
    "fixed": None,        # baselines: 'fixed' covers uniform + single_cvi
}
_DISK_WEIGHTING = {
    "raw_normalized": "raw", "softmax": "softmax_t05",
    "softmax_t05": "softmax_t05", "uniform": "uniform", "single": "single",
}
_DISK_DYNAMIC_K = {
    "none": "none", "predicted": "predicted", "real": "real", None: "none",
}


def _check_provenance(spec: MethodSpec, meta: Dict[str, Any]) -> str:
    """Return a ';'-joined mismatch description ('' when consistent).

    Compares the registry's declared taxonomy against the ``phaseD_metadata.json``
    block written *by the run itself*. Any disagreement is a hard signal that the
    registry mis-describes a method, so it is surfaced rather than tolerated.
    """
    if not meta:
        return ""
    problems: List[str] = []
    disk_src = _DISK_UTILITY_SOURCE.get(str(meta.get("utility_source")), "?")
    if disk_src is not None and disk_src != "?" and disk_src != spec.utility_source:
        problems.append(f"utility_source disk={meta.get('utility_source')!r} "
                        f"registry={spec.utility_source!r}")
    disk_w = _DISK_WEIGHTING.get(str(meta.get("weighting_mode")))
    if disk_w is not None and disk_w != spec.weighting_mode:
        problems.append(f"weighting_mode disk={meta.get('weighting_mode')!r} "
                        f"registry={spec.weighting_mode!r}")
    disk_k = _DISK_DYNAMIC_K.get(meta.get("dynamic_k_source"), "?")
    if disk_k != "?" and disk_k != spec.dynamic_k_source:
        problems.append(f"dynamic_k_source disk={meta.get('dynamic_k_source')!r} "
                        f"registry={spec.dynamic_k_source!r}")
    disk_name = meta.get("method_name")
    if disk_name and str(disk_name) != spec.on_disk_name:
        problems.append(f"method_name disk={disk_name!r} "
                        f"registry={spec.on_disk_name!r}")
    return ";".join(problems)


# --------------------------------------------------------------------------- #
# Row construction
# --------------------------------------------------------------------------- #
def _empty_row() -> Dict[str, Any]:
    return {c: None for c in RUN_FRAME_COLUMNS}


def _provenance(spec: MethodSpec) -> Dict[str, Any]:
    return {
        "method_name": spec.method_name,
        "method_full_name": spec.display_name,
        "method_short_label": spec.short_label,
        "paper_label": spec.paper_label,
        "on_disk_name": spec.on_disk_name,
        "schema": spec.schema,
        "framework": spec.framework,
        "method_family": spec.method_family,
        "utility_source": spec.utility_source,
        "k_policy": spec.k_policy,
        "fixed_metric_count": spec.fixed_metric_count,
        "dynamic_k_source": spec.dynamic_k_source,
        "weighting_mode": spec.weighting_mode,
        "softmax_temperature": spec.softmax_temperature,
        "metric_inventory": spec.metric_inventory,
        "metric_inventory_size": spec.metric_inventory_size,
        "selection_capability": ("metric_weights" if spec.metric_selection_capable
                                 else "none"),
        "oracle_level": spec.oracle_level,
        "uses_real_utility": spec.uses_real_utility,
        "uses_real_dynamic_k": spec.uses_real_dynamic_k,
        "is_oracle_assisted": spec.is_oracle_assisted,
        "is_utility_oracle": spec.is_utility_oracle,
        "is_meta_learning": spec.is_meta_learning,
        "is_external": spec.is_external,
        "is_paper_method": spec.is_paper_method,
        "same_search_space": spec.same_search_space,
        "search_budget": spec.search_budget,
    }


def _identity(ds: pd.Series, view_id: str) -> Dict[str, Any]:
    return {
        "dataset_id": ds.get("dataset_id"),
        "dataset_dir": str(ds.get("dataset_dir")),
        "split_id": int(ds.get("split_id")),
        "family": ds.get("family"),
        "family_id": ds.get("family_id"),
        "subfamily": ds.get("subfamily"),
        "difficulty": ds.get("difficulty"),
        "cluster_count": ds.get("cluster_count"),
        "n_points": ds.get("n_points"),
        "view_id": view_id,
    }


def _profiler_sec(timing: Dict[str, Any], key: str) -> Optional[float]:
    prof = timing.get("profiler_summary")
    if not isinstance(prof, dict):
        return None
    entry = prof.get(key)
    if isinstance(entry, dict) and entry.get("total_sec") is not None:
        try:
            return float(entry["total_sec"])
        except (TypeError, ValueError):
            return None
    return None


def _profiler_count(timing: Dict[str, Any], key: str) -> Optional[int]:
    prof = timing.get("profiler_summary")
    if not isinstance(prof, dict):
        return None
    entry = prof.get(key)
    if isinstance(entry, dict) and entry.get("count") is not None:
        try:
            return int(entry["count"])
        except (TypeError, ValueError):
            return None
    return None


def _collect_clustopt(run_dir: Path, row: Dict[str, Any], spec: MethodSpec,
                      stats: CollectionStats) -> Dict[str, Any]:
    status_payload = paths.read_json(run_dir / "status.json")
    if not status_payload:
        row["status"] = STATUS_MISSING
        stats.warn("clustopt_missing_status_json")
        return row
    status = str(status_payload.get("status", "")).lower() or STATUS_MISSING
    row["status"] = status
    if status != STATUS_SUCCESS:
        row["failure_reason"] = str(status_payload.get("error_message") or "")
        return row

    ext = paths.read_json(run_dir / "external_metrics.json") or {}
    best = paths.read_json(run_dir / "best_result.json") or {}
    sel = paths.read_json(run_dir / "selected_metrics.json") or {}
    timing = paths.read_json(run_dir / "timing_summary.json") or {}
    meta = paths.read_json(run_dir / "phaseD_metadata.json") or {}

    if not ext:
        stats.warn("clustopt_missing_external_metrics")
    if not best:
        stats.warn("clustopt_missing_best_result")
    if spec.metric_selection_capable and not sel:
        stats.warn("clustopt_missing_selected_metrics")
    if not timing:
        stats.warn("clustopt_missing_timing_summary")

    row.update({
        "selected_algorithm": best.get("selected_algorithm"),
        "selected_params": best.get("selected_params"),
        "selected_k": best.get("selected_k", status_payload.get("selected_k")),
        "true_k": best.get("true_k", status_payload.get("true_k")),
        "k_correct": best.get("k_correct"),
        "k_abs_error": best.get("k_abs_error"),
        "k_signed_error": best.get("k_signed_error"),
        "best_objective_score": best.get("best_objective_score"),
        "n_trials_completed": best.get("n_trials_completed"),
        "used_fallback": best.get("used_fallback", sel.get("used_fallback")),
        "runtime_sec": timing.get("runtime_sec", status_payload.get("runtime_sec")),
        "fit_predict_sec": timing.get("fit_predict_sec"),
        "cvi_eval_sec": timing.get("cvi_eval_sec"),
        "cvi_normalize_sec": _profiler_sec(timing, "cvi_normalize"),
        "cvi_aggregate_sec": _profiler_sec(timing, "cvi_aggregate"),
        "n_cvi_evaluations": _profiler_count(timing, "cvi_evaluate"),
        "cvi_type": sel.get("cvi_type") or best.get("cvi_type"),
    })
    for key in EXTERNAL_METRIC_COLUMNS:
        row[key] = ext.get(key)

    selected = sel.get("selected_metrics") or []
    weights = sel.get("selected_metric_weights") or {}
    utilities = (meta.get("selected_metric_utilities")
                 or (sel.get("resolver_debug") or {}).get("selected_utilities") or {})
    row.update({
        "selected_metrics": json.dumps(list(selected)),
        "selected_metric_weights": json.dumps(dict(weights)),
        "selected_metric_utilities": json.dumps(dict(utilities)),
        "selected_metric_count": len(selected) if selected else None,
        "metric_weight_entropy": sel.get("metric_weight_entropy"),
        "top_metric": sel.get("top_metric"),
        "dynamic_topk_policy": meta.get("dynamic_topk_policy")
        or (sel.get("resolver_debug") or {}).get("dynamic_topk_policy"),
        "executed_utility_source": meta.get("utility_source"),
        "executed_weighting_mode": meta.get("weighting_mode"),
        "executed_dynamic_k_source": meta.get("dynamic_k_source"),
    })

    mismatch = _check_provenance(spec, meta)
    row["provenance_mismatch"] = mismatch
    if mismatch:
        stats.n_provenance_mismatch += 1
        stats.warn("provenance_mismatch")
    return row


def _collect_external(run_dir: Path, row: Dict[str, Any], spec: MethodSpec,
                      stats: CollectionStats) -> Dict[str, Any]:
    result = paths.read_json(run_dir / "result.json")
    source = "result.json"
    if not result:
        result = paths.read_json(run_dir / "summary_metrics.json")
        source = "summary_metrics.json"
        if result:
            stats.warn("external_fell_back_to_summary_metrics")
    if not result:
        row["status"] = STATUS_MISSING
        stats.warn("external_missing_result_json")
        return row

    status = str(result.get("status", "")).lower() or STATUS_MISSING
    row["status"] = status
    if status != STATUS_SUCCESS:
        row["failure_reason"] = str(result.get("error_message")
                                    or result.get("failure_kind") or "")
        return row

    metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else result
    row.update({
        "selected_algorithm": result.get("selected_algorithm"),
        "selected_params": (json.dumps(result.get("best_configuration"))
                            if result.get("best_configuration") is not None else None),
        "selected_k": result.get("selected_k", metrics.get("selected_k")),
        "true_k": metrics.get("true_k"),
        "k_correct": metrics.get("k_correct", metrics.get("exact_k_match")),
        "k_abs_error": metrics.get("k_abs_error", metrics.get("abs_k_error")),
        "k_signed_error": metrics.get("k_signed_error", metrics.get("k_error")),
        "n_trials_completed": result.get("optimizer_evaluations"),
        "runtime_sec": result.get("runtime_sec", metrics.get("runtime_sec")),
        "cvi_type": result.get("optimizer"),
        "config_path": str(run_dir / source),
    })
    for key in EXTERNAL_METRIC_COLUMNS:
        row[key] = metrics.get(key)
    return row


def _collect_run(ds: pd.Series, spec: MethodSpec, view_id: str, split_id: int,
                 stats: CollectionStats) -> Dict[str, Any]:
    dataset_dir = Path(str(ds["dataset_dir"]))
    run_dir = (dataset_dir / "experiments"
               / paths.split_dir_name(split_id, spec.search_suffix)
               / spec.on_disk_name / spec.view_dirs[view_id])
    row = _empty_row()
    row.update(_identity(ds, view_id))
    row.update(_provenance(spec))
    row["status"] = STATUS_MISSING
    row["failure_reason"] = ""
    row["provenance_mismatch"] = ""
    row["run_dir"] = str(run_dir)
    if spec.schema == "clustopt":
        row["config_path"] = str(run_dir / "config_snapshot.json")
        row["trace_path"] = str(run_dir / "clustopt_results.csv")
        return _collect_clustopt(run_dir, row, spec, stats)
    row["config_path"] = str(run_dir / "result.json")
    row["trace_path"] = None
    return _collect_external(run_dir, row, spec, stats)


def _collect_dataset(ds: pd.Series, specs: Sequence[MethodSpec],
                     views: Sequence[str], split_id: int
                     ) -> Tuple[List[Dict[str, Any]], CollectionStats]:
    """All rows for one dataset (own stats object -- merged by the caller)."""
    stats = CollectionStats()
    rows: List[Dict[str, Any]] = []
    for spec in specs:
        for view_id in views:
            rows.append(_collect_run(ds, spec, view_id, split_id, stats))
    return rows, stats


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def collect_run_frame(
    *,
    assignments: pd.DataFrame,
    split_id: int,
    method_names: Optional[Sequence[str]] = None,
    views: Sequence[str] = VIEW_IDS,
    max_workers: Optional[int] = None,
    limit_datasets: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, CollectionStats]:
    """Return ``(run_frame, stats)`` for one split.

    ``max_workers > 1`` enables a dataset-level thread pool. The per-run reads are
    many small long-path JSON opens (I/O-bound, GIL-releasing), and
    ``ThreadPoolExecutor.map`` preserves input order, so the resulting row order
    is byte-identical to the serial path.
    """
    specs = select_methods(method_names)
    sub = assignments[assignments["split_id"].astype("Int64") == int(split_id)]
    dataset_rows = [r for _, r in sub.iterrows()]
    if limit_datasets:
        dataset_rows = dataset_rows[: int(limit_datasets)]
    n = len(dataset_rows)

    total = CollectionStats(n_datasets=n)
    rows: List[Dict[str, Any]] = []

    def _merge(result: Tuple[List[Dict[str, Any]], CollectionStats]) -> None:
        part_rows, part_stats = result
        rows.extend(part_rows)
        for key, count in part_stats.warnings.items():
            total.warnings[key] = total.warnings.get(key, 0) + count
        total.n_provenance_mismatch += part_stats.n_provenance_mismatch

    if verbose:
        print(f"[collect] split={split_id} datasets={n} methods={len(specs)} "
              f"views={len(views)} -> {n * len(specs) * len(views)} expected rows",
              flush=True)

    if max_workers and int(max_workers) > 1 and n:
        done = 0
        with ThreadPoolExecutor(max_workers=int(max_workers)) as pool:
            for result in pool.map(
                lambda ds: _collect_dataset(ds, specs, views, split_id),
                dataset_rows,
            ):
                _merge(result)
                done += 1
                if verbose and done % 100 == 0:
                    print(f"[collect] {done}/{n} datasets "
                          f"({int(max_workers)} workers)...", flush=True)
    else:
        for i, ds in enumerate(dataset_rows, start=1):
            _merge(_collect_dataset(ds, specs, views, split_id))
            if verbose and i % 100 == 0:
                print(f"[collect] {i}/{n} datasets...", flush=True)

    frame = pd.DataFrame(rows, columns=list(RUN_FRAME_COLUMNS))
    frame = _coerce_types(frame)
    total.n_rows = len(frame)
    total.n_success = int((frame["status"] == STATUS_SUCCESS).sum())
    total.n_missing = int((frame["status"] == STATUS_MISSING).sum())
    total.n_failed = int(total.n_rows - total.n_success - total.n_missing)
    if verbose:
        print(f"[collect] {total.n_rows} rows | success={total.n_success} "
              f"failed={total.n_failed} missing={total.n_missing} | "
              f"warnings={total.warnings or 'none'}", flush=True)
    return frame, total


_NUMERIC_COLUMNS: Tuple[str, ...] = (
    EXTERNAL_METRIC_COLUMNS + RUNTIME_COLUMNS
    + ("selected_k", "true_k", "k_abs_error", "k_signed_error",
       "best_objective_score", "n_trials_completed", "cluster_count",
       "n_points", "selected_metric_count", "metric_weight_entropy",
       "fixed_metric_count", "softmax_temperature", "metric_inventory_size",
       "search_budget")
)
_BOOL_COLUMNS: Tuple[str, ...] = (
    "k_correct", "used_fallback", "uses_real_utility", "uses_real_dynamic_k",
    "is_oracle_assisted", "is_utility_oracle", "is_meta_learning", "is_external",
    "is_paper_method", "same_search_space",
)


def _coerce_types(frame: pd.DataFrame) -> pd.DataFrame:
    """Numeric/boolean coercion so downstream code never re-parses strings."""
    for col in _NUMERIC_COLUMNS:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    for col in _BOOL_COLUMNS:
        if col in frame.columns:
            frame[col] = frame[col].map(_to_bool).astype("boolean")
    for col in ("status", "failure_reason", "provenance_mismatch"):
        if col in frame.columns:
            frame[col] = frame[col].fillna("").astype(str)
    return frame


def _to_bool(value: Any) -> Optional[bool]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in ("true", "1", "1.0", "yes"):
        return True
    if text in ("false", "0", "0.0", "no"):
        return False
    return None


# --------------------------------------------------------------------------- #
# Coverage audit
# --------------------------------------------------------------------------- #
def coverage_audit(frame: pd.DataFrame, *, expected_views: int = len(VIEW_IDS)
                   ) -> pd.DataFrame:
    """Per-method coverage table: datasets, views, success/failure/missing."""
    if frame.empty:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    for name, grp in frame.groupby("method_name", dropna=False):
        n_ds = int(grp["dataset_id"].nunique())
        per_ds_views = grp.groupby("dataset_id")["view_id"].nunique()
        succ = grp[grp["status"] == STATUS_SUCCESS]
        rows.append({
            "method_name": name,
            "method_short_label": grp["method_short_label"].iloc[0],
            "method_family": grp["method_family"].iloc[0],
            "n_datasets": n_ds,
            "n_runs": int(len(grp)),
            "n_success": int(len(succ)),
            "n_failed": int((~grp["status"].isin([STATUS_SUCCESS, STATUS_MISSING])).sum()),
            "n_missing": int((grp["status"] == STATUS_MISSING).sum()),
            "run_success_rate": float(len(succ) / len(grp)) if len(grp) else float("nan"),
            "n_datasets_with_all_views": int((per_ds_views >= expected_views).sum()),
            "n_datasets_with_any_success": int(succ["dataset_id"].nunique()),
            "n_datasets_with_no_success": int(n_ds - succ["dataset_id"].nunique()),
            "n_with_finite_ari": int(pd.to_numeric(succ["ari"], errors="coerce")
                                     .notna().sum()),
            "n_provenance_mismatch": int((grp["provenance_mismatch"].astype(str)
                                          .str.len() > 0).sum()),
        })
    out = pd.DataFrame(rows)
    return out.sort_values(["method_family", "method_name"]).reset_index(drop=True)


def observed_selected_metrics(frame: pd.DataFrame) -> List[str]:
    """Every distinct metric name that appears in any run's selection."""
    seen: set = set()
    for payload in frame.get("selected_metrics", pd.Series(dtype=object)).dropna():
        try:
            names = json.loads(payload) if isinstance(payload, str) else list(payload)
        except (TypeError, ValueError):
            continue
        seen.update(str(n) for n in names)
    return sorted(seen)


def parse_weights(payload: Any) -> Dict[str, float]:
    """Parse a ``selected_metric_weights`` cell into ``{metric: weight}``."""
    if payload is None or (isinstance(payload, float) and np.isnan(payload)):
        return {}
    try:
        data = json.loads(payload) if isinstance(payload, str) else dict(payload)
    except (TypeError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    out: Dict[str, float] = {}
    for key, value in data.items():
        try:
            out[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def parse_metric_list(payload: Any) -> List[str]:
    if payload is None or (isinstance(payload, float) and np.isnan(payload)):
        return []
    try:
        data = json.loads(payload) if isinstance(payload, str) else list(payload)
    except (TypeError, ValueError):
        return []
    return [str(n) for n in data] if isinstance(data, (list, tuple)) else []
