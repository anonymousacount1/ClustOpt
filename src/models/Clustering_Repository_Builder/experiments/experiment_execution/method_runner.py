"""Run a single (dataset, method, view) experiment and persist all outputs.

Resume/skip/failure-safety live here: a completed successful run is skipped
under ``--resume``; any failure is captured into ``status.json`` (with
traceback) and never propagates so the dataset/worker/subfamily run continues.
"""
from __future__ import annotations

import json
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from .clustopt_execution import run_experiment_clustopt
from .config import ExperimentExecutionConfig, MethodSpec
from .external_metrics import compute_external_metrics
from .k_metrics import compute_k_metrics
from .output_writer import ensure_dir, read_json, write_csv_atomic, write_json_atomic
from .view_builder import ClustOptView, LoadedDatasetForUtility


@dataclass
class RunRecord:
    """One row per (dataset, view, method) for progress + raw aggregation."""

    dataset_id: str
    dataset_dir: str
    split_id: int
    family: Optional[str]
    subfamily: Optional[str]
    difficulty: Optional[str]
    cluster_count: Optional[int]
    view_id: str
    method_name: str
    method_group: str
    status: str  # 'success' | 'failed' | 'skipped'
    failure_reason: str = ""
    runtime_sec: float = 0.0
    ari: Optional[float] = None
    selected_k: Optional[int] = None
    true_k: Optional[int] = None
    top_metric: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


def run_output_dir(
    dataset_dir: Path, split_dir_name: str, method_name: str, view_id: str
) -> Path:
    return Path(dataset_dir) / "experiments" / split_dir_name / method_name / view_id


def _status_path(out_dir: Path) -> Path:
    return out_dir / "status.json"


def is_run_complete(out_dir: Path) -> bool:
    payload = read_json(_status_path(out_dir))
    if not payload:
        return False
    return str(payload.get("status", "")).lower() == "success"


def _load_config_snapshot(config_path: Path) -> Dict[str, Any]:
    return read_json(config_path) or {}


def _early_stopping_enabled(snapshot: Dict[str, Any]) -> bool:
    """True iff the optuna block requests a (non-null) patience callback."""
    sa = snapshot.get("search_algorithm", {})
    params = sa.get("params", {}) if isinstance(sa, dict) else {}
    return params.get("patience") is not None


def _trials_requested(snapshot: Dict[str, Any]):
    sa = snapshot.get("search_algorithm", {})
    params = sa.get("params", {}) if isinstance(sa, dict) else {}
    return params.get("n_trials")


def build_phaseD_metadata(
    *, snapshot: Dict[str, Any], outcome, method: MethodSpec, config_path: Path,
) -> Dict[str, Any]:
    """Assemble Phase-D runtime metadata from the config descriptor + outcome.

    Returns ``{}`` when the config carries no ``phaseD`` block (i.e. a legacy
    run), so existing outputs are never altered.
    """
    desc = snapshot.get("phaseD")
    if not isinstance(desc, dict):
        return {}
    rd = outcome.resolver_debug or {}
    selected_metrics = list(outcome.selected_metrics or [])
    selected_count = rd.get("selected_k_metrics_count")
    if selected_count is None:
        selected_count = len(selected_metrics)
    return {
        "phase": desc.get("phase"),
        "method_name": method.name,
        "utility_source": desc.get("utility_source"),
        "metric_ranking_source": desc.get("metric_ranking_source"),
        "dynamic_k_source": rd.get("dynamic_k_source", desc.get("dynamic_k_source")),
        # selected_k_metrics_count != selected_k (predicted #clusters); kept distinct.
        "selected_k_metrics_count": int(selected_count),
        "selected_metric_count": int(selected_count),
        "selected_metrics": selected_metrics,
        "selected_metric_utilities": rd.get("selected_utilities", {}),
        "selected_metric_weights": dict(outcome.selected_metric_weights or {}),
        "weighting_mode": desc.get("weighting_mode"),
        "softmax_temperature": desc.get("softmax_temperature"),
        "dynamic_topk_policy": rd.get("dynamic_topk_policy", desc.get("dynamic_topk_policy")),
        "k_rel_raw": rd.get("k_rel_raw"),
        "regressor_path": desc.get("regressor_path"),
        "real_utility_path": rd.get("real_utility_path", desc.get("real_utility_path")),
        "optuna_trials_requested": _trials_requested(snapshot),
        "optuna_trials_completed": int(outcome.n_trials_completed),
        "early_stopping_enabled": _early_stopping_enabled(snapshot),
        "search_map_source": str(config_path),
        "fallback_used": bool(outcome.used_fallback),
    }


def run_method_view(
    *,
    loaded: LoadedDatasetForUtility,
    view: ClustOptView,
    method: MethodSpec,
    config_path: Path,
    dataset_meta: Dict[str, Any],
    cfg: ExperimentExecutionConfig,
) -> RunRecord:
    """Execute one method on one view; write outputs; return a RunRecord."""
    split_dir_name = cfg.split_dir_name()
    out_dir = run_output_dir(loaded.dataset_dir, split_dir_name, method.name, view.view_id)

    base_record = RunRecord(
        dataset_id=loaded.dataset_id,
        dataset_dir=str(loaded.dataset_dir),
        split_id=int(cfg.split_id),
        family=dataset_meta.get("family"),
        subfamily=dataset_meta.get("subfamily"),
        difficulty=dataset_meta.get("difficulty"),
        cluster_count=dataset_meta.get("cluster_count"),
        view_id=view.view_id,
        method_name=method.name,
        method_group=method.group,
        status="failed",
    )

    # Resume short-circuit.
    if cfg.resume and not cfg.overwrite and is_run_complete(out_dir):
        base_record.status = "skipped"
        base_record.failure_reason = ""
        return base_record

    ensure_dir(out_dir)
    started_at = datetime.now(timezone.utc).isoformat()

    try:
        outcome = run_experiment_clustopt(
            config_path=Path(config_path),
            dataset_dir=Path(loaded.dataset_dir),
            view_id=view.view_id,
            X_decision=view.X_decision,
            X_full=view.X_full,
            y_true=view.y_true,
            runtime_block=cfg.runtime_block(),
        )
    except Exception as exc:  # whole run failed before producing results
        return _finalize_failure(
            base_record, out_dir, stage="clustopt_run", error=exc,
            started_at=started_at, config_path=config_path,
        )

    finished_at = datetime.now(timezone.utc).isoformat()

    # External + K metrics from the recovered best-config labels.
    ext = compute_external_metrics(view.y_true, outcome.y_pred)
    true_k = dataset_meta.get("cluster_count")
    if true_k is None:
        true_k = loaded.cluster_count_from_generator
    kmx = compute_k_metrics(outcome.y_pred, true_k)

    # Phase-D metadata (empty dict for legacy configs -> no behaviour change).
    snapshot = _load_config_snapshot(config_path)
    phaseD_meta = build_phaseD_metadata(
        snapshot=snapshot, outcome=outcome, method=method, config_path=config_path)

    # ---- write outputs (atomic) ----
    try:
        write_csv_atomic(out_dir / "clustopt_results.csv", outcome.results_df)

        best_result = {
            "selected_algorithm": outcome.selected_algorithm,
            "selected_params": outcome.selected_params,
            "best_objective_score": outcome.best_objective_score,
            "n_trials_completed": int(outcome.n_trials_completed),
            "selected_k": kmx.get("selected_k"),
            "true_k": kmx.get("true_k"),
            "k_correct": kmx.get("k_correct"),
            "k_abs_error": kmx.get("k_abs_error"),
            "k_signed_error": kmx.get("k_signed_error"),
            "cvi_type": outcome.cvi_type,
            "used_fallback": outcome.used_fallback,
        }
        # Search-budget accounting (additive; None for legacy/unseeded configs
        # so historical outputs keep their exact shape when re-run).
        budget = {
            "requested_trials": outcome.requested_trials,
            "delivered_trials": int(outcome.n_trials_completed),
            "valid_trials": outcome.valid_trials,
            "failed_trials": outcome.failed_trials,
            "search_seed": outcome.search_seed,
        }
        if any(v is not None for v in budget.values()):
            best_result.update(budget)
        write_json_atomic(out_dir / "best_result.json", best_result)
        write_json_atomic(out_dir / "external_metrics.json", ext)
        selected_payload = {
            "cvi_type": outcome.cvi_type,
            "method_group": method.group,
            "selected_metrics": outcome.selected_metrics,
            "selected_metric_weights": outcome.selected_metric_weights,
            "top_metric": outcome.top_metric,
            "metric_weight_entropy": outcome.metric_weight_entropy,
            "used_fallback": outcome.used_fallback,
            "resolver_debug": outcome.resolver_debug,
        }
        if phaseD_meta:  # additive: only present for Phase-D configs
            selected_payload["phaseD"] = phaseD_meta
        write_json_atomic(out_dir / "selected_metrics.json", selected_payload)
        write_json_atomic(out_dir / "timing_summary.json", {
            "runtime_sec": float(outcome.runtime_sec),
            "fit_predict_sec": float(outcome.fit_predict_sec),
            "cvi_eval_sec": float(outcome.cvi_eval_sec),
            "profiler_summary": outcome.profiler_summary,
        })
        write_json_atomic(out_dir / "config_snapshot.json", snapshot)
        if phaseD_meta:  # additive Phase-D metadata sidecar
            write_json_atomic(out_dir / "phaseD_metadata.json", phaseD_meta)
        write_json_atomic(out_dir / "run_log.json", {
            "dataset_id": loaded.dataset_id,
            "view_id": view.view_id,
            "method_name": method.name,
            "method_group": method.group,
            "config_path": str(config_path),
            "runtime_block": cfg.runtime_block(),
            "n_points": int(loaded.n_points),
            "started_at": started_at,
            "finished_at": finished_at,
            "search_budget": budget,
        })

        status_payload = {
            "status": "success",
            "dataset_id": loaded.dataset_id,
            "method_name": method.name,
            "view_id": view.view_id,
            "started_at": started_at,
            "finished_at": finished_at,
            "search_budget": budget,
            "runtime_sec": float(outcome.runtime_sec),
            "ari": ext.get("ari"),
            "selected_k": kmx.get("selected_k"),
            "true_k": kmx.get("true_k"),
        }
        write_json_atomic(_status_path(out_dir), status_payload)
    except Exception as exc:
        return _finalize_failure(
            base_record, out_dir, stage="write_outputs", error=exc,
            started_at=started_at, config_path=config_path,
        )

    base_record.status = "success"
    base_record.runtime_sec = float(outcome.runtime_sec)
    base_record.ari = ext.get("ari")
    base_record.selected_k = kmx.get("selected_k")
    base_record.true_k = kmx.get("true_k")
    base_record.top_metric = outcome.top_metric
    if phaseD_meta:
        base_record.extra = {
            "selected_metric_count": phaseD_meta["selected_metric_count"],
            "weighting_mode": phaseD_meta["weighting_mode"],
            "dynamic_k_source": phaseD_meta["dynamic_k_source"],
            "utility_source": phaseD_meta["utility_source"],
            "metric_ranking_source": phaseD_meta["metric_ranking_source"],
            "early_stopping_enabled": phaseD_meta["early_stopping_enabled"],
            "optuna_trials_requested": phaseD_meta["optuna_trials_requested"],
            "optuna_trials_completed": phaseD_meta["optuna_trials_completed"],
            "fallback_used": phaseD_meta["fallback_used"],
        }
    return base_record


def _finalize_failure(
    base: RunRecord,
    out_dir: Path,
    *,
    stage: str,
    error: BaseException,
    started_at: str,
    config_path: Path,
) -> RunRecord:
    tb = "".join(traceback.format_exception(type(error), error, error.__traceback__))
    payload = {
        "status": "failed",
        "dataset_id": base.dataset_id,
        "method_name": base.method_name,
        "view_id": base.view_id,
        "stage": stage,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "traceback": tb,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
    }
    try:
        ensure_dir(out_dir)
        write_json_atomic(_status_path(out_dir), payload)
    except Exception:
        pass
    base.status = "failed"
    base.failure_reason = f"{type(error).__name__}: {error}"
    return base
