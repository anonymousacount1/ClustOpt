"""Convert a normalised AutoClust adapter result into on-disk experiment outputs.

Mirrors the AutoML4Clust / ML2DAC result converters (identical metrics + layout)
and adds AutoClust-specific fields: ``selected_algorithm``, ``optimizer``,
``optimizer_evaluations``, ``mlp_objective_used``, ``algorithm_selection_method``,
``selected_cvis``. AutoClust runs natively (no wall-clock timeout), so there are no
timeout fields.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from ..utils.metrics import compute_all_metrics
from ..utils.path_utils import (
    ensure_dir,
    read_json,
    write_json_atomic,
    write_npy_atomic,
    write_text_atomic,
)
from ..utils.view_selection import select_best_record_by_ari


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def record_output_dir(autoclust_dir: Path, record_type: str) -> Path:
    return Path(autoclust_dir) / record_type


def record_is_complete(out_dir: Path) -> bool:
    payload = read_json(Path(out_dir) / "result.json")
    if not payload:
        return False
    return str(payload.get("status", "")).lower() == "success"


def evaluate_record(adapter_result: Dict[str, Any], y_true, true_k: Optional[int]) -> Dict[str, Any]:
    status = adapter_result.get("status", "failed")
    labels = adapter_result.get("labels")
    if status == "success" and labels is not None:
        metrics = compute_all_metrics(y_true, np.asarray(labels), true_k)
    else:
        metrics = compute_all_metrics(None, None, true_k)
    metrics["status"] = status
    metrics["runtime_sec"] = float(adapter_result.get("runtime_sec", 0.0) or 0.0)
    return metrics


def build_result_json(adapter_result, metrics, *, dataset_meta, record_type, split_id, config) -> Dict[str, Any]:
    art = adapter_result.get("artifact_status") or {}
    return {
        "method": "AutoClust", "implementation": "ML2DAC_RelatedWork_Reimplementation",
        "record_type": record_type, "split_id": int(split_id),
        "status": adapter_result.get("status", "failed"),
        "failure_kind": adapter_result.get("failure_kind"),
        "dataset_id": dataset_meta.get("dataset_id"), "family": dataset_meta.get("family"),
        "subfamily": dataset_meta.get("subfamily"), "difficulty": dataset_meta.get("difficulty"),
        "selected_k": adapter_result.get("selected_k"),
        "selected_algorithm": adapter_result.get("selected_algorithm"),
        "algorithm_selection_method": adapter_result.get("algorithm_selection_method"),
        "selected_cvis": adapter_result.get("selected_cvis"),
        "mlp_objective_used": adapter_result.get("mlp_objective_used"),
        "optimizer": adapter_result.get("optimizer"),
        "optimizer_evaluations": adapter_result.get("optimizer_evaluations"),
        "best_configuration": adapter_result.get("best_configuration"),
        "runtime_sec": float(adapter_result.get("runtime_sec", 0.0) or 0.0),
        "metrics": {k: v for k, v in metrics.items() if k not in ("status", "runtime_sec")},
        "details": adapter_result.get("details"),
        "artifact_status": {
            "has_autoclust_implementation": art.get("has_autoclust_implementation"),
            "has_pretrained_artifacts": art.get("has_pretrained_artifacts"),
            "has_trained_mlp": art.get("has_trained_mlp"),
            "can_run_online_phase_without_offline_phase": art.get("can_run_online_phase_without_offline_phase"),
            "missing_required_artifacts": art.get("missing_required_artifacts"),
        },
        "config": config, "error": adapter_result.get("error"),
        "error_type": adapter_result.get("error_type"),
        "error_message": adapter_result.get("error_message"),
        "stage": adapter_result.get("stage"), "traceback": adapter_result.get("traceback"),
        "finished_at": _now(),
    }


def build_summary_metrics(metrics, record_type, adapter_result) -> Dict[str, Any]:
    keys = ["ari", "nmi", "ami", "v_measure", "homogeneity", "completeness", "fowlkes_mallows",
            "purity", "selected_k", "true_k", "k_error", "abs_k_error", "exact_k_match",
            "n_predicted_clusters_excluding_noise", "n_noise_points", "predicted_noise_ratio",
            "true_noise_ratio", "runtime_sec", "status"]
    out: Dict[str, Any] = {"record_type": record_type}
    for k in keys:
        out[k] = metrics.get(k)
    out["selected_algorithm"] = adapter_result.get("selected_algorithm")
    out["optimizer"] = adapter_result.get("optimizer")
    out["optimizer_evaluations"] = adapter_result.get("optimizer_evaluations")
    out["mlp_objective_used"] = adapter_result.get("mlp_objective_used")
    return out


def write_record_outputs(out_dir, *, adapter_result, metrics, dataset_meta, record_type,
                         split_id, config, run_log_text) -> Dict[str, Any]:
    out_dir = Path(out_dir)
    ensure_dir(out_dir)
    payload = build_result_json(adapter_result, metrics, dataset_meta=dataset_meta,
                                record_type=record_type, split_id=split_id, config=config)
    write_json_atomic(out_dir / "result.json", payload)
    write_json_atomic(out_dir / "summary_metrics.json",
                      build_summary_metrics(metrics, record_type, adapter_result))
    labels = adapter_result.get("labels")
    if labels is not None:
        try:
            write_npy_atomic(out_dir / "labels.npy", np.asarray(labels))
        except Exception:
            pass
    raw = adapter_result.get("raw_autoclust_result")
    if raw is not None:
        try:
            write_json_atomic(out_dir / "raw_result.json", raw)
        except Exception:
            write_json_atomic(out_dir / "raw_result.json", {"note": "raw result not JSON-serialisable"})
    if adapter_result.get("artifact_status") is not None:
        try:
            write_json_atomic(out_dir / "autoclust_artifact_report.json", adapter_result["artifact_status"])
        except Exception:
            pass
    try:
        write_text_atomic(out_dir / "run_log.txt", run_log_text)
    except Exception:
        pass
    return payload


def build_dataset_summary(dataset_meta, record_results, *, split_id, total_runtime_sec, config,
                          artifact_status=None) -> Dict[str, Any]:
    best_block = select_best_record_by_ari(record_results)
    statuses, failures, detail = {}, {}, {}
    for rtype, payload in record_results.items():
        st = payload.get("status", "unknown")
        statuses[rtype] = st
        detail[rtype] = {"status": st,
                         "selected_algorithm": payload.get("selected_algorithm"),
                         "optimizer_evaluations": payload.get("optimizer_evaluations"),
                         "runtime_sec": payload.get("runtime_sec")}
        if st != "success":
            msg = payload.get("error_message") or payload.get("error")
            if msg:
                failures[rtype] = str(msg).splitlines()[0]
    return {
        "method": "AutoClust", "implementation": "ML2DAC_RelatedWork_Reimplementation",
        "dataset_id": dataset_meta.get("dataset_id"), "split_id": int(split_id),
        "family": dataset_meta.get("family"), "subfamily": dataset_meta.get("subfamily"),
        "difficulty": dataset_meta.get("difficulty"), "true_k": dataset_meta.get("true_k"),
        "best_by_ari": best_block, "record_statuses": statuses, "records_detail": detail,
        "failures": failures, "total_runtime_sec": float(total_runtime_sec),
        "n_records": len(record_results),
        "artifact_status": {
            "has_pretrained_artifacts": (artifact_status or {}).get("has_pretrained_artifacts"),
            "has_trained_mlp": (artifact_status or {}).get("has_trained_mlp"),
            "missing_required_artifacts": (artifact_status or {}).get("missing_required_artifacts"),
        },
        "finished_at": _now(),
    }


__all__ = ["record_output_dir", "record_is_complete", "evaluate_record", "build_result_json",
           "build_summary_metrics", "write_record_outputs", "build_dataset_summary"]
