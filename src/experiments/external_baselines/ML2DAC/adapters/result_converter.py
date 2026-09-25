"""Convert a normalised ML2DAC adapter result into on-disk experiment outputs.

Mirrors the AutoML4Clust result converter (same metrics + layout) and adds the
ML2DAC-specific fields: ``selected_cvi``, ``selected_algorithm``,
``warmstart_count``, and a per-record ``ml2dac_artifact_report.json``.

Per-record output folder layout::

    <...>/ML2DAC/<record_type>/
        result.json                full normalised result + metrics + provenance
        labels.npy                 predicted labels (success only)
        summary_metrics.json       comparable metric subset
        raw_result.json            ML2DAC optimizer history / meta-learning info
        run_log.txt                human-readable per-record log
        ml2dac_artifact_report.json  MKR artifact availability report
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


def record_output_dir(ml2dac_dir: Path, record_type: str) -> Path:
    return Path(ml2dac_dir) / record_type


def record_is_complete(out_dir: Path) -> bool:
    """Resume key: a record is complete iff its result.json status == success."""
    payload = read_json(Path(out_dir) / "result.json")
    if not payload:
        return False
    return str(payload.get("status", "")).lower() == "success"


def evaluate_record(adapter_result: Dict[str, Any], y_true, true_k: Optional[int]) -> Dict[str, Any]:
    """Compute external + K metrics for a (possibly failed) adapter result."""
    status = adapter_result.get("status", "failed")
    labels = adapter_result.get("labels")
    if status == "success" and labels is not None:
        metrics = compute_all_metrics(y_true, np.asarray(labels), true_k)
    else:
        metrics = compute_all_metrics(None, None, true_k)
    metrics["status"] = status
    metrics["runtime_sec"] = float(adapter_result.get("runtime_sec", 0.0) or 0.0)
    return metrics


def build_result_json(
    adapter_result: Dict[str, Any],
    metrics: Dict[str, Any],
    *,
    dataset_meta: Dict[str, Any],
    record_type: str,
    split_id: int,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    artifact_status = adapter_result.get("artifact_status") or {}
    return {
        "method": "ML2DAC",
        "record_type": record_type,
        "split_id": int(split_id),
        "status": adapter_result.get("status", "failed"),
        "failure_kind": adapter_result.get("failure_kind"),
        "dataset_id": dataset_meta.get("dataset_id"),
        "family": dataset_meta.get("family"),
        "subfamily": dataset_meta.get("subfamily"),
        "difficulty": dataset_meta.get("difficulty"),
        "selected_k": adapter_result.get("selected_k"),
        "selected_cvi": adapter_result.get("selected_cvi"),
        "selected_algorithm": adapter_result.get("selected_algorithm"),
        "selected_algorithms": adapter_result.get("selected_algorithms"),
        "warmstart_count": adapter_result.get("warmstart_count"),
        "best_configuration": adapter_result.get("best_configuration"),
        "runtime_sec": float(adapter_result.get("runtime_sec", 0.0) or 0.0),
        "metrics": {k: v for k, v in metrics.items() if k not in ("status", "runtime_sec")},
        "details": adapter_result.get("details"),
        "artifact_status": {
            "has_pretrained_artifacts": artifact_status.get("has_pretrained_artifacts"),
            "can_run_full_ml2dac": artifact_status.get("can_run_full_ml2dac"),
            "missing_required_artifacts": artifact_status.get("missing_required_artifacts"),
            "mkr_path": artifact_status.get("mkr_path"),
            "mf_set_name": artifact_status.get("mf_set_name"),
        },
        "config": config,
        "error": adapter_result.get("error"),
        "error_type": adapter_result.get("error_type"),
        "error_message": adapter_result.get("error_message"),
        "stage": adapter_result.get("stage"),
        "traceback": adapter_result.get("traceback"),
        "finished_at": _now(),
    }


def build_summary_metrics(metrics: Dict[str, Any], record_type: str,
                          adapter_result: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "ari", "nmi", "ami", "v_measure", "homogeneity", "completeness",
        "fowlkes_mallows", "purity",
        "selected_k", "true_k", "k_error", "abs_k_error", "exact_k_match",
        "n_predicted_clusters_excluding_noise", "n_noise_points",
        "predicted_noise_ratio", "true_noise_ratio",
        "runtime_sec", "status",
    ]
    out: Dict[str, Any] = {"record_type": record_type}
    for k in keys:
        out[k] = metrics.get(k)
    out["selected_cvi"] = adapter_result.get("selected_cvi")
    out["selected_algorithm"] = adapter_result.get("selected_algorithm")
    out["warmstart_count"] = adapter_result.get("warmstart_count")
    return out


def write_record_outputs(
    out_dir: Path,
    *,
    adapter_result: Dict[str, Any],
    metrics: Dict[str, Any],
    dataset_meta: Dict[str, Any],
    record_type: str,
    split_id: int,
    config: Dict[str, Any],
    run_log_text: str,
) -> Dict[str, Any]:
    """Persist all per-record output files. Returns the result.json payload."""
    out_dir = Path(out_dir)
    ensure_dir(out_dir)

    result_payload = build_result_json(
        adapter_result, metrics, dataset_meta=dataset_meta,
        record_type=record_type, split_id=split_id, config=config,
    )
    write_json_atomic(out_dir / "result.json", result_payload)
    write_json_atomic(out_dir / "summary_metrics.json",
                      build_summary_metrics(metrics, record_type, adapter_result))

    labels = adapter_result.get("labels")
    if labels is not None:
        try:
            write_npy_atomic(out_dir / "labels.npy", np.asarray(labels))
        except Exception:
            pass

    raw = adapter_result.get("raw_ml2dac_result")
    if raw is not None:
        try:
            write_json_atomic(out_dir / "raw_result.json", raw)
        except Exception:
            write_json_atomic(out_dir / "raw_result.json",
                              {"note": "raw ML2DAC result was not JSON-serialisable"})

    if adapter_result.get("artifact_status") is not None:
        try:
            write_json_atomic(out_dir / "ml2dac_artifact_report.json",
                              adapter_result["artifact_status"])
        except Exception:
            pass

    try:
        write_text_atomic(out_dir / "run_log.txt", run_log_text)
    except Exception:
        pass

    return result_payload


def build_dataset_summary(
    dataset_meta: Dict[str, Any],
    record_results: Dict[str, Dict[str, Any]],
    *,
    split_id: int,
    total_runtime_sec: float,
    config: Dict[str, Any],
    artifact_status: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Dataset-level summary: best record by ARI, statuses, totals, ML2DAC fields."""
    best_block = select_best_record_by_ari(record_results)

    statuses: Dict[str, str] = {}
    failures: Dict[str, str] = {}
    for rtype, payload in record_results.items():
        statuses[rtype] = payload.get("status", "unknown")
        if payload.get("status") != "success":
            msg = payload.get("error_message") or payload.get("error")
            if msg:
                failures[rtype] = str(msg).splitlines()[0]

    return {
        "method": "ML2DAC",
        "dataset_id": dataset_meta.get("dataset_id"),
        "split_id": int(split_id),
        "family": dataset_meta.get("family"),
        "subfamily": dataset_meta.get("subfamily"),
        "difficulty": dataset_meta.get("difficulty"),
        "true_k": dataset_meta.get("true_k"),
        "best_by_ari": best_block,
        "record_statuses": statuses,
        "failures": failures,
        "total_runtime_sec": float(total_runtime_sec),
        "n_records": len(record_results),
        "artifact_status": {
            "has_pretrained_artifacts": (artifact_status or {}).get("has_pretrained_artifacts"),
            "can_run_full_ml2dac": (artifact_status or {}).get("can_run_full_ml2dac"),
            "missing_required_artifacts": (artifact_status or {}).get("missing_required_artifacts"),
        },
        "finished_at": _now(),
    }


__all__ = [
    "record_output_dir",
    "record_is_complete",
    "evaluate_record",
    "build_result_json",
    "build_summary_metrics",
    "write_record_outputs",
    "build_dataset_summary",
]
