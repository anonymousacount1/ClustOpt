"""Convert a normalised adapter result into on-disk experiment outputs.

Keeps the adapter (which only knows AutoML4Clust) decoupled from the output
layout (which only knows the ClustOpt-style folder convention). All metric
computation happens here, against the true labels -- which are used for
*external evaluation only*, after AutoML4Clust has predicted.

Per-record output folder layout (mirrors the ClustOpt per-run convention)::

    <...>/AutoML4Clust/<record_type>/
        result.json           full normalised result + metrics + provenance
        labels.npy            predicted labels (npy) -- omitted on failure
        summary_metrics.json  the comparable metric subset
        raw_result.json       AutoML4Clust optimizer history (serialisable)
        run_log.txt           human-readable per-record log
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..utils.metrics import compute_all_metrics
from ..utils.path_utils import (
    ensure_dir,
    path_exists,
    read_json,
    write_json_atomic,
    write_npy_atomic,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def record_output_dir(automl_dir: Path, record_type: str) -> Path:
    return Path(automl_dir) / record_type


def record_is_complete(out_dir: Path) -> bool:
    """Resume key: a record is complete iff its result.json status == success."""
    payload = read_json(Path(out_dir) / "result.json")
    if not payload:
        return False
    return str(payload.get("status", "")).lower() == "success"


def evaluate_record(
    adapter_result: Dict[str, Any],
    y_true,
    true_k: Optional[int],
) -> Dict[str, Any]:
    """Compute external + K metrics for a (possibly failed) adapter result.

    On a failed adapter result, metrics are NaN/None but ``status`` is preserved.
    """
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
    """Assemble the full result.json payload (metrics + provenance, no raw history)."""
    payload: Dict[str, Any] = {
        "method": adapter_result.get("method", "AutoML4Clust"),
        "optimizer": adapter_result.get("optimizer"),
        "record_type": record_type,
        "split_id": int(split_id),
        "status": adapter_result.get("status", "failed"),
        "dataset_id": dataset_meta.get("dataset_id"),
        "family": dataset_meta.get("family"),
        "subfamily": dataset_meta.get("subfamily"),
        "difficulty": dataset_meta.get("difficulty"),
        "selected_k": adapter_result.get("selected_k"),
        "best_configuration": adapter_result.get("best_configuration"),
        "runtime_sec": float(adapter_result.get("runtime_sec", 0.0) or 0.0),
        "metrics": {k: v for k, v in metrics.items() if k not in ("status", "runtime_sec")},
        "details": adapter_result.get("details"),
        "config": config,
        "error": adapter_result.get("error"),
        "error_type": adapter_result.get("error_type"),
        "error_message": adapter_result.get("error_message"),
        "stage": adapter_result.get("stage"),
        "traceback": adapter_result.get("traceback"),
        "finished_at": _now(),
    }
    return payload


def build_summary_metrics(metrics: Dict[str, Any], record_type: str) -> Dict[str, Any]:
    """The comparable metric subset written to summary_metrics.json."""
    keys = [
        "ari", "nmi", "ami", "v_measure", "homogeneity", "completeness",
        "fowlkes_mallows", "purity",
        "selected_k", "true_k", "k_error", "abs_k_error", "exact_k_match",
        "n_predicted_clusters_excluding_noise", "n_noise_points",
        "predicted_noise_ratio", "true_noise_ratio",
        "runtime_sec", "status",
    ]
    out = {"record_type": record_type}
    for k in keys:
        out[k] = metrics.get(k)
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
    """Persist all per-record output files. Returns the result.json payload.

    Never raises on a serialisation hiccup with the raw history -- that single
    optional artifact degrades gracefully so a heavy/odd optimizer object can't
    sink an otherwise-successful record.
    """
    out_dir = Path(out_dir)
    ensure_dir(out_dir)

    result_payload = build_result_json(
        adapter_result, metrics, dataset_meta=dataset_meta,
        record_type=record_type, split_id=split_id, config=config,
    )
    write_json_atomic(out_dir / "result.json", result_payload)
    write_json_atomic(out_dir / "summary_metrics.json",
                      build_summary_metrics(metrics, record_type))

    # labels.npy only when we actually have labels.
    labels = adapter_result.get("labels")
    if labels is not None:
        try:
            write_npy_atomic(out_dir / "labels.npy", np.asarray(labels))
        except Exception:
            pass

    # raw optimizer history -- serialisable JSON (already coerced by adapter).
    raw = adapter_result.get("raw_automl4clust_result")
    if raw is not None:
        try:
            write_json_atomic(out_dir / "raw_result.json", raw)
        except Exception:
            # Degrade: record that the raw history could not be serialised.
            write_json_atomic(out_dir / "raw_result.json",
                              {"note": "raw AutoML4Clust result was not JSON-serialisable"})

    # human-readable per-record log
    try:
        from ..utils.path_utils import write_text_atomic
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
) -> Dict[str, Any]:
    """Dataset-level summary: best record by ARI, per-record statuses, totals.

    ``record_results`` maps record_type -> the result.json payload for it.
    """
    best_record_type: Optional[str] = None
    best_ari = float("-inf")
    statuses: Dict[str, str] = {}
    failures: Dict[str, str] = {}
    best_payload: Optional[Dict[str, Any]] = None

    for rtype, payload in record_results.items():
        statuses[rtype] = payload.get("status", "unknown")
        if payload.get("status") != "success":
            msg = payload.get("error_message") or payload.get("error")
            if msg:
                failures[rtype] = str(msg).splitlines()[0]
        metrics = payload.get("metrics", {}) or {}
        ari = metrics.get("ari")
        if payload.get("status") == "success" and ari is not None:
            try:
                ari_f = float(ari)
            except (TypeError, ValueError):
                ari_f = float("nan")
            if ari_f == ari_f and ari_f > best_ari:  # not NaN
                best_ari = ari_f
                best_record_type = rtype
                best_payload = payload

    best_block: Dict[str, Any] = {
        "best_record_type": best_record_type,
        "best_ari": None,
        "best_nmi": None,
        "best_ami": None,
        "selected_k": None,
        "exact_k_match": None,
    }
    if best_payload is not None:
        m = best_payload.get("metrics", {}) or {}
        best_block.update({
            "best_ari": m.get("ari"),
            "best_nmi": m.get("nmi"),
            "best_ami": m.get("ami"),
            "selected_k": best_payload.get("selected_k"),
            "exact_k_match": m.get("exact_k_match"),
        })

    return {
        "method": "AutoML4Clust",
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
        "optimizer": config.get("optimizer"),
        "metric": config.get("metric"),
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
