"""Per-dataset worker: build all view records, evaluations, and meta-features.

This is the unit of multiprocessing (one dataset folder = three views). It
never raises for per-config / per-metric problems — those are recorded as
failures or NaNs — but a dataset-level fatal error (e.g. unreadable data) is
captured on the returned :class:`DatasetResult`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .clustering import NOISE_LABEL, ClusterResult, _summarise, run_configuration
from .configuration import resolve_view_configurations
from .dataset_scanner import SplitInfo
from .logging_utils import configure_process
from .metafeatures import extract_autoclust, extract_ml2dac
from .metric_registry import build_registry, unified_metric_order
from .schemas import (
    EVALUATIONS_FIXED_COLUMNS,
    FAILURES_COLUMNS,
    METAFEATURES_FIXED_COLUMNS,
    RECORDS_COLUMNS,
    SCHEMA_VERSION,
)
from .storage import DatasetArtifacts, UnifiedMKRStorage
from .view_loader import load_dataset_views


@dataclass
class BuildConfig:
    out_dir: str
    config_root: str
    repo_root: str
    created_at: str
    family: str = ""
    subfamily: str = ""
    metafeatures_mode: str = "auto"  # "auto" | "on" | "off"
    overwrite: bool = False
    noise_label: int = NOISE_LABEL
    reuse_precomputed: bool = True  # reuse ClustOpt metrics from utility/<view>/clustopt_results.csv


@dataclass
class DatasetResult:
    dataset_id: str
    status: str
    n_records: int = 0
    n_evaluations: int = 0
    n_valid: int = 0
    n_failures: int = 0
    runtime_sec: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    metrics_reused: int = 0
    metrics_computed: int = 0
    skipped: bool = False
    error: str = ""
    ml2dac_mf_status: str = ""
    autoclust_mf_status: str = ""
    extras: Dict[str, object] = field(default_factory=dict)


def _external_metrics(y_true, labels) -> Dict[str, float]:
    from models.Clustering_Repository_Builder.experiments.experiment_execution.external_metrics import (
        compute_external_metrics,
    )

    raw = compute_external_metrics(y_true, labels, noise_label=NOISE_LABEL)
    return {
        "ari": _f(raw.get("ari")),
        "ami": _f(raw.get("ami")),
        "nmi": _f(raw.get("nmi")),
        "fmi": _f(raw.get("fowlkes_mallows")),
        "v_measure": _f(raw.get("v_measure")),
        "homogeneity": _f(raw.get("homogeneity")),
        "completeness": _f(raw.get("completeness")),
    }


def _f(v) -> float:
    try:
        v = float(v)
        return v if math.isfinite(v) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _cluster_result_from_labels(labels: np.ndarray) -> ClusterResult:
    n_found, pred_k, noise = _summarise(labels)
    return ClusterResult(
        labels=labels,
        valid_result=pred_k >= 1,
        failure_reason="" if pred_k >= 1 else "no_non_noise_cluster",
        runtime_sec=float("nan"),  # loaded from cache; original runtime unknown
        n_clusters_found=n_found,
        predicted_k=pred_k,
        noise_fraction=noise,
    )


def process_dataset(
    dataset_dir: str, split: SplitInfo, cfg: BuildConfig
) -> DatasetResult:
    configure_process()
    import time

    t0 = time.perf_counter()
    ds_id = Path(dataset_dir).name
    storage = UnifiedMKRStorage(cfg.out_dir)

    if not cfg.overwrite and storage.dataset_complete(ds_id):
        status = storage.read_dataset_status(ds_id) or {}
        return DatasetResult(
            dataset_id=ds_id,
            status="complete",
            n_records=int(status.get("n_records", 0)),
            n_evaluations=int(status.get("n_evaluations", 0)),
            n_valid=int(status.get("n_valid", 0)),
            n_failures=int(status.get("n_failures", 0)),
            runtime_sec=float(status.get("runtime_sec", 0.0)),
            skipped=True,
        )

    try:
        dv = load_dataset_views(dataset_dir)
    except Exception as exc:  # noqa: BLE001
        return DatasetResult(
            dataset_id=ds_id,
            status="fatal",
            error=f"load_dataset_views: {type(exc).__name__}: {exc}"[:300],
            runtime_sec=time.perf_counter() - t0,
        )

    metric_order = unified_metric_order(cfg.config_root)
    records_rows: List[dict] = []
    mf_rows: List[dict] = []
    eval_rows: List[dict] = []
    fail_rows: List[dict] = []
    cache_hits = cache_misses = 0
    metrics_reused = metrics_computed = 0
    ml2dac_status = autoclust_status = "skipped"

    split_id = split.split_id
    family = cfg.family or split.family
    subfamily = cfg.subfamily or split.subfamily

    for view_type, view in dv.views.items():
        rc = resolve_view_configurations(cfg.config_root, view_type)
        from models.ClustOpt.search_space.Clustering_Search_Space import (
            ClusteringSearchSpace,
        )

        search_space = ClusteringSearchSpace.from_dict(rc.search_space_config)

        records_rows.append(
            {
                "record_id": view.record_id,
                "dataset_id": ds_id,
                "dataset_folder": dv.dataset_folder,
                "family": family,
                "subfamily": subfamily,
                "split_id": split_id,
                "difficulty": split.difficulty,
                "view_type": view_type,
                "n_samples": view.n_samples,
                "n_features": view.n_features,
                "true_k": view.true_k,
                "data_path": dv.data_path,
                "true_labels_path": dv.true_labels_path,
                "metadata_path": dv.metadata_path,
                "created_at": cfg.created_at,
                "schema_version": SCHEMA_VERSION,
            }
        )

        from .metric_runner import ViewMetricEngine

        engine = ViewMetricEngine(
            cfg.config_root, view.X_decision, view.X_full, repo_root=cfg.repo_root
        )

        precomputed = None
        if cfg.reuse_precomputed:
            from .precomputed_cache import PrecomputedView

            precomputed = PrecomputedView.load(
                dv.dataset_folder, view_type, engine.clustopt_names
            )

        for spec in rc.configs:
            # Clustering with label-file reuse (resume / cache).
            reused = False
            if not cfg.overwrite and storage.labels_exist(view.record_id, spec.config_id):
                cached = storage.load_labels(view.record_id, spec.config_id)
                if cached is not None and cached.shape[0] == view.n_samples:
                    result = _cluster_result_from_labels(np.asarray(cached))
                    reused = True
                    cache_hits += 1
            if not reused:
                result = run_configuration(search_space, spec, view.X_decision)
                cache_misses += 1

            row = _base_eval_row(view, spec, ds_id, split_id, metric_order)
            if result.valid_result and result.labels is not None:
                from .io_utils import array_hash, downcast_labels

                stored_labels = downcast_labels(result.labels)
                labels_rel = storage.write_labels(
                    view.record_id, spec.config_id, stored_labels
                )
                ext = _external_metrics(view.y_true, result.labels)
                match = None
                if precomputed is not None and precomputed.available:
                    match = precomputed.lookup(spec.to_search_config())
                if match is not None and precomputed.guard_ok(
                    match, result.predicted_k, result.noise_fraction
                ):
                    metrics = engine.compute_reused(result.labels, match)
                    metrics_reused += 1
                else:
                    metrics = engine.compute(result.labels)
                    metrics_computed += 1
                row.update(
                    {
                        "valid_result": True,
                        "failure_reason": result.failure_reason,
                        "runtime_sec": result.runtime_sec,
                        "n_clusters_found": result.n_clusters_found,
                        "predicted_k": result.predicted_k,
                        "noise_fraction": result.noise_fraction,
                        "labels_path": labels_rel,
                        "labels_hash": array_hash(stored_labels),
                        **ext,
                    }
                )
                for m in metric_order:
                    row[m] = metrics.get(m, float("nan"))
            else:
                row.update(
                    {
                        "valid_result": False,
                        "failure_reason": result.failure_reason,
                        "runtime_sec": result.runtime_sec,
                        "n_clusters_found": result.n_clusters_found,
                        "predicted_k": result.predicted_k,
                        "noise_fraction": result.noise_fraction,
                    }
                )
                fail_rows.append(
                    {
                        "record_id": view.record_id,
                        "config_id": spec.config_id,
                        "dataset_id": ds_id,
                        "view_type": view_type,
                        "split_id": split_id,
                        "stage": "clustering",
                        "algorithm": spec.algorithm,
                        "reason": result.failure_reason,
                    }
                )
            eval_rows.append(row)

        # Meta-features (once per view record, on the decision-space X).
        mf_row = {
            "record_id": view.record_id,
            "dataset_id": ds_id,
            "view_type": view_type,
            "split_id": split_id,
        }
        if cfg.metafeatures_mode == "off":
            mf_row.update(_mf_skipped())
        else:
            ml2dac = extract_ml2dac(view.X_decision, repo_root=cfg.repo_root)
            autoclust = extract_autoclust(view.X_decision, repo_root=cfg.repo_root)
            ml2dac_status = ml2dac.status
            autoclust_status = autoclust.status
            mf_row.update(
                {
                    "ml2dac_status": ml2dac.status,
                    "ml2dac_runtime_sec": ml2dac.runtime_sec,
                    "ml2dac_error": ml2dac.error,
                    "autoclust_status": autoclust.status,
                    "autoclust_runtime_sec": autoclust.runtime_sec,
                    "autoclust_error": autoclust.error,
                }
            )
            mf_row.update(ml2dac.features)
            mf_row.update(autoclust.features)
            # Only a genuine problem is a failure. ML2DAC "unavailable" is the
            # expected inline state (pymfe lives in the .mfenv pass, which fills
            # it in afterwards), so it is NOT logged as a failure. We flag only
            # a real pymfe error ("failed") or any non-success AutoClust status.
            ml2dac_problem = ml2dac.status == "failed"
            autoclust_problem = autoclust.status not in ("success", "skipped")
            if ml2dac_problem or autoclust_problem:
                fail_rows.append(
                    {
                        "record_id": view.record_id,
                        "config_id": "",
                        "dataset_id": ds_id,
                        "view_type": view_type,
                        "split_id": split_id,
                        "stage": "metafeatures",
                        "algorithm": "",
                        "reason": f"ml2dac={ml2dac.status}; autoclust={autoclust.status}",
                    }
                )
        mf_rows.append(mf_row)

    runtime = time.perf_counter() - t0
    n_valid = sum(1 for r in eval_rows if r["valid_result"])

    records_df = pd.DataFrame(records_rows, columns=RECORDS_COLUMNS)
    eval_cols = EVALUATIONS_FIXED_COLUMNS + metric_order
    evaluations_df = pd.DataFrame(eval_rows).reindex(columns=eval_cols)
    metafeatures_df = pd.DataFrame(mf_rows)
    # Stabilise meta-feature fixed columns first.
    mf_cols = METAFEATURES_FIXED_COLUMNS + [
        c for c in metafeatures_df.columns if c not in METAFEATURES_FIXED_COLUMNS
    ]
    metafeatures_df = metafeatures_df.reindex(columns=mf_cols)
    failures_df = pd.DataFrame(fail_rows, columns=FAILURES_COLUMNS)

    status = {
        "status": "complete",
        "dataset_id": ds_id,
        "n_records": len(records_rows),
        "n_evaluations": len(eval_rows),
        "n_valid": n_valid,
        "n_failures": len(fail_rows),
        "runtime_sec": runtime,
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "metrics_reused": metrics_reused,
        "metrics_computed": metrics_computed,
        "ml2dac_mf_status": ml2dac_status,
        "autoclust_mf_status": autoclust_status,
        "schema_version": SCHEMA_VERSION,
    }
    storage.stage_dataset(
        DatasetArtifacts(
            dataset_id=ds_id,
            records=records_df,
            metafeatures=metafeatures_df,
            evaluations=evaluations_df,
            failures=failures_df,
            status=status,
        )
    )

    return DatasetResult(
        dataset_id=ds_id,
        status="complete",
        n_records=len(records_rows),
        n_evaluations=len(eval_rows),
        n_valid=n_valid,
        n_failures=len(fail_rows),
        runtime_sec=runtime,
        cache_hits=cache_hits,
        cache_misses=cache_misses,
        metrics_reused=metrics_reused,
        metrics_computed=metrics_computed,
        ml2dac_mf_status=ml2dac_status,
        autoclust_mf_status=autoclust_status,
    )


def _base_eval_row(view, spec, ds_id, split_id, metric_order) -> dict:
    row = {
        "record_id": view.record_id,
        "config_id": spec.config_id,
        "dataset_id": ds_id,
        "view_type": view.view_type,
        "split_id": split_id,
        "algorithm": spec.algorithm,
        "hyperparameters_json": spec.hyperparameters_json(),
        "valid_result": False,
        "failure_reason": "",
        "runtime_sec": float("nan"),
        "n_clusters_found": 0,
        "predicted_k": 0,
        "noise_fraction": float("nan"),
        "labels_path": "",
        "labels_hash": "",
        "ari": float("nan"),
        "ami": float("nan"),
        "nmi": float("nan"),
        "fmi": float("nan"),
        "v_measure": float("nan"),
        "homogeneity": float("nan"),
        "completeness": float("nan"),
    }
    for m in metric_order:
        row[m] = float("nan")
    return row


def _mf_skipped() -> dict:
    return {
        "ml2dac_status": "skipped",
        "ml2dac_runtime_sec": 0.0,
        "ml2dac_error": "",
        "autoclust_status": "skipped",
        "autoclust_runtime_sec": 0.0,
        "autoclust_error": "",
    }


def build_configurations_table(config_root: str) -> pd.DataFrame:
    """The resolved 32-configuration table for all three views."""
    from .schemas import CONFIGURATIONS_COLUMNS, VIEW_TYPES

    rows: List[dict] = []
    for view_type in VIEW_TYPES:
        rc = resolve_view_configurations(config_root, view_type)
        for spec in rc.configs:
            rows.append(
                {
                    "config_id": spec.config_id,
                    "original_config_id": spec.original_config_id,
                    "view_type": view_type,
                    "algorithm": spec.algorithm,
                    "hyperparameters_json": spec.hyperparameters_json(),
                    "was_replaced": spec.was_replaced,
                    "replacement_reason": spec.replacement_reason,
                    "source_json_path": rc.source_json_path,
                    "schema_version": SCHEMA_VERSION,
                }
            )
    return pd.DataFrame(rows, columns=CONFIGURATIONS_COLUMNS)


def build_metric_registry_table(config_root: str) -> pd.DataFrame:
    return build_registry(config_root)
