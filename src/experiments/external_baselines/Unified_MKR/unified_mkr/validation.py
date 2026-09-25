"""Post-build validation: integrity, coverage, and cache reports.

Reads the aggregated subfamily tables and emits three JSON reports under
``validation/`` (see UNIFIED_MKR_DESIGN.md §25).
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from . import io_utils
from .schemas import N_CONFIGS_PER_VIEW
from .storage import UnifiedMKRStorage


def _metric_columns(evaluations: pd.DataFrame, metric_registry: pd.DataFrame) -> List[str]:
    reg_metrics = list(metric_registry["metric_id"])
    return [m for m in reg_metrics if m in evaluations.columns]


def run_validation(
    storage: UnifiedMKRStorage,
    *,
    configurations: pd.DataFrame,
    metric_registry: pd.DataFrame,
    dataset_results: List[dict],
) -> Dict[str, dict]:
    evaluations = io_utils.read_parquet(storage.out_dir / "evaluations.parquet")
    records = io_utils.read_parquet(storage.out_dir / "records.parquet")
    failures = io_utils.read_parquet(storage.out_dir / "failures.parquet")
    if evaluations is None:
        evaluations = pd.DataFrame()
    if records is None:
        records = pd.DataFrame()
    if failures is None:
        failures = pd.DataFrame()

    integrity = _integrity_report(evaluations, configurations, metric_registry)
    coverage = _coverage_report(records, evaluations)
    cache = _cache_report(dataset_results, failures)

    io_utils.ensure_dir(storage.validation_dir)
    io_utils.write_json_atomic(storage.validation_dir / "integrity_report.json", integrity)
    io_utils.write_json_atomic(storage.validation_dir / "coverage_report.json", coverage)
    io_utils.write_json_atomic(storage.validation_dir / "cache_report.json", cache)
    return {"integrity": integrity, "coverage": coverage, "cache": cache}


def _integrity_report(evaluations, configurations, metric_registry) -> dict:
    checks: Dict[str, object] = {}

    # Metric registry: no duplicate metric_id; one implementation per metric.
    metric_ids = list(metric_registry["metric_id"])
    dup_ids = sorted({m for m in metric_ids if metric_ids.count(m) > 1})
    checks["metric_registry_duplicate_ids"] = dup_ids
    checks["metric_registry_no_duplicates"] = not dup_ids
    impl_per_metric = (
        metric_registry.groupby("metric_id")["implementation_source"].nunique()
    )
    checks["metrics_with_multiple_implementations"] = sorted(
        impl_per_metric[impl_per_metric > 1].index.tolist()
    )
    checks["overlapping_metrics_computed_once"] = bool(
        (impl_per_metric <= 1).all()
    )

    # Configuration table: exactly 32 per view.
    per_view = configurations.groupby("view_type")["config_id"].count().to_dict()
    checks["configs_per_view"] = {k: int(v) for k, v in per_view.items()}
    checks["configs_per_view_ok"] = all(
        int(v) == N_CONFIGS_PER_VIEW for v in per_view.values()
    )
    checks["no_constrained_in_resolved"] = bool(
        not configurations["algorithm"].astype(str).str.startswith("hdbscan_constrained").any()
    )

    # Label storage: every valid evaluation has a labels_path.
    if not evaluations.empty:
        valid = evaluations[evaluations["valid_result"] == True]  # noqa: E712
        missing_labels = int((valid["labels_path"].fillna("") == "").sum())
        checks["valid_evaluations"] = int(len(valid))
        checks["valid_evaluations_missing_labels"] = missing_labels
        checks["all_valid_have_labels"] = missing_labels == 0

        # Metric coverage: valid rows whose ALL metric columns are NaN.
        metric_cols = _metric_columns(evaluations, metric_registry)
        if metric_cols and len(valid):
            all_nan = valid[metric_cols].isna().all(axis=1).sum()
            checks["valid_rows_all_metrics_nan"] = int(all_nan)
        else:
            checks["valid_rows_all_metrics_nan"] = 0
    else:
        checks["valid_evaluations"] = 0
        checks["valid_evaluations_missing_labels"] = 0
        checks["all_valid_have_labels"] = True
        checks["valid_rows_all_metrics_nan"] = 0

    return checks


def _coverage_report(records, evaluations) -> dict:
    report: Dict[str, object] = {}
    report["n_records"] = int(len(records))
    report["n_evaluations"] = int(len(evaluations))

    if not records.empty:
        per_dataset_views = records.groupby("dataset_id")["view_type"].nunique()
        report["n_datasets"] = int(per_dataset_views.shape[0])
        report["datasets_with_3_views"] = int((per_dataset_views == 3).sum())
        report["datasets_with_fewer_views"] = sorted(
            per_dataset_views[per_dataset_views < 3].index.tolist()
        )[:50]
    else:
        report["n_datasets"] = 0
        report["datasets_with_3_views"] = 0
        report["datasets_with_fewer_views"] = []

    if not evaluations.empty:
        per_record = evaluations.groupby("record_id")["config_id"].count()
        bad = per_record[per_record != N_CONFIGS_PER_VIEW]
        report["records_with_32_configs"] = int((per_record == N_CONFIGS_PER_VIEW).sum())
        report["records_with_wrong_config_count"] = {
            str(k): int(v) for k, v in bad.head(50).to_dict().items()
        }
        report["all_records_have_32_configs"] = bool(bad.empty)
        report["split_ids_present"] = sorted(
            int(s) for s in evaluations["split_id"].dropna().unique()
        )
    else:
        report["records_with_32_configs"] = 0
        report["records_with_wrong_config_count"] = {}
        report["all_records_have_32_configs"] = False
        report["split_ids_present"] = []

    return report


def _cache_report(dataset_results, failures) -> dict:
    hits = sum(int(r.get("cache_hits", 0)) for r in dataset_results)
    misses = sum(int(r.get("cache_misses", 0)) for r in dataset_results)
    total = hits + misses
    report: Dict[str, object] = {
        "cache_hits": hits,
        "cache_misses": misses,
        "cache_hit_rate": (hits / total) if total else 0.0,
        "datasets_processed": len(dataset_results),
        "datasets_skipped_resume": sum(1 for r in dataset_results if r.get("skipped")),
    }

    if failures is not None and not failures.empty:
        report["number_of_failures"] = int(len(failures))
        report["failures_by_stage"] = _counts(failures, "stage")
        report["failures_by_algorithm"] = _counts(failures, "algorithm")
        report["failures_by_view_type"] = _counts(failures, "view_type")
        report["failures_by_reason"] = _counts(failures, "reason", top=20)
    else:
        report["number_of_failures"] = 0
        report["failures_by_stage"] = {}
        report["failures_by_algorithm"] = {}
        report["failures_by_view_type"] = {}
        report["failures_by_reason"] = {}
    return report


def _counts(df: pd.DataFrame, col: str, top: int = 50) -> Dict[str, int]:
    if col not in df.columns:
        return {}
    vc = df[col].fillna("").astype(str).value_counts().head(top)
    return {str(k): int(v) for k, v in vc.to_dict().items()}
