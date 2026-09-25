"""Derived AutoClust repository variant builder.

Builds one variant (original_cvis / extended_cvis). The algorithm selector and
its labels are CVI-independent (they use only the AutoClust landmarking
meta-features + best-ARI algorithm), so they are computed once by the runner and
written identically into both variants. The ARI-predictor MLP differs per
variant because its input is the CVI candidate vector.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional

import joblib
import pandas as pd

from . import io_utils
from . import autoclust_models as acm
from . import autoclust_training_data as actd
from . import autoclust_validation as acval


def _log(log, msg):
    if log:
        log(msg)


def _candidates_frame(candidate_ids, registry, excluded) -> pd.DataFrame:
    rdir = dict(zip(registry["metric_id"], registry["direction"]))
    rsrc = dict(zip(registry["metric_id"], registry["implementation_source"]))
    rows = []
    for mid in candidate_ids:
        rows.append({"metric_id": mid, "direction": rdir.get(mid),
                     "implementation_source": rsrc.get(mid, "unknown"),
                     "included": True, "exclusion_reason": ""})
    for mid, reason in excluded.items():
        rows.append({"metric_id": mid, "direction": rdir.get(mid),
                     "implementation_source": rsrc.get(mid, "unknown"),
                     "included": False, "exclusion_reason": reason})
    return pd.DataFrame(rows)


def _attach_variant_spec(out_dir, validation: dict, variant_spec) -> dict:
    """Persist the frozen Stage-3A1 spec beside the variant and pin it.

    Additive: variants built through the historical default path pass
    ``variant_spec=None`` and are byte-identical to before.
    """
    if not variant_spec:
        validation["variant_spec"] = None
        return validation
    io_utils.write_json_atomic(out_dir / "variant_spec.json", variant_spec)
    validation["variant_spec"] = {
        "variant_name": variant_spec.get("variant_name"),
        "semantic_variant": variant_spec.get("semantic_variant"),
        "manifest_hash": variant_spec.get("manifest_hash"),
        "requested_metric_count": variant_spec.get("requested_metric_count"),
        "expected_live_metric_count": variant_spec.get("expected_live_metric_count"),
        "expected_dead_metric_ids": variant_spec.get("expected_dead_metric_ids"),
    }
    return validation


def build_variant(
    *,
    variant: str,
    out_dir: Path,
    train_records: pd.DataFrame,
    train_evaluations: pd.DataFrame,
    train_metafeatures: pd.DataFrame,
    metric_registry: pd.DataFrame,
    candidate_ids: List[str],
    excluded: Dict[str, str],
    autoclust_feature_cols: List[str],
    algorithm_labels: pd.DataFrame,
    selector_pipe,
    selector_meta: dict,
    split_policy: dict,
    seed: int,
    variant_spec: Optional[dict] = None,
    log: Optional[Callable[[str], None]] = None,
) -> dict:
    out_dir = Path(out_dir)
    io_utils.ensure_dir(out_dir)

    candidates_df = _candidates_frame(candidate_ids, metric_registry, excluded)
    excluded_df = candidates_df[~candidates_df["included"]][["metric_id", "exclusion_reason"]].copy()

    _log(log, f"[{variant}] assembling ARI regression dataset ({len(candidate_ids)} CVIs)")
    reg_df, reg_info = actd.build_ari_regression_dataset(
        train_evaluations, train_records, candidate_ids)

    _log(log, f"[{variant}] training ARI-predictor MLP on {len(reg_df):,} rows")
    predictor, pred_meta = acm.train_ari_predictor(reg_df, candidate_ids, seed)
    pred_meta["regression_dataset_info"] = reg_info

    validation = acval.validate_variant(
        variant=variant,
        heldout_split=split_policy["heldout_test_split"],
        excluded_splits=split_policy.get("excluded_splits", []),
        train_splits=split_policy["train_splits"],
        train_records=train_records, train_evaluations=train_evaluations,
        algorithm_labels=algorithm_labels, reg_df=reg_df,
        cvi_candidates=candidates_df, selector_meta=selector_meta,
        predictor_meta=pred_meta, autoclust_feature_cols=autoclust_feature_cols,
        metafeatures=train_metafeatures)

    # --- persist ---
    io_utils.write_parquet_atomic(out_dir / "train_records.parquet", train_records)
    io_utils.write_parquet_atomic(out_dir / "train_metafeatures.parquet", train_metafeatures)
    fixed = [c for c in [
        "record_id", "config_id", "dataset_id", "view_type", "split_id",
        "algorithm", "hyperparameters_json", "valid_result", "ari",
        "predicted_k", "n_clusters_found",
    ] if c in train_evaluations.columns]
    keep = fixed + [m for m in candidate_ids if m in train_evaluations.columns]
    io_utils.write_parquet_atomic(out_dir / "train_evaluations.parquet", train_evaluations[keep])
    io_utils.write_parquet_atomic(out_dir / "cvi_candidates.parquet", candidates_df)
    io_utils.write_parquet_atomic(out_dir / "excluded_metrics.parquet", excluded_df)
    io_utils.write_parquet_atomic(out_dir / "algorithm_labels.parquet", algorithm_labels)
    io_utils.write_parquet_atomic(out_dir / "ari_regression_dataset.parquet", reg_df)

    with open(io_utils.ext(out_dir / "algorithm_selector.pkl"), "wb") as f:
        joblib.dump(selector_pipe, f)
    with open(io_utils.ext(out_dir / "ari_predictor.pkl"), "wb") as f:
        joblib.dump(predictor, f)
    io_utils.write_json_atomic(out_dir / "algorithm_selector_metadata.json", selector_meta)
    io_utils.write_json_atomic(out_dir / "ari_predictor_metadata.json", pred_meta)
    io_utils.write_json_atomic(out_dir / "feature_columns.json", {
        "autoclust_meta_feature_columns": autoclust_feature_cols,
        "ari_predictor_cvi_columns": candidate_ids,
    })
    validation = _attach_variant_spec(out_dir, validation, variant_spec)
    io_utils.write_json_atomic(out_dir / "validation_report.json", validation)

    _write_build_report(out_dir, variant, candidates_df, algorithm_labels,
                        selector_meta, pred_meta, validation, excluded)
    return {
        "variant": variant, "candidates": candidates_df,
        "selector_meta": selector_meta, "predictor_meta": pred_meta,
        "algorithm_labels": algorithm_labels, "validation": validation,
        "n_regression_rows": int(len(reg_df)),
    }


def _write_build_report(out_dir, variant, candidates_df, algo_labels, sel_meta,
                        pred_meta, validation, excluded):
    vs = pred_meta["validation_scores"]
    dist = {str(k): int(v) for k, v in algo_labels["best_algorithm"].value_counts().items()}
    lines = [f"# AutoClust Derived Repository — {variant}", "",
             "## CVI candidates", "",
             f"- Included: {int(candidates_df['included'].sum())}",
             f"- Excluded: {int((~candidates_df['included']).sum())} {dict(excluded) if excluded else ''}", "",
             "## Algorithm selector (shared, CVI-independent)", "",
             f"- Model: {sel_meta['model_type']}",
             f"- Classes: {sel_meta['n_classes']}",
             f"- CV accuracy: {sel_meta['cv_accuracy']}",
             f"- CV balanced accuracy: {sel_meta['cv_balanced_accuracy']}",
             f"- best_algorithm distribution: {dist}", "",
             "## ARI predictor (MLP 60-30-10)", "",
             f"- Train rows: {pred_meta['train_row_count']}",
             f"- Validation (grouped 20% holdout): MAE={vs['mae']:.4f} RMSE={vs['rmse']:.4f} "
             f"R2={vs['r2']:.4f} Spearman={vs['spearman_pred_vs_true']:.4f}", "",
             "## Validation", "",
             f"- Leakage-free: {validation['leakage']['no_leakage']}",
             f"- All checks pass: {validation['all_checks_pass']}"]
    io_utils.write_text_atomic(out_dir / "build_report.md", "\n".join(lines))
