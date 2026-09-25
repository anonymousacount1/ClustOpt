"""Derived ML2DAC MKR variant builder.

Builds one variant (original_cvis or extended_cvis) from train-only master data:
CVI* labels, CVI rank correlations, the RandomForest CVI classifier, the
KDTree nearest-neighbor index, and the warmstart repository — plus a per-variant
validation report. Mirrors ML2DAC's LearningPhase model choices:

* classifier: ``RandomForestClassifier`` with NaN->0 imputation, no scaling
  (LearningPhase.train_model_not_for_dataset uses ``np.nan_to_num(X, 0)`` and a
  default RandomForest);
* NN index: ``sklearn.neighbors.KDTree`` over NaN->0 meta-features
  (MetaFeatureExtractor builds ``KDTree(np.nan_to_num(mfs))``).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neighbors import KDTree

from . import io_utils
from . import ml2dac_cvi_selection as cvisel
from . import ml2dac_warmstarts as warm
from . import ml2dac_validation as mlval

_MF_NON_FEATURE = {
    "record_id", "dataset_id", "view_type", "split_id",
    "ml2dac_status", "ml2dac_runtime_sec", "ml2dac_error",
    "autoclust_status", "autoclust_runtime_sec", "autoclust_error",
}


def _log(log, msg):
    if log:
        log(msg)


def ml2dac_feature_columns(metafeatures: pd.DataFrame) -> List[str]:
    return [
        c for c in metafeatures.columns
        if c.startswith("ml2dac_") and c not in _MF_NON_FEATURE
    ]


def _candidates_frame(
    candidate_ids: List[str],
    directions: Dict[str, str],
    metric_registry: pd.DataFrame,
    excluded: Dict[str, str],
) -> pd.DataFrame:
    src = dict(zip(metric_registry["metric_id"], metric_registry["implementation_source"]))
    rows = []
    for mid in candidate_ids:
        rows.append({
            "metric_id": mid, "direction": directions.get(mid),
            "implementation_source": src.get(mid, "unknown"),
            "included": True, "exclusion_reason": "",
        })
    for mid, reason in excluded.items():
        rows.append({
            "metric_id": mid, "direction": directions.get(mid),
            "implementation_source": src.get(mid, "unknown"),
            "included": False, "exclusion_reason": reason,
        })
    return pd.DataFrame(rows)


def _train_classifier(X: np.ndarray, y: np.ndarray, seed: int) -> tuple:
    clf = RandomForestClassifier(random_state=seed)
    clf.fit(X, y)
    # internal CV (train-only) for a sanity score
    classes, counts = np.unique(y, return_counts=True)
    min_class = int(counts.min())
    cv_acc = cv_bal = None
    cv_note = ""
    if len(classes) >= 2 and min_class >= 2:
        n_splits = int(min(5, min_class))
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        scores = cross_validate(
            RandomForestClassifier(random_state=seed), X, y, cv=skf,
            scoring=["accuracy", "balanced_accuracy"], n_jobs=-1,
        )
        cv_acc = float(np.mean(scores["test_accuracy"]))
        cv_bal = float(np.mean(scores["test_balanced_accuracy"]))
        cv_note = f"StratifiedKFold(n_splits={n_splits})"
    else:
        cv_note = "skipped (insufficient per-class samples)"
    return clf, cv_acc, cv_bal, cv_note, classes, counts


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
    split_policy: dict,
    seed: int,
    variant_spec: Optional[dict] = None,
    log: Optional[Callable[[str], None]] = None,
) -> dict:
    out_dir = Path(out_dir)
    io_utils.ensure_dir(out_dir)
    train_split_ids = sorted(int(s) for s in train_records["split_id"].unique())

    _log(log, f"[{variant}] resolving directions for {len(candidate_ids)} candidates")
    # resolve for included + excluded so the candidates artifact is fully populated
    directions = cvisel.resolve_directions(
        candidate_ids + list(excluded.keys()), metric_registry, log=log)
    candidates_df = _candidates_frame(candidate_ids, directions, metric_registry, excluded)

    _log(log, f"[{variant}] computing CVI* over {train_records['record_id'].nunique()} records")
    cvi_star, correlations = cvisel.compute_cvi_star(
        train_evaluations, candidate_ids, directions, train_records, log=log,
    )

    # --- features + classifier (resolved records only) ---
    feat_cols = ml2dac_feature_columns(train_metafeatures)
    mf = train_metafeatures.set_index("record_id")
    resolved = cvi_star[cvi_star["resolved"]].copy()
    Xdf = mf.reindex(resolved["record_id"])[feat_cols]
    imputer = SimpleImputer(strategy="constant", fill_value=0.0)
    X = imputer.fit_transform(Xdf.to_numpy(dtype=float))
    y = resolved["best_cvi"].to_numpy()

    _log(log, f"[{variant}] training RandomForest CVI classifier on {len(y)} records")
    clf, cv_acc, cv_bal, cv_note, classes, counts = _train_classifier(X, y, seed)
    classifier_meta = {
        "trained": True,
        "classifier_type": "RandomForestClassifier",
        "hyperparameters": {k: clf.get_params()[k] for k in ("n_estimators", "criterion", "max_depth", "random_state")},
        "feature_columns": feat_cols,
        "n_features": len(feat_cols),
        "n_classes": int(len(classes)),
        "target_distribution": {str(c): int(n) for c, n in zip(classes, counts)},
        "train_record_count": int(len(y)),
        "random_seed": seed,
        "imputer": {"strategy": "constant", "fill_value": 0.0,
                    "note": "matches ML2DAC np.nan_to_num(X, 0)"},
        "scaler": None,
        "scaler_note": "ML2DAC uses no scaler for the RF CVI classifier",
        "cv_accuracy": cv_acc,
        "cv_balanced_accuracy": cv_bal,
        "cv_note": cv_note,
        "train_split_ids": train_split_ids,
    }

    # --- NN index over ALL train records (ML2DAC builds kdtree over all) ---
    _log(log, f"[{variant}] building KDTree NN index")
    nn_ids = list(train_metafeatures["record_id"])
    Xnn_df = mf.reindex(nn_ids)[feat_cols]
    Xnn = imputer.transform(Xnn_df.to_numpy(dtype=float))
    kdtree = KDTree(Xnn)  # default metric=euclidean, leaf_size=40 (mirrors ML2DAC)
    assert split_policy["heldout_test_split"] not in train_split_ids
    nn_meta = {
        "built": True,
        "index_type": "sklearn.neighbors.KDTree",
        "distance_metric": "euclidean (KDTree default)",
        "leaf_size": 40,
        "feature_columns": feat_cols,
        "n_train_records": int(len(nn_ids)),
        "train_record_ids": nn_ids,
        "train_split_ids": train_split_ids,
        "imputer": "constant(0.0)",
        "scaler": None,
    }

    # --- warmstart repository (all train records, all 32 configs ranked) ---
    _log(log, f"[{variant}] building warmstart repository")
    warmstarts = warm.build_warmstart_repository(train_evaluations)

    # --- persist artifacts ---
    io_utils.write_parquet_atomic(out_dir / "train_records.parquet", train_records)
    io_utils.write_parquet_atomic(out_dir / "train_metafeatures.parquet", train_metafeatures)
    # train_evaluations reduced to fixed cols + this variant's candidate metrics
    fixed = [c for c in [
        "record_id", "config_id", "dataset_id", "view_type", "split_id",
        "algorithm", "hyperparameters_json", "valid_result", "ari",
        "predicted_k", "n_clusters_found",
    ] if c in train_evaluations.columns]
    keep = fixed + [m for m in candidate_ids if m in train_evaluations.columns]
    io_utils.write_parquet_atomic(out_dir / "train_evaluations.parquet", train_evaluations[keep])
    io_utils.write_parquet_atomic(out_dir / "cvi_candidates.parquet", candidates_df)
    io_utils.write_parquet_atomic(out_dir / "cvi_star.parquet", cvi_star)
    io_utils.write_parquet_atomic(out_dir / "cvi_rank_correlations.parquet", correlations)
    io_utils.write_parquet_atomic(out_dir / "warmstart_repository.parquet", warmstarts)
    io_utils.write_parquet_atomic(out_dir / "config_rankings.parquet", warmstarts)

    with open(io_utils.ext(out_dir / "cvi_classifier.pkl"), "wb") as f:
        joblib.dump(clf, f)
    with open(io_utils.ext(out_dir / "imputer.pkl"), "wb") as f:
        joblib.dump(imputer, f)
    with open(io_utils.ext(out_dir / "nearest_neighbor_index.pkl"), "wb") as f:
        joblib.dump({"kdtree": kdtree, "train_record_ids": nn_ids, "feature_columns": feat_cols}, f)
    io_utils.write_json_atomic(out_dir / "cvi_classifier_metadata.json", classifier_meta)
    io_utils.write_json_atomic(out_dir / "nearest_neighbor_metadata.json", nn_meta)
    io_utils.write_json_atomic(out_dir / "feature_columns.json", {"feature_columns": feat_cols})

    # --- validation ---
    validation = mlval.validate_variant(
        variant=variant,
        heldout_split=split_policy["heldout_test_split"],
        train_splits=split_policy["train_splits"],
        excluded_splits=split_policy.get("excluded_splits", []),
        train_records=train_records,
        train_evaluations=train_evaluations,
        cvi_candidates=candidates_df,
        cvi_star=cvi_star,
        warmstarts=warmstarts,
        classifier_meta=classifier_meta,
        nn_meta=nn_meta,
    )
    validation = _attach_variant_spec(out_dir, validation, variant_spec)
    io_utils.write_json_atomic(out_dir / "validation_report.json", validation)

    # --- per-variant build report ---
    _write_build_report(out_dir, variant, candidates_df, cvi_star, classifier_meta,
                        nn_meta, validation, excluded)

    return {
        "variant": variant,
        "candidates": candidates_df,
        "cvi_star": cvi_star,
        "classifier_meta": classifier_meta,
        "nn_meta": nn_meta,
        "validation": validation,
        "n_warmstart_rows": int(len(warmstarts)),
    }


def _cvi_star_distribution(cvi_star: pd.DataFrame) -> Dict[str, int]:
    res = cvi_star[cvi_star["resolved"]]
    return {str(k): int(v) for k, v in res["best_cvi"].value_counts().items()}


def _write_build_report(out_dir, variant, candidates_df, cvi_star, clf_meta, nn_meta, validation, excluded):
    dist = _cvi_star_distribution(cvi_star)
    lines = [f"# ML2DAC Derived MKR — {variant}", "",
             "## CVI candidates", "",
             f"- Included: {int(candidates_df['included'].sum())}",
             f"- Excluded: {int((~candidates_df['included']).sum())} {dict(excluded) if excluded else ''}", "",
             "## CVI* distribution (resolved train records)", ""]
    lines += [f"- `{k}`: {v}" for k, v in sorted(dist.items(), key=lambda kv: -kv[1])]
    lines += ["", "## Classifier", "",
              f"- Type: {clf_meta['classifier_type']}",
              f"- Classes: {clf_meta['n_classes']}",
              f"- Train records: {clf_meta['train_record_count']}",
              f"- CV accuracy: {clf_meta['cv_accuracy']}",
              f"- CV balanced accuracy: {clf_meta['cv_balanced_accuracy']} ({clf_meta['cv_note']})", "",
              "## NN index", "",
              f"- Type: {nn_meta['index_type']} ({nn_meta['distance_metric']})",
              f"- Train records: {nn_meta['n_train_records']}", "",
              "## Validation", "",
              f"- Leakage-free: {validation['leakage']['no_leakage']}",
              f"- All checks pass: {validation['all_checks_pass']}"]
    io_utils.write_text_atomic(out_dir / "build_report.md", "\n".join(lines))
