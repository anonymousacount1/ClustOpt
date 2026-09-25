"""Validation for the derived AutoClust repositories."""
from __future__ import annotations

from typing import List

import pandas as pd

N_CONFIGS = 32


def validate_variant(
    *,
    variant: str,
    heldout_split: int,
    excluded_splits: List[int],
    train_splits: List[int],
    train_records: pd.DataFrame,
    train_evaluations: pd.DataFrame,
    algorithm_labels: pd.DataFrame,
    reg_df: pd.DataFrame,
    cvi_candidates: pd.DataFrame,
    selector_meta: dict,
    predictor_meta: dict,
    autoclust_feature_cols: List[str],
    metafeatures: pd.DataFrame,
) -> dict:
    rep: dict = {"variant": variant}
    held = heldout_split

    def _has(df, col, val):
        return bool(col in df.columns and (df[col] == val).any())

    leak = {
        "heldout_split": held,
        "train_records_contain_heldout": _has(train_records, "split_id", held),
        "train_evaluations_contain_heldout": _has(train_evaluations, "split_id", held),
        "algorithm_labels_contain_heldout": _has(algorithm_labels, "split_id", held),
        "regression_rows_contain_heldout": _has(reg_df, "split_id", held),
        "selector_trained_on_heldout": bool(held in selector_meta.get("train_split_ids", [])),
        "train_contains_excluded_split16": bool(any(s in train_splits for s in excluded_splits)),
    }
    leak["no_leakage"] = not (
        leak["train_records_contain_heldout"] or leak["train_evaluations_contain_heldout"]
        or leak["algorithm_labels_contain_heldout"] or leak["regression_rows_contain_heldout"]
        or leak["selector_trained_on_heldout"] or leak["train_contains_excluded_split16"]
    )
    rep["leakage"] = leak

    n_records = int(train_records["record_id"].nunique())
    n_eval = int(len(train_evaluations))
    rep["counts"] = {
        "train_splits": sorted(train_splits),
        "heldout_split": held,
        "excluded_splits": sorted(excluded_splits),
        "n_train_records": n_records,
        "n_train_evaluations": n_eval,
        "n_algorithm_labels": int(len(algorithm_labels)),
        "n_regression_rows": int(len(reg_df)),
        "n_cvi_candidates": int(cvi_candidates["included"].sum()),
        "evaluations_eq_records_x32": n_eval == n_records * N_CONFIGS,
        "algorithm_labels_eq_records": int(len(algorithm_labels)) == n_records,
        "regression_rows_approx_evaluations": abs(len(reg_df) - n_eval) <= (n_eval * 0.001 + 100),
    }

    have_ac = all(c in metafeatures.columns for c in autoclust_feature_cols)
    cand_cols = list(cvi_candidates[cvi_candidates["included"]]["metric_id"])
    cand_present = all(c in train_evaluations.columns for c in cand_cols)
    rep["feature_coverage"] = {
        "autoclust_features_present": bool(have_ac),
        "n_autoclust_features": len(autoclust_feature_cols),
        "candidate_cvi_columns_present": bool(cand_present),
        "n_candidates_included": len(cand_cols),
        "n_candidates_excluded": int((~cvi_candidates["included"]).sum()),
    }

    rep["models"] = {
        "algorithm_selector_saved": bool(selector_meta.get("trained", False)),
        "selector_cv_accuracy": selector_meta.get("cv_accuracy"),
        "selector_cv_balanced_accuracy": selector_meta.get("cv_balanced_accuracy"),
        "ari_predictor_saved": bool(predictor_meta.get("trained", False)),
        "ari_validation": predictor_meta.get("validation_scores"),
    }

    rep["all_checks_pass"] = bool(
        leak["no_leakage"]
        and rep["counts"]["evaluations_eq_records_x32"]
        and rep["counts"]["algorithm_labels_eq_records"]
        and rep["counts"]["regression_rows_approx_evaluations"]
        and rep["feature_coverage"]["autoclust_features_present"]
        and rep["feature_coverage"]["candidate_cvi_columns_present"]
        and rep["models"]["algorithm_selector_saved"]
        and rep["models"]["ari_predictor_saved"]
    )
    return rep
