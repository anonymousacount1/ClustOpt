"""Validation for the derived ML2DAC MKRs (leakage, counts, coverage, models)."""
from __future__ import annotations

from typing import Dict, List

import pandas as pd

N_CONFIGS = 32


def validate_variant(
    *,
    variant: str,
    heldout_split: int,
    train_splits: List[int],
    excluded_splits: List[int],
    train_records: pd.DataFrame,
    train_evaluations: pd.DataFrame,
    cvi_candidates: pd.DataFrame,
    cvi_star: pd.DataFrame,
    warmstarts: pd.DataFrame,
    classifier_meta: dict,
    nn_meta: dict,
) -> dict:
    rep: Dict[str, object] = {"variant": variant}

    # --- leakage: split 1 (heldout) must appear nowhere ---
    held = heldout_split
    leak = {
        "heldout_split": held,
        "train_records_contain_heldout": bool((train_records["split_id"] == held).any()),
        "train_evaluations_contain_heldout": bool((train_evaluations["split_id"] == held).any()),
        "cvi_star_contains_heldout": bool((cvi_star["split_id"] == held).any()),
        "warmstarts_contain_heldout": bool((warmstarts["split_id"] == held).any()),
        "nn_index_contains_heldout": bool(held in nn_meta.get("train_split_ids", [])),
        "classifier_trained_on_heldout": bool(held in classifier_meta.get("train_split_ids", [])),
    }
    leak["no_leakage"] = not any(v for k, v in leak.items() if k.endswith(("heldout",)) or "contain" in k or "trained_on" in k)
    # explicit recompute of no_leakage over the boolean checks
    leak["no_leakage"] = not (
        leak["train_records_contain_heldout"]
        or leak["train_evaluations_contain_heldout"]
        or leak["cvi_star_contains_heldout"]
        or leak["warmstarts_contain_heldout"]
        or leak["nn_index_contains_heldout"]
        or leak["classifier_trained_on_heldout"]
    )
    rep["leakage"] = leak

    # --- counts ---
    n_train_records = int(train_records["record_id"].nunique())
    n_train_eval = int(len(train_evaluations))
    rep["counts"] = {
        "train_splits": sorted(train_splits),
        "heldout_split": held,
        "excluded_splits": sorted(excluded_splits),
        "n_train_records": n_train_records,
        "n_train_evaluations": n_train_eval,
        "n_cvi_candidates": int(len(cvi_candidates)),
        "n_cvi_star_records": int(len(cvi_star)),
        "n_cvi_star_resolved": int(cvi_star["resolved"].sum()),
        "n_cvi_star_unresolved": int((~cvi_star["resolved"]).sum()),
        "n_warmstart_rows": int(len(warmstarts)),
        "evaluations_eq_records_x32": n_train_eval == n_train_records * N_CONFIGS,
    }

    # --- CVI coverage (direction required only for INCLUDED candidates;
    #     excluded metrics are unusable and need no direction) ---
    included = cvi_candidates[cvi_candidates["included"]]
    rep["cvi_coverage"] = {
        "all_candidates_have_direction": bool(included["direction"].notna().all()),
        "candidates_included": int(cvi_candidates["included"].sum()),
        "candidates_excluded": int((~cvi_candidates["included"]).sum()),
        "excluded": cvi_candidates[~cvi_candidates["included"]][
            ["metric_id", "exclusion_reason"]
        ].to_dict("records"),
    }

    # --- classifier ---
    rep["classifier"] = {
        "trained": bool(classifier_meta.get("trained", False)),
        "classifier_type": classifier_meta.get("classifier_type"),
        "n_classes": classifier_meta.get("n_classes"),
        "cv_accuracy": classifier_meta.get("cv_accuracy"),
        "cv_balanced_accuracy": classifier_meta.get("cv_balanced_accuracy"),
        "train_record_count": classifier_meta.get("train_record_count"),
    }

    # --- NN index ---
    rep["nn_index"] = {
        "built": bool(nn_meta.get("built", False)),
        "n_train_records": nn_meta.get("n_train_records"),
        "matches_expected": nn_meta.get("n_train_records") == classifier_meta.get("train_record_count"),
        "distance_metric": nn_meta.get("distance_metric"),
    }

    # --- warmstarts ---
    per_record = warmstarts.groupby("record_id")["config_id"].count()
    rep["warmstarts"] = {
        "all_records_32_configs": bool((per_record == N_CONFIGS).all()),
        "records_with_wrong_count": int((per_record != N_CONFIGS).sum()),
        "n_records": int(per_record.shape[0]),
    }

    # --- overall ---
    rep["all_checks_pass"] = bool(
        leak["no_leakage"]
        and rep["counts"]["evaluations_eq_records_x32"]
        and rep["cvi_coverage"]["all_candidates_have_direction"]
        and rep["classifier"]["trained"]
        and rep["nn_index"]["built"]
        and rep["nn_index"]["matches_expected"]
        and rep["warmstarts"]["all_records_32_configs"]
    )
    return rep
