"""Consolidate raw rows to one row per (dataset, method) via best-view selection.

The best view is chosen by **highest ARI** -- this is an *oracle* best-view
selection because it uses ground-truth ARI to pick the view. It is documented
as such (plan §49) and is intended as an upper-bound style comparison.
"""
from __future__ import annotations

import pandas as pd

DATASET_LEVEL_COLUMNS = [
    "dataset_id", "method_name", "method_group", "family", "subfamily",
    "difficulty", "cluster_count", "best_view_id", "best_ari", "best_nmi",
    "best_ami", "best_v_measure", "best_selected_algorithm", "best_selected_k",
    "true_k", "best_k_correct", "best_k_abs_error", "best_runtime_sec",
    "total_runtime_sec_all_views",
]


def build_dataset_level(raw: pd.DataFrame) -> pd.DataFrame:
    """Return one row per (dataset, method), choosing the highest-ARI view."""
    if raw.empty:
        return pd.DataFrame(columns=DATASET_LEVEL_COLUMNS)

    success = raw[raw["status"] == "success"].copy()
    rows = []
    group_cols = ["dataset_id", "method_name"]
    for (dataset_id, method_name), grp in raw.groupby(group_cols):
        succ = grp[grp["status"] == "success"]
        total_runtime = pd.to_numeric(grp["runtime_sec"], errors="coerce").sum()
        if succ.empty:
            # No successful view: emit a row carrying identity + NaNs.
            first = grp.iloc[0]
            rows.append({
                "dataset_id": dataset_id, "method_name": method_name,
                "method_group": first.get("method_group"),
                "family": first.get("family"), "subfamily": first.get("subfamily"),
                "difficulty": first.get("difficulty"),
                "cluster_count": first.get("cluster_count"),
                "best_view_id": None, "best_ari": None, "best_nmi": None,
                "best_ami": None, "best_v_measure": None,
                "best_selected_algorithm": None, "best_selected_k": None,
                "true_k": first.get("true_k"), "best_k_correct": None,
                "best_k_abs_error": None, "best_runtime_sec": None,
                "total_runtime_sec_all_views": float(total_runtime),
            })
            continue
        succ = succ.copy()
        succ["_ari"] = pd.to_numeric(succ["ari"], errors="coerce")
        best = succ.sort_values("_ari", ascending=False).iloc[0]
        rows.append({
            "dataset_id": dataset_id, "method_name": method_name,
            "method_group": best.get("method_group"),
            "family": best.get("family"), "subfamily": best.get("subfamily"),
            "difficulty": best.get("difficulty"),
            "cluster_count": best.get("cluster_count"),
            "best_view_id": best.get("view_id"),
            "best_ari": best.get("ari"), "best_nmi": best.get("nmi"),
            "best_ami": best.get("ami"), "best_v_measure": best.get("v_measure"),
            "best_selected_algorithm": best.get("selected_algorithm"),
            "best_selected_k": best.get("selected_k"),
            "true_k": best.get("true_k"),
            "best_k_correct": best.get("k_correct"),
            "best_k_abs_error": best.get("k_abs_error"),
            "best_runtime_sec": best.get("runtime_sec"),
            "total_runtime_sec_all_views": float(total_runtime),
        })
    return pd.DataFrame(rows, columns=DATASET_LEVEL_COLUMNS)
