"""Frozen 20-policy target order and the per-view target contract.

The target vector is the 20 learned deployable ClustOpt policies of
``paper_analysis.portfolio_registry.strict_deployable_clustopt_vbs`` -- 10 MLP and
10 KNN. All 20 are retained: Stage 2A-2 showed a 10-policy greedy portfolio
recovers 99.9% of the achievable oracle gain, but pruning the *target space* would
discard information the predictor may need, and costs nothing to keep.

Targets are produced per DATASET x VIEW, not per dataset. No Best-View aggregation
belongs inside the primary targets.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List

from models.Clustering_Repository_Builder.experiments.paper_analysis.method_registry import (
    METHOD_BY_NAME,
)
from models.Clustering_Repository_Builder.experiments.paper_analysis.portfolio_registry import (
    PORTFOLIO_BY_ID,
)

PORTFOLIO_ID = "strict_deployable_clustopt_vbs"
ARI_PREFIX = "policy_ari__"
REGRET_PREFIX = "policy_regret__"

#: Diagnostic / target-only columns derived from the 20 ARIs.
DIAGNOSTIC_COLUMNS: tuple[str, ...] = (
    "view_policy_vbs", "winner_set", "n_exact_winners",
    "top1_minus_top2_policy_score", "top1_minus_second_distinct_score",
    "n_within_0_001", "n_within_0_005", "n_within_0_01",
)

#: Sentinel used when every policy ties, so no strictly-lower score exists.
ALL_TIED_SENTINEL = -1.0


def policy_order() -> List[str]:
    """The canonical ordered 20 policy ids. Order is part of the schema."""
    pf = PORTFOLIO_BY_ID[PORTFOLIO_ID]
    ids = list(pf.methods)
    validate_policy_order(ids)
    return ids


def validate_policy_order(ids: List[str]) -> None:
    if len(ids) != 20:
        raise ValueError(f"expected 20 policies, got {len(ids)}")
    if len(set(ids)) != 20:
        raise ValueError("duplicate policy ids in the target order")
    n_mlp = sum(1 for i in ids if METHOD_BY_NAME[i].method_family == "clustopt_mlp")
    n_knn = sum(1 for i in ids if METHOD_BY_NAME[i].method_family == "clustopt_knn")
    if (n_mlp, n_knn) != (10, 10):
        raise ValueError(f"expected 10 MLP + 10 KNN, got {n_mlp}/{n_knn}")
    for i in ids:
        s = METHOD_BY_NAME[i]
        if s.uses_real_utility or s.is_oracle_assisted:
            raise ValueError(f"{i}: oracle policy must not be a target")
        if s.method_family == "clustopt_fixed":
            raise ValueError(f"{i}: fixed baseline must not be a target")


def policy_manifest() -> List[Dict[str, Any]]:
    """Ordered, self-describing target manifest."""
    out = []
    for idx, pid in enumerate(policy_order()):
        s = METHOD_BY_NAME[pid]
        out.append({
            "index": idx,
            "policy_id": pid,
            "short_label": s.short_label,
            "paper_label": s.paper_label,
            "source_family": "MLP" if s.method_family == "clustopt_mlp" else "KNN",
            "top_k_rule": ("dynamic_predict_k" if s.k_policy == "dynamic_predict"
                           else f"fixed_top_{s.fixed_metric_count}"),
            "fixed_metric_count": s.fixed_metric_count,
            "weighting": "RAW" if s.weighting_mode == "raw" else "SOFTMAX",
            "weighting_impl": s.weighting_mode,
            "dynamic": s.k_policy == "dynamic_predict",
            "k_policy": s.k_policy,
            "ari_target_column": f"{ARI_PREFIX}{pid}",
            "regret_target_column": f"{REGRET_PREFIX}{pid}",
        })
    return out


def ari_target_columns() -> List[str]:
    return [f"{ARI_PREFIX}{p}" for p in policy_order()]


def regret_target_columns() -> List[str]:
    return [f"{REGRET_PREFIX}{p}" for p in policy_order()]


def target_schema_hash() -> str:
    payload = [(m["index"], m["policy_id"], m["source_family"], m["top_k_rule"],
                m["weighting"]) for m in policy_manifest()]
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":")).encode()).hexdigest()
