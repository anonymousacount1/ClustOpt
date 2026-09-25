"""The frozen 20-policy -> 10-reranker mapping.

Two policies sharing (utility source, K regime) share a reranker because their
candidate-CVI observability mask is identical and the frozen C2 interface exposes
no weighting mode, no metric weights and no objective J. They nonetheless remain
SEPARATE online runs: RAW and SOFTMAX drive different Optuna trajectories and can
visit different candidates.

The mapping is derived from the production registry via ``build_policy_set()``,
never from string matching on policy names.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Tuple

from models.Clustering_Repository_Builder.experiments.experiment_execution.oof_policy_replay import (  # noqa: E501
    build_policy_set, policy_semantics_hash,
)

RERANKER_IDS: Tuple[str, ...] = (
    "MLP_TOP1", "MLP_TOP3", "MLP_TOP5", "MLP_TOP10", "MLP_DYNAMIC",
    "KNN_TOP1", "KNN_TOP3", "KNN_TOP5", "KNN_TOP10", "KNN_DYNAMIC",
)
N_POLICIES = 20
N_RERANKERS = 10


def _k_mode(top_k: Any) -> str:
    return "dynamic" if top_k == "dynamic" else f"top{int(top_k)}"


def reranker_id(source: str, k_mode: str) -> str:
    return f"{source.upper()}_{k_mode.upper()}"


def build_mapping() -> List[Dict[str, Any]]:
    """One row per production policy, bound to its reranker."""
    rows: List[Dict[str, Any]] = []
    for p in build_policy_set():
        km = _k_mode(p.top_k)
        rid = reranker_id(p.source, km)
        rows.append({
            "policy_id": p.policy_id, "short_label": p.short_label,
            "utility_source": p.source, "k_mode": km,
            "weighting_mode": ("RAW" if p.weighting == "normalized_positive"
                               else "SOFTMAX_T05"),
            "weighting_impl": p.weighting, "k_policy": p.k_policy,
            "reranker_id": rid, "regime_id": f"{p.source}__{km}",
            "online_run_dir": p.policy_id,
        })
    validate(rows)
    return rows


def validate(rows: List[Dict[str, Any]]) -> None:
    if len(rows) != N_POLICIES:
        raise ValueError(f"expected {N_POLICIES} policies, got {len(rows)}")
    if len({r["policy_id"] for r in rows}) != N_POLICIES:
        raise ValueError("duplicate policy_id")
    rids = {r["reranker_id"] for r in rows}
    if rids != set(RERANKER_IDS):
        raise ValueError(f"reranker set mismatch: {sorted(rids)}")
    # Every policy must resolve to exactly one reranker -- no ambiguity.
    per_policy = {r["policy_id"]: r["reranker_id"] for r in rows}
    if len(per_policy) != N_POLICIES:
        raise ValueError("a policy mapped ambiguously")
    # Each reranker must serve exactly one RAW and one SOFTMAX policy.
    from collections import defaultdict
    grp = defaultdict(list)
    for r in rows:
        grp[r["reranker_id"]].append(r["weighting_mode"])
    for rid, ws in grp.items():
        if sorted(ws) != ["RAW", "SOFTMAX_T05"]:
            raise ValueError(f"{rid} serves {sorted(ws)}, expected RAW+SOFTMAX")


def mapping_hash(rows: List[Dict[str, Any]]) -> str:
    payload = sorted((r["policy_id"], r["reranker_id"]) for r in rows)
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()
                          ).hexdigest()[:16]


def registry_semantics_hash() -> str:
    return policy_semantics_hash(build_policy_set())
