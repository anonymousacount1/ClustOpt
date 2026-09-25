"""The 20 learned deployable ClustOpt policy contexts, over 10 stored regimes.

The portfolio factorises exactly:

    2 utility sources x 5 K regimes x 2 weighting modes = 20 policies

and the first two factors alone determine the metric SET. Two policies sharing
(source, K) therefore share their candidate-CVI availability mask and differ only
in the weight vector over the same selected metrics. The package stores 10 mask
rows per slate with both weight vectors attached, and materialises none of the
20 as a separate candidate matrix.

The mapping is not hand-listed: it is derived from the frozen registry portfolio
``strict_deployable_clustopt_vbs`` and asserted to cover all 20.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Tuple

from models.Clustering_Repository_Builder.experiments.experiment_execution.oof_policy_replay import (  # noqa: E501
    build_policy_set, policy_semantics_hash,
)

from .masking_regimes import (
    K_MODES, SOFTMAX_TEMPERATURE, SOURCES, WEIGHTING_IMPL, regime_id,
)


def _k_mode_of(top_k: Any) -> str:
    if top_k == "dynamic":
        return "dynamic"
    return f"top{int(top_k)}"


def all_policy_contexts() -> List[Dict[str, Any]]:
    """Each of the 20 policies bound to its stored regime and weight column."""
    out: List[Dict[str, Any]] = []
    for p in build_policy_set():
        km = _k_mode_of(p.top_k)
        wm = "RAW" if p.weighting == "normalized_positive" else "SOFTMAX_T05"
        out.append({
            "policy_id": p.policy_id,
            "short_label": p.short_label,
            "utility_source": p.source,
            "k_mode": km,
            "weighting_mode": wm,
            "weighting_impl": p.weighting,
            "softmax_temperature": (SOFTMAX_TEMPERATURE
                                    if wm == "SOFTMAX_T05" else None),
            "regime_id": regime_id(p.source, km),
            "weight_columns": ("w_raw_" if wm == "RAW" else "w_soft_"),
            "objective": "J = sum_m w_m * <metric>_norm over selected metrics",
        })
    validate_policy_contexts(out)
    return out


def validate_policy_contexts(ctx: List[Dict[str, Any]]) -> None:
    if len(ctx) != 20:
        raise ValueError(f"expected 20 policy contexts, got {len(ctx)}")
    if len({c["policy_id"] for c in ctx}) != 20:
        raise ValueError("duplicate policy_id in contexts")
    cells = {(c["utility_source"], c["k_mode"], c["weighting_mode"]) for c in ctx}
    expected = {(s, k, w) for s in SOURCES for k in K_MODES
                for w in ("RAW", "SOFTMAX_T05")}
    if cells != expected:
        missing = expected - cells
        raise ValueError(f"policy grid incomplete, missing {sorted(missing)[:4]}")
    regimes = {c["regime_id"] for c in ctx}
    if len(regimes) != 10:
        raise ValueError(f"expected 20 policies over 10 regimes, got {len(regimes)}")
    for c in ctx:
        if WEIGHTING_IMPL[c["weighting_mode"]] != c["weighting_impl"]:
            raise ValueError(f"{c['policy_id']}: weighting impl mismatch")


def policy_context_hash(ctx: List[Dict[str, Any]]) -> str:
    payload = [[c["policy_id"], c["utility_source"], c["k_mode"],
                c["weighting_mode"], c["regime_id"]] for c in ctx]
    return hashlib.sha256(json.dumps(payload, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()[:16]


def registry_semantics_hash() -> str:
    """Hash of the upstream registry policy set, for cross-stage binding."""
    return policy_semantics_hash(build_policy_set())


def regime_to_policies() -> Dict[str, Tuple[str, str]]:
    """regime_id -> (RAW policy_id, SOFTMAX policy_id)."""
    out: Dict[str, Dict[str, str]] = {}
    for c in all_policy_contexts():
        out.setdefault(c["regime_id"], {})[c["weighting_mode"]] = c["policy_id"]
    return {k: (v["RAW"], v["SOFTMAX_T05"]) for k, v in out.items()}
