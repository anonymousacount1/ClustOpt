"""The 10 canonical observability regimes: {MLP, KNN} x {1, 3, 5, 10, Dynamic}.

A regime answers exactly one question: *which of the 60 CVIs would a deployed
ClustOpt search actually have computed for a visited candidate?*

That question has a hard answer in production source. ``DynamicCVI`` builds a
``GenericCVIEvaluator(metrics=selected_keys)`` and ``GenericCVIEvaluator.evaluate``
loops over ``self.metrics`` only. An online trial therefore computes ONLY its
selected Top-K CVIs -- the unselected 59/57/55/50 are never evaluated and cannot
be recovered without re-clustering. Masking is the true deployment state, not an
experimental restriction.

Selection reuses the production resolvers verbatim (``select_top_k``,
``compute_dynamic_k``); nothing here reimplements ClustOpt science.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from models.ClustOpt.cluster_validity_indices.dynamic_metric_selection.dynamic_k import (
    DYNAMIC_TOPK_ALPHA, DYNAMIC_TOPK_MAX_K, DYNAMIC_TOPK_POLICY,
    DYNAMIC_TOPK_SOFT_CAP_BASE, compute_dynamic_k,
)
from models.ClustOpt.cluster_validity_indices.dynamic_metric_selection.weighting import (
    compute_weights, select_top_k,
)

SOURCES: Tuple[str, ...] = ("mlp", "knn")
K_MODES: Tuple[str, ...] = ("top1", "top3", "top5", "top10", "dynamic")
K_OF_MODE: Dict[str, Any] = {"top1": 1, "top3": 3, "top5": 5, "top10": 10,
                             "dynamic": "dynamic"}
WEIGHTING_MODES: Tuple[str, ...] = ("RAW", "SOFTMAX_T05")
WEIGHTING_IMPL: Dict[str, str] = {"RAW": "normalized_positive",
                                  "SOFTMAX_T05": "softmax"}
SOFTMAX_TEMPERATURE = 0.5
# Max metrics any regime can select -- Dynamic-K is clipped to MAX_K = 10.
MAX_SELECTED = 10


def regime_id(source: str, k_mode: str) -> str:
    return f"{source}__{k_mode}"


def all_regimes() -> List[Dict[str, Any]]:
    """The 10 canonical observability regimes, in frozen order."""
    out: List[Dict[str, Any]] = []
    for src in SOURCES:
        for km in K_MODES:
            out.append({
                "regime_id": regime_id(src, km),
                "utility_source": src,
                "k_mode": km,
                "k_rule": ("dynamic_predict_k" if km == "dynamic"
                           else f"fixed_top_{K_OF_MODE[km]}"),
                "fixed_k": None if km == "dynamic" else int(K_OF_MODE[km]),
                "dynamic_policy": DYNAMIC_TOPK_POLICY if km == "dynamic" else None,
                "dynamic_alpha": DYNAMIC_TOPK_ALPHA if km == "dynamic" else None,
                "dynamic_soft_cap_base": (DYNAMIC_TOPK_SOFT_CAP_BASE
                                          if km == "dynamic" else None),
                "dynamic_max_k": DYNAMIC_TOPK_MAX_K if km == "dynamic" else None,
                "max_selected": MAX_SELECTED,
            })
    if len(out) != 10:
        raise AssertionError(f"expected 10 regimes, built {len(out)}")
    return out


def regime_registry_hash(regimes: Sequence[Dict[str, Any]]) -> str:
    payload = [[r["regime_id"], r["utility_source"], r["k_mode"], r["k_rule"],
                str(r["fixed_k"]), str(r["dynamic_policy"])] for r in regimes]
    return hashlib.sha256(json.dumps(payload, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()[:16]


def resolve_regime(utilities: np.ndarray, metric_names: Sequence[str],
                   k_mode: str) -> Dict[str, Any]:
    """Selected metric indices, weights and the 60-dim mask for one regime.

    ``utilities`` is the source's 60-vector in canonical metric order. RAW and
    SOFTMAX weights are BOTH returned: they share this selection exactly (Top-K
    depends only on utilities and K), so the mask is computed once and only the
    weighting differs. That is why the package stores 10 mask rows per slate and
    never 20.
    """
    umap = {m: float(u) for m, u in zip(metric_names, utilities)}
    if k_mode == "dynamic":
        k, k_rel_raw = compute_dynamic_k(umap)
    else:
        k, k_rel_raw = int(K_OF_MODE[k_mode]), None

    selected = select_top_k(umap, k)                 # production resolver
    names = [n for n, _ in selected]
    idx = [metric_names.index(n) for n in names]
    w_raw = compute_weights(selected, weighting="normalized_positive")
    w_soft = compute_weights(selected, weighting="softmax",
                             softmax_temperature=SOFTMAX_TEMPERATURE)

    mask = np.zeros(len(metric_names), dtype=np.int8)
    mask[idx] = 1
    return {
        "actual_k": int(k), "k_rel_raw": k_rel_raw,
        "selected_indices": idx, "selected_names": names,
        "mask": mask,
        "w_raw": [float(w_raw[n]) for n in names],
        "w_softmax": [float(w_soft[n]) for n in names],
    }


def pad_indices(idx: Sequence[int]) -> List[int]:
    """Ranked selected indices padded to MAX_SELECTED with -1."""
    out = list(idx)[:MAX_SELECTED]
    return out + [-1] * (MAX_SELECTED - len(out))


def pad_weights(w: Sequence[float]) -> List[float]:
    out = list(w)[:MAX_SELECTED]
    return out + [0.0] * (MAX_SELECTED - len(out))


def apply_mask(cvi: np.ndarray, mask: np.ndarray, *,
               fill: float = np.nan) -> np.ndarray:
    """Masked CVI values. Unavailable positions become ``fill``, never 0.

    A silent zero would be indistinguishable from a genuine normalised score of
    zero, so the default fill is NaN and every consumer must read the companion
    mask.
    """
    out = np.asarray(cvi, dtype=np.float64).copy()
    out[np.asarray(mask) == 0] = fill
    return out
