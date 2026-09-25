"""Dynamic Top-K selection for the Phase-D ablation.

Wraps the already-implemented and validated heuristic
``dynamic_topk_relsoft_a092_k5_ceil`` (see
``models/ClustOpt/dynamic_topk_selection/heuristics/selected_dynamic_topk.py``)
so the dynamic CVI resolvers can choose K from a utility vector instead of a
fixed integer.

Crucially, the *number* of metrics K and the *ranking* used to pick the top-K
can come from different utility vectors (the Phase-D ``realK`` variants rank by
predicted utility but size K from the real/oracle utility). This module keeps
those two responsibilities separate so the raw and softmax weighting variants of
the same dynamic method always select the *same* metric subset.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

# Reuse the canonical, validated heuristic (do not reimplement the formula).
from models.ClustOpt.dynamic_topk_selection.heuristics.selected_dynamic_topk import (
    ALPHA as _ALPHA,
    HEURISTIC_NAME as _HEURISTIC_NAME,
    MAX_K as _MAX_K,
    SOFT_CAP_BASE as _SOFT_CAP_BASE,
    clean_utilities,
    k_rel_count,
    soft_cap,
)
import numpy as np

from .weighting import select_top_k

# Public, config-facing policy identifier.
DYNAMIC_TOPK_POLICY = _HEURISTIC_NAME            # "dynamic_topk_relsoft_a092_k5_ceil"
DYNAMIC_TOPK_ALPHA = _ALPHA
DYNAMIC_TOPK_SOFT_CAP_BASE = _SOFT_CAP_BASE
DYNAMIC_TOPK_MAX_K = _MAX_K

# Strings that select the dynamic policy in a config ``top_k`` field.
DYNAMIC_TOPK_TOKENS = {"dynamic", "dynamic_topk", DYNAMIC_TOPK_POLICY}


def is_dynamic_top_k(top_k) -> bool:
    """True when a config ``top_k`` value requests the dynamic policy."""
    return isinstance(top_k, str) and top_k.strip().lower() in DYNAMIC_TOPK_TOKENS


def compute_dynamic_k(k_source_utilities: Dict[str, float]) -> Tuple[int, int]:
    """Return ``(selected_k, k_rel_raw)`` from the K-source utility vector.

    ``selected_k`` is the clipped heuristic output in ``[1, MAX_K]``; ``k_rel_raw``
    is the pre-cap relative-threshold count (recorded for metadata).
    """
    vals = np.asarray(list(k_source_utilities.values()), dtype=float)
    u = clean_utilities(vals)
    u_max = float(u.max()) if u.size else 0.0
    k_rel = int(k_rel_count(u, u_max))
    return int(soft_cap(k_rel)), k_rel


def select_dynamic_top_k(
    ranking_utilities: Dict[str, float],
    k_source_utilities: Optional[Dict[str, float]] = None,
) -> Tuple[List[Tuple[str, float]], int, int]:
    """Select the dynamic top-K metrics.

    ``ranking_utilities`` orders the metrics; ``k_source_utilities`` (defaults to
    the ranking vector) sizes K. Returns ``(selected_pairs, selected_k,
    k_rel_raw)`` where ``selected_pairs`` are ``(name, ranking_score)`` ordered by
    descending ranking score -- identical regardless of downstream weighting.
    """
    k_src = k_source_utilities if k_source_utilities is not None else ranking_utilities
    selected_k, k_rel_raw = compute_dynamic_k(k_src)
    selected = select_top_k(ranking_utilities, selected_k)
    return selected, selected_k, k_rel_raw
