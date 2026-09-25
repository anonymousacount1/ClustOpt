"""Top-K selection and utility -> weight conversion.

Given a ``{metric_name: utility_score}`` mapping, these helpers select the
Top-K metrics and convert their scores into normalised weights using either a
``softmax`` (temperature-controlled) or a ``normalized_positive`` scheme.

The output is always a ``{metric_name: weight}`` dict whose weights sum to 1
(or is empty if no metric survives selection).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def select_top_k(
    utilities: Dict[str, float],
    top_k: int,
) -> List[Tuple[str, float]]:
    """Return the ``top_k`` ``(name, score)`` pairs ordered by descending score.

    Ties are broken deterministically by metric name so results are
    reproducible across runs. Non-finite scores are treated as ``-inf`` so they
    sink to the bottom of the ranking.
    """
    if top_k <= 0:
        raise ValueError(f"top_k must be a positive integer, got {top_k}.")

    def _sort_key(item: Tuple[str, float]) -> Tuple[float, str]:
        name, score = item
        s = float(score)
        if not np.isfinite(s):
            s = float("-inf")
        # Negate score for descending; name ascending as a stable tiebreaker.
        return (-s, name)

    ordered = sorted(utilities.items(), key=_sort_key)
    return [(name, float(score)) for name, score in ordered[:top_k]]


def _softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError(f"softmax_temperature must be > 0, got {temperature}.")
    scaled = values / float(temperature)
    scaled = scaled - np.max(scaled)  # numerical stability
    exp = np.exp(scaled)
    total = exp.sum()
    if not np.isfinite(total) or total <= 0:
        # Degenerate fallback: uniform over the selected metrics.
        return np.full_like(values, 1.0 / len(values))
    return exp / total


def _normalized_positive(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, a_min=0.0, a_max=None)
    total = clipped.sum()
    if total <= 0:
        # All non-positive -> fall back to a uniform distribution.
        return np.full_like(values, 1.0 / len(values))
    return clipped / total


def compute_weights(
    selected: List[Tuple[str, float]],
    weighting: str = "softmax",
    softmax_temperature: float = 0.5,
) -> Dict[str, float]:
    """Convert selected ``(name, score)`` pairs into a ``{name: weight}`` dict.

    Supported ``weighting`` schemes:

    * ``softmax``              - ``softmax(scores / softmax_temperature)``.
    * ``normalized_positive``  - clip negatives to 0 and normalise to sum 1.
    """
    if not selected:
        return {}

    names = [name for name, _ in selected]
    scores = np.asarray([score for _, score in selected], dtype=np.float64)
    # Guard against non-finite scores leaking into the math.
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    scheme = str(weighting).lower()
    if scheme == "softmax":
        weights = _softmax(scores, softmax_temperature)
    elif scheme == "normalized_positive":
        weights = _normalized_positive(scores)
    else:
        raise ValueError(
            f"Unsupported weighting scheme '{weighting}'. "
            f"Options: 'softmax', 'normalized_positive'."
        )

    return {name: float(w) for name, w in zip(names, weights)}
