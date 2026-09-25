"""Adapter for the ML2DAC-only internal CVIs (Dunn, Coggins-Jain, COP).

These three CVIs are *not* present in the ClustOpt metric registry, so they are
computed here using the **original ML2DAC implementation**
(``external/ml2dac/src/ClusterValidityIndices/CVIHandler.py``). The four CVIs
shared with ClustOpt (CH/DBI/SIL/DBCV) are intentionally NOT computed here —
they come from ClustOpt to honour the de-duplication rule.

Lazy import + graceful degradation: if the ML2DAC source is unavailable in the
running interpreter the adapter reports ``available == False`` and every value
is NaN with a recorded reason (the build continues).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .metric_registry import ML2DAC_CVIS, ml2dac_only_metrics

# Map our canonical names -> ML2DAC CVI abbreviations for lookup.
_CANON_TO_ABBREV = {name: abbrev for name, (_, abbrev) in ML2DAC_CVIS.items()}


@dataclass
class _AdapterState:
    available: bool = False
    error: str = ""
    cvi_by_abbrev: Dict[str, object] = field(default_factory=dict)


_state: Optional[_AdapterState] = None


def _ensure_loaded(repo_root: Optional[str] = None) -> _AdapterState:
    global _state
    if _state is not None:
        return _state
    state = _AdapterState()
    try:
        if repo_root is not None:
            from .io_utils import ensure_ml2dac_on_path

            ensure_ml2dac_on_path(repo_root)
        from ClusterValidityIndices.CVIHandler import CVICollection

        state.cvi_by_abbrev = {
            cvi.get_abbrev(): cvi for cvi in CVICollection.internal_cvis
        }
        state.available = True
    except Exception as exc:  # noqa: BLE001
        state.available = False
        state.error = f"{type(exc).__name__}: {exc}"[:300]
    _state = state
    return state


def required_metric_names() -> List[str]:
    return ml2dac_only_metrics()


def compute_ml2dac_only(
    X: np.ndarray,
    labels: np.ndarray,
    *,
    repo_root: Optional[str] = None,
) -> Dict[str, float]:
    """Compute the ML2DAC-only CVIs on (X, labels). NaN where undefined/unavailable."""
    names = required_metric_names()
    out: Dict[str, float] = {n: float("nan") for n in names}
    state = _ensure_loaded(repo_root)
    if not state.available:
        return out

    n_clusters = int(np.unique(labels[labels != -1]).size)
    if n_clusters < 2:
        return out  # CVIs undefined for < 2 clusters

    for name in names:
        abbrev = _CANON_TO_ABBREV.get(name)
        cvi = state.cvi_by_abbrev.get(abbrev)
        if cvi is None:
            continue
        try:
            val = float(cvi.score_cvi(X, labels))
            out[name] = val if np.isfinite(val) else float("nan")
        except Exception:  # noqa: BLE001
            out[name] = float("nan")
    return out


def adapter_status(repo_root: Optional[str] = None) -> Dict[str, object]:
    state = _ensure_loaded(repo_root)
    return {"available": state.available, "error": state.error}
