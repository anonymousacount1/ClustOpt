"""Standalone implementation of the selected Dynamic Top-K heuristic.

This package implements, runs and post-analyses the heuristic
``dynamic_topk_relsoft_a092_k5_ceil`` selected in Phase 2. It is intentionally
separate from ``pre_analysis`` (which is exploratory) and does NOT modify any
ClustOpt optimization code -- the heuristic is evaluated as a standalone selector.
"""

from .selected_dynamic_topk import (  # noqa: F401
    ALPHA,
    HEURISTIC_NAME,
    MAX_K,
    SOFT_CAP_BASE,
    Selection,
    select,
    select_batch,
)
