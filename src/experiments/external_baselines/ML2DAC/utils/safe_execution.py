"""Safe-execution helpers for the ML2DAC baseline (re-exported from AutoML4Clust).

A failure in any single (dataset, record) unit is captured into a structured
``failed`` result (with traceback) and never propagates. The optional
thread-based timeout is best-effort (SMAC cannot be interrupted cleanly);
ML2DAC also enforces its own SMAC ``wallclock-limit``, which is the primary
budget guard.
"""
from __future__ import annotations

from experiments.external_baselines.AutoML4Clust.utils.safe_execution import (  # noqa: F401
    build_failure,
    safe_call,
    TimeoutExceeded,
)

__all__ = ["build_failure", "safe_call", "TimeoutExceeded"]
