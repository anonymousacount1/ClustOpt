"""Safe-execution helpers for the AutoClust baseline (re-exported from AutoML4Clust).

A failure in any single (dataset, record) unit is captured into a structured
``failed`` result (with traceback) and never propagates.

NOTE: AutoClust runs **natively** — its budget is a fixed evaluation count
(``optimizer_evaluations`` -> SMAC ``runcount-limit``), NOT a wall-clock timeout
(the original paper uses 40 trials; the ML2DAC re-implementation uses 100; neither
imposes a wall-clock cut). SMAC's own native ceilings (7200s wallclock, 300s
per-config cutoff) are the only time bounds. We therefore do **not** add an
external hard timeout / process-kill wrapper here.
"""
from __future__ import annotations

from experiments.external_baselines.AutoML4Clust.utils.safe_execution import (  # noqa: F401
    build_failure,
    safe_call,
    TimeoutExceeded,
)

__all__ = ["build_failure", "safe_call", "TimeoutExceeded"]
