"""Logging helpers for the AutoClust baseline (re-exported from AutoML4Clust).

``configure_process`` makes stdout/stderr UTF-8 tolerant and silences known noisy
warnings (AutoClust/SMAC are very chatty); ``RunLogger`` accumulates timestamped
progress lines and flushes them to ``run_log.txt``.
"""
from __future__ import annotations

from experiments.external_baselines.AutoML4Clust.utils.logging_utils import (  # noqa: F401
    configure_process,
    RunLogger,
)

__all__ = ["configure_process", "RunLogger"]
