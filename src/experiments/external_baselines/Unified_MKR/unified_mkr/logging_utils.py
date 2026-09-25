"""Process-level setup + lightweight logging for the Unified MKR builder.

``configure_process`` mirrors ``utility_generation/worker_setup.configure_process``:
it makes stdio UTF-8 tolerant (Windows narrow code pages) and silences the
known-noisy third-party warnings that fire on degenerate 1D/collapsed
partitions. It is idempotent and safe to call in every worker.
"""
from __future__ import annotations

import sys
import warnings

_DONE = "_unified_mkr_process_configured"


def configure_process() -> None:
    if getattr(sys, _DONE, False):
        return

    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

    # Cluster runs produce a flood of benign warnings on tiny / collapsed
    # partitions. Silence the known ones so real signal stays visible.
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message=".*force_all_finite.*")
    warnings.filterwarnings("ignore", message=".*ConstantInputWarning.*")
    try:
        from sklearn.exceptions import ConvergenceWarning

        warnings.filterwarnings("ignore", category=ConvergenceWarning)
    except Exception:
        pass

    setattr(sys, _DONE, True)


def log(message: str) -> None:
    """Print + flush a single progress line."""
    print(message, flush=True)
