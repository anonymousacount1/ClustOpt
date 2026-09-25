"""Logging helpers for the AutoML4Clust baseline.

* :func:`configure_process` makes stdout/stderr UTF-8 tolerant (so a stray
  glyph from old third-party code cannot crash a worker under a narrow Windows
  code page) and silences the well-known noisy warnings -- mirroring the
  philosophy of ``utility_generation/worker_setup.py`` but kept self-contained
  so the baseline has no hard dependency on that module.
* :class:`RunLogger` accumulates timestamped progress lines, echoes them to the
  console with a consistent prefix, and can flush them to a ``run_log.txt``.
"""
from __future__ import annotations

import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from .path_utils import write_text_atomic

_DONE_FLAG = "_automl4clust_baseline_process_setup_done"


def _reconfigure_stdio() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def _install_warning_filters() -> None:
    # sklearn convergence / BIRCH-style noise from degenerate tiny slices.
    try:
        from sklearn.exceptions import ConvergenceWarning
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
    except Exception:
        pass
    warnings.filterwarnings(
        "ignore", message=r".*invalid value encountered in.*", category=RuntimeWarning
    )
    warnings.filterwarnings(
        "ignore", message=r".*Number of distinct clusters.*", category=UserWarning
    )


def configure_process() -> None:
    """Idempotent per-process setup for the main runner and every worker."""
    if getattr(sys, _DONE_FLAG, False):
        return
    _reconfigure_stdio()
    _install_warning_filters()
    setattr(sys, _DONE_FLAG, True)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


class RunLogger:
    """Accumulate + echo progress lines; flush to ``run_log.txt`` on demand.

    Parameters
    ----------
    prefix:
        Bracketed tag prepended to every console line, e.g. ``"AutoML4Clust 2d"``.
    echo:
        When True (default) each line is also printed to the console.
    """

    def __init__(self, prefix: str = "AutoML4Clust", *, echo: bool = True) -> None:
        self.prefix = prefix
        self.echo = echo
        self._lines: List[str] = []

    def log(self, message: str) -> None:
        line = f"{_now()} [{self.prefix}] {message}"
        self._lines.append(line)
        if self.echo:
            print(line, flush=True)

    def section(self, title: str) -> None:
        self.log("-" * 8 + f" {title} " + "-" * 8)

    @property
    def text(self) -> str:
        return "\n".join(self._lines) + ("\n" if self._lines else "")

    def dump(self, path: str | Path) -> Path:
        return write_text_atomic(path, self.text)


__all__ = ["configure_process", "RunLogger"]
