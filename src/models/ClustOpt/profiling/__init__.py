"""Optional, off-by-default profiling primitives for ClustOpt.

Usage::

    from models.ClustOpt.profiling import Profiler

    profiler = Profiler(enabled=True)
    search_algorithm.profiler = profiler

    search_algorithm.search(X, X_full)

    profiler.write_csv("phase_timings.csv")
    profiler.summary_by_name()

If ``profiler`` is left ``None`` (the default), no timing instrumentation
runs and there is no overhead.
"""
from __future__ import annotations

from models.ClustOpt.profiling.profiler import Profiler, NullProfiler

__all__ = ["Profiler", "NullProfiler"]
