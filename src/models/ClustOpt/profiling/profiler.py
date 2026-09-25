"""Lightweight phase / section profiler.

The profiler exposes a context-manager :meth:`section` for timing arbitrary
phases (``fit_predict``, ``cvi_evaluate``, ``log``, ...). It records one
event per section enter/exit pair. Nested sections are allowed and
recorded as separate flat events; aggregation across runs is left to
post-processing.
"""
from __future__ import annotations

import csv
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


class Profiler:
    def __init__(self, enabled: bool = True):
        self.enabled = bool(enabled)
        self.events: List[Dict[str, Any]] = []

    @contextmanager
    def section(self, name: str, **meta: Any) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.events.append(
                {
                    "name": str(name),
                    "duration_sec": time.perf_counter() - t0,
                    **meta,
                }
            )

    def event(self, name: str, **meta: Any) -> None:
        """Record an instantaneous event with zero duration."""
        if not self.enabled:
            return
        self.events.append({"name": str(name), "duration_sec": 0.0, **meta})

    # ----------------------------------------------------------- aggregation
    def summary_by_name(self) -> Dict[str, Dict[str, float]]:
        agg: Dict[str, Dict[str, float]] = defaultdict(lambda: {"count": 0, "total_sec": 0.0})
        for ev in self.events:
            slot = agg[ev["name"]]
            slot["count"] += 1
            slot["total_sec"] += float(ev["duration_sec"])
        return {k: dict(v) for k, v in agg.items()}

    # ------------------------------------------------------------------ I/O
    def write_csv(self, path: str | Path) -> str:
        path = Path(path)
        if not self.events:
            path.write_text("name,duration_sec\n", encoding="utf-8")
            return str(path)
        fieldnames = sorted({k for ev in self.events for k in ev})
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for ev in self.events:
                writer.writerow(ev)
        return str(path)

    def reset(self) -> None:
        self.events.clear()


class NullProfiler(Profiler):
    """No-op profiler — drop-in replacement when instrumentation is undesired."""

    def __init__(self) -> None:
        super().__init__(enabled=False)
