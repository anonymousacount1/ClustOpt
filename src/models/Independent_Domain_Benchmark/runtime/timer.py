"""The one common external timing boundary. See configs/runtime_protocol.json."""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Dict


class RuntimeAccount:
    """ONLINE_RUNTIME and MODEL_INIT_RUNTIME, kept strictly separate.

    The headline is ONLINE_RUNTIME. At T0 the frozen X for one view is already
    materialised in memory; benchmark preparation and artifact loading are
    outside it by construction.
    """

    def __init__(self) -> None:
        self.online_sec = 0.0
        self.init_sec = 0.0

    @contextmanager
    def model_init(self):
        t = time.perf_counter()
        try:
            yield
        finally:
            self.init_sec += time.perf_counter() - t

    @contextmanager
    def online(self):
        t = time.perf_counter()
        try:
            yield
        finally:
            self.online_sec += time.perf_counter() - t

    def as_dict(self) -> Dict[str, float]:
        return {"online_runtime_sec": self.online_sec,
                "init_runtime_sec": self.init_sec,
                "total_cold_runtime_sec": self.online_sec + self.init_sec}
