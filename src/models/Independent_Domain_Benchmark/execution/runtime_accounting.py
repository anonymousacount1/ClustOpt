"""Runtime fields and the shared-work attribution rule.

T0 begins **after** the frozen ``X_decision``/``X_full`` are materialised in
memory and the required frozen model artifacts are loaded. T1 is when the final
clustering labels for the scientific method are materialised. Dataset
acquisition, benchmark-input preprocessing, storage serialisation and offline
evaluation all sit outside T0..T1 and are recorded in their own fields.

The attribution rule matters more than the plumbing. Six physical traces serve
fourteen scientific arms, so an arm consuming a shared trace pays for it **in
full**:

    method_runtime_equivalent = physical_trace_runtime + arm_scoring_runtime

The physical work is executed once operationally; dividing its cost across the
consuming arms would hand the shared families an artificial speed advantage that
has nothing to do with the methods. Actual deduplicated wall-clock is recorded
separately and must never be substituted for the equivalent runtime.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Iterator, Mapping, Optional

RUNTIME_CONTRACT_VERSION = "idb_stage4_runtime_v1"

T0_DEFINITION = ("after the frozen X_decision and X_full are materialised in "
                 "memory and the required frozen model artifacts are loaded")
T1_DEFINITION = ("when the final clustering labels for the scientific method are "
                 "materialised")
EXCLUDED_FROM_T0_T1 = (
    "dataset acquisition and loading",
    "benchmark-input preprocessing (imputation / scaling / PCA)",
    "serialisation and deserialisation performed only for storage",
    "offline ground-truth evaluation",
)
SHARED_WORK_RULE = (
    "method_runtime_equivalent = full physical_trace_runtime + that arm's own "
    "scoring/selection runtime. Shared physical cost is NEVER divided across the "
    "arms that consume it.")


@dataclass
class Stopwatch:
    """Accumulating timer. Explicit about which field it feeds."""

    name: str
    seconds: float = 0.0

    @contextmanager
    def running(self) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.seconds += time.perf_counter() - t0


@dataclass
class RuntimeRecord:
    """Every timing field the protocol names, kept separate on purpose."""

    preparation_runtime: float = 0.0        # outside T0..T1
    model_init_runtime: float = 0.0         # outside T0..T1
    metafeature_runtime: float = 0.0        # recorded separately
    physical_trace_runtime: float = 0.0     # inside T0..T1
    arm_scoring_runtime: float = 0.0        # inside T0..T1
    persistence_runtime: float = 0.0        # outside T0..T1
    extra: Dict[str, float] = field(default_factory=dict)

    @property
    def method_runtime_equivalent(self) -> float:
        """The frozen headline field. Shared physical cost is paid in full."""
        return float(self.physical_trace_runtime) + float(self.arm_scoring_runtime)

    def as_dict(self, *, wall_clock: Optional[float] = None) -> Mapping[str, object]:
        d = {
            "runtime_contract_version": RUNTIME_CONTRACT_VERSION,
            "t0_definition": T0_DEFINITION, "t1_definition": T1_DEFINITION,
            "excluded_from_t0_t1": list(EXCLUDED_FROM_T0_T1),
            "shared_work_rule": SHARED_WORK_RULE,
            "preparation_runtime": round(self.preparation_runtime, 6),
            "model_init_runtime": round(self.model_init_runtime, 6),
            "metafeature_runtime": round(self.metafeature_runtime, 6),
            "physical_trace_runtime": round(self.physical_trace_runtime, 6),
            "arm_scoring_runtime": round(self.arm_scoring_runtime, 6),
            "persistence_runtime": round(self.persistence_runtime, 6),
            "method_runtime_equivalent": round(self.method_runtime_equivalent, 6),
            "headline_field": "method_runtime_equivalent",
        }
        if wall_clock is not None:
            d["actual_wall_clock_runtime"] = round(float(wall_clock), 6)
            d["wall_clock_note"] = (
                "deduplicated actual compute for the whole group; NOT comparable "
                "to method_runtime_equivalent and never substituted for it")
        d.update({k: round(v, 6) for k, v in self.extra.items()})
        return d


def arms_pay_full_shared_cost(physical_runtime: float,
                              arm_runtimes: Mapping[str, float]
                              ) -> Dict[str, float]:
    """Equivalent runtime per arm. No division by the number of consumers."""
    return {arm: float(physical_runtime) + float(sec)
            for arm, sec in arm_runtimes.items()}
