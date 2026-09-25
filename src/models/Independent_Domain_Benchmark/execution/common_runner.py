"""The common Stage-4 benchmark runner: physical first, then scientific.

    DatasetView -> PhysicalTraceGroup(s) -> ScientificArmResult(s)

Six physical groups per dataset/view serve fourteen scientific arms. A group is
executed once; every arm in it consumes the same immutable trace and pays the
full physical cost in its equivalent runtime (see ``runtime_accounting``).

Three invariants the runner owns, so no method can get them wrong:

* **View contract.** The runner constructs ``DatasetView`` and passes
  ``X_decision`` and ``X_full`` explicitly to every builder and scorer. No scorer
  default is ever relied upon.
* **Label isolation.** Nothing the method layer touches carries ground truth.
  Evaluation lives behind a separate boundary and runs only after arms are
  COMPLETE.
* **Authoritative COMPLETE.** A provenance-valid COMPLETE record is reused
  exactly. Valid work is never recomputed, corrupt work is never silently
  regenerated, and mismatched provenance never reuses.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np

from . import hashing as H
from .dataset_view import DatasetView, assert_row_alignment
from .groups import PhysicalGroup, physical_groups
from .runtime_accounting import RuntimeRecord, Stopwatch, arms_pay_full_shared_cost
from .store import IntegrityError, Layer, ProvenanceMismatch, ResultStore, State

_ROOT = Path(__file__).resolve().parents[1]
_REPO = _ROOT.parents[1]

SCIENTIFIC_ARM_CONTRACT = "scientific_arm_result_v2"
WATCHDOG_SECONDS = 1800
OUTER_WORKERS = 8
THREAD_ENV = {
    "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
    "PYTHONHASHSEED": "0",
}


class RunnerFailure:
    """Infrastructure failures, kept distinct from scientific method failures."""

    WATCHDOG_TIMEOUT = "WATCHDOG_TIMEOUT"          # infrastructure, not science
    PHYSICAL_TRACE_FAILED = "PHYSICAL_TRACE_FAILED"
    ARM_SCORING_FAILED = "ARM_SCORING_FAILED"
    TRACE_IDENTITY_MISMATCH = "TRACE_IDENTITY_MISMATCH"
    VIEW_CONTRACT_VIOLATION = "VIEW_CONTRACT_VIOLATION"
    ALL = ("WATCHDOG_TIMEOUT", "PHYSICAL_TRACE_FAILED", "ARM_SCORING_FAILED",
           "TRACE_IDENTITY_MISMATCH", "VIEW_CONTRACT_VIOLATION")


def apply_thread_controls() -> Dict[str, str]:
    """Pin numeric-library threads. Reduces variation; never chases bit identity."""
    for k, v in THREAD_ENV.items():
        os.environ.setdefault(k, v)
    applied = {k: os.environ.get(k, "") for k in THREAD_ENV}
    try:
        import cv2
        cv2.setNumThreads(0)
        applied["cv2_num_threads"] = str(cv2.getNumThreads())
    except Exception:  # noqa: BLE001
        applied["cv2_num_threads"] = "unavailable"
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
        applied["threadpoolctl"] = "1"
    except Exception:  # noqa: BLE001
        applied["threadpoolctl"] = "unavailable"
    return applied


def _label_payload(labels, view) -> Dict[str, Any]:
    """The evaluable payload. A successful arm may not be COMPLETE without it.

    Stage 4B-4 proved storage integrity but never proved that the offline
    evaluator's required input was present, so ARI could not be computed from a
    finished run. The invariants below make that failure mode impossible: the
    vector must exist, cover every sample, and hash to what the arm recorded.
    """
    from Independent_Domain_Benchmark.methods.clustopt_trace import partition_hash
    if labels is None:
        raise RunnerContractError(
            "a successful scientific arm has no final label vector")
    a = np.asarray(labels, dtype=np.int64).reshape(-1)
    if a.shape[0] != int(view.sample_count):
        raise RunnerContractError(
            "final labels cover %d samples, view has %d"
            % (a.shape[0], view.sample_count))
    return {"final_labels": [int(v) for v in a],
            "final_labels_raw_hash": _raw_hash(a),
            "canonical_partition_hash": partition_hash(a),
            "sample_count": int(a.shape[0])}


def _raw_hash(a: np.ndarray) -> str:
    import hashlib
    return hashlib.sha256(np.asarray(a, dtype=np.int64).tobytes()).hexdigest()[:16]


class RunnerContractError(RuntimeError):
    """The scientific result contract was violated. Never downgraded."""


@dataclass
class Watchdog:
    """A wall-clock budget for genuine hangs, not a scientific timeout."""

    seconds: int = WATCHDOG_SECONDS
    started: float = field(default_factory=time.perf_counter)

    def check(self, what: str) -> None:
        if time.perf_counter() - self.started > self.seconds:
            raise TimeoutError("%s: %s after %ds"
                               % (RunnerFailure.WATCHDOG_TIMEOUT, what, self.seconds))


@dataclass
class ArmOutcome:
    method_id: str
    group_id: str
    status: str
    failure_kind: Optional[str]
    selected_algorithm: Optional[str]
    selected_configuration: Optional[Mapping[str, Any]]
    final_labels_hash: Optional[str]
    n_evaluations: int
    runtime: Mapping[str, Any]
    details: Mapping[str, Any]
    #: the final partition itself. Stage 4C found that persisting only the hash
    #: left the offline evaluator unable to compute ARI without re-executing the
    #: method, so a successful arm now carries its labels through to storage.
    final_labels: Optional[np.ndarray] = None


# --------------------------------------------------------------- executors
class FamilyExecutor:
    """One physical build + N arm scorings for a method family."""

    family = "UNSET"

    def build(self, view: DatasetView, budget: Optional[int]) -> Tuple[Any, str, float]:
        raise NotImplementedError

    def score(self, trace: Any, method_ids: Tuple[str, ...],
              view: DatasetView) -> Tuple[Dict[str, ArmOutcome], Dict[str, float]]:
        raise NotImplementedError


class AutoClustExecutor(FamilyExecutor):
    family = "AutoClust"

    def build(self, view, budget):
        from Independent_Domain_Benchmark.methods import autoclust as AC
        t0 = time.perf_counter()
        tr = AC.build_autoclust_candidate_trace(
            view.decision_for_algorithm(), view.dataset_id, view.view_id,
            budget=budget)
        return tr, tr.trace_id, time.perf_counter() - t0

    def score(self, trace, method_ids, view):
        from Independent_Domain_Benchmark.methods import autoclust_scoring as AS
        out, secs = {}, {}
        for mid in method_ids:
            t0 = time.perf_counter()
            # X_full is EXPLICIT: the frozen V2 contract, never a scorer default
            r = AS.score_trace(trace, mid,
                               X_decision=view.decision_for_algorithm(),
                               X_full=view.full_for_algorithm())
            secs[mid] = time.perf_counter() - t0
            out[mid] = ArmOutcome(
                method_id=mid, group_id=trace.trace_id, status=r.status,
                failure_kind=r.failure_kind,
                selected_algorithm=r.selected_algorithm,
                selected_configuration=(None if r.selected_configuration is None
                                        else dict(r.selected_configuration)),
                final_labels_hash=r.final_labels_hash,
                n_evaluations=r.n_candidates, runtime={},
                final_labels=(None if r.selected_candidate_index is None else
                              trace.candidates[r.selected_candidate_index].labels),
                details={"metric_inventory_id": r.metric_inventory_id,
                         "predictor_artifact_hash": r.predictor_artifact_hash,
                         "feature_schema_hash": r.feature_schema_hash,
                         "selected_candidate_index": r.selected_candidate_index})
        return out, secs


class ML2DACExecutor(FamilyExecutor):
    family = "ML2DAC"

    def build(self, view, budget):
        from Independent_Domain_Benchmark.methods import ml2dac as ML
        t0 = time.perf_counter()
        tr = ML.build_ml2dac_candidate_trace(
            view.decision_for_algorithm(), view.dataset_id, view.view_id,
            budget=budget)
        return tr, tr.trace_id, time.perf_counter() - t0

    def score(self, trace, method_ids, view):
        from Independent_Domain_Benchmark.methods import ml2dac_scoring as MS
        out, secs = {}, {}
        for mid in method_ids:
            t0 = time.perf_counter()
            r = MS.score_trace(trace, mid,
                               X_decision=view.decision_for_algorithm(),
                               X_full=view.full_for_algorithm())
            secs[mid] = time.perf_counter() - t0
            out[mid] = ArmOutcome(
                method_id=mid, group_id=trace.trace_id, status=r.status,
                failure_kind=r.failure_kind,
                selected_algorithm=r.selected_algorithm,
                selected_configuration=(None if r.selected_configuration is None
                                        else dict(r.selected_configuration)),
                final_labels_hash=r.final_labels_hash,
                n_evaluations=r.n_evaluations, runtime={},
                final_labels=(None if r.selected_candidate_index is None else
                              trace.candidates[r.selected_candidate_index].labels),
                details={"scoring_semantics_id": r.scoring_semantics_id,
                         "predicted_cvi_metric_id":
                             r.decision.predicted_cvi_metric_id,
                         "direction": r.decision.direction,
                         "classifier_artifact_hash":
                             r.decision.classifier_artifact_hash[:16],
                         "metric_inventory_id": r.metric_inventory_id,
                         "selected_candidate_index": r.selected_candidate_index})
        return out, secs


class ClustOptExecutor(FamilyExecutor):
    """ClustOpt: one physical search per validated trace group, N selectors.

    The ClustOpt adapter owns its own process-wide trace cache keyed by seed
    identity, which is what makes C0/C3 and C1/C4 share a search. ``build``
    therefore runs the group's first member end to end (populating that cache)
    and ``score`` replays every member against the cached trace, so each member's
    measured time is its selector alone. The search cost is recovered as
    ``build_time - first_member_selector_time`` and is then charged IN FULL to
    every arm in the group by the shared-work rule.
    """

    family = "ClustOpt"
    K_MIN, K_MAX = 2, 5

    def __init__(self) -> None:
        from Independent_Domain_Benchmark.methods import clustopt_adapter as CA
        self._CA = CA
        self._adapters = CA.build_adapters()
        for a in self._adapters.values():
            a.artifacts.resolve()
        self._group_of: Dict[str, str] = {}
        self._build_seconds: Dict[str, float] = {}

    def _kw(self, view: DatasetView, budget: Optional[int]) -> Dict[str, Any]:
        from Independent_Domain_Benchmark.search import protocol as PROTO
        return dict(seed=PROTO.BASE_SEED,
                    budget=int(budget or PROTO.BUDGET_EVALUATIONS),
                    k_min=self.K_MIN, k_max=self.K_MAX,
                    dataset_id=view.dataset_id, view_id=view.view_id)

    def build(self, view, budget, method_ids: Tuple[str, ...] = ()):
        first = method_ids[0]
        t0 = time.perf_counter()
        res = self._adapters[first].run(view.full_for_algorithm(),
                                        **self._kw(view, budget))
        secs = time.perf_counter() - t0
        trace_id = res.extra["search_trace_id"]
        self._build_seconds[trace_id] = secs
        return (view, budget), trace_id, secs

    def score(self, trace, method_ids, view):
        ctx_view, budget = trace
        out, secs = {}, {}
        trace_id = None
        for mid in method_ids:
            t0 = time.perf_counter()
            res = self._adapters[mid].run(ctx_view.full_for_algorithm(),
                                          **self._kw(ctx_view, budget))
            secs[mid] = time.perf_counter() - t0
            trace_id = res.extra["search_trace_id"]
            out[mid] = ArmOutcome(
                method_id=mid, group_id=res.extra["trace_identity"],
                status="success", failure_kind=None,
                selected_algorithm=res.selected_algorithm,
                selected_configuration=dict(res.best_configuration),
                final_labels_hash=res.extra.get("final_labels_hash"),
                n_evaluations=int(res.n_evaluations), runtime={},
                final_labels=res.labels,
                details={"search_trace_id": trace_id,
                         "trace_identity": res.extra["trace_identity"],
                         "final_selection_mechanism":
                             res.extra["final_selection_mechanism"],
                         "selected_candidate_index":
                             res.extra["selected_candidate_index"],
                         "selected_metric_ids": res.extra["selected_metric_ids"]})
        return out, secs


# ------------------------------------------------------------------ runner
@dataclass
class BenchmarkRunner:
    """Executes one or more DatasetViews under the frozen plan."""

    store: ResultStore
    budget: Optional[int] = None            # None = frozen protocol budget
    watchdog_seconds: int = WATCHDOG_SECONDS
    counters: Dict[str, int] = field(default_factory=lambda: {
        "physical_built": 0, "physical_reused": 0,
        "arms_scored": 0, "arms_reused": 0})
    thread_controls: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.thread_controls = apply_thread_controls()
        self.groups = physical_groups()

    # ------------------------------------------------------- provenance
    def _base_provenance(self, view: DatasetView) -> Dict[str, Any]:
        import json as _json
        from Independent_Domain_Benchmark.search import protocol as PROTO
        cfg = _ROOT / "configs"
        return {
            "corpus_hash": _json.loads(
                (cfg / "dataset_manifest.json").read_text(encoding="utf-8")
            ).get("manifest_sha256_16"),
            "search_space_fingerprint": "8ddf86c79888ebe9",
            "protocol_hash": H.canonical_text_hash(
                (cfg / "benchmark_protocol.json").read_bytes()),
            "method_registry_hash": H.canonical_text_hash(
                (cfg / "method_registry.json").read_bytes()),
            "numerical_policy_hash": H.canonical_text_hash(
                (cfg / "numerical_reproducibility_policy.json").read_bytes()),
            "budget": int(self.budget or PROTO.BUDGET_EVALUATIONS),
            "hash_conventions_version": H.HASH_CONVENTIONS_VERSION,
            **dict(view.identity()),
        }

    def _physical_provenance(self, view, group: PhysicalGroup) -> Dict[str, Any]:
        from Independent_Domain_Benchmark.search import protocol as PROTO
        p = self._base_provenance(view)
        p.update({"group_id": group.group_id, "family": group.family,
                  "seed_identity": group.group_id,
                  "resolved_seed": PROTO.derive_seed(
                      view.dataset_id, view.view_id, group.method_ids[0]),
                  "method_ids": list(group.method_ids)})
        return p

    def _arm_provenance(self, view, group, method_id, trace_id) -> Dict[str, Any]:
        p = self._base_provenance(view)
        p.update({"method_id": method_id, "group_id": group.group_id,
                  "physical_trace_id": trace_id,
                  "result_contract_version": "idb_stage4_arm_v1"})
        if group.family == "ML2DAC":
            from Independent_Domain_Benchmark.methods import ml2dac_scoring as MS
            p["scoring_semantics_id"] = MS.SCORING_SEMANTICS_ID
        return p

    # ------------------------------------------------------------- keys
    @staticmethod
    def _pkey(view: DatasetView, group: PhysicalGroup) -> str:
        return "%s__%s__%s" % (view.dataset_id, view.view_id, group.group_id)

    @staticmethod
    def _akey(view: DatasetView, method_id: str) -> str:
        return "%s__%s__%s" % (view.dataset_id, view.view_id, method_id)

    def _arm_is_authoritative(self, view: DatasetView, group: PhysicalGroup,
                              method_id: str) -> bool:
        """Is this arm already a valid COMPLETE result?

        The physical trace id is excluded from the comparison because it is only
        known after the trace is built, and the whole point of this check is to
        avoid building one. Every other frozen identity is still compared, and the
        record must pass full integrity; a corrupt record still raises.
        """
        key = self._akey(view, method_id)
        rec = self.store.load(Layer.SCIENTIFIC, key)
        if rec is None or rec.get("state") != State.COMPLETE.value:
            return False
        self.store.check_integrity(Layer.SCIENTIFIC, key, rec)   # raises if corrupt
        expected = dict(self._arm_provenance(view, group, method_id, ""))
        stored = dict(rec.get("provenance", {}))
        expected.pop("physical_trace_id", None)
        stored.pop("physical_trace_id", None)
        if H.semantic_object_hash(expected) != H.semantic_object_hash(stored):
            raise ProvenanceMismatch(
                "%s belongs to a different frozen identity" % key)
        return True

    def _build(self, ex: "FamilyExecutor", view: DatasetView,
               group: PhysicalGroup):
        """Some executors need the group membership to pick the search owner."""
        try:
            return ex.build(view, self.budget, group.method_ids)
        except TypeError:
            return ex.build(view, self.budget)

    # ------------------------------------------------------------- run
    def run_view(self, view: DatasetView,
                 executors: Mapping[str, FamilyExecutor]) -> Dict[str, ArmOutcome]:
        assert_row_alignment(view)
        dog = Watchdog(self.watchdog_seconds)
        results: Dict[str, ArmOutcome] = {}

        prep = Stopwatch("preparation")
        with prep.running():
            self.store.complete(
                Layer.PREPARATION, "%s__%s" % (view.dataset_id, view.view_id),
                provenance=self._base_provenance(view),
                content={"view_id": view.view_id,
                         "sample_count": view.sample_count,
                         "decision_shape": list(view.X_decision.shape),
                         "full_shape": list(view.X_full.shape),
                         "view_space_contract": "V2",
                         "x_full_explicit": True})

        for gid, group in sorted(self.groups.items()):
            ex = executors.get(group.family)
            if ex is None:
                continue
            dog.check("group %s" % gid)
            pkey = self._pkey(view, group)
            pprov = self._physical_provenance(view, group)

            # Decide what this group actually owes BEFORE building anything. A
            # group whose arms are all authoritative COMPLETE needs no physical
            # work at all; building it first and discovering that afterwards
            # regenerates an entire search for nothing.
            pending_first = [m for m in group.method_ids
                             if not self._arm_is_authoritative(view, group, m)]
            if not pending_first:
                for m in group.method_ids:
                    rec = self.store.load(Layer.SCIENTIFIC, self._akey(view, m))
                    self.counters["arms_reused"] += 1
                    c = rec["content"]
                    results[m] = ArmOutcome(
                        m, gid, c["status"], c.get("failure_kind"),
                        c.get("selected_algorithm"), c.get("selected_configuration"),
                        c.get("final_labels_hash"), int(c.get("n_evaluations", 0)),
                        c.get("runtime", {}), c.get("details", {}))
                continue

            reuse = self.store.try_reuse(Layer.PHYSICAL, pkey, pprov)
            trace = None
            if reuse.reusable:
                self.counters["physical_reused"] += 1
                trace_id = reuse.record["content"]["trace_id"]
                phys_secs = float(reuse.record["content"]["physical_trace_runtime"])
            else:
                self.store.mark(Layer.PHYSICAL, pkey, State.RUNNING)
                trace, trace_id, phys_secs = self._build(ex, view, group)
                self.counters["physical_built"] += 1
                self.store.complete(
                    Layer.PHYSICAL, pkey, provenance=pprov,
                    content={"trace_id": trace_id,
                             "candidate_count": int(getattr(trace, "candidate_count",
                                                            len(getattr(trace, "candidates", ())))),
                             "physical_trace_runtime": round(phys_secs, 6),
                             "consumed_by": list(group.method_ids),
                             "validated_sharing": group.validated_sharing})

            pending = [m for m in group.method_ids
                       if not self.store.try_reuse(
                           Layer.SCIENTIFIC, self._akey(view, m),
                           self._arm_provenance(view, group, m, trace_id)).reusable]
            for m in group.method_ids:
                if m in pending:
                    continue
                rec = self.store.load(Layer.SCIENTIFIC, self._akey(view, m))
                self.counters["arms_reused"] += 1
                c = rec["content"]
                results[m] = ArmOutcome(
                    m, gid, c["status"], c.get("failure_kind"),
                    c.get("selected_algorithm"), c.get("selected_configuration"),
                    c.get("final_labels_hash"), int(c.get("n_evaluations", 0)),
                    c.get("runtime", {}), c.get("details", {}))

            if not pending:
                continue
            if trace is None:
                # arms need scoring but the physical trace was only referenced:
                # rebuild it, which is INCOMPLETE work, never a valid COMPLETE arm
                trace, trace_id, phys_secs = self._build(ex, view, group)
                self.counters["physical_built"] += 1

            dog.check("scoring %s" % gid)
            outs, arm_secs = ex.score(trace, tuple(pending), view)
            if getattr(ex, "family", "") == "ClustOpt" and pending:
                # build() ran one member end to end; remove that member's
                # selector time so phys_secs is the SEARCH cost alone
                owner = group.method_ids[0]
                phys_secs = max(0.0, phys_secs - arm_secs.get(owner, 0.0))
            equiv = arms_pay_full_shared_cost(phys_secs, arm_secs)
            for m, outcome in outs.items():
                rr = RuntimeRecord(preparation_runtime=prep.seconds,
                                   physical_trace_runtime=phys_secs,
                                   arm_scoring_runtime=arm_secs[m])
                runtime = dict(rr.as_dict(wall_clock=phys_secs + sum(arm_secs.values())))
                # the stored field is rounded to 6 dp by as_dict(); compare at
                # that resolution rather than pretending the record is exact
                assert abs(runtime["method_runtime_equivalent"] - equiv[m]) < 1e-6, (
                    "shared-work formula disagrees with the stored field")
                outcome = ArmOutcome(
                    outcome.method_id, gid, outcome.status, outcome.failure_kind,
                    outcome.selected_algorithm, outcome.selected_configuration,
                    outcome.final_labels_hash, outcome.n_evaluations, runtime,
                    outcome.details, outcome.final_labels)
                content = {"status": outcome.status,
                           "failure_kind": outcome.failure_kind,
                           "physical_trace_id": trace_id,
                           "selected_algorithm": outcome.selected_algorithm,
                           "selected_configuration": outcome.selected_configuration,
                           "final_labels_hash": outcome.final_labels_hash,
                           "n_evaluations": outcome.n_evaluations,
                           "runtime": runtime, "details": dict(outcome.details),
                           "result_contract": SCIENTIFIC_ARM_CONTRACT,
                           "result_origin": "EXECUTED"}
                if outcome.status == "success":
                    content.update(_label_payload(outcome.final_labels, view))
                self.store.complete(
                    Layer.SCIENTIFIC, self._akey(view, m),
                    provenance=self._arm_provenance(view, group, m, trace_id),
                    content=content)
                self.counters["arms_scored"] += 1
                results[m] = outcome

            trace = None                      # release the candidate partitions
        return results
