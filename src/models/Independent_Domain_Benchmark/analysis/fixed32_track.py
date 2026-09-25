"""Track B — the independent fixed32 metric-analysis track.

One common candidate bank of 32 clusterings per (dataset, view); every CVI scores
exactly the same partitions. That is the whole point: it lets metric quality be
compared without confounding by any method's search, warmstarts or learned policy.

This track is **diagnostic**. Nothing computed here may reach ClustOpt, AutoClust
or ML2DAC inference, candidate generation, metric selection, model weights,
training or tuning. Ground truth enters only in the offline layer, after the bank
and the metric evaluations are complete.

The frozen V2 view semantics apply unchanged: candidates are fitted in
``X_decision(view)``; a metric is evaluated in its declared ``metric.space`` --
the view for decision-space metrics, the full 2-D representation of the same
observations for full-space metrics.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_REPO = _ROOT.parents[1]

FIXED32_VERSION = "idb_fixed32_v1"
FIXED32_SIZE = 32
SEARCH_CONFIG_ROOT = "models/ClustOpt/configs/utilities_maps"
#: the two New46 metrics that were all-NaN on the internal repository. They stay
#: DESIGNED metrics and are attempted normally in the external analysis; they are
#: never learned-model features.
HISTORICALLY_DEAD_NEW46 = ("convexity_ratio_image_based", "hough_arc_circle_strength")
MODERN_PRIMARY = ("cdbw", "cvnn", "cvdd", "dcsi")
DIAGNOSTIC_ONLY = ("dcsi_author_compat", "cdbw_package_compat")


class Fixed32Failure:
    CONFIG_RESOLUTION_FAILED = "CONFIG_RESOLUTION_FAILED"
    BANK_SIZE_MISMATCH = "BANK_SIZE_MISMATCH"
    CANDIDATE_EXECUTION_FAILED = "CANDIDATE_EXECUTION_FAILED"
    METRIC_EVALUATION_FAILED = "METRIC_EVALUATION_FAILED"
    ALL = ("CONFIG_RESOLUTION_FAILED", "BANK_SIZE_MISMATCH",
           "CANDIDATE_EXECUTION_FAILED", "METRIC_EVALUATION_FAILED")


class Fixed32Error(RuntimeError):
    def __init__(self, kind: str, detail: str = "") -> None:
        super().__init__("%s: %s" % (kind, detail) if detail else kind)
        self.kind, self.detail = kind, detail


# --------------------------------------------------------------- the bank
@dataclass(frozen=True)
class Fixed32Candidate:
    candidate_index: int
    candidate_id: str
    algorithm: str
    configuration: Mapping[str, Any]
    status: str                       # "valid" | "invalid"
    failure_kind: Optional[str]
    raw_labels_hash: Optional[str]
    canonical_partition_hash: Optional[str]
    labels: Optional[np.ndarray]      # write-protected

    def identity(self) -> Tuple:
        return (self.candidate_index, self.candidate_id, self.algorithm,
                json.dumps(dict(self.configuration), sort_keys=True),
                self.status, self.raw_labels_hash,
                self.canonical_partition_hash)


@dataclass(frozen=True)
class Fixed32CandidateBank:
    """One immutable candidate bank. Holds no ARI, no CVI value, no utility."""

    dataset_id: str
    view_id: str
    fixed32_version: str
    fixed32_config_hash: str
    decision_hash: str
    full_hash: str
    candidate_count: int
    candidates: Tuple[Fixed32Candidate, ...]
    bank_id: str
    provenance: Mapping[str, Any]

    def valid_indices(self) -> Tuple[int, ...]:
        return tuple(c.candidate_index for c in self.candidates
                     if c.status == "valid" and c.labels is not None)

    def content_hash(self) -> str:
        payload = json.dumps({
            "bank_id": self.bank_id, "dataset_id": self.dataset_id,
            "view_id": self.view_id, "fixed32_config_hash": self.fixed32_config_hash,
            "candidates": [list(c.identity()) for c in self.candidates]},
            sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _freeze_labels(labels) -> Optional[np.ndarray]:
    if labels is None:
        return None
    a = np.array(labels, dtype=np.int64, copy=True)
    a.setflags(write=False)
    return a


def resolve_fixed32_configs(view_id: str, repo: Path = _REPO):
    """The authoritative 32 configurations for one view, from the Unified MKR.

    Both compositions are recorded by :func:`fixed32_composition`: the raw
    enumeration (which contains 4 constrained-HDBSCAN configs) and the resolved
    bank after the frozen ``replace_constrained`` policy swaps those for extra
    non-constrained configs while holding the total at 32. They are the same
    definition observed before and after that documented step.
    """
    import sys
    mkr = str(repo / "experiments/external_baselines/Unified_MKR")
    if mkr not in sys.path:
        sys.path.insert(0, mkr)
    from unified_mkr.configuration import resolve_view_configurations
    rv = resolve_view_configurations(str(repo / SEARCH_CONFIG_ROOT), view_id)
    if len(rv.configs) != FIXED32_SIZE:
        raise Fixed32Error(Fixed32Failure.BANK_SIZE_MISMATCH,
                           "%d configs for %s" % (len(rv.configs), view_id))
    return rv


def fixed32_composition(view_id: str, repo: Path = _REPO) -> Dict[str, Any]:
    """Pre- and post-replacement composition, so the difference is never drift."""
    import collections
    import sys
    mkr = str(repo / "experiments/external_baselines/Unified_MKR")
    if mkr not in sys.path:
        sys.path.insert(0, mkr)
    from unified_mkr.configuration import (enumerate_configs, load_utility_map,
                                           replace_constrained)
    raw, _ = load_utility_map(str(repo / SEARCH_CONFIG_ROOT), view_id)
    ss = raw["search_space"]
    ms = int(raw.get("search_algorithm", {}).get("params", {})
             .get("max_samples_per_param", 1))
    enum = enumerate_configs(ss, view_id, ms)
    pre = collections.Counter(
        ("constrained" if s.is_constrained() else s.algorithm) for s in enum)
    post = collections.Counter(s.algorithm for s in replace_constrained(enum, ss, view_id))
    return {"enumerated_total": len(enum), "pre_replacement": dict(sorted(pre.items())),
            "resolved_total": sum(post.values()),
            "post_replacement": dict(sorted(post.items())),
            "replacement_policy": (
                "configuration.replace_constrained: the 4 constrained-HDBSCAN "
                "configs are swapped for extra non-constrained configs by fair "
                "round-robin, holding the bank at exactly 32")}


def fixed32_config_hash(view_id: str, repo: Path = _REPO) -> str:
    rv = resolve_fixed32_configs(view_id, repo)
    payload = json.dumps([[c.algorithm, c.original_config_id,
                           json.dumps(c.params, sort_keys=True)]
                          for c in rv.configs], sort_keys=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def build_fixed32_bank(view, *, repo: Path = _REPO) -> Fixed32CandidateBank:
    """Execute the 32 frozen configurations on one DatasetView. Label-free.

    ``view`` is a ``DatasetView``: it has no ground-truth field, so true K cannot
    be forced and no density algorithm can be handed the true cluster count.
    """
    import sys
    mkr = str(repo / "experiments/external_baselines/Unified_MKR")
    if mkr not in sys.path:
        sys.path.insert(0, mkr)
    from unified_mkr.clustering import run_configuration
    from models.ClustOpt.search_space.Clustering_Search_Space import (
        ClusteringSearchSpace)
    from Independent_Domain_Benchmark.methods.clustopt_trace import partition_hash

    rv = resolve_fixed32_configs(view.view_id, repo)
    ss = ClusteringSearchSpace.from_dict(rv.search_space_config)
    # third-party clustering cores need a writable buffer (Stage 4B-4 finding)
    Xd = view.decision_for_algorithm()

    cands: List[Fixed32Candidate] = []
    for i, spec in enumerate(rv.configs):
        try:
            res = run_configuration(ss, spec, Xd)
            labels = _freeze_labels(getattr(res, "labels", None))
            status = "valid" if res.valid_result else "invalid"
            fk = (None if res.valid_result
                  else (res.failure_reason or Fixed32Failure.CANDIDATE_EXECUTION_FAILED))
        except Exception as exc:  # noqa: BLE001
            labels, status = None, "invalid"
            fk = "%s:%s" % (Fixed32Failure.CANDIDATE_EXECUTION_FAILED,
                            type(exc).__name__)
        # the slot is ALWAYS kept: no replacement, no resampling, no budget growth
        cands.append(Fixed32Candidate(
            candidate_index=i, candidate_id=str(spec.original_config_id),
            algorithm=str(spec.algorithm),
            configuration=MappingProxyType(dict(spec.params)),
            status=status, failure_kind=fk,
            raw_labels_hash=(None if labels is None else hashlib.sha256(
                np.asarray(labels, dtype=np.int64).tobytes()).hexdigest()[:16]),
            canonical_partition_hash=(None if labels is None
                                      else partition_hash(labels)),
            labels=labels))
    if len(cands) != FIXED32_SIZE:
        raise Fixed32Error(Fixed32Failure.BANK_SIZE_MISMATCH, str(len(cands)))

    cfg_hash = fixed32_config_hash(view.view_id, repo)
    bank_id = hashlib.sha256(json.dumps({
        "dataset_id": view.dataset_id, "view_id": view.view_id,
        "fixed32_version": FIXED32_VERSION, "fixed32_config_hash": cfg_hash,
        "decision_hash": view.decision_hash, "full_hash": view.full_hash},
        sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return Fixed32CandidateBank(
        dataset_id=view.dataset_id, view_id=view.view_id,
        fixed32_version=FIXED32_VERSION, fixed32_config_hash=cfg_hash,
        decision_hash=view.decision_hash, full_hash=view.full_hash,
        candidate_count=len(cands), candidates=tuple(cands), bank_id=bank_id,
        provenance=MappingProxyType({
            "authoritative_source": ("experiments/external_baselines/Unified_MKR/"
                                     "unified_mkr/configuration.py::"
                                     "resolve_view_configurations"),
            "composition": fixed32_composition(view.view_id, repo),
            "independent_of_method_search": True,
            "label_free": True, "true_k_forced": False}))


# ------------------------------------------------------- metric evaluation
METRIC_EVALUATION_SEMANTICS = "fixed32_metric_evaluation_v2"

# The registry declares three implementation kinds. The v1 harness branched on
# modern_comparator only and sent everything else to the ML2DAC engine, which
# (a) could not resolve the two clustopt_registry metrics outside the ML2DAC
# 61-class collection, and (b) applied float() to the MetricResult dataclass the
# modern comparators return. Both faults were at the call site; no metric
# implementation is changed here.
IMPLEMENTATION_KINDS = ("clustopt_registry", "ml2dac_cvi_collection",
                        "modern_comparator")

# CVNN is normalised by max(sep) and max(comp) ACROSS a compared set, so it has
# no single-candidate value. It is evaluated once per bank over that bank's
# candidate partitions, which is exactly the comparison set fpc assumes.
BANK_LEVEL_METRICS = ("cvnn",)


class _ClustOptRegistry:
    """Lazy handle on the authoritative ClustOpt metric registry."""

    _reg = None

    @classmethod
    def get(cls, repo: Path):
        if cls._reg is None:
            import sys
            for extra in (str(repo), str(repo / "models"), str(repo / "models/ClustOpt")):
                if extra not in sys.path:
                    sys.path.insert(0, extra)
            from cluster_validity_indices.metrics.metrics_registry import (
                METRIC_REGISTRY)
            cls._reg = METRIC_REGISTRY
        return cls._reg


def _space_array(view, metric_space: str):
    return (view.full_for_algorithm() if metric_space == "full"
            else view.decision_for_algorithm())


def _bank_level_cvnn(bank, view, entry, repo):
    """One CVNN pass over the bank. Returns {candidate_index: (value, status, fk)}.

    Degenerate partitions stay in the comparison set: fpc includes them when it
    takes max(sep) and max(comp), so dropping them would change the normaliser
    and therefore every other candidate's value. The frozen implementation
    judges their validity separately.
    """
    from Independent_Domain_Benchmark.metric_validation import external_cvis as MC
    idx = [c.candidate_index for c in bank.candidates
           if c.status == "valid" and c.labels is not None]
    labs = [np.asarray(c.labels) for c in bank.candidates
            if c.status == "valid" and c.labels is not None]
    out: Dict[int, Any] = {}
    if not labs:
        return out
    X = np.asarray(_space_array(view, entry["metric_space"]), dtype=float)
    try:
        results = MC.cvnn_index(X, labs)
    except Exception as exc:  # noqa: BLE001
        fk = "%s:%s" % (Fixed32Failure.METRIC_EVALUATION_FAILED, type(exc).__name__)
        return {i: (None, "invalid", fk) for i in idx}
    for i, res in zip(idx, results):
        if getattr(res, "valid", False) and res.value is not None \
                and np.isfinite(float(res.value)):
            out[i] = (float(res.value), "valid", None)
        else:
            reason = getattr(res, "invalid_reason", "") or "non_finite_value"
            out[i] = (None, "invalid", "%s:%s"
                      % (Fixed32Failure.METRIC_EVALUATION_FAILED, reason))
    return out


def _evaluate_one(entry, mid, labels, view, eng, repo):
    """Dispatch a single metric on its DECLARED implementation kind."""
    impl = entry["implementation"]
    if impl == "modern_comparator":
        from Independent_Domain_Benchmark.metric_validation import external_cvis as MC
        fn = getattr(MC, mid)
        res = fn(np.asarray(_space_array(view, entry["metric_space"]), dtype=float),
                 np.asarray(labels))
        # the modern comparators return a MetricResult, never a bare float
        if not getattr(res, "valid", False) or res.value is None:
            reason = getattr(res, "invalid_reason", "") or "invalid"
            raise _MetricInvalid(reason)
        return float(res.value)
    if impl == "clustopt_registry":
        obj = _ClustOptRegistry.get(repo)[mid]
        return float(obj.evaluate(
            np.asarray(_space_array(view, entry["metric_space"]), dtype=float),
            np.asarray(labels)))
    if impl == "ml2dac_cvi_collection":
        from Independent_Domain_Benchmark.methods import ml2dac_scoring as MS
        return float(MS.canonical_raw_value(mid, labels, engine=eng, repo=repo))
    raise ValueError("unknown implementation kind %r for %s" % (impl, mid))


class _MetricInvalid(RuntimeError):
    """The implementation itself reported the value invalid, with a reason."""


def evaluate_bank_metrics(bank: Fixed32CandidateBank, view, registry: Mapping[str, Any],
                          *, include_diagnostics: bool = True,
                          only_metrics: Optional[Sequence[str]] = None,
                          repo: Path = _REPO) -> List[Dict[str, Any]]:
    """Evaluate CVIs on ONE bank, each through its declared implementation kind.

    Uses only ``X_decision``, ``X_full`` and the candidate labels. Ground-truth
    labels are not reachable from here.

    ``only_metrics`` restricts the evaluation to named canonical ids, which is
    what the targeted metric-dispatch repair uses: the untouched metrics keep
    their original values rather than being recomputed.
    """
    from Independent_Domain_Benchmark.methods import ml2dac_scoring as MS

    entries = list(registry["entries"])
    if include_diagnostics:
        entries += list(registry.get("diagnostics_entries", []))
    if only_metrics is not None:
        want = set(only_metrics)
        entries = [e for e in entries if e["canonical_metric_id"] in want]

    need_engine = any(e["implementation"] == "ml2dac_cvi_collection" for e in entries)
    eng = (MS._engine(view.decision_for_algorithm(), view.full_for_algorithm(), repo)
           if need_engine else None)

    bank_level = {}
    for e in entries:
        if e["canonical_metric_id"] in BANK_LEVEL_METRICS:
            bank_level[e["canonical_metric_id"]] = _bank_level_cvnn(
                bank, view, e, repo)

    rows: List[Dict[str, Any]] = []
    for c in bank.candidates:
        if c.status != "valid" or c.labels is None:
            continue
        for e in entries:
            mid = e["canonical_metric_id"]
            if mid in bank_level:
                val, status, fk = bank_level[mid].get(
                    c.candidate_index,
                    (None, "invalid", "%s:candidate_absent_from_bank_pass"
                     % Fixed32Failure.METRIC_EVALUATION_FAILED))
            else:
                try:
                    val = _evaluate_one(e, mid, c.labels, view, eng, repo)
                    status, fk = ("valid" if np.isfinite(val) else "invalid"), None
                    if not np.isfinite(val):
                        fk = Fixed32Failure.METRIC_EVALUATION_FAILED
                        val = None
                except _MetricInvalid as exc:
                    val, status = None, "invalid"
                    fk = "%s:%s" % (Fixed32Failure.METRIC_EVALUATION_FAILED, exc)
                except Exception as exc:  # noqa: BLE001
                    val, status = None, "invalid"
                    fk = "%s:%s" % (Fixed32Failure.METRIC_EVALUATION_FAILED,
                                    type(exc).__name__)
            rows.append({
                "dataset_id": bank.dataset_id, "view_id": bank.view_id,
                "fixed32_bank_id": bank.bank_id, "candidate_id": c.candidate_id,
                "candidate_index": c.candidate_index, "metric_id": mid,
                "metric_group": e["group"], "primary": e["primary"],
                "orientation": e["orientation"],
                "evaluation_space": e["metric_space"],
                "status": status, "failure_kind": fk,
                "raw_metric_value": val,
                "implementation": e["implementation"]})
    return rows
