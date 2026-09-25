"""AutoClust physical candidate-trace execution — label-free, immutable, shared.

This is the PHYSICAL layer only: it turns ``X`` into the candidate clusterings
that A0–A3 will later score. It does not compute CVI vectors, does not call an
ARI predictor and does not select a final candidate; that is Stage 4B-3A2.

**One trace, four arms.** Stage 4B-3A1 validated that A0–A3 produce byte-identical
candidate traces under the amended common-seed protocol (15 traces, 7/7
algorithms, 3000 candidate executions, zero mismatches of any kind). The four
arms therefore share one physical trace and differ only in scoring. The trace id
deliberately does not depend on the METHOD_ID, so requesting it from an A0 or an
A3 context returns the same object identity.

**Labels never enter.** The public API takes ``X`` and protocol identifiers. There
is no ground-truth parameter to pass, not even ``y=None``: the historical
``run_view`` wrote ARI/NMI/AMI/FMI into every candidate record, and none of that
reaches here.

**No repository membership.** The historical path required the dataset to exist in
the master meta-feature parquet. An external dataset does not, so the seven
landmark features are computed from ``X`` directly with the frozen extractor.

Every scientific component is called, not reimplemented: ``extract_autoclust``,
the frozen algorithm selector, the production seed resolver, the authoritative
search-space construction, the CONSTRAINED sampler and ``run_configuration``.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_REPO = _ROOT.parents[1]

#: the seven frozen landmark features, in artifact order
LANDMARK_FEATURES: Tuple[str, ...] = (
    "autoclust_CH", "autoclust_DBI", "autoclust_SIL", "autoclust_DBCV",
    "autoclust_DI", "autoclust_CJI", "autoclust_COP")
#: the four scientific arms share this physical trace
ARM_METHOD_IDS: Tuple[str, ...] = (
    "IDB_AutoClust_A0_original_cvis",
    "IDB_AutoClust_A1_original_plus_established",
    "IDB_AutoClust_A2_original_plus_new46",
    "IDB_AutoClust_A3_extended_cvis")
#: the selector is scientifically identical across arms (df5af99); A0's artifact
#: is the canonical one to load
SELECTOR_VARIANT_DIR = ("experiments/external_baselines/Unified_MKR/derived/"
                        "autoclust_split1/original_cvis")
SEARCH_CONFIG_ROOT = "models/ClustOpt/configs/utilities_maps"
SEARCH_SPACE_FINGERPRINT = "8ddf86c79888ebe9"
SEED_IDENTITY = "autoclust_shared_candidate_slate"


class AutoClustFailure:
    """Physical-layer failure kinds. No silent fallback of any sort."""

    ARTIFACT_MISSING = "ARTIFACT_MISSING"
    ARTIFACT_HASH_MISMATCH = "ARTIFACT_HASH_MISMATCH"
    FEATURE_SCHEMA_MISMATCH = "FEATURE_SCHEMA_MISMATCH"
    LANDMARK_FEATURE_FAILED = "LANDMARK_FEATURE_FAILED"
    ALGORITHM_SELECTOR_FAILED = "ALGORITHM_SELECTOR_FAILED"
    SEARCH_SPACE_MISMATCH = "SEARCH_SPACE_MISMATCH"
    SEED_IDENTITY_MISMATCH = "SEED_IDENTITY_MISMATCH"
    CONFIG_GENERATION_FAILED = "CONFIG_GENERATION_FAILED"
    CANDIDATE_EXECUTION_FAILED = "CANDIDATE_EXECUTION_FAILED"
    NO_VALID_CANDIDATE = "NO_VALID_CANDIDATE"
    ALL = ("ARTIFACT_MISSING", "ARTIFACT_HASH_MISMATCH",
           "FEATURE_SCHEMA_MISMATCH", "LANDMARK_FEATURE_FAILED",
           "ALGORITHM_SELECTOR_FAILED", "SEARCH_SPACE_MISMATCH",
           "SEED_IDENTITY_MISMATCH", "CONFIG_GENERATION_FAILED",
           "CANDIDATE_EXECUTION_FAILED", "NO_VALID_CANDIDATE")


class AutoClustTraceError(RuntimeError):
    """Whole-trace failure. Candidate-level failures live IN the trace instead."""

    def __init__(self, kind: str, detail: str = "") -> None:
        super().__init__("%s: %s" % (kind, detail) if detail else kind)
        self.kind = kind
        self.detail = detail


# ------------------------------------------------------------------ immutability
def _freeze_labels(labels) -> Optional[np.ndarray]:
    """A private, write-protected copy. Callers cannot mutate the trace's labels."""
    if labels is None:
        return None
    a = np.array(labels, dtype=np.int64, copy=True)
    a.setflags(write=False)
    return a


def _freeze_config(cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    """Read-only view over a private copy, so the caller cannot edit the config."""
    return MappingProxyType(dict(cfg))


def _raw_labels_hash(labels) -> Optional[str]:
    if labels is None:
        return None
    return hashlib.sha256(np.asarray(labels, dtype=np.int64).tobytes()
                          ).hexdigest()[:16]


@dataclass(frozen=True)
class AutoClustCandidate:
    """One evaluated clustering configuration. Failures are kept, never dropped."""

    candidate_index: int
    algorithm: str
    configuration: Mapping[str, Any]
    k: Optional[int]
    random_state: Optional[int]
    status: str                       # "valid" | "invalid"
    failure_kind: Optional[str]
    raw_labels_hash: Optional[str]
    canonical_partition_hash: Optional[str]
    labels: Optional[np.ndarray]      # write-protected

    def identity(self) -> Tuple:
        return (self.candidate_index, self.algorithm,
                json.dumps(dict(self.configuration), sort_keys=True),
                self.status, self.raw_labels_hash,
                self.canonical_partition_hash)


@dataclass(frozen=True)
class AutoClustCandidateTrace:
    """The immutable physical trace shared by A0–A3.

    Contains no ARI, NMI, AMI, FMI, true K or evaluation label — a scorer that
    could see any of those would not be scoring blind.
    """

    trace_id: str
    dataset_id: str
    view_id: str
    selected_algorithm: str
    landmark_features: Mapping[str, float]
    landmark_features_hash: str
    selector_artifact_hash: str
    seed_identity: str
    resolved_seed: int
    protocol_hash: str
    search_space_fingerprint: str
    candidate_count: int
    candidates: Tuple[AutoClustCandidate, ...]
    budget: int
    shared_by: Tuple[str, ...] = field(default=ARM_METHOD_IDS)

    # -- read-only consumer interface for the Stage-4B-3A2 scorers
    def get_candidates_for_scoring(self) -> Tuple[AutoClustCandidate, ...]:
        return self.candidates

    def availability_mask(self) -> Tuple[bool, ...]:
        """Which candidates are scoreable. Identical for all four arms."""
        return tuple(c.status == "valid" for c in self.candidates)

    def content_hash(self) -> str:
        """Stable hash over everything a consumer must not change."""
        payload = json.dumps({
            "trace_id": self.trace_id, "dataset_id": self.dataset_id,
            "view_id": self.view_id, "selected_algorithm": self.selected_algorithm,
            "landmark_features_hash": self.landmark_features_hash,
            "resolved_seed": self.resolved_seed,
            "search_space_fingerprint": self.search_space_fingerprint,
            "candidates": [list(c.identity()) for c in self.candidates],
        }, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# ------------------------------------------------------------------- execution
def _selector_and_hash(repo: Path):
    import joblib
    p = repo / SELECTOR_VARIANT_DIR / "algorithm_selector.pkl"
    if not p.is_file():
        raise AutoClustTraceError(AutoClustFailure.ARTIFACT_MISSING, str(p))
    h = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    return joblib.load(p), h


def _protocol_hash(repo: Path) -> str:
    p = repo / "models/Independent_Domain_Benchmark/configs/benchmark_protocol.json"
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()[:16]


def build_autoclust_candidate_trace(
    X: np.ndarray, dataset_id: str, view_id: str, *,
    repo: Path = _REPO, budget: Optional[int] = None,
) -> AutoClustCandidateTrace:
    """Execute one physical AutoClust candidate trace from ``X`` alone.

    There is deliberately no ground-truth parameter and no repository-membership
    requirement: the landmark features come from ``X`` via the frozen extractor.
    """
    import sys
    mkr = str(repo / "experiments/external_baselines/Unified_MKR")
    if mkr not in sys.path:
        sys.path.insert(0, mkr)

    from unified_mkr import metafeatures as MF
    from unified_mkr.clustering import run_configuration
    from unified_mkr.configuration import ConfigSpec, resolve_view_configurations
    from models.ClustOpt.search_space.Clustering_Search_Space import (
        ClusteringSearchSpace)
    from models.ClustOpt.configs.experiments.generate_experiment_configs import (
        build_search_space)
    #: the CONSTRAINED sampler, never the native one -- the native path reaches
    #: _sample_params, which draws random_state = rng.integers(0, 10000) per
    #: candidate and would silently destroy the shared-slate property
    from experiments.external_baselines.ML2DAC.indomain.search import (
        sample_optimizer_configs_constrained)
    from Independent_Domain_Benchmark.methods.clustopt_trace import partition_hash
    from Independent_Domain_Benchmark.search import protocol as PROTO

    X = np.asarray(X, dtype=float)

    # ---- 1. landmark features, from X only
    try:
        res = MF.extract_autoclust(X, repo_root=str(repo))
        if res.status != "success":
            raise AutoClustTraceError(AutoClustFailure.LANDMARK_FEATURE_FAILED,
                                      res.error or res.status)
        feats = {k: float(res.features[k]) for k in LANDMARK_FEATURES}
    except AutoClustTraceError:
        raise
    except Exception as exc:
        raise AutoClustTraceError(AutoClustFailure.LANDMARK_FEATURE_FAILED,
                                  "%s: %s" % (type(exc).__name__, str(exc)[:180]))
    fvec = np.array([[feats[k] for k in LANDMARK_FEATURES]], dtype=float)
    fhash = hashlib.sha256(fvec.tobytes()).hexdigest()[:16]

    # ---- 2. frozen algorithm selector (identical across arms)
    selector, sel_hash = _selector_and_hash(repo)
    try:
        algorithm = str(selector.predict(fvec)[0])
    except Exception as exc:
        raise AutoClustTraceError(AutoClustFailure.ALGORITHM_SELECTOR_FAILED,
                                  "%s: %s" % (type(exc).__name__, str(exc)[:180]))

    # ---- 3. the production seed, identical for all four arms by protocol
    seeds = {m: PROTO.derive_seed(dataset_id, view_id, m) for m in ARM_METHOD_IDS}
    if len(set(seeds.values())) != 1:
        raise AutoClustTraceError(
            AutoClustFailure.SEED_IDENTITY_MISMATCH,
            "A0-A3 resolved %d distinct seeds" % len(set(seeds.values())))
    seed = int(next(iter(seeds.values())))
    if PROTO.seed_identity(ARM_METHOD_IDS[0]) != SEED_IDENTITY:
        raise AutoClustTraceError(AutoClustFailure.SEED_IDENTITY_MISMATCH,
                                  PROTO.seed_identity(ARM_METHOD_IDS[0]))

    # ---- 4. authoritative search-space objects (two, kept distinct)
    try:
        space = build_search_space(view_id)                     # sampler constraint
        ss = ClusteringSearchSpace.from_dict(                   # instantiation
            resolve_view_configurations(str(repo / SEARCH_CONFIG_ROOT),
                                        view_id).search_space_config)
    except Exception as exc:
        raise AutoClustTraceError(AutoClustFailure.SEARCH_SPACE_MISMATCH,
                                  "%s: %s" % (type(exc).__name__, str(exc)[:180]))

    n = int(budget if budget is not None else PROTO.BUDGET_EVALUATIONS)
    try:
        cfgs = sample_optimizer_configs_constrained(
            [algorithm], space, n, (PROTO.K_MIN, PROTO.K_MAX), seed)
    except Exception as exc:
        raise AutoClustTraceError(AutoClustFailure.CONFIG_GENERATION_FAILED,
                                  "%s: %s" % (type(exc).__name__, str(exc)[:180]))
    if len(cfgs) != n:
        raise AutoClustTraceError(
            AutoClustFailure.CONFIG_GENERATION_FAILED,
            "budget %d but sampler produced %d" % (n, len(cfgs)))

    # ---- 5. execute every candidate; a failure is RECORDED, never resampled
    cands = []
    for i, c in enumerate(cfgs):
        params = dict(c["hyperparameters"])
        try:
            r = run_configuration(
                ss, ConfigSpec(algorithm=c["algorithm"], params=dict(params),
                               original_config_id=""), X)
            labels = _freeze_labels(getattr(r, "labels", None))
            status = "valid" if r.valid_result else "invalid"
            fk = (None if r.valid_result
                  else (r.failure_reason or AutoClustFailure.CANDIDATE_EXECUTION_FAILED))
        except Exception as exc:
            labels, status = None, "invalid"
            fk = "%s:%s" % (AutoClustFailure.CANDIDATE_EXECUTION_FAILED,
                            type(exc).__name__)
        cands.append(AutoClustCandidate(
            candidate_index=i, algorithm=str(c["algorithm"]),
            configuration=_freeze_config(params),
            k=next((int(params[p]) for p in ("n_clusters", "n_components")
                    if p in params), None),
            random_state=(int(params["random_state"])
                          if "random_state" in params else None),
            status=status, failure_kind=fk,
            raw_labels_hash=_raw_labels_hash(labels),
            canonical_partition_hash=(None if labels is None
                                      else partition_hash(labels)),
            labels=labels))

    if not any(c.status == "valid" for c in cands):
        raise AutoClustTraceError(AutoClustFailure.NO_VALID_CANDIDATE,
                                  "all %d candidates failed" % len(cands))

    # ---- 6. trace identity: physical, never arm-specific
    ident = json.dumps({
        "dataset_id": dataset_id, "view_id": view_id,
        "seed_identity": SEED_IDENTITY, "resolved_seed": seed,
        "search_space_fingerprint": SEARCH_SPACE_FINGERPRINT,
        "selected_algorithm": algorithm,
        "configs": [[c["algorithm"], json.dumps(c["hyperparameters"], sort_keys=True)]
                    for c in cfgs]}, sort_keys=True)
    trace_id = hashlib.sha256(ident.encode("utf-8")).hexdigest()[:16]

    return AutoClustCandidateTrace(
        trace_id=trace_id, dataset_id=dataset_id, view_id=view_id,
        selected_algorithm=algorithm,
        landmark_features=MappingProxyType(dict(feats)),
        landmark_features_hash=fhash, selector_artifact_hash=sel_hash,
        seed_identity=SEED_IDENTITY, resolved_seed=seed,
        protocol_hash=_protocol_hash(repo),
        search_space_fingerprint=SEARCH_SPACE_FINGERPRINT,
        candidate_count=len(cands), candidates=tuple(cands), budget=n)
