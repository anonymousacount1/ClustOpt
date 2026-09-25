"""ML2DAC physical candidate-trace execution — label-free, immutable, shared.

This is the PHYSICAL layer only: it turns ``X`` into the candidate clusterings
that M0–M3 will later score. It does not load a CVI classifier, does not predict
a CVI, does not orient or score anything and does not select a final candidate;
that is the next substage.

**No classifier in the physical layer.** Stage 4B-3B1 proved that the ML2DAC
candidate chain — 52 meta-features, nearest neighbour, warmstarts,
``ws_algorithms``, constrained sampling — is arm-independent, and that
``selected_cvi = mkr.predict_cvi(feats)`` is consumed only by later scoring. So
``InDomainMKR`` is deliberately NOT used here: it eagerly loads
``cvi_classifier.pkl`` and requires the dataset to exist in the master
meta-feature parquet. This module loads only the common generation state
(imputer, KDTree, warmstart repository) and asserts their frozen hashes.

**One trace, four arms.** Under the pre-run amendment of 2026-09-10 the four
METHOD_IDs share the seed identity ``ml2dac_shared_candidate_slate``. The trace
id deliberately does not depend on the METHOD_ID, so requesting it from an M0 or
an M3 context returns the same identity.

**Authoritative meta-features.** Extraction runs in the isolated ``.mfenv``
interpreter through ``unified_mkr/pymfe_worker.py::_extract`` — the frozen path
that produced the master parquet — never through the
``unified_mkr/metafeatures.py::extract_ml2dac`` fallback.

**Labels never enter.** The public API takes ``X`` and protocol identifiers.
There is no ground-truth parameter, not even ``y=None``. The historical
``run_view`` wrote ARI/NMI/AMI/FMI into every candidate record; none of that
reaches here. The warmstart repository's training-time ARI rank is carried only
as an integer ``warmstart_rank`` — frozen artifact provenance about the training
split, never a metric of the dataset being evaluated — and the training ARI
value itself is dropped.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_REPO = _ROOT.parents[1]

#: the four scientific arms share this physical trace
ARM_METHOD_IDS: Tuple[str, ...] = (
    "IDB_ML2DAC_M0_original_cvis",
    "IDB_ML2DAC_M1_original_plus_established",
    "IDB_ML2DAC_M2_original_plus_new46",
    "IDB_ML2DAC_M3_extended_cvis")
SEED_IDENTITY = "ml2dac_shared_candidate_slate"
SEARCH_SPACE_FINGERPRINT = "8ddf86c79888ebe9"
SEARCH_CONFIG_ROOT = "models/ClustOpt/configs/utilities_maps"

#: generation state is arm-independent (Stage 4B-3B1); M0's directory is the
#: canonical copy. imputer.pkl and nearest_neighbor_index.pkl are byte-identical
#: across all four arms; warmstart_repository.parquet is content-identical.
COMMON_STATE_DIR = ("experiments/external_baselines/Unified_MKR/derived/"
                    "ml2dac_split1/original_cvis")
IMPUTER_HASH = "953dc6215b21535c"
NN_INDEX_HASH = "d06053c3abf4168d"
WARMSTART_SEMANTIC_HASH = "bb3a1306b4562217"

#: frozen classifier input schema (Stage 4B-3B1, configs/ml2dac_metafeature_schema.json)
META_FEATURE_SCHEMA_HASH = "f33a72e081fff983"
N_META_FEATURES = 52

#: frozen extraction environment; the master parquet was produced by exactly this
FROZEN_MF_ENV: Mapping[str, str] = MappingProxyType({
    "python": "3.12.2", "pymfe": "0.4.4", "numpy": "2.4.6", "pandas": "3.0.3"})
MFENV_DIR = "experiments/external_baselines/Unified_MKR/.mfenv"
AUTHORITATIVE_WORKER = ("experiments/external_baselines/Unified_MKR/unified_mkr/"
                        "pymfe_worker.py")

#: matched ML2DAC budget semantics, verified against 216/216 historical records:
#: 25 warmstarts + 25 sampled configurations = 50 evaluations.
WARMSTART_FRACTION = 0.5

#: the algorithms ML2DAC can actually reach: the warmstart repository's
#: algorithms intersected with the ClustOpt phaseE2 space. optics, spectral and
#: the hdbscan_constrained-* variants exist in the space but no ML2DAC warmstart
#: names them, so they are unreachable and deliberately absent.
REACHABLE_ALGORITHMS: Tuple[str, ...] = (
    "agglomerative", "birch", "dbscan", "gmm", "hdbscan", "kmeans",
    "minibatch_kmeans")


class ML2DACFailure:
    """Physical-layer failure kinds. No silent fallback of any sort."""

    METAFEATURE_ENV_MISMATCH = "METAFEATURE_ENV_MISMATCH"
    METAFEATURE_FAILED = "METAFEATURE_FAILED"
    METAFEATURE_SCHEMA_MISMATCH = "METAFEATURE_SCHEMA_MISMATCH"
    ARTIFACT_MISSING = "ARTIFACT_MISSING"
    ARTIFACT_HASH_MISMATCH = "ARTIFACT_HASH_MISMATCH"
    NEIGHBOUR_LOOKUP_FAILED = "NEIGHBOUR_LOOKUP_FAILED"
    WARMSTART_LOOKUP_FAILED = "WARMSTART_LOOKUP_FAILED"
    SEARCH_SPACE_MISMATCH = "SEARCH_SPACE_MISMATCH"
    SEED_IDENTITY_MISMATCH = "SEED_IDENTITY_MISMATCH"
    CONFIG_GENERATION_FAILED = "CONFIG_GENERATION_FAILED"
    CANDIDATE_EXECUTION_FAILED = "CANDIDATE_EXECUTION_FAILED"
    NO_VALID_CANDIDATE = "NO_VALID_CANDIDATE"
    ALL = ("METAFEATURE_ENV_MISMATCH", "METAFEATURE_FAILED",
           "METAFEATURE_SCHEMA_MISMATCH", "ARTIFACT_MISSING",
           "ARTIFACT_HASH_MISMATCH", "NEIGHBOUR_LOOKUP_FAILED",
           "WARMSTART_LOOKUP_FAILED", "SEARCH_SPACE_MISMATCH",
           "SEED_IDENTITY_MISMATCH", "CONFIG_GENERATION_FAILED",
           "CANDIDATE_EXECUTION_FAILED", "NO_VALID_CANDIDATE")


class ML2DACTraceError(RuntimeError):
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


def _freeze_vector(v) -> np.ndarray:
    a = np.array(v, dtype=float, copy=True)
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
class ML2DACWarmstart:
    """One frozen warmstart, projected onto the ClustOpt space."""

    warmstart_index: int
    algorithm: str
    configuration: Mapping[str, Any]
    warmstart_rank: int          # rank inside the frozen TRAINING repository

    def identity(self) -> Tuple:
        return (self.warmstart_index, self.algorithm,
                json.dumps(dict(self.configuration), sort_keys=True),
                self.warmstart_rank)


@dataclass(frozen=True)
class ML2DACCandidate:
    """One evaluated clustering configuration. Failures are kept, never dropped."""

    candidate_index: int
    origin: str                       # "warmstart" | "sampled"
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
        return (self.candidate_index, self.origin, self.algorithm,
                json.dumps(dict(self.configuration), sort_keys=True),
                self.status, self.raw_labels_hash,
                self.canonical_partition_hash)


@dataclass(frozen=True)
class ML2DACCandidateTrace:
    """The immutable physical trace shared by M0–M3.

    Contains no predicted CVI, no classifier hash, no candidate CVI score, no
    final selection, and no ARI/NMI/AMI/FMI/true K — a scorer that could see any
    of those would not be scoring blind.
    """

    trace_id: str
    dataset_id: str
    view_id: str

    meta_feature_schema_hash: str
    meta_feature_names: Tuple[str, ...]
    meta_feature_vector: np.ndarray          # write-protected
    meta_feature_vector_hash: str
    meta_feature_env: Mapping[str, str]

    nearest_neighbor_id: str
    nearest_neighbor_distance: float
    imputer_artifact_hash: str
    nn_index_artifact_hash: str
    warmstart_repository_semantic_hash: str

    warmstarts: Tuple[ML2DACWarmstart, ...]
    warmstarts_hash: str
    warmstarts_skipped_due_to_k_range: int
    warmstarts_dropped_not_in_space: int
    ws_algorithms: Tuple[str, ...]
    avoid_set_hash: str

    seed_identity: str
    resolved_seed: int
    protocol_hash: str
    search_space_fingerprint: str

    candidate_count: int
    candidates: Tuple[ML2DACCandidate, ...]
    budget: int
    n_warmstart_slots: int
    n_sampled_slots: int
    shared_by: Tuple[str, ...] = field(default=ARM_METHOD_IDS)

    # -- read-only consumer interface for the next substage's scorers
    def get_candidates_for_scoring(self) -> Tuple[ML2DACCandidate, ...]:
        return self.candidates

    def availability_mask(self) -> Tuple[bool, ...]:
        """Which candidates are scoreable. Identical for all four arms."""
        return tuple(c.status == "valid" for c in self.candidates)

    def content_hash(self) -> str:
        """Stable hash over everything a consumer must not change."""
        payload = json.dumps({
            "trace_id": self.trace_id, "dataset_id": self.dataset_id,
            "view_id": self.view_id,
            "meta_feature_vector_hash": self.meta_feature_vector_hash,
            "nearest_neighbor_id": self.nearest_neighbor_id,
            "warmstarts_hash": self.warmstarts_hash,
            "ws_algorithms": list(self.ws_algorithms),
            "resolved_seed": self.resolved_seed,
            "search_space_fingerprint": self.search_space_fingerprint,
            "candidates": [list(c.identity()) for c in self.candidates],
        }, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# ------------------------------------------------------- authoritative features
def resolve_mfenv_python(repo: Path = _REPO) -> Path:
    """The isolated interpreter that produced the frozen master parquet."""
    base = repo / MFENV_DIR
    for c in (base / "Scripts" / "python.exe", base / "bin" / "python"):
        if c.is_file():
            return c
    raise ML2DACTraceError(ML2DACFailure.METAFEATURE_ENV_MISMATCH,
                           "no .mfenv interpreter under %s" % base)


def extract_authoritative_metafeatures(
    X: np.ndarray, *, repo: Path = _REPO, check_env: bool = True,
) -> Tuple[List[str], np.ndarray, Dict[str, str]]:
    """Run ``pymfe_worker._extract`` in ``.mfenv``. Never the fallback extractor."""
    py = resolve_mfenv_python(repo)
    worker = repo / AUTHORITATIVE_WORKER
    if not worker.is_file():
        raise ML2DACTraceError(ML2DACFailure.ARTIFACT_MISSING, str(worker))
    driver = Path(__file__).resolve().parent / "_mfenv_extract_worker.py"
    with tempfile.TemporaryDirectory() as td:
        xp, op = Path(td) / "X.npy", Path(td) / "out.json"
        np.save(xp, np.asarray(X, dtype=float))
        proc = subprocess.run([str(py), str(driver), str(worker), str(xp), str(op)],
                              capture_output=True, text=True)
        if not op.is_file():
            raise ML2DACTraceError(
                ML2DACFailure.METAFEATURE_FAILED,
                (proc.stderr or proc.stdout or "no output")[-300:])
        res = json.loads(op.read_text(encoding="utf-8"))
    env = dict(res.get("env") or {})
    if check_env:
        bad = {k: (env.get(k), v) for k, v in FROZEN_MF_ENV.items() if env.get(k) != v}
        if bad:
            raise ML2DACTraceError(ML2DACFailure.METAFEATURE_ENV_MISMATCH, json.dumps(bad))
    if res.get("status") != "success":
        raise ML2DACTraceError(ML2DACFailure.METAFEATURE_FAILED,
                               str(res.get("error"))[:300])
    names = ["ml2dac_%s" % n for n in res["names"]]
    return names, np.asarray(res["values"], dtype=float), env


# ------------------------------------------------------------- common state
def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def _protocol_hash(repo: Path) -> str:
    p = repo / "models/Independent_Domain_Benchmark/configs/benchmark_protocol.json"
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()[:16]


def load_common_generation_state(repo: Path = _REPO) -> Dict[str, Any]:
    """Imputer + KDTree + warmstart repository. The CVI classifier is NOT loaded."""
    import joblib
    import pandas as pd

    d = repo / COMMON_STATE_DIR
    imp_p, nn_p, wr_p = (d / "imputer.pkl", d / "nearest_neighbor_index.pkl",
                         d / "warmstart_repository.parquet")
    for p in (imp_p, nn_p, wr_p):
        if not p.is_file():
            raise ML2DACTraceError(ML2DACFailure.ARTIFACT_MISSING, str(p))
    ih, nh = _sha(imp_p), _sha(nn_p)
    if ih != IMPUTER_HASH or nh != NN_INDEX_HASH:
        raise ML2DACTraceError(ML2DACFailure.ARTIFACT_HASH_MISMATCH,
                               "imputer %s nn %s" % (ih, nh))
    nn = joblib.load(nn_p)
    wr = pd.read_parquet(wr_p).sort_values(["record_id", "rank_by_ari"],
                                           kind="mergesort")
    return {"imputer": joblib.load(imp_p), "kdtree": nn["kdtree"],
            "record_ids": list(nn["train_record_ids"]),
            "feature_columns": list(nn["feature_columns"]),
            "warmstarts": {rid: g for rid, g in wr.groupby("record_id", sort=False)},
            "imputer_hash": ih, "nn_index_hash": nh}


def _ranked_warmstarts(state: Dict[str, Any], nn_record_id: str) -> List[dict]:
    """Frozen ranked warmstarts of the neighbour. The training ARI value is dropped."""
    g = state["warmstarts"].get(nn_record_id)
    if g is None:
        raise ML2DACTraceError(ML2DACFailure.WARMSTART_LOOKUP_FAILED, nn_record_id)
    return [{"config_id": r.config_id, "algorithm": r.algorithm,
             "hyperparameters_json": r.hyperparameters_json,
             "rank_by_ari": int(r.rank_by_ari)}
            for r in g.itertuples(index=False)]


# ------------------------------------------------------------------- execution
def build_ml2dac_candidate_trace(
    X: np.ndarray, dataset_id: str, view_id: str, *,
    repo: Path = _REPO, budget: Optional[int] = None,
) -> ML2DACCandidateTrace:
    """Execute one physical ML2DAC candidate trace from ``X`` alone.

    There is deliberately no ground-truth parameter and no repository-membership
    requirement: the 52 meta-features come from ``X`` via the authoritative
    isolated extractor, so an external dataset needs no master-parquet row.

    ``budget`` is a NON-SCIENTIFIC test override only; production uses the frozen
    protocol budget (50 = 25 warmstart slots + 25 sampled slots).
    """
    mkr_path = str(repo / "experiments/external_baselines/Unified_MKR")
    if mkr_path not in sys.path:
        sys.path.insert(0, mkr_path)
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

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
        sample_optimizer_configs_constrained, select_warmstarts, snap_to_space)
    from Independent_Domain_Benchmark.methods.clustopt_trace import partition_hash
    from Independent_Domain_Benchmark.search import protocol as PROTO

    X = np.asarray(X, dtype=float)

    # ---- 1. authoritative 52 meta-features, from X only, in the frozen env
    names, values, env = extract_authoritative_metafeatures(X, repo=repo)
    state = load_common_generation_state(repo)
    cols = state["feature_columns"]
    if len(names) != N_META_FEATURES or list(names) != list(cols):
        raise ML2DACTraceError(
            ML2DACFailure.METAFEATURE_SCHEMA_MISMATCH,
            "extractor emitted %d features; classifier state expects %d"
            % (len(names), len(cols)))
    schema_hash = hashlib.sha256("\n".join(cols).encode("utf-8")).hexdigest()[:16]
    if schema_hash != META_FEATURE_SCHEMA_HASH:
        raise ML2DACTraceError(ML2DACFailure.METAFEATURE_SCHEMA_MISMATCH, schema_hash)
    fvec = np.asarray(values, dtype=float).reshape(1, -1)
    fhash = hashlib.sha256(fvec.tobytes()).hexdigest()[:16]

    # ---- 2. common nearest neighbour (imputer -> KDTree), no classifier involved
    try:
        Xi = state["imputer"].transform(fvec)
        dist, idx = state["kdtree"].query(Xi, k=1)
        nn_id = str(state["record_ids"][int(idx[0][0])])
        nn_dist = float(dist[0][0])
    except Exception as exc:  # noqa: BLE001
        raise ML2DACTraceError(ML2DACFailure.NEIGHBOUR_LOOKUP_FAILED,
                               "%s: %s" % (type(exc).__name__, str(exc)[:180]))

    # ---- 3. the production seed, identical for all four arms by protocol
    seeds = {m: PROTO.derive_seed(dataset_id, view_id, m) for m in ARM_METHOD_IDS}
    if len(set(seeds.values())) != 1:
        raise ML2DACTraceError(
            ML2DACFailure.SEED_IDENTITY_MISMATCH,
            "M0-M3 resolved %d distinct seeds" % len(set(seeds.values())))
    seed = int(next(iter(seeds.values())))
    if PROTO.seed_identity(ARM_METHOD_IDS[0]) != SEED_IDENTITY:
        raise ML2DACTraceError(ML2DACFailure.SEED_IDENTITY_MISMATCH,
                               PROTO.seed_identity(ARM_METHOD_IDS[0]))

    # ---- 4. authoritative search-space objects (two, kept distinct)
    try:
        space = build_search_space(view_id)                     # sampler constraint
        ss = ClusteringSearchSpace.from_dict(                   # instantiation
            resolve_view_configurations(str(repo / SEARCH_CONFIG_ROOT),
                                        view_id).search_space_config)
    except Exception as exc:  # noqa: BLE001
        raise ML2DACTraceError(ML2DACFailure.SEARCH_SPACE_MISMATCH,
                               "%s: %s" % (type(exc).__name__, str(exc)[:180]))

    n = int(budget if budget is not None else PROTO.BUDGET_EVALUATIONS)
    if n % 2:
        raise ML2DACTraceError(ML2DACFailure.CONFIG_GENERATION_FAILED,
                               "budget must be even (warmstart/sampled halves)")
    n_ws = n_opt = int(n * WARMSTART_FRACTION)

    # ---- 5. common warmstarts, K-filtered then projected onto the ClustOpt space
    ranked = _ranked_warmstarts(state, nn_id)
    selected, skipped_k = select_warmstarts(ranked, (PROTO.K_MIN, PROTO.K_MAX), n_ws)
    warms: List[ML2DACWarmstart] = []
    dropped = 0
    for w in selected:
        snapped = snap_to_space(w["algorithm"], w["hyperparameters"], space,
                                (PROTO.K_MIN, PROTO.K_MAX))
        if snapped is None:      # algorithm outside the ClustOpt space
            dropped += 1
            continue
        warms.append(ML2DACWarmstart(
            warmstart_index=len(warms), algorithm=str(w["algorithm"]),
            configuration=_freeze_config(snapped),
            warmstart_rank=int(w["rank_by_ari"])))
    ws_algorithms = tuple(sorted({w.algorithm for w in warms})
                          or sorted({r["algorithm"] for r in ranked}))
    avoid = {w.algorithm + "|" + json.dumps(dict(w.configuration), sort_keys=True)
             for w in warms}

    # ---- 6. sampled fill, CONSTRAINED sampler, deterministic under the seed
    try:
        cfgs = sample_optimizer_configs_constrained(
            list(ws_algorithms), space, n_opt, (PROTO.K_MIN, PROTO.K_MAX), seed,
            avoid=avoid)
    except Exception as exc:  # noqa: BLE001
        raise ML2DACTraceError(ML2DACFailure.CONFIG_GENERATION_FAILED,
                               "%s: %s" % (type(exc).__name__, str(exc)[:180]))
    if len(cfgs) != n_opt:
        raise ML2DACTraceError(
            ML2DACFailure.CONFIG_GENERATION_FAILED,
            "asked %d sampled configs, sampler produced %d" % (n_opt, len(cfgs)))

    plan = ([("warmstart", w.algorithm, dict(w.configuration)) for w in warms]
            + [("sampled", c["algorithm"], dict(c["hyperparameters"])) for c in cfgs])

    # ---- 7. execute every candidate; a failure is RECORDED, never resampled
    cands = []
    for i, (origin, algo, params) in enumerate(plan):
        try:
            r = run_configuration(
                ss, ConfigSpec(algorithm=algo, params=dict(params),
                               original_config_id=""), X)
            labels = _freeze_labels(getattr(r, "labels", None))
            status = "valid" if r.valid_result else "invalid"
            fk = (None if r.valid_result
                  else (r.failure_reason or ML2DACFailure.CANDIDATE_EXECUTION_FAILED))
        except Exception as exc:  # noqa: BLE001
            labels, status = None, "invalid"
            fk = "%s:%s" % (ML2DACFailure.CANDIDATE_EXECUTION_FAILED,
                            type(exc).__name__)
        cands.append(ML2DACCandidate(
            candidate_index=i, origin=origin, algorithm=str(algo),
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
        raise ML2DACTraceError(ML2DACFailure.NO_VALID_CANDIDATE,
                               "all %d candidates failed" % len(cands))

    ws_hash = hashlib.sha256(
        json.dumps([list(w.identity()) for w in warms]).encode("utf-8")
    ).hexdigest()[:16]
    avoid_hash = hashlib.sha256(
        "\n".join(sorted(avoid)).encode("utf-8")).hexdigest()[:16]

    # ---- 8. trace identity: physical, never arm-specific
    ident = json.dumps({
        "dataset_id": dataset_id, "view_id": view_id,
        "meta_feature_schema_hash": schema_hash,
        "meta_feature_vector_hash": fhash,
        "nearest_neighbor_id": nn_id,
        "warmstarts_hash": ws_hash,
        "ws_algorithms": list(ws_algorithms),
        "seed_identity": SEED_IDENTITY, "resolved_seed": seed,
        "search_space_fingerprint": SEARCH_SPACE_FINGERPRINT,
        "plan": [[o, a, json.dumps(p, sort_keys=True)] for o, a, p in plan],
    }, sort_keys=True)
    trace_id = hashlib.sha256(ident.encode("utf-8")).hexdigest()[:16]

    return ML2DACCandidateTrace(
        trace_id=trace_id, dataset_id=dataset_id, view_id=view_id,
        meta_feature_schema_hash=schema_hash,
        meta_feature_names=tuple(cols),
        meta_feature_vector=_freeze_vector(fvec.ravel()),
        meta_feature_vector_hash=fhash,
        meta_feature_env=MappingProxyType(dict(env)),
        nearest_neighbor_id=nn_id, nearest_neighbor_distance=nn_dist,
        imputer_artifact_hash=state["imputer_hash"],
        nn_index_artifact_hash=state["nn_index_hash"],
        warmstart_repository_semantic_hash=WARMSTART_SEMANTIC_HASH,
        warmstarts=tuple(warms), warmstarts_hash=ws_hash,
        warmstarts_skipped_due_to_k_range=int(skipped_k),
        warmstarts_dropped_not_in_space=int(dropped),
        ws_algorithms=ws_algorithms, avoid_set_hash=avoid_hash,
        seed_identity=SEED_IDENTITY, resolved_seed=seed,
        protocol_hash=_protocol_hash(repo),
        search_space_fingerprint=SEARCH_SPACE_FINGERPRINT,
        candidate_count=len(cands), candidates=tuple(cands), budget=n,
        n_warmstart_slots=len(warms), n_sampled_slots=len(cfgs))
