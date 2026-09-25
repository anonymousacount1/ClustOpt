"""Corrected Stage-4 ML2DAC arm scoring — M0/M1/M2/M3 over one shared trace.

Scoring semantics id: ``ml2dac_corrected_cvi_scoring_v1`` (pre-run amendment of
2026-09-10, project_lead Option 2).

The intended ML2DAC mechanism is: the frozen classifier predicts one canonical
CVI, that CVI is evaluated for every eligible candidate, one explicit direction
is applied, and the best candidate wins. Stage-3A implemented that for only 7 of
the 61 classifier classes and defaulted the rest to NaN, which made every
candidate tie and left the top warmstart as the answer. This module implements
the intended semantics instead. It is a **separate** module: the legacy
``phase._selected_cvi_value`` / ``_oriented`` remain untouched and auditable, and
a behavioural gate proves this path never calls them.

Three things are corrected relative to Stage-3A, all driven by frozen metadata
rather than by special-casing:

1. **Coverage.** All 61 classes are evaluated through their frozen canonical
   implementation — 58 via the ClustOpt metric registry, 3 via the vendored
   ML2DAC CVICollection.
2. **Direction.** Every class carries an explicit MAXIMIZE/MINIMIZE from
   ``configs/ml2dac_direction_registry.json``. There is no default, no
   name/sign/data inference, and an unknown class fails the arm cleanly.
3. **Canonical raw values.** The vendored ``CVI.score_cvi`` returns an *optimizer
   loss*: it negates MAXIMIZE-objective scores and substitutes ``2147483647`` for
   NaN. The upstream ``cvi_objective`` is read from the CVI object itself and the
   convention is inverted exactly once, so a canonical raw value reaches the
   direction registry and no double orientation can occur.

Nothing here mutates the physical trace, re-extracts meta-features, reruns
neighbour or warmstart lookup, regenerates candidates, or reruns clustering. No
ground-truth label, ARI, or CVI* utility is read anywhere.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_REPO = _ROOT.parents[1]

SCORING_SEMANTICS_ID = "ml2dac_corrected_cvi_scoring_v1"
DIRECTION_REGISTRY = "models/Independent_Domain_Benchmark/configs/ml2dac_direction_registry.json"
APPLICABILITY_POLICY = ("models/Independent_Domain_Benchmark/configs/"
                        "ml2dac_cvi_applicability_policy.json")
SEARCH_CONFIG_ROOT = "models/ClustOpt/configs/utilities_maps"

#: arm -> frozen artifact directory
ARM_SPEC: Mapping[str, Mapping[str, Any]] = MappingProxyType({
    "IDB_ML2DAC_M0_original_cvis": MappingProxyType(
        {"variant": "original_cvis", "n_classes": 7, "inventory": "original"}),
    "IDB_ML2DAC_M1_original_plus_established": MappingProxyType(
        {"variant": "original_plus_established", "n_classes": 17,
         "inventory": "original_plus_established"}),
    "IDB_ML2DAC_M2_original_plus_new46": MappingProxyType(
        {"variant": "original_plus_new46", "n_classes": 51,
         "inventory": "original_plus_new46"}),
    "IDB_ML2DAC_M3_extended_cvis": MappingProxyType(
        {"variant": "extended_cvis", "n_classes": 61, "inventory": "extended"}),
})
DERIVED = "experiments/external_baselines/Unified_MKR/derived/ml2dac_split1"

#: the Stage-4 modern comparators are ANALYSIS-ONLY and must never be reachable
#: from a learned ML2DAC class
FORBIDDEN_METRICS = frozenset({"cdbw", "cvnn", "cvdd", "dcsi",
                               "dcsi_author_compat", "cdbw_package_compat"})

#: the vendored CVIHandler substitutes this integer for a NaN score
UPSTREAM_NAN_SENTINEL = 2147483647


class ML2DACScoreFailure:
    """Scoring-layer failure kinds. No fallback of any sort."""

    ARTIFACT_MISSING = "ARTIFACT_MISSING"
    ARTIFACT_HASH_MISMATCH = "ARTIFACT_HASH_MISMATCH"
    FEATURE_SCHEMA_MISMATCH = "FEATURE_SCHEMA_MISMATCH"
    CLASSIFIER_FAILED = "CLASSIFIER_FAILED"
    CVI_CLASS_UNKNOWN = "CVI_CLASS_UNKNOWN"
    DIRECTION_UNKNOWN = "DIRECTION_UNKNOWN"
    FORBIDDEN_METRIC = "FORBIDDEN_METRIC"
    CVI_EVALUATION_FAILED = "CVI_EVALUATION_FAILED"
    #: the frozen classifier predicted a CVI whose STRUCTURAL applicability rule
    #: excludes this view. Distinct from NO_ELIGIBLE_CANDIDATE: nothing about the
    #: candidates is at fault, so no candidate is even scored.
    PREDICTED_CVI_INAPPLICABLE_FOR_VIEW = "PREDICTED_CVI_INAPPLICABLE_FOR_VIEW"
    #: the predicted CVI IS structurally applicable to this view, but no candidate
    #: in the frozen trace yielded an eligible score.
    NO_ELIGIBLE_CANDIDATE = "NO_ELIGIBLE_CANDIDATE"
    ALL = ("ARTIFACT_MISSING", "ARTIFACT_HASH_MISMATCH", "FEATURE_SCHEMA_MISMATCH",
           "CLASSIFIER_FAILED", "CVI_CLASS_UNKNOWN", "DIRECTION_UNKNOWN",
           "FORBIDDEN_METRIC", "CVI_EVALUATION_FAILED",
           "PREDICTED_CVI_INAPPLICABLE_FOR_VIEW", "NO_ELIGIBLE_CANDIDATE")


class ML2DACScoreError(RuntimeError):
    def __init__(self, kind: str, detail: str = "") -> None:
        super().__init__("%s: %s" % (kind, detail) if detail else kind)
        self.kind = kind
        self.detail = detail


# --------------------------------------------------------------- registry
_REG_CACHE: Dict[str, Any] = {}


def load_direction_registry(repo: Path = _REPO) -> Dict[str, Any]:
    if "reg" not in _REG_CACHE:
        p = repo / DIRECTION_REGISTRY
        if not p.is_file():
            raise ML2DACScoreError(ML2DACScoreFailure.ARTIFACT_MISSING, str(p))
        d = json.loads(p.read_text(encoding="utf-8"))
        by_id = {e["canonical_metric_id"]: e for e in d["entries"]}
        if len(by_id) != len(d["entries"]):
            raise ML2DACScoreError(ML2DACScoreFailure.CVI_CLASS_UNKNOWN,
                                   "duplicate canonical_metric_id in the registry")
        bad = sorted(set(by_id) & FORBIDDEN_METRICS)
        if bad:
            raise ML2DACScoreError(ML2DACScoreFailure.FORBIDDEN_METRIC, ", ".join(bad))
        _REG_CACHE["reg"] = d
        _REG_CACHE["by_id"] = by_id
        _REG_CACHE["hash"] = hashlib.sha256(
            json.dumps(d, indent=2, default=str).encode("utf-8")).hexdigest()[:16]
    return _REG_CACHE["reg"]


_APP_CACHE: Dict[str, Any] = {}


def load_applicability_policy(repo: Path = _REPO) -> Dict[str, Any]:
    """The frozen structural applicability map. NO FALLBACK follows a failure."""
    if "pol" not in _APP_CACHE:
        p = repo / APPLICABILITY_POLICY
        if not p.is_file():
            raise ML2DACScoreError(ML2DACScoreFailure.ARTIFACT_MISSING, str(p))
        d = json.loads(p.read_text(encoding="utf-8"))
        _APP_CACHE["pol"] = d
        _APP_CACHE["by_id"] = {e["canonical_metric_id"]: e for e in d["entries"]}
    return _APP_CACHE["pol"]


def is_applicable(metric_id: str, n_features: int, repo: Path = _REPO) -> bool:
    """Structural applicability of one class to a view of this dimensionality.

    Structural only: whether the metric CAN evaluate this kind of view at all.
    Whether a particular partition yields a finite value is candidate-dependent
    and is handled by NO_ELIGIBLE_CANDIDATE instead.
    """
    load_applicability_policy(repo)
    e = _APP_CACHE["by_id"].get(metric_id)
    if e is None:
        raise ML2DACScoreError(ML2DACScoreFailure.CVI_CLASS_UNKNOWN, metric_id)
    return bool(e["applicable_2d"] if int(n_features) >= 2 else e["applicable_1d"])


def direction_of(metric_id: str, repo: Path = _REPO) -> str:
    """Explicit MAXIMIZE/MINIMIZE. Never a default; unknown ids raise."""
    load_direction_registry(repo)
    e = _REG_CACHE["by_id"].get(metric_id)
    if e is None:
        raise ML2DACScoreError(ML2DACScoreFailure.CVI_CLASS_UNKNOWN, metric_id)
    d = e.get("orientation")
    if d not in ("MAXIMIZE", "MINIMIZE"):
        raise ML2DACScoreError(ML2DACScoreFailure.DIRECTION_UNKNOWN,
                               "%s -> %r" % (metric_id, d))
    return d


def orient(metric_id: str, raw: float, repo: Path = _REPO) -> float:
    """Canonical raw value -> higher-is-better score. Applied exactly once."""
    if raw is None or not np.isfinite(raw):
        return float("-inf")
    return float(raw) if direction_of(metric_id, repo) == "MAXIMIZE" else -float(raw)


# --------------------------------------------------- canonical raw values
def _upstream_objectives(repo: Path) -> Dict[str, str]:
    """Read each ML2DAC-only CVI's MetricObjective from the vendored source.

    Systematic, not per-metric special-casing: whatever the upstream declares is
    what drives the single inversion below.
    """
    import sys
    mkr = str(repo / "experiments/external_baselines/Unified_MKR")
    if mkr not in sys.path:
        sys.path.insert(0, mkr)
    from unified_mkr.ml2dac_cvi_adapter import _CANON_TO_ABBREV, _ensure_loaded

    st = _ensure_loaded(str(repo))
    if not st.available:
        raise ML2DACScoreError(ML2DACScoreFailure.CVI_EVALUATION_FAILED, st.error)
    out = {}
    for canon, abbrev in _CANON_TO_ABBREV.items():
        cvi = st.cvi_by_abbrev.get(abbrev)
        if cvi is None:
            raise ML2DACScoreError(ML2DACScoreFailure.CVI_CLASS_UNKNOWN, canon)
        out[canon] = str(cvi.cvi_objective).upper()
    return out


def canonical_ml2dac_only(X: np.ndarray, labels: np.ndarray, *,
                          repo: Path = _REPO) -> Dict[str, float]:
    """The three ML2DAC-only CVIs as CANONICAL RAW values.

    ``CVI.score_cvi`` returns an optimizer loss: MAXIMIZE-objective scores are
    negated and NaN becomes ``2147483647``. Both conventions are undone here,
    exactly once, using the upstream objective metadata.
    """
    import sys
    mkr = str(repo / "experiments/external_baselines/Unified_MKR")
    if mkr not in sys.path:
        sys.path.insert(0, mkr)
    from unified_mkr.ml2dac_cvi_adapter import compute_ml2dac_only

    obj = _upstream_objectives(repo)
    got = compute_ml2dac_only(np.asarray(X), np.asarray(labels), repo_root=str(repo))
    out: Dict[str, float] = {}
    for name, v in got.items():
        if v is None or not np.isfinite(v) or abs(float(v)) == UPSTREAM_NAN_SENTINEL:
            out[name] = float("nan")
            continue
        v = float(v)
        # score_cvi negated a MAXIMIZE objective to make it a loss -> undo once
        out[name] = -v if obj.get(name) == "MAXIMIZE" else v
    return out


def _engine(X_decision, X_full, repo: Path):
    """Build the authoritative metric engine.

    ``X_full`` is NOT optional and must never be defaulted to ``X_decision``.
    46 of the 61 classifier classes declare ``space="full"`` and are evaluated on
    the FULL dataset matrix regardless of which view is the decision space.
    Passing a 1-D view as ``X_full`` silently starves those 46 metrics of the
    second dimension and makes them report NaN, which looks exactly like
    structural inapplicability but is a wiring error.
    """
    import sys
    mkr = str(repo / "experiments/external_baselines/Unified_MKR")
    if mkr not in sys.path:
        sys.path.insert(0, mkr)
    from unified_mkr.metric_runner import ViewMetricEngine
    return ViewMetricEngine(str(repo / SEARCH_CONFIG_ROOT), np.asarray(X_decision),
                            np.asarray(X_full), repo_root=str(repo))


def canonical_raw_value(metric_id: str, labels: np.ndarray, *, engine,
                        repo: Path = _REPO) -> float:
    """The canonical raw value of one classifier class on one partition.

    58 classes go through the frozen ClustOpt registry, 3 through the vendored
    ML2DAC implementation. There is no NaN gate: an unknown id raises.
    """
    load_direction_registry(repo)
    e = _REG_CACHE["by_id"].get(metric_id)
    if e is None:
        raise ML2DACScoreError(ML2DACScoreFailure.CVI_CLASS_UNKNOWN, metric_id)
    if metric_id in FORBIDDEN_METRICS:
        raise ML2DACScoreError(ML2DACScoreFailure.FORBIDDEN_METRIC, metric_id)
    labels = np.asarray(labels)
    if e["engine_entrypoint"] == "ViewMetricEngine._ml2dac":
        return float(canonical_ml2dac_only(engine.X_decision, labels,
                                           repo=repo).get(metric_id, float("nan")))
    v = engine._clustopt_raw(labels, [metric_id]).get(metric_id, float("nan"))
    return float(v)


# ------------------------------------------------------------- arm artifacts
_ARM_CACHE: Dict[str, Dict[str, Any]] = {}


def load_arm(method_id: str, repo: Path = _REPO) -> Dict[str, Any]:
    """The arm's frozen classifier and class order. Never fitted here."""
    if method_id in _ARM_CACHE:
        return _ARM_CACHE[method_id]
    if method_id not in ARM_SPEC:
        raise ML2DACScoreError(ML2DACScoreFailure.ARTIFACT_MISSING, method_id)
    import joblib
    spec = ARM_SPEC[method_id]
    d = repo / DERIVED / spec["variant"]
    cp, ip = d / "cvi_classifier.pkl", d / "imputer.pkl"
    for p in (cp, ip):
        if not p.is_file():
            raise ML2DACScoreError(ML2DACScoreFailure.ARTIFACT_MISSING, str(p))
    clf = joblib.load(cp)
    classes = [str(c) for c in clf.classes_]
    if len(classes) != spec["n_classes"]:
        raise ML2DACScoreError(ML2DACScoreFailure.ARTIFACT_HASH_MISMATCH,
                               "%d classes, expected %d" % (len(classes),
                                                            spec["n_classes"]))
    arm = {
        "method_id": method_id, "variant": spec["variant"],
        "inventory": spec["inventory"], "classifier": clf,
        "imputer": joblib.load(ip), "classes": tuple(classes),
        "classifier_artifact_hash": hashlib.sha256(cp.read_bytes()).hexdigest(),
        "classifier_artifact_sha256_16": hashlib.sha256(
            cp.read_bytes()).hexdigest()[:16],
        "imputer_artifact_sha256_16": hashlib.sha256(ip.read_bytes()).hexdigest()[:16],
        "class_order_hash": hashlib.sha256(
            "\n".join(classes).encode("utf-8")).hexdigest()[:16],
    }
    _ARM_CACHE[method_id] = arm
    return arm


# ---------------------------------------------------------------- records
@dataclass(frozen=True)
class ML2DACArmDecision:
    """Arm-specific classification state. Deliberately NOT in the physical trace."""

    method_id: str
    physical_trace_id: str
    meta_feature_schema_hash: str
    classifier_artifact_hash: str
    classifier_class_order_hash: str
    predicted_cvi_metric_id: Optional[str]
    predicted_cvi_abbrev: Optional[str]
    direction: Optional[str]
    structurally_applicable_to_view: Optional[bool]
    status: str
    failure_kind: Optional[str]


@dataclass(frozen=True)
class ML2DACCandidateScore:
    method_id: str
    trace_id: str
    candidate_index: int
    predicted_cvi_metric_id: str
    raw_cvi_value: Optional[float]
    orientation: str
    oriented_score: Optional[float]
    score_status: str                 # "scored" | "ineligible" | "failed"
    failure_kind: Optional[str]

    def identity(self) -> Tuple:
        return (self.candidate_index, self.predicted_cvi_metric_id,
                self.score_status, self.oriented_score)


@dataclass(frozen=True)
class ML2DACArmResult:
    method_id: str
    dataset_id: str
    view_id: str
    status: str
    failure_kind: Optional[str]
    scoring_semantics_id: str
    shared_physical_trace_id: str
    decision: ML2DACArmDecision
    scores: Tuple[ML2DACCandidateScore, ...]
    n_eligible: int
    selected_candidate_index: Optional[int]
    selected_algorithm: Optional[str]
    selected_configuration: Optional[Mapping[str, Any]]
    final_labels_hash: Optional[str]
    n_evaluations: int
    search_space_fingerprint: str
    seed_identity: str
    resolved_seed: int
    direction_registry_hash: str
    score_vector_hash: str
    metric_inventory_id: str = field(default="")


#: Frozen Stage-4 selection rule. Uniform higher-is-better representation: the
#: raw value is oriented exactly once, then maximised with a STRICT comparison,
#: so equal scores leave the incumbent in place and the LOWEST candidate index
#: wins a tie. Non-finite oriented scores are ineligible. There is no
#: unknown-direction fallback.
SELECTION_RULE = ("maximise the oriented predicted-CVI score over eligible "
                  "candidates; strict improvement only, so the lowest candidate "
                  "index wins a tie; non-finite scores are ineligible; if no "
                  "candidate is eligible the arm fails with NO_ELIGIBLE_CANDIDATE")


def select_final(scores) -> Optional[int]:
    best_i, best_v = None, None
    for s in scores:
        if s.score_status != "scored" or s.oriented_score is None:
            continue
        v = float(s.oriented_score)
        if not np.isfinite(v):
            continue
        if best_v is None or v > best_v:      # strict: ties keep the incumbent
            best_i, best_v = s.candidate_index, v
    return best_i


def predict_cvi(trace, method_id: str, *, repo: Path = _REPO,
                arm: Optional[Dict[str, Any]] = None,
                n_features: Optional[int] = None) -> ML2DACArmDecision:
    """Predict this arm's CVI from the trace's already-frozen meta-features.

    The trace vector is consumed as-is: nothing is re-extracted here.
    """
    from experiments.external_baselines.ML2DAC.indomain.dataset import CVI_ABBREV
    a = arm or load_arm(method_id, repo)
    vec = np.asarray(trace.meta_feature_vector, dtype=float).reshape(1, -1)
    if vec.shape[1] != len(trace.meta_feature_names):
        raise ML2DACScoreError(ML2DACScoreFailure.FEATURE_SCHEMA_MISMATCH,
                               "%d values" % vec.shape[1])
    try:
        cvi = str(a["classifier"].predict(a["imputer"].transform(vec))[0])
    except Exception as exc:  # noqa: BLE001
        raise ML2DACScoreError(ML2DACScoreFailure.CLASSIFIER_FAILED,
                               "%s: %s" % (type(exc).__name__, str(exc)[:180]))
    d = direction_of(cvi, repo)          # raises on unknown / unresolved
    app = None if n_features is None else is_applicable(cvi, n_features, repo)
    return ML2DACArmDecision(
        method_id=method_id, physical_trace_id=trace.trace_id,
        meta_feature_schema_hash=trace.meta_feature_schema_hash,
        classifier_artifact_hash=a["classifier_artifact_hash"],
        classifier_class_order_hash=a["class_order_hash"],
        predicted_cvi_metric_id=cvi, predicted_cvi_abbrev=CVI_ABBREV.get(cvi),
        direction=d, structurally_applicable_to_view=app,
        status="success", failure_kind=None)


def score_trace(trace, method_id: str, *, X_decision: np.ndarray,
                X_full: np.ndarray, repo: Path = _REPO,
                engine=None) -> ML2DACArmResult:
    """Score one immutable physical trace with one arm. No labels, no clustering.

    ``X_full`` is REQUIRED: the majority of the class inventory is evaluated on
    the full dataset matrix, not on the view. See :func:`_engine`.
    """
    if X_full is None:
        raise ML2DACScoreError(
            ML2DACScoreFailure.FEATURE_SCHEMA_MISMATCH,
            "X_full is required; 46 of 61 classes are full-space metrics")
    load_direction_registry(repo)
    a = load_arm(method_id, repo)
    _nf = int(np.asarray(X_decision).reshape(len(np.asarray(X_decision)), -1).shape[1])
    dec = predict_cvi(trace, method_id, repo=repo, arm=a, n_features=_nf)
    cvi = dec.predicted_cvi_metric_id
    orientation = dec.direction
    n_feat = int(np.asarray(X_decision).reshape(len(np.asarray(X_decision)), -1).shape[1])
    if not is_applicable(cvi, n_feat, repo):
        # The classifier predicted a CVI that structurally cannot evaluate this
        # view. Fail the method/view here: scoring 50 candidates to collect 50
        # identical NaNs would misrepresent a CVI-selection failure as ordinary
        # candidate failure. NO FALLBACK: no second-best class, no substitute CVI,
        # no top-warmstart default, no padding of the view.
        return ML2DACArmResult(
            method_id=method_id, dataset_id=trace.dataset_id, view_id=trace.view_id,
            status="failed",
            failure_kind=ML2DACScoreFailure.PREDICTED_CVI_INAPPLICABLE_FOR_VIEW,
            scoring_semantics_id=SCORING_SEMANTICS_ID,
            shared_physical_trace_id=trace.trace_id, decision=dec, scores=tuple(),
            n_eligible=0, selected_candidate_index=None, selected_algorithm=None,
            selected_configuration=None, final_labels_hash=None,
            n_evaluations=trace.candidate_count,
            search_space_fingerprint=trace.search_space_fingerprint,
            seed_identity=trace.seed_identity, resolved_seed=trace.resolved_seed,
            direction_registry_hash=_REG_CACHE["hash"], score_vector_hash="",
            metric_inventory_id=a["inventory"])

    eng = engine if engine is not None else _engine(X_decision, X_full, repo)

    scores = []
    for c in trace.candidates:
        if c.status != "valid" or c.labels is None:
            scores.append(ML2DACCandidateScore(
                method_id, trace.trace_id, c.candidate_index, cvi, None,
                orientation, None, "ineligible", c.failure_kind))
            continue
        try:
            raw = canonical_raw_value(cvi, c.labels, engine=eng, repo=repo)
        except ML2DACScoreError:
            raise
        except Exception as exc:  # noqa: BLE001
            scores.append(ML2DACCandidateScore(
                method_id, trace.trace_id, c.candidate_index, cvi, None,
                orientation, None, "failed",
                "%s:%s" % (ML2DACScoreFailure.CVI_EVALUATION_FAILED,
                           type(exc).__name__)))
            continue
        ov = orient(cvi, raw, repo)
        finite = raw is not None and np.isfinite(raw)
        scores.append(ML2DACCandidateScore(
            method_id, trace.trace_id, c.candidate_index, cvi,
            (float(raw) if raw is not None and not math.isnan(raw) else None),
            orientation, (float(ov) if finite else None),
            "scored" if finite else "ineligible",
            None if finite else ML2DACScoreFailure.CVI_EVALUATION_FAILED))

    scores = tuple(scores)
    n_elig = sum(1 for s in scores if s.score_status == "scored")
    sel = select_final(scores)
    svh = hashlib.sha256(json.dumps(
        [[s.candidate_index, s.score_status,
          None if s.oriented_score is None else repr(s.oriented_score)]
         for s in scores]).encode("utf-8")).hexdigest()[:16]
    if sel is None:
        # Every candidate is ineligible for the predicted CVI -- e.g. an
        # image-based class predicted on a 1-D view. §24 forbids falling back to
        # another CVI, so the ARM fails cleanly and the other three arms are
        # unaffected. The failure is a scientific outcome, not an exception.
        return ML2DACArmResult(
            method_id=method_id, dataset_id=trace.dataset_id, view_id=trace.view_id,
            status="failed", failure_kind=ML2DACScoreFailure.NO_ELIGIBLE_CANDIDATE,
            scoring_semantics_id=SCORING_SEMANTICS_ID,
            shared_physical_trace_id=trace.trace_id, decision=dec, scores=scores,
            n_eligible=0, selected_candidate_index=None, selected_algorithm=None,
            selected_configuration=None, final_labels_hash=None,
            n_evaluations=trace.candidate_count,
            search_space_fingerprint=trace.search_space_fingerprint,
            seed_identity=trace.seed_identity, resolved_seed=trace.resolved_seed,
            direction_registry_hash=_REG_CACHE["hash"], score_vector_hash=svh,
            metric_inventory_id=a["inventory"])
    chosen = trace.candidates[sel]
    return ML2DACArmResult(
        method_id=method_id, dataset_id=trace.dataset_id, view_id=trace.view_id,
        status="success", failure_kind=None,
        scoring_semantics_id=SCORING_SEMANTICS_ID,
        shared_physical_trace_id=trace.trace_id, decision=dec, scores=scores,
        n_eligible=n_elig, selected_candidate_index=sel,
        selected_algorithm=chosen.algorithm,
        selected_configuration=chosen.configuration,
        final_labels_hash=chosen.raw_labels_hash,
        n_evaluations=trace.candidate_count,
        search_space_fingerprint=trace.search_space_fingerprint,
        seed_identity=trace.seed_identity, resolved_seed=trace.resolved_seed,
        direction_registry_hash=_REG_CACHE["hash"], score_vector_hash=svh,
        metric_inventory_id=a["inventory"])


def score_all_arms(trace, *, X_decision: np.ndarray, X_full: np.ndarray,
                   repo: Path = _REPO) -> Dict[str, ML2DACArmResult]:
    """All four arms over ONE shared physical trace. The engine is built once.

    ``X_full`` is REQUIRED for the same reason as in :func:`score_trace`.
    """
    if X_full is None:
        raise ML2DACScoreError(
            ML2DACScoreFailure.FEATURE_SCHEMA_MISMATCH,
            "X_full is required; 46 of 61 classes are full-space metrics")
    eng = _engine(X_decision, X_full, repo)
    return {m: score_trace(trace, m, X_decision=X_decision, X_full=X_full,
                           repo=repo, engine=eng) for m in ARM_SPEC}
