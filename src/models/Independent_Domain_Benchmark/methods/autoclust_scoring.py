"""AutoClust arm-specific scoring — CVI vector, frozen predictor, final selection.

Consumes the immutable physical ``AutoClustCandidateTrace`` produced by
``autoclust.py`` and produces one scientific result per arm. It never reruns
clustering, never rebuilds landmark features, never re-resolves a seed and never
mutates the trace: the physical trace is an input boundary.

The CVI vector is built by the frozen Stage-3A routine ``phase._cvi_vector`` over
a ``ViewMetricEngine``, so the predictors receive exactly the metric semantics
they were trained on. In particular an undefined CVI stays ``NaN`` and is imputed
downstream by the pipeline's own ``SimpleImputer(median)`` — converting NaN to
zero here would silently feed the predictor something it never saw in training.

Final selection reproduces the historical rule: the highest predicted ARI wins,
and because the historical loop keeps a candidate only on strictly greater score,
an exact tie is won by the LOWEST candidate index.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_REPO = _ROOT.parents[1]

DERIVED = ("experiments/external_baselines/Unified_MKR/derived/autoclust_split1")
SEARCH_CONFIG_ROOT = "models/ClustOpt/configs/utilities_maps"

#: METHOD_ID -> (variant directory, metric inventory id, expected feature count)
ARM_SPEC: Mapping[str, Tuple[str, str, int]] = MappingProxyType({
    "IDB_AutoClust_A0_original_cvis": ("original_cvis", "original", 7),
    "IDB_AutoClust_A1_original_plus_established":
        ("original_plus_established", "original_plus_established", 17),
    "IDB_AutoClust_A2_original_plus_new46":
        ("original_plus_new46", "original_plus_new46_live", 51),
    "IDB_AutoClust_A3_extended_cvis": ("extended_cvis", "extended_live", 61),
})

#: analysis-only Stage-4 comparators; these must never enter a learned schema
FORBIDDEN_FEATURES = frozenset({
    "cdbw", "cvnn", "cvdd", "dcsi", "dcsi_author_compat", "cdbw_package_compat"})


class AutoClustScoreFailure:
    ARTIFACT_MISSING = "ARTIFACT_MISSING"
    ARTIFACT_HASH_MISMATCH = "ARTIFACT_HASH_MISMATCH"
    FEATURE_SCHEMA_MISMATCH = "FEATURE_SCHEMA_MISMATCH"
    FORBIDDEN_FEATURE = "FORBIDDEN_FEATURE"
    CVI_COMPUTATION_FAILED = "CVI_COMPUTATION_FAILED"
    PREDICTOR_FAILED = "PREDICTOR_FAILED"
    FULL_EVALUATION_SPACE_REQUIRED = "FULL_EVALUATION_SPACE_REQUIRED"
    NO_ELIGIBLE_CANDIDATE = "NO_ELIGIBLE_CANDIDATE"
    ALL = ("ARTIFACT_MISSING", "ARTIFACT_HASH_MISMATCH",
           "FEATURE_SCHEMA_MISMATCH", "FORBIDDEN_FEATURE",
           "CVI_COMPUTATION_FAILED", "PREDICTOR_FAILED",
           "FULL_EVALUATION_SPACE_REQUIRED", "NO_ELIGIBLE_CANDIDATE")


class AutoClustScoreError(RuntimeError):
    def __init__(self, kind: str, detail: str = "") -> None:
        super().__init__("%s: %s" % (kind, detail) if detail else kind)
        self.kind, self.detail = kind, detail


@dataclass(frozen=True)
class AutoClustCandidateScore:
    """One arm's opinion of one candidate. Lives OUTSIDE the physical trace."""

    method_id: str
    trace_id: str
    candidate_index: int
    feature_schema_hash: str
    cvi_vector_hash: Optional[str]
    predictor_artifact_hash: str
    predicted_ari: Optional[float]
    score_status: str                 # "scored" | "ineligible" | "failed"
    failure_kind: Optional[str]
    n_nan_features: Optional[int]


@dataclass(frozen=True)
class AutoClustArmResult:
    """One scientific AutoClust output. No ground-truth field exists here."""

    method_id: str
    trace_id: str
    dataset_id: str
    view_id: str
    metric_inventory_id: str
    feature_schema_hash: str
    predictor_artifact_hash: str
    n_candidates: int
    n_eligible: int
    scores: Tuple[AutoClustCandidateScore, ...]
    selected_candidate_index: Optional[int]
    selected_algorithm: Optional[str]
    selected_configuration: Optional[Mapping[str, Any]]
    final_labels_hash: Optional[str]
    selection_rule: str
    status: str
    failure_kind: Optional[str]


# ------------------------------------------------------------------- artifacts
def _sha16(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def load_arm(method_id: str, repo: Path = _REPO) -> Dict[str, Any]:
    """Feature schema + frozen predictor for one arm, with hashes and checks."""
    if method_id not in ARM_SPEC:
        raise AutoClustScoreError(AutoClustScoreFailure.ARTIFACT_MISSING,
                                  "unknown method_id %r" % method_id)
    import joblib
    vdir, inventory, expected = ARM_SPEC[method_id]
    d = repo / DERIVED / vdir
    fcp, mp = d / "feature_columns.json", d / "ari_predictor.pkl"
    for p in (fcp, mp):
        if not p.is_file():
            raise AutoClustScoreError(AutoClustScoreFailure.ARTIFACT_MISSING, str(p))
    cols = list(json.loads(fcp.read_text(encoding="utf-8"))
                ["ari_predictor_cvi_columns"])
    if len(cols) != expected:
        raise AutoClustScoreError(
            AutoClustScoreFailure.FEATURE_SCHEMA_MISMATCH,
            "%s expects %d features, artifact has %d" % (method_id, expected,
                                                         len(cols)))
    if len(set(cols)) != len(cols):
        raise AutoClustScoreError(AutoClustScoreFailure.FEATURE_SCHEMA_MISMATCH,
                                  "duplicate feature columns")
    bad = sorted(set(c.lower() for c in cols) & FORBIDDEN_FEATURES)
    if bad:
        raise AutoClustScoreError(AutoClustScoreFailure.FORBIDDEN_FEATURE, str(bad))
    model = joblib.load(mp)
    n_in = getattr(model, "n_features_in_", None)
    if n_in is not None and int(n_in) != len(cols):
        raise AutoClustScoreError(
            AutoClustScoreFailure.FEATURE_SCHEMA_MISMATCH,
            "predictor takes %d features, schema has %d" % (n_in, len(cols)))
    return {"method_id": method_id, "inventory": inventory,
            "feature_columns": tuple(cols),
            "feature_schema_hash": hashlib.sha256(
                json.dumps(cols).encode()).hexdigest()[:16],
            "feature_columns_file_hash": _sha16(fcp),
            "predictor": model, "predictor_hash": _sha16(mp),
            "variant_dir": str((repo / DERIVED / vdir).relative_to(repo)
                               ).replace("\\", "/")}


# ------------------------------------------------------------------ CVI vector
def build_autoclust_cvi_vector(engine, labels: np.ndarray,
                               feature_columns: Tuple[str, ...]):
    """The frozen Stage-3A CVI vector, in artifact order. NaN is preserved."""
    from experiments.external_baselines.AutoClust.indomain.phase import _cvi_vector
    vec = _cvi_vector(engine, np.asarray(labels), list(feature_columns))
    a = np.asarray(vec, dtype=float)
    if a.shape[0] != len(feature_columns):
        raise AutoClustScoreError(
            AutoClustScoreFailure.FEATURE_SCHEMA_MISMATCH,
            "built %d values for %d columns" % (a.shape[0], len(feature_columns)))
    return a


def _vec_hash(a: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(a, dtype=np.float64).tobytes()
                          ).hexdigest()[:16]


# ------------------------------------------------------------------- selection
#: Frozen from the Stage-3A source: the loop keeps a candidate only on
#: ``pred_ari > best``, so equal scores leave the incumbent in place and the
#: LOWEST candidate index wins a tie. Recorded rather than assumed.
SELECTION_RULE = ("maximise predicted ARI over eligible candidates; strict "
                  "improvement only, so an exact tie is won by the lowest "
                  "candidate index")


def select_final(scores: Tuple[AutoClustCandidateScore, ...]) -> Optional[int]:
    best_i, best_v = None, None
    for s in scores:
        if s.score_status != "scored" or s.predicted_ari is None:
            continue
        v = float(s.predicted_ari)
        if not np.isfinite(v):
            continue
        if best_v is None or v > best_v:      # strict: ties keep the incumbent
            best_i, best_v = s.candidate_index, v
    return best_i


# --------------------------------------------------------------------- scoring
def score_trace(trace, method_id: str, *, X_decision: np.ndarray,
                X_full: np.ndarray, repo: Path = _REPO,
                arm: Optional[Dict[str, Any]] = None) -> AutoClustArmResult:
    """Score one immutable physical trace with one arm. No labels, no clustering.

    ``X_full`` is REQUIRED and is never aliased to ``X_decision``.

    The frozen V2 view semantics (audit ``4ef3f57``) are: candidates are fitted in
    ``X_decision(v)``, and each metric is evaluated in its declared
    ``metric.space`` -- decision-space metrics see the view, full-space metrics see
    the frozen full 2-D representation of the SAME observations in the same row
    order. Aliasing ``X_full`` to a 1-D view would silently evaluate the
    full-space metrics on the decision view, which is not the frozen benchmark
    definition. Omission therefore fails loudly.
    """
    import sys
    mkr = str(repo / "experiments/external_baselines/Unified_MKR")
    if mkr not in sys.path:
        sys.path.insert(0, mkr)
    from unified_mkr.metric_runner import ViewMetricEngine

    if X_full is None:
        raise AutoClustScoreError(
            AutoClustScoreFailure.FULL_EVALUATION_SPACE_REQUIRED,
            "X_full must be supplied explicitly for scientific AutoClust scoring")
    a = arm or load_arm(method_id, repo)
    Xd = np.asarray(X_decision, dtype=float)
    Xf = np.asarray(X_full, dtype=float)
    if Xd.shape[0] != Xf.shape[0]:
        raise AutoClustScoreError(
            AutoClustScoreFailure.FULL_EVALUATION_SPACE_REQUIRED,
            "sample-count mismatch: X_decision %d rows, X_full %d rows"
            % (Xd.shape[0], Xf.shape[0]))
    engine = ViewMetricEngine(str(repo / SEARCH_CONFIG_ROOT), Xd, Xf,
                              repo_root=str(repo))

    scores = []
    for c in trace.candidates:
        # eligibility mirrors the historical guard: a valid clustering with labels
        if c.status != "valid" or c.labels is None:
            scores.append(AutoClustCandidateScore(
                method_id, trace.trace_id, c.candidate_index,
                a["feature_schema_hash"], None, a["predictor_hash"], None,
                "ineligible", c.failure_kind, None))
            continue
        try:
            vec = build_autoclust_cvi_vector(engine, c.labels,
                                             a["feature_columns"])
        except Exception as exc:
            scores.append(AutoClustCandidateScore(
                method_id, trace.trace_id, c.candidate_index,
                a["feature_schema_hash"], None, a["predictor_hash"], None,
                "failed", "%s:%s" % (AutoClustScoreFailure.CVI_COMPUTATION_FAILED,
                                     type(exc).__name__), None))
            continue
        try:
            # NaN is deliberately passed through: the pipeline imputes it
            pred = float(np.asarray(
                a["predictor"].predict(vec.reshape(1, -1))).ravel()[0])
            st, fk = "scored", None
        except Exception as exc:
            pred, st = None, "failed"
            fk = "%s:%s" % (AutoClustScoreFailure.PREDICTOR_FAILED,
                            type(exc).__name__)
        scores.append(AutoClustCandidateScore(
            method_id, trace.trace_id, c.candidate_index,
            a["feature_schema_hash"], _vec_hash(vec), a["predictor_hash"],
            pred, st, fk, int(np.isnan(vec).sum())))

    scores = tuple(scores)
    n_elig = sum(1 for s in scores if s.score_status == "scored")
    pick = select_final(scores)
    if pick is None:
        return AutoClustArmResult(
            method_id, trace.trace_id, trace.dataset_id, trace.view_id,
            a["inventory"], a["feature_schema_hash"], a["predictor_hash"],
            len(scores), n_elig, scores, None, None, None, None,
            SELECTION_RULE, "failed",
            AutoClustScoreFailure.NO_ELIGIBLE_CANDIDATE)
    cand = trace.candidates[pick]
    return AutoClustArmResult(
        method_id, trace.trace_id, trace.dataset_id, trace.view_id,
        a["inventory"], a["feature_schema_hash"], a["predictor_hash"],
        len(scores), n_elig, scores, pick, cand.algorithm,
        MappingProxyType(dict(cand.configuration)), cand.raw_labels_hash,
        SELECTION_RULE, "success", None)


def score_all_arms(trace, *, X_decision: np.ndarray, X_full: np.ndarray,
                   repo: Path = _REPO) -> Dict[str, AutoClustArmResult]:
    """All four scientific arms over ONE shared physical trace.

    ``X_full`` is REQUIRED for the same reason as in :func:`score_trace`.
    """
    if X_full is None:
        raise AutoClustScoreError(
            AutoClustScoreFailure.FULL_EVALUATION_SPACE_REQUIRED,
            "X_full must be supplied explicitly for scientific AutoClust scoring")
    return {m: score_trace(trace, m, X_decision=X_decision, X_full=X_full,
                           repo=repo) for m in ARM_SPEC}
