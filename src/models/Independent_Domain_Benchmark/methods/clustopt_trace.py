"""The ClustOpt candidate trace, and the adapter's failure vocabulary.

A trace is everything one ClustOpt search produced, in a form rich enough for
BOTH final selectors -- the original objective J and the Candidate Reranker --
so that a ``+R`` arm can reuse the ``noR`` arm's search rather than repeating it.

Ground truth is not part of the trace. There is no ARI field, no true K and no
evaluation label anywhere in this module: an ARI column would let a selector see
the answer, which is exactly what the shared-trace design must not permit.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


class ClustOptFailure:
    """Deterministic adapter failure kinds.

    A failure is always explicit. The adapter never falls back from PP to MLP,
    from MLP to KNN, or from the reranker to J: no such fallback exists in the
    frozen pipeline, so inventing one here would silently change the method.
    """

    ARTIFACT_MISSING = "ARTIFACT_MISSING"
    ARTIFACT_HASH_MISMATCH = "ARTIFACT_HASH_MISMATCH"
    ARTIFACT_AMBIGUOUS = "ARTIFACT_AMBIGUOUS"
    FEATURE_SCHEMA_MISMATCH = "FEATURE_SCHEMA_MISMATCH"
    META_FEATURE_EXTRACTION_FAILED = "META_FEATURE_EXTRACTION_FAILED"
    POLICY_UNRESOLVED = "POLICY_UNRESOLVED"
    SEARCH_FAILED = "SEARCH_FAILED"
    NO_VALID_CANDIDATE = "NO_VALID_CANDIDATE"
    RERANKER_SCHEMA_MISMATCH = "RERANKER_SCHEMA_MISMATCH"
    RERANKER_FAILED = "RERANKER_FAILED"

    ALL = ("ARTIFACT_MISSING", "ARTIFACT_HASH_MISMATCH", "ARTIFACT_AMBIGUOUS",
           "FEATURE_SCHEMA_MISMATCH", "META_FEATURE_EXTRACTION_FAILED",
           "POLICY_UNRESOLVED", "SEARCH_FAILED", "NO_VALID_CANDIDATE",
           "RERANKER_SCHEMA_MISMATCH", "RERANKER_FAILED")


class AdapterError(RuntimeError):
    """Carries a failure kind so the runner can record it verbatim."""

    def __init__(self, kind: str, detail: str = "") -> None:
        super().__init__("%s: %s" % (kind, detail) if detail else kind)
        self.kind = kind
        self.detail = detail


def partition_hash(labels: np.ndarray) -> str:
    """Identity of a partition, invariant to how its clusters are named.

    Two candidates that induce the same grouping hash the same even if one calls
    a cluster 0 and the other calls it 3, which is what makes a trace comparison
    a statement about partitions rather than about labelling accidents.
    """
    y = np.asarray(labels).ravel()
    noise = y < 0
    out = np.full(y.shape, -1, dtype=np.int64)
    _, first = np.unique(y[~noise], return_index=True)
    order = y[~noise][np.sort(first)]
    remap = {int(c): i for i, c in enumerate(order)}
    out[~noise] = [remap[int(v)] for v in y[~noise]]
    return hashlib.sha256(out.tobytes()).hexdigest()[:16]


@dataclass
class ClustOptTrace:
    """One physical ClustOpt search, shareable between a noR and a +R arm."""

    trace_id: str
    trace_identity: str
    dataset_id: str
    view_id: str
    search_seed: int
    #: the resolved pre-search decision: which metrics, at what weights
    utility_source: str
    top_k: Any
    weighting: str
    policy_id: Optional[str]
    selected_metrics: List[str]
    selected_metric_weights: Dict[str, float]
    utility_vector_hash: Optional[str]
    #: the search log, one row per evaluated candidate
    trial_log: pd.DataFrame
    n_evaluations: int
    requested_trials: Optional[int]
    valid_trials: Optional[int]
    failed_trials: Optional[int]
    #: partition hash per candidate, aligned to trial_log rows
    partition_hashes: List[Optional[str]]
    #: the 120-dim reranker utility context (60 MLP + 60 KNN), if built
    reranker_utility_vector: Optional[np.ndarray] = None
    runtime_sec: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------- identity
    def candidate_frame(self) -> pd.DataFrame:
        """The comparable projection of the trace, for trace-sharing proofs."""
        cols = ["algorithm", "config_str", "aggregate_score", "valid"]
        present = [c for c in cols if c in self.trial_log.columns]
        out = self.trial_log[present].copy()
        out.insert(0, "candidate_index", np.arange(len(out)))
        out["partition_hash"] = self.partition_hashes
        return out.reset_index(drop=True)

    def trace_fingerprint(self) -> str:
        """A single hash over everything a shared trace must reproduce."""
        f = self.candidate_frame()
        payload = f.to_csv(index=False, float_format="%.17g")
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def summary(self) -> Dict[str, Any]:
        return {"trace_id": self.trace_id, "trace_identity": self.trace_identity,
                "dataset_id": self.dataset_id, "view_id": self.view_id,
                "search_seed": self.search_seed,
                "utility_source": self.utility_source, "top_k": self.top_k,
                "weighting": self.weighting, "policy_id": self.policy_id,
                "n_evaluations": self.n_evaluations,
                "n_selected_metrics": len(self.selected_metrics),
                "trace_fingerprint": self.trace_fingerprint()}


def assert_no_ground_truth(frame: pd.DataFrame) -> None:
    """Refuse to carry an evaluation column into a selector's input."""
    banned = {"ari", "nmi", "ami", "adjusted_rand", "true_k", "y_true", "labels_true"}
    hit = sorted(c for c in frame.columns if str(c).strip().lower() in banned)
    if hit:
        raise AdapterError(ClustOptFailure.FEATURE_SCHEMA_MISMATCH,
                           "ground-truth column(s) present: %s" % hit)
