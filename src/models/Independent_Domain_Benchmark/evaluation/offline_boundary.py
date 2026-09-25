"""The offline evaluation boundary — structure only, no execution here.

Ground truth lives on this side of the wall and nowhere else. The method layer
receives a ``DatasetView`` that has no ``y_true`` field at all, so labels cannot
be reached by ordinary object navigation; an evaluator is constructed separately
and only after the scientific arm records are COMPLETE.

Stage 4B-4 defines the contract. It does not evaluate anything: no ARI is
computed, and the frozen external corpus is not touched.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np

EVALUATION_CONTRACT_VERSION = "idb_stage4_evaluation_v1"

#: offline metrics the evaluator may write. They live ONLY in evaluation records.
OFFLINE_METRICS = ("ari", "nmi", "ami", "fmi")


class EvaluationBoundaryError(RuntimeError):
    """An attempt to cross the boundary in the wrong direction."""


@dataclass(frozen=True)
class GroundTruth:
    """Deliberately a separate object from DatasetView. Never passed to methods."""

    dataset_id: str
    view_id: str
    y_true: np.ndarray
    true_k: Optional[int]
    sample_identity_hash: str          # must equal the DatasetView's


@dataclass(frozen=True)
class EvaluationRequest:
    """What the evaluator consumes: an immutable scientific result reference."""

    method_id: str
    dataset_id: str
    view_id: str
    scientific_record_key: str
    scientific_content_hash: str
    final_labels_hash: Optional[str]
    physical_trace_id: str


def assert_method_phase_complete(record: Mapping[str, Any]) -> None:
    """Evaluation may only read a COMPLETE scientific record."""
    if record.get("state") != "COMPLETE":
        raise EvaluationBoundaryError(
            "evaluation requires a COMPLETE scientific record; got %r"
            % record.get("state"))


def assert_sample_alignment(gt: GroundTruth, view_identity: Mapping[str, Any]) -> None:
    """Labels must describe the same samples in the same order as the view."""
    if gt.sample_identity_hash != view_identity.get("sample_identity_hash"):
        raise EvaluationBoundaryError(
            "ground truth does not belong to this dataset/view")
    if gt.view_id != view_identity.get("view_id"):
        raise EvaluationBoundaryError("view mismatch")


def evaluation_record_schema() -> Mapping[str, Any]:
    """The shape of an offline evaluation record. Written by a later stage."""
    return {
        "evaluation_contract_version": EVALUATION_CONTRACT_VERSION,
        "references": ["method_id", "dataset_id", "view_id",
                       "scientific_record_key", "scientific_content_hash",
                       "physical_trace_id"],
        "metrics": list(OFFLINE_METRICS),
        "rules": [
            "evaluation records are a SEPARATE layer; offline metrics never "
            "appear in a scientific arm record",
            "a failed view is not evaluated and is never assigned ARI = 0 or any "
            "other numeric sentinel",
            "BestView is the maximum over SUCCESSFULLY produced view outputs; a "
            "failed view is not a BestView candidate",
            "if all three views fail, the dataset-level method result is FAILURE "
            "and no artificial value is created",
            "xy_2d is additionally reported on its own",
        ],
    }


def best_view(successful: Mapping[str, float]) -> Optional[str]:
    """BestView over successfully produced views only. No sentinels."""
    if not successful:
        return None
    return max(successful, key=lambda v: successful[v])
