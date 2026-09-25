"""The runner-owned dataset/view contract. Ground truth cannot reach it.

Methods never construct or guess a view: the runner builds one ``DatasetView``
per (dataset, view) and hands it to every physical builder and scorer. That is
what makes the frozen V2 semantics enforceable in one place -- ``X_decision`` is
the view the candidates are fitted in, ``X_full`` is the frozen full 2-D
evaluation representation of the SAME observations in the SAME row order, and a
metric receives one or the other according to its declared ``metric.space``.

There is deliberately no ``y_true`` field, not even ``None``. Ground truth lives
behind the offline-evaluation boundary in a physically separate object, so a
method cannot reach labels by ordinary attribute navigation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Optional, Tuple

import numpy as np

from . import hashing as H

VIEW_IDS: Tuple[str, ...] = ("x_only", "y_only", "xy_2d")
#: fields that must never appear on anything the method layer can see
FORBIDDEN_FIELDS = frozenset({
    "y", "y_true", "labels_true", "target", "true_k", "ari", "nmi", "ami",
    "fmi", "ground_truth", "class_distribution", "cvi_star"})


class ViewContractError(RuntimeError):
    """The view contract was violated. Never downgraded to a warning."""


@dataclass(frozen=True)
class DatasetView:
    """One scientific (dataset, view) unit. Label-free by construction."""

    dataset_id: str
    source: str
    view_id: str
    X_decision: np.ndarray            # write-protected
    X_full: np.ndarray                # write-protected
    sample_count: int
    sample_identity_hash: str         # semantic_array_v1 over X_full
    decision_hash: str                # semantic_array_v1 over X_decision
    full_hash: str                    # semantic_array_v1 over X_full
    dataset_dir: Optional[str] = None
    metadata: Mapping[str, Any] = MappingProxyType({})

    # Some clustering backends (hdbscan's Cython core, for one) require a
    # WRITABLE input buffer and fail with "Cannot create writable memory view
    # from read-only memoryview" otherwise. That would silently turn valid
    # candidates into failures and change scientific output, so algorithm input
    # is always a fresh writable copy while the stored matrices stay read-only.
    def decision_for_algorithm(self) -> np.ndarray:
        return np.array(self.X_decision, dtype=float, copy=True)

    def full_for_algorithm(self) -> np.ndarray:
        return np.array(self.X_full, dtype=float, copy=True)

    def identity(self) -> Mapping[str, Any]:
        return MappingProxyType({
            "dataset_id": self.dataset_id, "source": self.source,
            "view_id": self.view_id, "sample_count": self.sample_count,
            "sample_identity_hash": self.sample_identity_hash,
            "decision_hash": self.decision_hash, "full_hash": self.full_hash,
            "hash_convention": H.SEMANTIC_ARRAY})


def _freeze(a: np.ndarray) -> np.ndarray:
    out = np.array(a, dtype=float, copy=True)
    out.setflags(write=False)
    return out


def build_dataset_view(*, dataset_id: str, source: str, view_id: str,
                       X_full: np.ndarray, dataset_dir: Optional[Path] = None,
                       metadata: Optional[Mapping[str, Any]] = None
                       ) -> DatasetView:
    """Derive the view from the frozen full matrix. The runner owns this.

    ``X_decision`` is a projection of ``X_full``, so identical rows in identical
    order are guaranteed by construction rather than by a downstream check.
    """
    if view_id not in VIEW_IDS:
        raise ViewContractError("unknown view_id %r" % view_id)
    Xf = np.asarray(X_full, dtype=float)
    if Xf.ndim != 2 or Xf.shape[1] < 2:
        raise ViewContractError(
            "X_full must be the 2-D evaluation representation; got shape %s"
            % (Xf.shape,))
    if view_id == "x_only":
        Xd = Xf[:, :1]
    elif view_id == "y_only":
        Xd = Xf[:, 1:2]
    else:
        Xd = Xf
    md = dict(metadata or {})
    bad = sorted(k for k in md if k.lower() in FORBIDDEN_FIELDS)
    if bad:
        raise ViewContractError("ground-truth field(s) in view metadata: %s" % bad)
    Xd, Xf = _freeze(Xd), _freeze(Xf)
    return DatasetView(
        dataset_id=dataset_id, source=source, view_id=view_id,
        X_decision=Xd, X_full=Xf, sample_count=int(Xf.shape[0]),
        sample_identity_hash=H.semantic_array_hash(Xf),
        decision_hash=H.semantic_array_hash(Xd),
        full_hash=H.semantic_array_hash(Xf),
        dataset_dir=None if dataset_dir is None else str(dataset_dir),
        metadata=MappingProxyType(md))


def assert_row_alignment(view: DatasetView) -> None:
    """A feature-space difference must never hide a row/sample mismatch."""
    if view.X_decision.shape[0] != view.X_full.shape[0]:
        raise ViewContractError(
            "sample-count mismatch: X_decision %d rows, X_full %d rows"
            % (view.X_decision.shape[0], view.X_full.shape[0]))
    if view.X_decision.shape[0] != view.sample_count:
        raise ViewContractError("sample_count disagrees with the matrices")
    col = {"x_only": 0, "y_only": 1}.get(view.view_id)
    if col is not None:
        if not np.array_equal(view.X_decision[:, 0], view.X_full[:, col]):
            raise ViewContractError(
                "row order differs between X_decision and X_full")
    elif not np.array_equal(view.X_decision, view.X_full):
        raise ViewContractError("xy_2d decision space must equal X_full")


def assert_label_free(obj: Any) -> None:
    """No ground-truth attribute may be reachable on a method-facing object."""
    hits = sorted(a for a in dir(obj)
                  if not a.startswith("_") and a.lower() in FORBIDDEN_FIELDS)
    if hits:
        raise ViewContractError("ground-truth attribute(s) reachable: %s" % hits)
