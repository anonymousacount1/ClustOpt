"""Final-partition materialisation and migration into an amended run.

The parent Stage-4C run recorded ``final_labels_hash`` but not the label vector,
so its successful arms cannot be evaluated offline. This module recovers the
missing vectors **without re-selecting anything**: it re-instantiates the exact
configuration the parent arm already chose and then proves, cryptographically,
that the resulting partition is the same one.

The acceptance rule is the whole point. A reconstruction is migrated only if it
reproduces the parent's stored hash under the convention that arm recorded --
raw label identity where the parent stored it, canonical partition identity where
cluster ids may legitimately permute. Canonical equality is scientifically
sufficient for ARI, because relabelling does not change a partition. Anything
else -- same cluster count, close agreement, "looks right" -- is rejected and the
arm goes to the targeted-rerun set.

Nothing here decides what to rerun using outcomes: the rule is reconstruct,
verify, and fall back on verification failure alone. No ground truth is read.
"""
from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_REPO = _ROOT.parents[1]

RECOVERY_CONTRACT = "final_partition_recovery_v1"
ORIGIN_RECOVERED = "VERIFIED_PARENT_PARTITION_RECOVERY"
ORIGIN_RERUN = "TARGETED_RERUN"
ORIGIN_EXECUTED = "EXECUTED"


class RecoveryOutcome:
    VERIFIED = "VERIFIED"
    HASH_MISMATCH = "HASH_MISMATCH"
    NOT_RECONSTRUCTIBLE = "NOT_RECONSTRUCTIBLE"
    ERROR = "ERROR"


@dataclass(frozen=True)
class RecoveryResult:
    outcome: str
    labels: Optional[np.ndarray]
    raw_hash: Optional[str]
    canonical_hash: Optional[str]
    matched_convention: Optional[str]
    detail: str = ""


def _raw_hash(a) -> str:
    return hashlib.sha256(np.asarray(a, dtype=np.int64).tobytes()).hexdigest()[:16]


def _canonical_hash(a) -> str:
    from Independent_Domain_Benchmark.methods.clustopt_trace import partition_hash
    return partition_hash(np.asarray(a))


def family_of(method_id: str) -> str:
    for f in ("ClustOpt", "AutoClust", "ML2DAC"):
        if f in method_id:
            return f
    raise ValueError("unknown family for %r" % method_id)


def _clustopt_labels(view, selected_configuration: Mapping[str, Any]):
    """Re-instantiate a ClustOpt selection through ClustOpt's own executor."""
    from models.Clustering_Repository_Builder.experiments.experiment_execution import (
        clustopt_execution as CE)
    from models.ClustOpt.search_space.Clustering_Search_Space import (
        ClusteringSearchSpace)
    import sys
    mkr = str(_REPO / "experiments/external_baselines/Unified_MKR")
    if mkr not in sys.path:
        sys.path.insert(0, mkr)
    from unified_mkr.configuration import resolve_view_configurations

    cfg_str = selected_configuration.get("config_str")
    if not cfg_str:
        return None
    ss = ClusteringSearchSpace.from_dict(
        resolve_view_configurations(
            str(_REPO / "models/ClustOpt/configs/utilities_maps"),
            view.view_id).search_space_config)
    return CE._predict_best_labels(ss, {"config_str": cfg_str},
                                   view.decision_for_algorithm())


def _generic_labels(view, algorithm: str, params: Mapping[str, Any]):
    """Re-instantiate an AutoClust / ML2DAC selection through the shared executor."""
    import sys
    mkr = str(_REPO / "experiments/external_baselines/Unified_MKR")
    if mkr not in sys.path:
        sys.path.insert(0, mkr)
    from unified_mkr.clustering import run_configuration
    from unified_mkr.configuration import ConfigSpec, resolve_view_configurations
    from models.ClustOpt.search_space.Clustering_Search_Space import (
        ClusteringSearchSpace)

    ss = ClusteringSearchSpace.from_dict(
        resolve_view_configurations(
            str(_REPO / "models/ClustOpt/configs/utilities_maps"),
            view.view_id).search_space_config)
    res = run_configuration(ss, ConfigSpec(algorithm=algorithm,
                                           params=dict(params),
                                           original_config_id=""),
                            view.decision_for_algorithm())
    return getattr(res, "labels", None)


def recover_partition(parent_content: Mapping[str, Any], method_id: str,
                      view) -> RecoveryResult:
    """Materialise the parent arm's final partition and verify it.

    No search, no policy resolution, no CVI scoring, no reranking: only the
    configuration the parent already selected is instantiated.
    """
    stored = parent_content.get("final_labels_hash")
    algo = parent_content.get("selected_algorithm")
    cfg = parent_content.get("selected_configuration") or {}
    if not stored or not algo:
        return RecoveryResult(RecoveryOutcome.NOT_RECONSTRUCTIBLE, None, None, None,
                              None, "parent record lacks hash or algorithm")
    try:
        if family_of(method_id) == "ClustOpt":
            labels = _clustopt_labels(view, cfg)
        else:
            labels = _generic_labels(view, algo, cfg)
    except Exception as exc:  # noqa: BLE001
        return RecoveryResult(RecoveryOutcome.ERROR, None, None, None, None,
                              "%s: %s" % (type(exc).__name__, str(exc)[:160]))
    if labels is None:
        return RecoveryResult(RecoveryOutcome.NOT_RECONSTRUCTIBLE, None, None, None,
                              None, "executor returned no labels for %s" % algo)
    a = np.asarray(labels, dtype=np.int64).reshape(-1)
    if a.shape[0] != int(view.sample_count):
        return RecoveryResult(RecoveryOutcome.NOT_RECONSTRUCTIBLE, None, None, None,
                              None, "sample count %d != %d"
                              % (a.shape[0], view.sample_count))
    raw, canon = _raw_hash(a), _canonical_hash(a)
    if stored == raw:
        return RecoveryResult(RecoveryOutcome.VERIFIED, a, raw, canon, "raw")
    if stored == canon:
        return RecoveryResult(RecoveryOutcome.VERIFIED, a, raw, canon, "canonical")
    return RecoveryResult(RecoveryOutcome.HASH_MISMATCH, None, raw, canon, None,
                          "stored %s != raw %s / canonical %s" % (stored, raw, canon))


def migrate_record(parent_rec: Mapping[str, Any], rec: RecoveryResult,
                   view) -> Dict[str, Any]:
    """The amended-run content for a verified recovery. Parent metadata is kept."""
    if rec.outcome != RecoveryOutcome.VERIFIED or rec.labels is None:
        raise ValueError("only a VERIFIED recovery may be migrated")
    c = dict(parent_rec["content"])
    c.update({
        "final_labels": [int(v) for v in rec.labels],
        "final_labels_raw_hash": rec.raw_hash,
        "canonical_partition_hash": rec.canonical_hash,
        "sample_count": int(view.sample_count),
        "result_contract": "scientific_arm_result_v2",
        "result_origin": ORIGIN_RECOVERED,
        "recovery": {
            "contract": RECOVERY_CONTRACT,
            "verified_against": rec.matched_convention,
            "parent_final_labels_hash": parent_rec["content"]["final_labels_hash"],
            "parent_provenance_hash": parent_rec.get("provenance_hash"),
            "parent_content_hash": parent_rec.get("content_hash"),
            "note": ("the parent-selected configuration was re-instantiated and the "
                     "resulting partition proved identical to the parent result; no "
                     "search, scoring or selection was repeated"),
        },
    })
    return c
