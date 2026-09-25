"""Stage 3A-1: the canonical 2x2 metric-inventory partition.

The decomposition experiment needs three disjoint metric sets, and each one is
resolved from the source that *owns* it -- never from a hand-written list and
never from a metric name:

``ORIGINAL``
    The 7 internal CVIs the ML2DAC re-implementation (and therefore AutoClust)
    ships with. Owner: the master ``metric_registry.parquet`` flag
    ``used_by_ml2dac``, cross-checked against
    :data:`unified_mkr.metric_registry.ML2DAC_CVIS`.

``NEW46``
    The 46 newly designed ClustOpt image/pattern indices. Owner:
    ``models/Clustering_Repository_Builder/experiments/paper_analysis/
    metric_registry.py``, where ``is_new := not is_classic_head_group`` -- i.e.
    membership is decided by the *implementation package* of the metric class
    and by the MLP head group the utility predictor was trained with, then
    cross-checked against the live ClustOpt ``METRIC_REGISTRY`` and against
    ``models/metric_utility_mlp/head_groups.METRIC_HEAD_GROUPS``.

``ESTABLISHED``
    Defined residually, exactly as the brief specifies:
    ``EXTENDED_REQUESTED - ORIGINAL - NEW46``. ``EXTENDED_REQUESTED`` is the
    builders' own extended pool (``used_by_ml2dac`` OR ``used_by_clustopt``), so
    the residual cannot drift from what the historical Extended build used.

Two NEW46 members are all-NaN on the training repository and are deterministically
dropped by the builders' existing ``_exclude_all_nan`` rule. They are recorded
here, but the scientific statement stays *"46 newly designed metrics, of which 44
are evaluable on this training repository and 2 are deterministically excluded as
all-NaN"* -- never "New44".
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from . import io_utils

#: The two canonical NEW46 members that are 100% NaN on the training repository.
#: Recorded under their CANONICAL registry ids; the head-group spelling is the
#: alias (see :func:`alias_map`).
EXPECTED_DEAD: Tuple[str, ...] = (
    "convexity_ratio_image_based",
    "hough_arc_circle_strength",
)
DEAD_EXCLUSION_RULE = (
    "builder `_exclude_all_nan` / `_exclude_all_nan_on_train`: a candidate whose "
    "column is absent from the training evaluations, or is NaN for every "
    "valid_result==True training row, is dropped with reason '100% NaN on train "
    "(dead/renamed or unavailable metric)'"
)

VARIANT_ORIGINAL_PLUS_ESTABLISHED = "original_plus_established"
VARIANT_ORIGINAL_PLUS_NEW46 = "original_plus_new46"


def _sha(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]


# --------------------------------------------------------------- canonical sets
def original_set(registry: pd.DataFrame) -> List[str]:
    """The 7 baseline-internal CVIs, from the master registry flag."""
    from .metric_registry import ML2DAC_CVIS

    ids = sorted(registry[registry["used_by_ml2dac"] == True]["metric_id"])  # noqa: E712
    if sorted(ML2DAC_CVIS) != ids:
        raise AssertionError(
            "master registry used_by_ml2dac (%s) disagrees with ML2DAC_CVIS (%s)"
            % (ids, sorted(ML2DAC_CVIS)))
    return ids


def extended_requested_set(registry: pd.DataFrame) -> List[str]:
    """The builders' extended pool: ``used_by_ml2dac OR used_by_clustopt``.

    Order is the registry column order, exactly as ``_extended_pool`` builds it,
    so a spec generated here can be compared to the historical build without any
    ordering caveat.
    """
    ml2dac = set(registry[registry["used_by_ml2dac"] == True]["metric_id"])  # noqa: E712
    by_id = registry.set_index("metric_id")
    pool: List[str] = []
    for mid in registry["metric_id"]:
        if mid in ml2dac or bool(by_id.loc[mid, "used_by_clustopt"]):
            if mid not in pool:
                pool.append(mid)
    return pool


def new46_set(repo_root: str | Path) -> Tuple[List[str], Dict[str, Any]]:
    """The 46 newly designed ClustOpt indices, plus its provenance payload.

    Three independent sources must agree before the set is returned:
    the paper-analysis taxonomy (``is_new``), the MLP head groups
    (``full60 - classic_cvi``), and the live ClustOpt ``METRIC_REGISTRY``.
    """
    io_utils.ensure_repo_on_path(repo_root)
    from models.Clustering_Repository_Builder.experiments.paper_analysis import (  # noqa: E501
        metric_registry as PMR,
    )
    from models.metric_utility_mlp.head_groups import METRIC_HEAD_GROUPS
    from models.ClustOpt.cluster_validity_indices.metrics.metrics_registry import (  # noqa: E501
        METRIC_REGISTRY as CLUSTOPT_REGISTRY,
    )

    taxonomy = sorted(m for m in PMR.METRIC_NAMES if PMR.IS_NEW[m])

    full60 = [m for ms in METRIC_HEAD_GROUPS.values() for m in ms]
    classic = list(METRIC_HEAD_GROUPS[PMR.CLASSIC_HEAD_GROUP])
    head_derived = sorted({PMR.canonical_metric_name(m)
                           for m in full60 if m not in classic})

    if taxonomy != head_derived:
        raise AssertionError(
            "NEW46 disagreement: taxonomy-only=%s head-only=%s"
            % (sorted(set(taxonomy) - set(head_derived)),
               sorted(set(head_derived) - set(taxonomy))))
    missing = [m for m in taxonomy if m not in CLUSTOPT_REGISTRY]
    if missing:
        raise AssertionError("NEW46 members absent from ClustOpt "
                             "METRIC_REGISTRY: %s" % missing)
    if len(taxonomy) != 46:
        raise AssertionError("NEW46 resolved to %d metrics, expected 46"
                             % len(taxonomy))

    prov = {
        "primary_source": ("models/Clustering_Repository_Builder/experiments/"
                           "paper_analysis/metric_registry.py :: "
                           "is_new = not is_classic_head_group"),
        "cross_check_head_groups": ("models/metric_utility_mlp/head_groups.py :: "
                                    "METRIC_HEAD_GROUPS full60 minus the "
                                    "'%s' group" % PMR.CLASSIC_HEAD_GROUP),
        "cross_check_implementation": ("models/ClustOpt/cluster_validity_indices/"
                                       "metrics/metrics_registry.METRIC_REGISTRY"),
        "sources_agree": True,
        "name_inference_used": False,
        "n_clustopt_metrics": len(CLUSTOPT_REGISTRY),
        "n_full60": len(full60),
        "n_classic_head_group": len(classic),
        "counts_by_category": {
            c: int(sum(1 for m in taxonomy if PMR.CATEGORY_OF[m] == c))
            for c in sorted({PMR.CATEGORY_OF[m] for m in taxonomy})},
        "taxonomy_audit_hash": _sha({
            "n_new": PMR.METRIC_AUDIT["n_new"],
            "n_classic_head_group": PMR.METRIC_AUDIT["n_classic_head_group"],
            "n_metrics": PMR.METRIC_AUDIT["n_metrics"],
            "aliases_applied": PMR.METRIC_AUDIT["aliases_applied"],
            "ambiguous_classic": PMR.METRIC_AUDIT["ambiguous_classic"]}),
        "set_hash": _sha(taxonomy),
    }
    return taxonomy, prov


def alias_map(repo_root: str | Path) -> Dict[str, str]:
    """Head-group spelling -> canonical registry id, from the canonical source."""
    io_utils.ensure_repo_on_path(repo_root)
    from models.Clustering_Repository_Builder.experiments.paper_analysis import (  # noqa: E501
        metric_registry as PMR,
    )
    return dict(PMR.HEAD_GROUP_ALIASES)


# ------------------------------------------------------------------ the design
def resolve_partitions(repo_root: str | Path,
                       registry: pd.DataFrame) -> Dict[str, Any]:
    """Resolve all three partitions and run every §5 factorial gate.

    Raises ``AssertionError`` on any gate failure. Nothing here is repairable by
    hand: a failure means the canonical sources disagree and the experiment must
    stop.
    """
    ORIG = original_set(registry)
    EXT_REQ = extended_requested_set(registry)
    NEW46, new46_prov = new46_set(repo_root)
    EST = sorted(set(EXT_REQ) - set(ORIG) - set(NEW46))

    checks: List[Dict[str, Any]] = []

    def gate(name: str, ok: bool, detail: str) -> None:
        checks.append({"gate": name, "passed": bool(ok), "detail": detail})
        if not ok:
            raise AssertionError("metric-partition gate failed: %s -- %s"
                                 % (name, detail))

    gate("original_count_7", len(ORIG) == 7, "%d" % len(ORIG))
    gate("established_count_10", len(EST) == 10, "%d" % len(EST))
    gate("new46_count_46", len(NEW46) == 46, "%d" % len(NEW46))
    gate("original_disjoint_established", not (set(ORIG) & set(EST)),
         "overlap=%s" % sorted(set(ORIG) & set(EST)))
    gate("original_disjoint_new46", not (set(ORIG) & set(NEW46)),
         "overlap=%s" % sorted(set(ORIG) & set(NEW46)))
    gate("established_disjoint_new46", not (set(EST) & set(NEW46)),
         "overlap=%s" % sorted(set(EST) & set(NEW46)))
    union = set(ORIG) | set(EST) | set(NEW46)
    gate("union_equals_extended_requested", union == set(EXT_REQ),
         "sym_diff=%s" % sorted(union ^ set(EXT_REQ)))
    gate("extended_requested_count_63", len(EXT_REQ) == 63, "%d" % len(EXT_REQ))
    gate("dead_metrics_are_new46_members",
         set(EXPECTED_DEAD) <= set(NEW46),
         "dead=%s" % list(EXPECTED_DEAD))

    a1 = sorted(set(ORIG) | set(EST))
    a2 = sorted(set(ORIG) | set(NEW46))
    gate("a1_requested_17", len(a1) == 17, "%d" % len(a1))
    gate("a2_requested_53", len(a2) == 53, "%d" % len(a2))
    a1_live = sorted(set(a1) - set(EXPECTED_DEAD))
    a2_live = sorted(set(a2) - set(EXPECTED_DEAD))
    gate("a1_expected_live_17", len(a1_live) == 17, "%d" % len(a1_live))
    gate("a2_expected_live_51", len(a2_live) == 51, "%d" % len(a2_live))

    return {
        "original": ORIG, "established": EST, "new46": NEW46,
        "extended_requested": sorted(EXT_REQ),
        "extended_requested_order": list(EXT_REQ),
        "a1_requested": a1, "a1_expected_live": a1_live,
        "a2_requested": a2, "a2_expected_live": a2_live,
        "expected_dead": list(EXPECTED_DEAD),
        "alias_map": alias_map(repo_root),
        "new46_provenance": new46_prov,
        "gates": checks,
        "hashes": {"original": _sha(ORIG), "established": _sha(EST),
                   "new46": _sha(NEW46),
                   "extended_requested": _sha(sorted(EXT_REQ))},
    }


def verify_against_existing(parts: Dict[str, Any], derived_root: Path,
                            baseline: str) -> List[Dict[str, Any]]:
    """§10 backward-compatibility: the resolved sets must reproduce A0/A3.

    Read-only. Compares the canonically resolved ORIGINAL / EXTENDED-live sets to
    the *materialised* candidate lists of the existing A0/A3 variants, with zero
    tolerance for a symmetric difference. Never retrains or overwrites anything.
    """
    out: List[Dict[str, Any]] = []
    live_ext = sorted(set(parts["extended_requested"]) - set(EXPECTED_DEAD))
    for variant, expected_req, expected_live in (
            ("original_cvis", parts["original"], parts["original"]),
            ("extended_cvis", parts["extended_requested"], live_ext)):
        d = derived_root / variant
        cand = io_utils.read_parquet(d / "cvi_candidates.parquet")
        if cand is None:
            raise AssertionError("cannot read %s/cvi_candidates.parquet" % d)
        got_live = sorted(cand[cand["included"]]["metric_id"])
        got_dead = sorted(cand[~cand["included"]]["metric_id"])
        got_req = sorted(set(got_live) | set(got_dead))
        dirs = {r["metric_id"]: r["direction"] for _, r in cand.iterrows()}
        out.append({
            "baseline": baseline, "variant": variant,
            "requested_match": got_req == sorted(expected_req),
            "requested_sym_diff": sorted(set(got_req) ^ set(expected_req)),
            "live_match": got_live == sorted(expected_live),
            "live_sym_diff": sorted(set(got_live) ^ set(expected_live)),
            "n_requested_existing": len(got_req), "n_live_existing": len(got_live),
            "dead_match": (got_dead == sorted(EXPECTED_DEAD)
                           if variant == "extended_cvis" else got_dead == []),
            "dead_existing": got_dead,
            "all_directions_present": all(
                d_ is not None and str(d_) != "nan" for d_ in dirs.values()),
        })
    return out


__all__ = [
    "EXPECTED_DEAD", "DEAD_EXCLUSION_RULE",
    "VARIANT_ORIGINAL_PLUS_ESTABLISHED", "VARIANT_ORIGINAL_PLUS_NEW46",
    "original_set", "extended_requested_set", "new46_set", "alias_map",
    "resolve_partitions", "verify_against_existing",
]
