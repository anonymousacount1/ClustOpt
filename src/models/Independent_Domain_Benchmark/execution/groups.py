"""The frozen physical-trace groups, derived from the registry and protocol.

Groups are never inferred from similar method names. The 14 scientific METHOD_IDs
come from ``configs/method_registry.json`` and the ClustOpt sharing groups come
from ``search/protocol.py::TRACE_IDENTITY``, which is the same frozen mapping the
seed resolver uses. AutoClust and ML2DAC each form one validated group.

Per dataset/view: ClustOpt 4 + AutoClust 1 + ML2DAC 1 = 6 physical traces serving
14 scientific outputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import json

_ROOT = Path(__file__).resolve().parents[1]
_REPO = _ROOT.parents[1]

EXPECTED_METHODS = 14
EXPECTED_PHYSICAL_GROUPS_PER_VIEW = 6
FAMILIES = ("ClustOpt", "AutoClust", "ML2DAC")


class GroupError(RuntimeError):
    """The frozen plan does not match the registry. Never worked around."""


@dataclass(frozen=True)
class PhysicalGroup:
    group_id: str
    family: str
    method_ids: Tuple[str, ...]
    validated_sharing: bool


def frozen_method_ids() -> List[str]:
    reg = json.loads((_ROOT / "configs" / "method_registry.json")
                     .read_text(encoding="utf-8"))
    ids = [m["method_id"] for m in reg["methods"]]
    if len(ids) != EXPECTED_METHODS or len(set(ids)) != EXPECTED_METHODS:
        raise GroupError("registry holds %d method ids, expected %d"
                         % (len(ids), EXPECTED_METHODS))
    return ids


def physical_groups() -> Dict[str, PhysicalGroup]:
    """Read the groups; do not infer them."""
    import sys
    if str(_REPO / "models") not in sys.path:
        sys.path.insert(0, str(_REPO / "models"))
    from Independent_Domain_Benchmark.search import protocol as PROTO

    ids = frozen_method_ids()
    buckets: Dict[str, List[str]] = {}
    for mid in ids:
        buckets.setdefault(PROTO.seed_identity(mid), []).append(mid)

    groups: Dict[str, PhysicalGroup] = {}
    for gid, members in buckets.items():
        family = next((f for f in FAMILIES if f in members[0]), "UNKNOWN")
        if family == "UNKNOWN":
            raise GroupError("cannot resolve family for %s" % members[0])
        if any(f in m for m in members for f in FAMILIES if f != family):
            raise GroupError("group %s mixes method families" % gid)
        groups[gid] = PhysicalGroup(
            group_id=gid, family=family, method_ids=tuple(sorted(members)),
            validated_sharing=len(members) > 1)

    per_family = {f: sum(1 for g in groups.values() if g.family == f)
                  for f in FAMILIES}
    if per_family != {"ClustOpt": 4, "AutoClust": 1, "ML2DAC": 1}:
        raise GroupError("physical group plan is %s, expected "
                         "{'ClustOpt': 4, 'AutoClust': 1, 'ML2DAC': 1}"
                         % per_family)
    if len(groups) != EXPECTED_PHYSICAL_GROUPS_PER_VIEW:
        raise GroupError("%d physical groups, expected %d"
                         % (len(groups), EXPECTED_PHYSICAL_GROUPS_PER_VIEW))
    if sum(len(g.method_ids) for g in groups.values()) != EXPECTED_METHODS:
        raise GroupError("groups do not cover all 14 methods exactly once")
    return groups


def plan_counts(n_datasets: int, n_views: int) -> Mapping[str, int]:
    """Structural counts, asserted rather than observed."""
    return {
        "physical_traces_per_dataset_view": EXPECTED_PHYSICAL_GROUPS_PER_VIEW,
        "scientific_outputs_per_dataset_view": EXPECTED_METHODS,
        "physical_traces": n_datasets * n_views * EXPECTED_PHYSICAL_GROUPS_PER_VIEW,
        "scientific_outputs": n_datasets * n_views * EXPECTED_METHODS,
    }
