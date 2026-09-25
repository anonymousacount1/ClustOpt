"""Family planning: load planning JSONs, expand them into dataset jobs.

A *family planning config* is a higher-level config that describes a whole
family of synthetic datasets at once. It is **not** an atomic HYBRID_SCM
dataset config. The planning runner expands it into a flat list of
``DatasetJob`` records, each describing one raw dataset to be generated.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

from .difficulty import DIFFICULTIES


_DEFAULT_DIFFICULTY_POLICY: Dict[str, float] = {
    "easy": 0.20,
    "medium": 0.35,
    "hard": 0.30,
    "very_hard": 0.15,
}


@dataclass
class FamilySubfamilyPlan:
    subfamily_id: str
    cluster_budget: Dict[int, int]
    component_strategy: Dict[str, Any]
    parameter_ranges: Dict[str, Any] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return int(sum(self.cluster_budget.values()))


@dataclass
class FamilyPlan:
    family_id: str
    family_name: str
    family_seed: int
    n_features: int
    feature_names: List[str]
    bounds: Dict[str, List[float]]
    n_points_policy: Dict[str, Any]
    difficulty_policy: Dict[str, float]
    default_rendering: Dict[str, Any]
    subfamilies: List[FamilySubfamilyPlan]
    postprocess: Dict[str, Any] = field(default_factory=dict)
    raw_config: Dict[str, Any] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return int(sum(s.total for s in self.subfamilies))


@dataclass
class DatasetJob:
    family_id: str
    family_name: str
    family_seed: int
    subfamily: FamilySubfamilyPlan
    cluster_count: int
    difficulty: str
    local_idx: int           # index inside (subfamily, cluster_count, difficulty) cell
    cell_size: int           # how many datasets share the same cell
    subfamily_idx: int       # 1-based index of subfamily inside the family
    subfamily_local: int     # 1-based index of this job inside the subfamily
    global_idx: int          # 1-based index of this job inside the family run
    dataset_seed: int


# =========================================================
# Loading and validation
# =========================================================

def _require(d: Mapping[str, Any], keys: List[str], where: str) -> None:
    missing = [k for k in keys if k not in d]
    if missing:
        raise KeyError(f"Missing keys {missing} in {where}.")


def load_family_plan(path: Path) -> FamilyPlan:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(
        raw,
        [
            "family_id",
            "family_name",
            "family_seed",
            "n_features",
            "feature_names",
            "bounds",
            "subfamilies",
        ],
        str(path),
    )

    n_features = int(raw["n_features"])
    feature_names = list(raw["feature_names"])
    if len(feature_names) != n_features:
        raise ValueError("feature_names length must equal n_features.")

    bounds = dict(raw["bounds"])
    if "low" not in bounds or "high" not in bounds:
        raise ValueError("bounds must contain 'low' and 'high'.")
    if len(bounds["low"]) != n_features or len(bounds["high"]) != n_features:
        raise ValueError("bounds low/high must have length n_features.")

    difficulty_policy = dict(raw.get("difficulty_policy") or _DEFAULT_DIFFICULTY_POLICY)
    for d in DIFFICULTIES:
        if d not in difficulty_policy:
            raise ValueError(f"difficulty_policy missing key '{d}' in {path}.")
    total_share = sum(float(difficulty_policy[d]) for d in DIFFICULTIES)
    if abs(total_share - 1.0) > 1e-6:
        raise ValueError(
            f"difficulty_policy must sum to 1.0, got {total_share:.6f} in {path}.")

    subfamilies: List[FamilySubfamilyPlan] = []
    for s in raw["subfamilies"]:
        _require(s, ["subfamily_id", "cluster_budget", "component_strategy"], str(path))
        cluster_budget_raw = s["cluster_budget"]
        cluster_budget: Dict[int, int] = {}
        for k, v in cluster_budget_raw.items():
            cluster_budget[int(k)] = int(v)
        if not cluster_budget:
            raise ValueError(
                f"subfamily '{s['subfamily_id']}' has empty cluster_budget in {path}.")
        subfamilies.append(
            FamilySubfamilyPlan(
                subfamily_id=str(s["subfamily_id"]),
                cluster_budget=cluster_budget,
                component_strategy=dict(s["component_strategy"]),
                parameter_ranges=dict(s.get("parameter_ranges") or {}),
            )
        )

    return FamilyPlan(
        family_id=str(raw["family_id"]),
        family_name=str(raw["family_name"]),
        family_seed=int(raw["family_seed"]),
        n_features=n_features,
        feature_names=feature_names,
        bounds=bounds,
        n_points_policy=dict(raw.get("n_points_policy") or {}),
        difficulty_policy=difficulty_policy,
        default_rendering=dict(raw.get("default_rendering") or {}),
        subfamilies=subfamilies,
        postprocess=dict(raw.get("postprocess") or {}),
        raw_config=raw,
    )


# =========================================================
# Difficulty splitting (deterministic rounding)
# =========================================================

def split_difficulty(total: int, policy: Mapping[str, float]) -> Dict[str, int]:
    """Largest-remainder split that preserves the exact total.

    Returns a dict ``{difficulty: count}`` for the four difficulties.
    """
    if total <= 0:
        return {d: 0 for d in DIFFICULTIES}

    shares = {d: float(policy[d]) for d in DIFFICULTIES}
    raw = {d: total * shares[d] for d in DIFFICULTIES}
    floors = {d: int(raw[d]) for d in DIFFICULTIES}
    remaining = total - sum(floors.values())
    if remaining > 0:
        # Order difficulties by the largest fractional remainder, breaking ties
        # by the spec's canonical difficulty order.
        order = sorted(
            DIFFICULTIES,
            key=lambda d: (-(raw[d] - floors[d]), DIFFICULTIES.index(d)),
        )
        for d in order[:remaining]:
            floors[d] += 1
    return floors


# =========================================================
# Deterministic seed derivation
# =========================================================

_UINT32_MAX = 2**32 - 1


def derive_dataset_seed(
    family_seed: int,
    family_id: str,
    subfamily_id: str,
    cluster_count: int,
    difficulty: str,
    local_idx: int,
) -> int:
    """Reproducible 32-bit seed from the planning coordinates."""
    payload = (
        f"{int(family_seed)}|{family_id}|{subfamily_id}|"
        f"c{int(cluster_count)}|{difficulty}|i{int(local_idx)}"
    )
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") & _UINT32_MAX


# =========================================================
# Job expansion
# =========================================================

def _iter_cluster_counts(budget: Mapping[int, int]) -> List[int]:
    return sorted(int(k) for k in budget.keys())


def expand_jobs(plan: FamilyPlan) -> List[DatasetJob]:
    """Expand a family plan into the full ordered list of dataset jobs.

    Order is stable: subfamilies in declaration order; cluster counts ascending;
    difficulties in canonical order ``easy → medium → hard → very_hard``;
    ``local_idx`` ascending inside each cell.
    """
    jobs: List[DatasetJob] = []
    global_idx = 0
    for sub_idx, sub in enumerate(plan.subfamilies, start=1):
        subfamily_local = 0
        for cluster_count in _iter_cluster_counts(sub.cluster_budget):
            cell_total = int(sub.cluster_budget[cluster_count])
            split = split_difficulty(cell_total, plan.difficulty_policy)
            for difficulty in DIFFICULTIES:
                count = int(split[difficulty])
                for local_idx in range(count):
                    subfamily_local += 1
                    global_idx += 1
                    seed = derive_dataset_seed(
                        plan.family_seed,
                        plan.family_id,
                        sub.subfamily_id,
                        cluster_count,
                        difficulty,
                        local_idx,
                    )
                    jobs.append(
                        DatasetJob(
                            family_id=plan.family_id,
                            family_name=plan.family_name,
                            family_seed=plan.family_seed,
                            subfamily=sub,
                            cluster_count=cluster_count,
                            difficulty=difficulty,
                            local_idx=local_idx,
                            cell_size=count,
                            subfamily_idx=sub_idx,
                            subfamily_local=subfamily_local,
                            global_idx=global_idx,
                            dataset_seed=seed,
                        )
                    )
    return jobs


def family_job_counts(plan: FamilyPlan) -> Tuple[int, Dict[str, Dict[int, int]]]:
    """Return ``(total, {subfamily_id: {cluster_count: cell_total}})``."""
    counts: Dict[str, Dict[int, int]] = {}
    total = 0
    for sub in plan.subfamilies:
        per_cluster: Dict[int, int] = {}
        for cluster_count in _iter_cluster_counts(sub.cluster_budget):
            cell_total = int(sub.cluster_budget[cluster_count])
            per_cluster[cluster_count] = cell_total
            total += cell_total
        counts[sub.subfamily_id] = per_cluster
    return total, counts
