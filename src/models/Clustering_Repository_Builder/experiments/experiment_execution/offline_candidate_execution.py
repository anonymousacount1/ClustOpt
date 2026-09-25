"""Offline fixed-candidate execution for the Stage-1 2x3 ablation.

This is the offline counterpart of :mod:`clustopt_execution`. Where the online
module builds a ClustOpt pipeline and *searches* for candidates with Optuna,
this module scores an existing, immutable candidate set:

    <dataset_dir>/utility/<view_id>/clustopt_results.csv    (32 fixed candidates)

Every arm sees exactly the same 32 candidates for a given (dataset, view), so
differences between arms are attributable to the objective alone -- no Optuna
trajectory, no wall-clock cap, no delivered-trial difference, no clustering
randomness.

It never reimplements ClustOpt science. Metric selection and weighting come from
the production resolvers
(:mod:`models.ClustOpt.cluster_validity_indices.dynamic_metric_selection`), and
the objective is the same weighted sum over the *oriented/normalised* per-CVI
columns that :class:`GenericCVIEvaluator` aggregates online.

Numerics note: the ``*_norm`` columns are used, never the raw ones. Stage-0
measured ``-inf`` sentinels in the raw CVI columns (100 % of cells on
``valid=False`` rows, 0.42 % on valid rows) while the normalised columns carry
no Inf/NaN on valid candidates.

ARI is *never* an input to selection. It is read only after the argmax has been
chosen, and for the arm-independent candidate ceiling.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

# Objective tie tolerance. Candidates whose J is within this of the max are
# considered tied; the selector still returns the first such candidate in stored
# order, mirroring pandas' ``idxmax`` first-wins behaviour used online.
TIE_TOL: float = 1e-12

CANDIDATE_CSV = "clustopt_results.csv"
UTILITY_VECTOR = "utility/<view_id>/utility_vector.json"
WEIGHTING_RAW = "normalized_positive"
TOP_K = 5

# Base (non-metric) columns every candidate row must carry.
REQUIRED_BASE_COLUMNS: Tuple[str, ...] = (
    "algorithm", "config_str", "valid", "n_clusters_wo_noise", "ARI",
)


class OfflineCandidateError(RuntimeError):
    """Raised when the stored candidate set violates the Stage-1C contract."""


@dataclass
class ArmSpec:
    """One factorial arm: metric inventory x utility strategy."""

    method_id: str
    inventory: str            # 'head14' | 'full60'
    strategy: str             # 'uniform' | 'predicted' | 'oracle'
    deployable: bool
    model_run_dir: Optional[str] = None      # predicted arms only


@dataclass
class OfflineRunOutcome:
    """Result of scoring one (dataset, view, arm)."""

    method_id: str
    inventory: str
    strategy: str
    deployable: bool
    selected_metrics: List[str] = field(default_factory=list)
    selected_weights: Dict[str, float] = field(default_factory=dict)
    resolver_debug: Dict[str, Any] = field(default_factory=dict)
    n_candidates: int = 0
    n_valid_candidates: int = 0
    candidate_set_fingerprint: str = ""
    selected_candidate_index: Optional[int] = None
    selected_algorithm: Optional[str] = None
    selected_config_str: Optional[str] = None
    objective_j: Optional[float] = None
    max_objective_j: Optional[float] = None
    n_tied_at_max: int = 0
    tie_flag: bool = False
    ari: Optional[float] = None
    observed_k: Optional[int] = None
    best_possible_ari: Optional[float] = None
    ari_regret: Optional[float] = None
    ceiling_hit: Optional[bool] = None
    # Diagnostics over the tied set. NEVER used to choose a candidate.
    tie_ari_min: Optional[float] = None
    tie_ari_mean: Optional[float] = None
    tie_ari_max: Optional[float] = None


# --------------------------------------------------------------------------- io
def candidate_csv_path(dataset_dir: Path, view_id: str) -> Path:
    return Path(dataset_dir) / "utility" / view_id / CANDIDATE_CSV


def _fnum(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def _is_valid(value: Any) -> bool:
    return str(value).strip().lower() in ("true", "1")


def load_candidate_set(dataset_dir: Path, view_id: str, *,
                       expected_count: int = 32) -> Dict[str, Any]:
    """Load and validate the fixed candidate set for one (dataset, view).

    Returns ``{"frame", "metrics", "fingerprint"}``. Raises
    :class:`OfflineCandidateError` on any contract violation -- incomplete
    records are never silently dropped.
    """
    path = candidate_csv_path(dataset_dir, view_id)
    if not path.is_file():
        raise OfflineCandidateError(f"candidate file missing: {path}")

    df = pd.read_csv(path)
    if len(df) != expected_count:
        raise OfflineCandidateError(
            f"{path}: expected {expected_count} candidates, found {len(df)}")

    missing_base = [c for c in REQUIRED_BASE_COLUMNS if c not in df.columns]
    if missing_base:
        raise OfflineCandidateError(f"{path}: missing base columns {missing_base}")

    metrics = [c.split(" (w=")[0] for c in df.columns if "(w=" in c]
    missing_norm = [m for m in metrics if f"{m}_norm" not in df.columns]
    if missing_norm:
        raise OfflineCandidateError(
            f"{path}: {len(missing_norm)} metrics lack a _norm column, e.g. "
            f"{missing_norm[:3]}")

    # Candidate identity must be unique and stable.
    identity = (df["algorithm"].astype(str) + "|" + df["config_str"].astype(str))
    if identity.duplicated().any():
        dups = identity[identity.duplicated()].tolist()[:3]
        raise OfflineCandidateError(f"{path}: duplicate candidate identities {dups}")

    # Normalised CVI values must be finite on VALID candidates (invalid rows are
    # excluded from selection anyway).
    valid_mask = df["valid"].map(_is_valid)
    norm_cols = [f"{m}_norm" for m in metrics]
    if valid_mask.any():
        block = df.loc[valid_mask, norm_cols].apply(pd.to_numeric, errors="coerce")
        if not block.notna().all().all():
            bad = block.columns[block.isna().any()].tolist()[:3]
            raise OfflineCandidateError(
                f"{path}: NaN/Inf in normalised CVI columns on valid candidates, "
                f"e.g. {bad}")

    import hashlib
    fingerprint = hashlib.sha256(
        "\n".join(identity.tolist()).encode("utf-8")).hexdigest()[:16]
    return {"frame": df, "metrics": metrics, "fingerprint": fingerprint}


# ------------------------------------------------------------------- weighting
def uniform_weights(metric_names: Sequence[str]) -> Dict[str, float]:
    """Normalised uniform weights: 1/N each, summing to 1."""
    w = 1.0 / len(metric_names)
    return {m: w for m in metric_names}


def resolve_arm_weights(arm: ArmSpec, *, dataset_dir: Path, view_id: str,
                        head14: Sequence[str], full60: Sequence[str],
                        repo_root: Path) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Resolve one arm's metric weights using the PRODUCTION resolvers."""
    inventory = list(head14 if arm.inventory == "head14" else full60)

    if arm.strategy == "uniform":
        weights = uniform_weights(inventory)
        debug = {"mode": "uniform_normalized", "inventory": arm.inventory,
                 "n_metrics": len(inventory),
                 "selected_metrics": list(weights), "final_weights": dict(weights)}
        return weights, debug

    if arm.strategy == "oracle":
        from models.ClustOpt.cluster_validity_indices.dynamic_metric_selection.resolvers import (
            resolve_oracle_weights,
        )
        params: Dict[str, Any] = {
            "top_k": TOP_K, "weighting": WEIGHTING_RAW,
            "metric_name_prefix": "utility__", "utility_source": UTILITY_VECTOR,
        }
        if arm.inventory == "head14":
            params["candidate_metrics"] = list(head14)
        res = resolve_oracle_weights(params=params, dataset_dir=str(dataset_dir),
                                     view_id=view_id)
        return dict(res.weights), dict(res.debug)

    if arm.strategy == "predicted":
        from models.ClustOpt.cluster_validity_indices.dynamic_metric_selection.resolvers import (
            resolve_regressor_weights,
        )
        if not arm.model_run_dir:
            raise OfflineCandidateError(
                f"arm {arm.method_id} is a predicted arm but has no model_run_dir")
        params = {
            "model_run_dir": arm.model_run_dir, "checkpoint_policy": "fold_ensemble",
            "top_k": TOP_K, "weighting": WEIGHTING_RAW,
            "metric_name_prefix": "utility__",
            "feature_source": "features/features_records.csv", "view_aware": True,
        }
        res = resolve_regressor_weights(params=params, dataset_dir=str(dataset_dir),
                                        view_id=view_id)
        return dict(res.weights), dict(res.debug)

    raise OfflineCandidateError(f"unknown utility strategy: {arm.strategy}")


# ------------------------------------------------------------------- objective
def score_candidates(frame: pd.DataFrame, weights: Dict[str, float]) -> List[Optional[float]]:
    """J = sum_m w_m * <metric>_norm, per candidate row (None if unusable)."""
    out: List[Optional[float]] = []
    cols = {m: f"{m}_norm" for m in weights}
    for _, row in frame.iterrows():
        acc, ok = 0.0, True
        for m, w in weights.items():
            v = _fnum(row.get(cols[m]))
            if v is None:
                ok = False
                break
            acc += w * v
        out.append(acc if ok else None)
    return out


def run_arm_offline(*, dataset_dir: Path, view_id: str, arm: ArmSpec,
                    candidates: Dict[str, Any], head14: Sequence[str],
                    full60: Sequence[str], repo_root: Path) -> OfflineRunOutcome:
    """Score the fixed candidate set for one arm and select the argmax."""
    frame: pd.DataFrame = candidates["frame"]
    weights, debug = resolve_arm_weights(
        arm, dataset_dir=dataset_dir, view_id=view_id,
        head14=head14, full60=full60, repo_root=repo_root)

    missing = [m for m in weights if f"{m}_norm" not in frame.columns]
    if missing:
        raise OfflineCandidateError(
            f"{dataset_dir.name}/{view_id}/{arm.method_id}: selected metrics absent "
            f"from the candidate file: {missing}")

    js = score_candidates(frame, weights)
    valid_mask = frame["valid"].map(_is_valid).tolist()
    aris = [_fnum(v) for v in frame["ARI"].tolist()]
    ks = [_fnum(v) for v in frame["n_clusters_wo_noise"].tolist()]

    eligible = [i for i, (j, v) in enumerate(zip(js, valid_mask)) if v and j is not None]
    if not eligible:
        raise OfflineCandidateError(
            f"{dataset_dir.name}/{view_id}/{arm.method_id}: no scorable valid candidate")

    max_j = max(js[i] for i in eligible)
    tied = [i for i in eligible if abs(js[i] - max_j) <= TIE_TOL]
    # First-in-stored-order among the tied set: matches pandas idxmax semantics
    # used by the online best-view aggregation, and is deterministic because the
    # candidate file order is fixed.
    chosen = tied[0]

    # Candidate ceiling is ARM-INDEPENDENT: best ARI over valid candidates.
    ceiling_pool = [aris[i] for i in range(len(frame))
                    if valid_mask[i] and aris[i] is not None]
    ceiling = max(ceiling_pool) if ceiling_pool else None

    tie_aris = [aris[i] for i in tied if aris[i] is not None]
    sel_ari = aris[chosen]
    sel_k = ks[chosen]

    return OfflineRunOutcome(
        method_id=arm.method_id, inventory=arm.inventory, strategy=arm.strategy,
        deployable=arm.deployable,
        selected_metrics=list(weights), selected_weights=dict(weights),
        resolver_debug=debug,
        n_candidates=len(frame), n_valid_candidates=int(sum(valid_mask)),
        candidate_set_fingerprint=candidates["fingerprint"],
        selected_candidate_index=int(chosen),
        selected_algorithm=str(frame.iloc[chosen]["algorithm"]),
        selected_config_str=str(frame.iloc[chosen]["config_str"]),
        objective_j=float(js[chosen]), max_objective_j=float(max_j),
        n_tied_at_max=len(tied), tie_flag=len(tied) > 1,
        ari=sel_ari, observed_k=None if sel_k is None else int(sel_k),
        best_possible_ari=ceiling,
        ari_regret=None if (sel_ari is None or ceiling is None) else ceiling - sel_ari,
        ceiling_hit=None if (sel_ari is None or ceiling is None)
        else bool(abs(ceiling - sel_ari) <= 1e-12),
        tie_ari_min=min(tie_aris) if tie_aris else None,
        tie_ari_mean=(sum(tie_aris) / len(tie_aris)) if tie_aris else None,
        tie_ari_max=max(tie_aris) if tie_aris else None,
    )
