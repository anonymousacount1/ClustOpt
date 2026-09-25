"""Stage 2A-2: offline OOF replay of the 20 learned deployable ClustOpt policies.

Replays each policy over the ALREADY-PERSISTED fixed-32 candidate pool at
``<dataset>/utility/<view>/clustopt_results.csv``. **No clustering of any kind is
invoked** -- the policy's objective ``J`` is evaluated over stored candidates and
the argmax is selected. The persisted ARI is read only AFTER selection and never
influences utility prediction, metric selection, weights, the objective or the
choice.

Utility sources are the Stage-2A-1 leave-one-split-out artifacts: for a dataset in
split ``s`` the utilities come from components trained on ``{2..16} \\ {s}``. This
is asserted per record, not assumed.

Policy set: the 20 members of ``portfolio_registry.strict_deployable_clustopt_vbs``
(10 MLP + 10 KNN). The 6 fixed baselines, the 10 ``real_*`` utility oracles and the
4 ``*_dynamic_realK_*`` oracle-assisted variants are excluded by construction.

Sits beside the existing wet-run execution modules and reuses their conventions:
``offline_candidate_execution`` for the candidate contract, ``output_writer`` for
long-path-safe atomic writes, ``worker_setup`` for thread caps.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from models.ClustOpt.cluster_validity_indices.dynamic_metric_selection.dynamic_k import (
    DYNAMIC_TOPK_POLICY, select_dynamic_top_k,
)
from models.ClustOpt.cluster_validity_indices.dynamic_metric_selection.weighting import (
    compute_weights, select_top_k,
)

from ..paper_analysis.method_registry import METHOD_BY_NAME
from ..paper_analysis.portfolio_registry import PORTFOLIO_BY_ID
from .offline_candidate_execution import (
    OfflineCandidateError, load_candidate_set, score_candidates,
)
from .output_writer import _ext, ensure_dir, path_exists, read_json, write_json_atomic

EXECUTION_MODE = "offline_oof_policy20"
EXPERIMENT_VERSION = "stage2a2.v1"
RESULT_SCHEMA_VERSION = "stage2a2.result.v1"
SPLIT_DIR_TEMPLATE = "split_{split:02d}_offline_oof_policy20"
VIEW_IDS: Tuple[str, ...] = ("x_only", "y_only", "xy_2d")
DEV_SPLITS: Tuple[int, ...] = tuple(range(2, 17))
EXPECTED_CANDIDATES = 32

# ONE authoritative artifact per (policy, view). It carries the same six logical
# blocks the accepted ClustOpt wet-run schema splits across separate files
# (status / best_result / external_metrics / selected_metrics / timing_summary /
# config_snapshot), consolidated so the replay writes ~960k files instead of
# ~5.8M. Nothing is dropped: every field needed to reconstruct the historical
# per-file view is retained under its own block, and the completion barrier
# validates the whole record atomically.
RESULT_FILE = "result.json"
RESULT_BLOCKS: Tuple[str, ...] = (
    "identity", "status", "best_result", "external_metrics",
    "selected_metrics", "timing", "provenance",
)
# The pre-consolidation layout, retained only so existing artifacts can be
# migrated in place.
LEGACY_FILES: Tuple[str, ...] = (
    "status.json", "best_result.json", "external_metrics.json",
    "selected_metrics.json", "timing_summary.json", "config_snapshot.json",
)


# --------------------------------------------------------------------------- #
# Policy specifications -- frozen from the accepted registry, never hand-listed
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PolicySpec:
    policy_id: str
    short_label: str
    source: str                 # 'mlp' | 'knn'
    top_k: Any                  # int or 'dynamic'
    weighting: str              # 'normalized_positive' | 'softmax'
    k_policy: str
    softmax_temperature: float = 0.5

    @property
    def is_dynamic(self) -> bool:
        return self.top_k == "dynamic"

    def to_record(self) -> Dict[str, Any]:
        spec = METHOD_BY_NAME[self.policy_id]
        return {
            "policy_id": self.policy_id, "short_label": self.short_label,
            "policy_name": spec.display_name, "paper_label": spec.paper_label,
            "source_type": self.source.upper(),
            "top_k_rule": ("dynamic_predict_k" if self.is_dynamic
                           else f"fixed_top_{self.top_k}"),
            "weighting": "RAW" if self.weighting == "normalized_positive" else "SOFTMAX",
            "weighting_impl": self.weighting,
            "softmax_temperature": (self.softmax_temperature
                                    if self.weighting == "softmax" else None),
            "dynamic": self.is_dynamic,
            "dynamic_policy": DYNAMIC_TOPK_POLICY if self.is_dynamic else None,
            "k_policy": self.k_policy,
            "utility_source": ("stage2a1 OOF MLP (full60)" if self.source == "mlp"
                               else "stage2a1 OOF KNN repository"),
            "deployable": True,
            "uses_real_utility": bool(spec.uses_real_utility),
            "is_oracle_assisted": bool(spec.is_oracle_assisted),
            "method_family": spec.method_family,
            "historical_portfolio": "strict_deployable_clustopt_vbs",
            "oof_binding": ("combined/oof_mlp_utilities.csv.gz" if self.source == "mlp"
                            else "combined/oof_knn_utilities.csv.gz"),
        }


def _weighting_of(mode: str) -> str:
    return "normalized_positive" if mode == "raw" else "softmax"


def build_policy_set() -> List[PolicySpec]:
    """The frozen 20 learned deployable policies, read from the registry."""
    pf = PORTFOLIO_BY_ID["strict_deployable_clustopt_vbs"]
    out: List[PolicySpec] = []
    for name in pf.methods:
        s = METHOD_BY_NAME[name]
        out.append(PolicySpec(
            policy_id=name, short_label=s.short_label,
            source="mlp" if s.method_family == "clustopt_mlp" else "knn",
            top_k=("dynamic" if s.k_policy == "dynamic_predict"
                   else int(s.fixed_metric_count)),
            weighting=_weighting_of(s.weighting_mode),
            k_policy=s.k_policy))
    validate_policy_set(out)
    return out


def validate_policy_set(policies: Sequence[PolicySpec]) -> None:
    """Hard contract: exactly 20 learned deployable policies, 10 MLP + 10 KNN."""
    if len(policies) != 20:
        raise ValueError(f"expected 20 policies, got {len(policies)}")
    n_mlp = sum(1 for p in policies if p.source == "mlp")
    n_knn = sum(1 for p in policies if p.source == "knn")
    if (n_mlp, n_knn) != (10, 10):
        raise ValueError(f"expected 10 MLP + 10 KNN, got {n_mlp}/{n_knn}")
    for p in policies:
        spec = METHOD_BY_NAME[p.policy_id]
        if spec.uses_real_utility:
            raise ValueError(f"{p.policy_id}: real-utility oracle is forbidden")
        if spec.is_oracle_assisted:
            raise ValueError(f"{p.policy_id}: oracle-assisted (realK) is forbidden")
        if spec.method_family == "clustopt_fixed":
            raise ValueError(f"{p.policy_id}: fixed baseline is excluded by design")
        if spec.framework != "clustopt":
            raise ValueError(f"{p.policy_id}: not a ClustOpt policy")
        if p.k_policy == "dynamic_real":
            raise ValueError(f"{p.policy_id}: dynamic_realK is forbidden")


def policy_semantics_hash(policies: Sequence[PolicySpec]) -> str:
    payload = [[p.policy_id, str(p.top_k), p.weighting, p.k_policy,
                p.softmax_temperature] for p in policies]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# OOF utility index (per split, loaded once per worker process)
# --------------------------------------------------------------------------- #
@dataclass
class OofUtilityIndex:
    """(dataset_id, view_id) -> 60-dim utility vector, for one held-out split."""
    split_id: int
    metric_names: List[str]
    mlp: Dict[Tuple[str, str], np.ndarray] = field(default_factory=dict)
    knn: Dict[Tuple[str, str], np.ndarray] = field(default_factory=dict)
    artifact_id: str = ""

    def get(self, source: str, dataset_id: str, view_id: str) -> np.ndarray:
        table = self.mlp if source == "mlp" else self.knn
        key = (dataset_id, view_id)
        if key not in table:
            raise KeyError(f"no OOF {source} utility for {dataset_id}/{view_id} "
                           f"in split {self.split_id}")
        return table[key]

    def utilities(self, source: str, dataset_id: str, view_id: str
                  ) -> Dict[str, float]:
        vec = self.get(source, dataset_id, view_id)
        return {m: float(v) for m, v in zip(self.metric_names, vec)}


def load_oof_index(oof_root: Path, split_id: int,
                   metric_names: Sequence[str]) -> OofUtilityIndex:
    """Load the Stage-2A-1 OOF utilities for one held-out split.

    Reads the per-fold prediction files (not the combined ones) so the held-out
    split of the producing model is verified from its own manifest.
    """
    fold = oof_root / f"holdout_split_{split_id:02d}"
    kman = read_json(fold / "knn/fold_manifest.json") or {}
    mman = read_json(fold / "mlp/fold_manifest.json") or {}
    for man, label in ((kman, "knn"), (mman, "mlp")):
        if int(man.get("heldout_split", -1)) != split_id:
            raise ValueError(f"{label} fold manifest heldout_split "
                             f"{man.get('heldout_split')} != {split_id}")
        if split_id in list(man.get("training_splits") or []):
            raise ValueError(f"{label} fold {split_id}: held-out split present in "
                             f"its own training splits")
        if 1 in list(man.get("training_splits") or []):
            raise ValueError(f"{label} fold {split_id}: Split 1 in training splits")

    idx = OofUtilityIndex(split_id=split_id, metric_names=list(metric_names))
    idx.artifact_id = f"{mman.get('feature_schema_hash','')[:12]}/{split_id:02d}"

    knn = pd.read_csv(_ext(fold / "knn/oof_predictions.csv.gz"))
    if set(knn["split_id"].unique()) != {split_id}:
        raise ValueError(f"KNN OOF rows for split {split_id} contain other splits")
    kvals = knn[list(metric_names)].to_numpy(np.float64)
    for (ds, vw), row in zip(zip(knn["dataset_id"], knn["view_id"]), kvals):
        idx.knn[(ds, vw)] = row

    mlp = pd.read_csv(_ext(fold / "mlp/oof_predictions.csv.gz"))
    pcols = [f"pred_{m}" for m in metric_names]
    missing = [c for c in pcols if c not in mlp.columns]
    if missing:
        raise ValueError(f"MLP OOF file missing {len(missing)} pred columns, "
                         f"e.g. {missing[:3]}")
    mvals = mlp[pcols].to_numpy(np.float64)
    for (ds, vw), row in zip(zip(mlp["dataset_id"], mlp["view_mode"]), mvals):
        idx.mlp[(ds, vw)] = row
    return idx


# --------------------------------------------------------------------------- #
# Replay
# --------------------------------------------------------------------------- #
def run_output_dir(dataset_dir: Path, split_id: int, policy_id: str,
                   view_id: str) -> Path:
    return (Path(dataset_dir) / "experiments"
            / SPLIT_DIR_TEMPLATE.format(split=split_id) / policy_id / view_id)


def replay_is_complete(dataset_dir: Path, split_id: int, policy_id: str,
                       view_id: str, semantics_hash: str) -> bool:
    """Authoritative completion barrier for one (dataset, policy, view).

    Validates the ENTIRE consolidated record atomically: the file must exist,
    parse, carry the current result-schema version, contain every required block,
    report success, and agree with this run's experiment version, policy-semantics
    hash and held-out split. A partially written or stale record fails and the
    unit is rebuilt.
    """
    rec = read_json(run_output_dir(dataset_dir, split_id, policy_id, view_id)
                    / RESULT_FILE)
    if not rec or rec.get("schema_version") != RESULT_SCHEMA_VERSION:
        return False
    if any(b not in rec for b in RESULT_BLOCKS):
        return False
    if (rec.get("status") or {}).get("status") != "success":
        return False
    prov = rec.get("provenance") or {}
    ident = rec.get("identity") or {}
    return (prov.get("experiment_version") == EXPERIMENT_VERSION
            and prov.get("policy_semantics_hash") == semantics_hash
            and int(prov.get("oof_heldout_split", -1)) == split_id
            and ident.get("policy_id") == policy_id
            and ident.get("view_id") == view_id)


def replay_policy_view(*, dataset_dir: Path, dataset_id: str, split_id: int,
                       family: str, subfamily: str, view_id: str,
                       policy: PolicySpec, oof: OofUtilityIndex,
                       semantics_hash: str, source_commit: str,
                       candidate_cache: Optional[Dict[str, Any]] = None,
                       ) -> Dict[str, Any]:
    """Replay one policy on one view. Writes artifacts, returns a record row."""
    t0 = time.perf_counter()
    out = ensure_dir(run_output_dir(dataset_dir, split_id, policy.policy_id, view_id))
    row: Dict[str, Any] = {
        "dataset_id": dataset_id, "split_id": split_id, "family": family,
        "subfamily": subfamily, "view_id": view_id,
        "policy_id": policy.policy_id, "policy_short_label": policy.short_label,
        "policy_source": policy.source.upper(),
        "top_k_rule": ("dynamic" if policy.is_dynamic else policy.top_k),
        "weighting": "RAW" if policy.weighting == "normalized_positive" else "SOFTMAX",
        "oof_heldout_split": split_id, "execution_mode": EXECUTION_MODE,
        "status": "failed",
    }
    try:
        # ---- 1. OOF utilities (never the full-train model, never Split 1) ----
        utilities = oof.utilities(policy.source, dataset_id, view_id)
        if len(utilities) != 60:
            raise ValueError(f"expected 60 utilities, got {len(utilities)}")

        # ---- 2. Top-K selection + weighting (production semantics) -----------
        if policy.is_dynamic:
            selected, selected_k, k_rel_raw = select_dynamic_top_k(utilities)
        else:
            selected = select_top_k(utilities, int(policy.top_k))
            selected_k, k_rel_raw = len(selected), None
        weights = compute_weights(selected, policy.weighting,
                                  policy.softmax_temperature)
        wsum = float(sum(weights.values()))
        if not np.isclose(wsum, 1.0, atol=1e-9):
            raise ValueError(f"weights sum to {wsum}, expected 1")

        # ---- 3. Fixed-32 candidate pool (already persisted; no clustering) ---
        cand = candidate_cache if candidate_cache is not None else load_candidate_set(
            dataset_dir, view_id, expected_count=EXPECTED_CANDIDATES)
        frame = cand["frame"]

        # ---- 4. Objective over stored candidates; argmax. ARI NOT consulted --
        J = score_candidates(frame, weights)
        usable = [(j, i) for i, j in enumerate(J) if j is not None]
        if not usable:
            raise ValueError("no candidate produced a usable objective")
        best_j, best_i = max(usable)
        best = frame.iloc[best_i]

        # ---- 5. ARI read ONLY now, purely to evaluate the choice -------------
        ari = float(best["ARI"])
        n_tied = int(sum(1 for j, _ in usable if abs(j - best_j) <= 1e-12))

        row.update({
            "status": "success",
            "selected_candidate_index": int(best_i),
            "selected_candidate_id": str(best.get("config_str", best_i)),
            "selected_algorithm": str(best.get("algorithm")),
            "objective_J": float(best_j), "selected_ari": ari,
            "selected_k": _int_or_none(best.get("n_clusters_wo_noise")),
            "noise_ratio": _float_or_none(best.get("noise_ratio")),
            "n_labels_unique": _int_or_none(best.get("n_labels_unique")),
            "n_candidates": int(len(frame)),
            "n_valid_candidates": int(len(usable)),
            "n_tied_at_max_J": n_tied,
            "selected_metrics": "|".join(n for n, _ in selected),
            "selected_metric_count": int(selected_k),
            "dynamic_k_rel_raw": k_rel_raw,
            "candidate_fingerprint": cand.get("fingerprint"),
        })

        runtime = time.perf_counter() - t0
        # ---- 6. Persist ONE consolidated, atomically-written record ----------
        write_json_atomic(out / RESULT_FILE, _build_record(
            policy=policy, dataset_id=dataset_id, split_id=split_id, family=family,
            subfamily=subfamily, view_id=view_id, oof=oof,
            semantics_hash=semantics_hash, source_commit=source_commit,
            status="success", ari=ari, runtime=runtime, row=row, best_j=best_j,
            best_i=best_i, n_frame=len(frame), n_usable=len(usable),
            n_tied=n_tied, selected=selected, weights=weights, wsum=wsum,
            selected_k=selected_k, k_rel_raw=k_rel_raw,
            fingerprint=cand.get("fingerprint")))
        row["runtime_sec"] = runtime
        return row
    except (OfflineCandidateError, Exception) as exc:                # noqa: BLE001
        row["status"] = "failed"
        row["failure_reason"] = f"{type(exc).__name__}: {exc}"
        row["runtime_sec"] = time.perf_counter() - t0
        write_json_atomic(out / RESULT_FILE, {
            "schema_version": RESULT_SCHEMA_VERSION,
            "identity": {"dataset_id": dataset_id, "split_id": split_id,
                         "family": family, "subfamily": subfamily,
                         "view_id": view_id, "policy_id": policy.policy_id},
            "status": {"status": "failed",
                       "failure_reason": row["failure_reason"],
                       "execution_mode": EXECUTION_MODE},
            "best_result": {}, "external_metrics": {}, "selected_metrics": {},
            "timing": {"runtime_sec": row["runtime_sec"]},
            "provenance": {"experiment_version": EXPERIMENT_VERSION,
                           "policy_semantics_hash": semantics_hash,
                           "oof_heldout_split": split_id,
                           "source_commit": source_commit},
        })
        return row


def _build_record(*, policy, dataset_id, split_id, family, subfamily, view_id,
                  oof, semantics_hash, source_commit, status, ari, runtime, row,
                  best_j, best_i, n_frame, n_usable, n_tied, selected, weights,
                  wsum, selected_k, k_rel_raw, fingerprint) -> Dict[str, Any]:
    """The single authoritative per-(policy, view) record.

    Blocks correspond one-to-one to the six files the accepted wet-run schema
    writes separately, so the historical per-file view is reconstructible without
    loss.
    """
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "identity": {
            "dataset_id": dataset_id, "split_id": split_id, "family": family,
            "subfamily": subfamily, "view_id": view_id,
            "policy_id": policy.policy_id, "policy_short_label": policy.short_label,
            "policy_source": policy.source.upper(),
        },
        "status": {                                    # was status.json
            "status": status, "dataset_id": dataset_id, "split_id": split_id,
            "policy_id": policy.policy_id, "view_id": view_id,
            "ari": ari, "selected_k": row["selected_k"],
            "runtime_sec": runtime, "execution_mode": EXECUTION_MODE,
        },
        "best_result": {                               # was best_result.json
            "selected_algorithm": row["selected_algorithm"],
            "selected_params": row["selected_candidate_id"],
            "best_objective_score": float(best_j),
            "selected_candidate_index": int(best_i),
            "selected_k": row["selected_k"],
            "noise_ratio": row.get("noise_ratio"),
            "n_labels_unique": row.get("n_labels_unique"),
            "n_candidates_scored": int(n_frame),
            "n_valid_candidates": int(n_usable),
            "n_tied_at_max_objective": int(n_tied),
            "cvi_type": f"{policy.source}_oof_dynamic",
            "top_k": ("dynamic" if policy.is_dynamic else policy.top_k),
            "selected_metric_count": int(selected_k),
            "dynamic_topk_policy": DYNAMIC_TOPK_POLICY if policy.is_dynamic else None,
            "used_fallback": False,
        },
        "external_metrics": {                          # was external_metrics.json
            "ari": ari,
            "note": "read after candidate selection; never an input to selection",
        },
        "selected_metrics": {                          # was selected_metrics.json
            "cvi_type": f"{policy.source}_oof_dynamic",
            "method_group": policy.source,
            "selected_metrics": [n for n, _ in selected],
            "selected_metric_weights": {k: float(v) for k, v in weights.items()},
            "selected_utilities": {n: float(s) for n, s in selected},
            "weighting": policy.weighting,
            "softmax_temperature": (policy.softmax_temperature
                                    if policy.weighting == "softmax" else None),
            "weights_sum": wsum,
            "resolver_debug": {
                "mode": f"{policy.source}_oof",
                "utility_source": "stage2a1_oof",
                "oof_heldout_split": split_id,
                "oof_artifact_id": oof.artifact_id,
                "dynamic_top_k": policy.is_dynamic,
                "dynamic_topk_policy": DYNAMIC_TOPK_POLICY if policy.is_dynamic else None,
                "selected_k_metrics_count": int(selected_k),
                "k_rel_raw": k_rel_raw, "view_id": view_id,
            },
        },
        "timing": {                                    # was timing_summary.json
            "runtime_sec": runtime,
            "note": "replay-only cost: utility lookup, weighting and scoring of "
                    "already-persisted candidates. No clustering was executed.",
        },
        "provenance": {                                # was config_snapshot.json
            "experiment_version": EXPERIMENT_VERSION,
            "execution_mode": EXECUTION_MODE,
            "policy_semantics_hash": semantics_hash,
            "policy": policy.to_record(),
            "oof_heldout_split": split_id,
            "oof_artifact_id": oof.artifact_id,
            "candidate_pool": {
                "source": "utility/<view_id>/clustopt_results.csv",
                "expected_count": EXPECTED_CANDIDATES,
                "fingerprint": fingerprint,
            },
            "clustering_executed": False,
            "source_commit": source_commit,
        },
    }


def _int_or_none(v):
    try:
        return int(v) if v is not None and not pd.isna(v) else None
    except Exception:
        return None


def _float_or_none(v):
    try:
        return float(v) if v is not None and not pd.isna(v) else None
    except Exception:
        return None


def replay_dataset(*, dataset_dir: Path, dataset_id: str, split_id: int,
                   family: str, subfamily: str, policies: Sequence[PolicySpec],
                   oof: OofUtilityIndex, semantics_hash: str,
                   source_commit: str, overwrite: bool = False
                   ) -> List[Dict[str, Any]]:
    """All 20 policies x 3 views for one dataset. Candidate pools loaded once."""
    rows: List[Dict[str, Any]] = []
    cache: Dict[str, Any] = {}
    for view in VIEW_IDS:
        for policy in policies:
            if not overwrite and replay_is_complete(dataset_dir, split_id,
                                                    policy.policy_id, view,
                                                    semantics_hash):
                rows.append(_row_from_disk(dataset_dir, dataset_id, split_id, family,
                                           subfamily, view, policy))
                continue
            if view not in cache:
                try:
                    cache[view] = load_candidate_set(
                        dataset_dir, view, expected_count=EXPECTED_CANDIDATES)
                except Exception:                                     # noqa: BLE001
                    cache[view] = None
            rows.append(replay_policy_view(
                dataset_dir=dataset_dir, dataset_id=dataset_id, split_id=split_id,
                family=family, subfamily=subfamily, view_id=view, policy=policy,
                oof=oof, semantics_hash=semantics_hash, source_commit=source_commit,
                candidate_cache=cache[view]))
    return rows


def _row_from_disk(dataset_dir, dataset_id, split_id, family, subfamily, view,
                   policy) -> Dict[str, Any]:
    """Reconstruct a record row from a completed artifact (resume path)."""
    out = run_output_dir(dataset_dir, split_id, policy.policy_id, view)
    rec = read_json(out / RESULT_FILE) or {}
    st = rec.get("status") or {}
    br = rec.get("best_result") or {}
    sm = rec.get("selected_metrics") or {}
    return {
        "dataset_id": dataset_id, "split_id": split_id, "family": family,
        "subfamily": subfamily, "view_id": view, "policy_id": policy.policy_id,
        "policy_short_label": policy.short_label,
        "policy_source": policy.source.upper(),
        "top_k_rule": ("dynamic" if policy.is_dynamic else policy.top_k),
        "weighting": "RAW" if policy.weighting == "normalized_positive" else "SOFTMAX",
        "oof_heldout_split": split_id, "execution_mode": EXECUTION_MODE,
        "status": st.get("status", "failed"),
        "selected_candidate_index": br.get("selected_candidate_index"),
        "selected_candidate_id": br.get("selected_params"),
        "selected_algorithm": br.get("selected_algorithm"),
        "objective_J": br.get("best_objective_score"),
        "selected_ari": st.get("ari"), "selected_k": st.get("selected_k"),
        "n_candidates": br.get("n_candidates_scored"),
        "n_valid_candidates": br.get("n_valid_candidates"),
        "n_tied_at_max_J": br.get("n_tied_at_max_objective"),
        "selected_metrics": "|".join(sm.get("selected_metrics") or []),
        "selected_metric_count": br.get("selected_metric_count"),
        "dynamic_k_rel_raw": (sm.get("resolver_debug") or {}).get("k_rel_raw"),
        "runtime_sec": st.get("runtime_sec"),
        "resumed": True,
    }


# --------------------------------------------------------------------------- #
# Migration: pre-consolidation six-file layout -> single result.json
# --------------------------------------------------------------------------- #
def migrate_legacy_result(out_dir: Path) -> str:
    """Fold a legacy six-file unit into one ``result.json``. Idempotent.

    Returns 'migrated', 'already', 'absent' or 'incomplete'. Nothing is deleted
    until the consolidated record has been written and re-read successfully, so an
    interruption can never lose a completed unit.
    """
    if path_exists(out_dir / RESULT_FILE):
        return "already"
    present = [f for f in LEGACY_FILES if path_exists(out_dir / f)]
    if not present:
        return "absent"
    if len(present) != len(LEGACY_FILES):
        return "incomplete"

    st = read_json(out_dir / "status.json") or {}
    br = read_json(out_dir / "best_result.json") or {}
    em = read_json(out_dir / "external_metrics.json") or {}
    sm = read_json(out_dir / "selected_metrics.json") or {}
    tm = read_json(out_dir / "timing_summary.json") or {}
    cs = read_json(out_dir / "config_snapshot.json") or {}
    pol = cs.get("policy") or {}
    rec = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "identity": {
            "dataset_id": st.get("dataset_id"), "split_id": st.get("split_id"),
            "family": None, "subfamily": None, "view_id": st.get("view_id"),
            "policy_id": st.get("policy_id"),
            "policy_short_label": pol.get("short_label"),
            "policy_source": pol.get("source_type"),
        },
        "status": st, "best_result": br, "external_metrics": em,
        "selected_metrics": sm, "timing": tm,
        "provenance": {**cs, "migrated_from": "six-file legacy layout"},
    }
    write_json_atomic(out_dir / RESULT_FILE, rec)
    if not read_json(out_dir / RESULT_FILE):                       # verify first
        raise IOError(f"consolidated record unreadable after write: {out_dir}")
    for f in LEGACY_FILES:
        try:
            Path(_ext(out_dir / f)).unlink()
        except OSError:
            pass
    return "migrated"


def migrate_dataset_tree(dataset_dir: Path, split_id: int,
                         policies: Sequence[PolicySpec]) -> Dict[str, int]:
    """Migrate every (policy, view) unit under one dataset's Stage-2A-2 tree."""
    counts: Dict[str, int] = {}
    for policy in policies:
        for view in VIEW_IDS:
            r = migrate_legacy_result(
                run_output_dir(dataset_dir, split_id, policy.policy_id, view))
            counts[r] = counts.get(r, 0) + 1
    return counts
