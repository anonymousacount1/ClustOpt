"""Persist one offline (dataset, method, view) result.

Offline counterpart of :mod:`method_runner`. It deliberately mirrors that
module's contract so the existing aggregation layer works unchanged:

* the same :class:`~.method_runner.RunRecord` dataclass,
* the same output-directory convention via
  :func:`~.method_runner.run_output_dir`,
* the same ``status.json``-based completion/resume semantics via
  :func:`~.method_runner.is_run_complete`,
* the same atomic writers from :mod:`output_writer`.

Online and offline results cannot collide: offline runs set
``split_dir_suffix='offline_fixed32'``, so they land in
``experiments/split_01_offline_fixed32/`` while the historical online runs stay
in ``experiments/split_01/``. Historical discovery therefore never sees them,
and a future ``online_fixed50`` sweep gets its own suffix.

``method_runner`` itself is NOT modified -- the historical online path is byte
identical.
"""
from __future__ import annotations

import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .config import ExperimentExecutionConfig
from .method_runner import RunRecord, is_run_complete, run_output_dir
from .offline_candidate_execution import (
    ArmSpec, OfflineCandidateError, load_candidate_set, run_arm_offline,
)
from .output_writer import ensure_dir, path_exists, write_json_atomic

EXECUTION_MODE = "offline_fixed32"

# Per-run output files (offline analogue of config.PER_RUN_OUTPUT_FILES).
OFFLINE_PER_RUN_OUTPUT_FILES = (
    "offline_result.json",
    "selected_metrics.json",
    "stage1_metadata.json",
    "status.json",
)


def _finalize_failure(record: RunRecord, out_dir: Path, *, stage: str,
                      error: BaseException, started_at: str) -> RunRecord:
    """Mirror method_runner's failure contract: capture, never propagate."""
    ensure_dir(out_dir)
    record.status = "failed"
    record.failure_reason = f"{type(error).__name__}: {error}"
    payload = {
        "status": "failed",
        "execution_mode": EXECUTION_MODE,
        "dataset_id": record.dataset_id,
        "method_name": record.method_name,
        "view_id": record.view_id,
        "stage": stage,
        "error": record.failure_reason,
        "traceback": "".join(
            traceback.format_exception(type(error), error, error.__traceback__)),
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        write_json_atomic(out_dir / "status.json", payload)
    except Exception:                      # never let logging break the sweep
        pass
    return record


def run_offline_method_view(
    *,
    cfg: ExperimentExecutionConfig,
    dataset_dir: Path,
    dataset_meta: Dict[str, Any],
    view_id: str,
    arm: ArmSpec,
    candidates: Dict[str, Any],
    head14: Sequence[str],
    full60: Sequence[str],
) -> RunRecord:
    """Score one arm on one (dataset, view); write outputs; return a RunRecord."""
    split_dir_name = cfg.split_dir_name()
    out_dir = run_output_dir(dataset_dir, split_dir_name, arm.method_id, view_id)

    base = RunRecord(
        dataset_id=Path(dataset_dir).name,
        dataset_dir=str(dataset_dir),
        split_id=int(cfg.split_id),
        family=dataset_meta.get("family"),
        subfamily=dataset_meta.get("subfamily"),
        difficulty=dataset_meta.get("difficulty"),
        cluster_count=dataset_meta.get("cluster_count"),
        view_id=view_id,
        method_name=arm.method_id,
        method_group=("oracle" if arm.strategy == "oracle"
                      else "regressor" if arm.strategy == "predicted" else "baseline"),
        status="failed",
    )

    # Resume short-circuit -- identical semantics to the online runner.
    if cfg.resume and not cfg.overwrite and is_run_complete(out_dir):
        base.status = "skipped"
        base.failure_reason = ""
        return base

    ensure_dir(out_dir)
    started_at = datetime.now(timezone.utc).isoformat()
    try:
        outcome = run_arm_offline(
            dataset_dir=Path(dataset_dir), view_id=view_id, arm=arm,
            candidates=candidates, head14=head14, full60=full60,
            repo_root=cfg.repo_root,
        )
    except (OfflineCandidateError, Exception) as exc:
        return _finalize_failure(base, out_dir, stage="offline_scoring",
                                 error=exc, started_at=started_at)

    finished_at = datetime.now(timezone.utc).isoformat()
    true_k = dataset_meta.get("cluster_count")
    exact_k = (None if (outcome.observed_k is None or true_k is None)
               else int(int(outcome.observed_k) == int(true_k)))

    result = {
        "status": "success",
        "execution_mode": EXECUTION_MODE,
        "method_id": arm.method_id,
        "split_id": int(cfg.split_id),
        "dataset_id": base.dataset_id,
        "family": base.family,
        "subfamily": base.subfamily,
        "difficulty": base.difficulty,
        "view_id": view_id,
        "metric_inventory": arm.inventory,
        "utility_strategy": arm.strategy,
        "deployable": arm.deployable,
        "predictor": arm.model_run_dir,
        "selected_metrics": outcome.selected_metrics,
        "selected_weights": outcome.selected_weights,
        "candidate_count": outcome.n_candidates,
        "valid_candidate_count": outcome.n_valid_candidates,
        "candidate_set_fingerprint": outcome.candidate_set_fingerprint,
        "selected_candidate_index": outcome.selected_candidate_index,
        "selected_algorithm": outcome.selected_algorithm,
        "selected_configuration": outcome.selected_config_str,
        "objective_J": outcome.objective_j,
        "max_objective_J": outcome.max_objective_j,
        "n_tied_at_max": outcome.n_tied_at_max,
        "tie_flag": outcome.tie_flag,
        "ari": outcome.ari,
        "observed_K": outcome.observed_k,
        "true_K": true_k,
        "exact_K": exact_k,
        "best_possible_ari_in_candidate_set": outcome.best_possible_ari,
        "ari_regret": outcome.ari_regret,
        "ceiling_hit": outcome.ceiling_hit,
        # Diagnostics only -- never used to select a candidate.
        "tie_ari_min": outcome.tie_ari_min,
        "tie_ari_mean": outcome.tie_ari_mean,
        "tie_ari_max": outcome.tie_ari_max,
        "started_at": started_at,
        "finished_at": finished_at,
    }

    try:
        write_json_atomic(out_dir / "offline_result.json", result)
        write_json_atomic(out_dir / "selected_metrics.json", {
            "cvi_type": {"uniform": "generic", "predicted": "regressor_dynamic",
                         "oracle": "oracle_dynamic"}[arm.strategy],
            "method_group": base.method_group,
            "selected_metrics": outcome.selected_metrics,
            "selected_metric_weights": outcome.selected_weights,
            "resolver_debug": outcome.resolver_debug,
        })
        write_json_atomic(out_dir / "stage1_metadata.json", {
            "execution_mode": EXECUTION_MODE,
            "candidate_source": "utility/<view_id>/clustopt_results.csv",
            "candidate_contract": {"expected_candidates": 32,
                                   "candidates_seen": outcome.n_candidates},
            "policy_id_note": (
                "The '_fixed50' suffix denotes the shared POLICY identity and the "
                "future online protocol. Stage-1C scores 32 stored candidates and "
                "runs no Optuna trials."),
            "metric_inventory": arm.inventory,
            "metric_inventory_size": 14 if arm.inventory == "head14" else 60,
            "utility_strategy": arm.strategy,
            "top_k": None if arm.strategy == "uniform" else 5,
            "weighting_mode": ("uniform_normalized" if arm.strategy == "uniform"
                               else "normalized_positive"),
            "predictor_run_dir": arm.model_run_dir,
            "deployable": arm.deployable,
        })
        write_json_atomic(out_dir / "status.json", {
            "status": "success",
            "execution_mode": EXECUTION_MODE,
            "dataset_id": base.dataset_id,
            "method_name": arm.method_id,
            "view_id": view_id,
            "started_at": started_at,
            "finished_at": finished_at,
            "runtime_sec": 0.0,
            "ari": outcome.ari,
            "selected_k": outcome.observed_k,
            "true_k": true_k,
        })
    except Exception as exc:
        return _finalize_failure(base, out_dir, stage="offline_persist",
                                 error=exc, started_at=started_at)

    base.status = "success"
    base.failure_reason = ""
    base.runtime_sec = 0.0
    base.ari = outcome.ari
    base.selected_k = outcome.observed_k
    base.true_k = true_k
    base.top_metric = (max(outcome.selected_weights, key=outcome.selected_weights.get)
                       if outcome.selected_weights else None)
    base.extra = {
        "execution_mode": EXECUTION_MODE,
        "selected_metric_count": len(outcome.selected_metrics),
        "weighting_mode": ("uniform_normalized" if arm.strategy == "uniform"
                           else "normalized_positive"),
        "utility_source": arm.strategy,
        "metric_ranking_source": arm.strategy,
        "objective_J": outcome.objective_j,
        "n_tied_at_max": outcome.n_tied_at_max,
        "best_possible_ari_in_candidate_set": outcome.best_possible_ari,
        "ari_regret": outcome.ari_regret,
        "ceiling_hit": outcome.ceiling_hit,
        "candidate_set_fingerprint": outcome.candidate_set_fingerprint,
        "exact_K": exact_k,
    }
    return base


def offline_run_is_complete(dataset_dir: Path, split_dir_name: str,
                            method_id: str, view_id: str) -> bool:
    r"""True iff this offline run wrote a complete, non-partial output set.

    Must use :func:`output_writer.path_exists`, never ``Path.is_file()``: the
    output tree exceeds the legacy Windows MAX_PATH (a full offline run path is
    ~288 chars) and this machine has ``LongPathsEnabled=0``, so the plain
    pathlib API silently reports missing files. ``output_writer`` routes through
    the extended-length ``\\?\`` prefix, which is why the writes succeed.
    """
    out_dir = run_output_dir(dataset_dir, split_dir_name, method_id, view_id)
    if not is_run_complete(out_dir):
        return False
    return all(path_exists(out_dir / f) for f in OFFLINE_PER_RUN_OUTPUT_FILES)
