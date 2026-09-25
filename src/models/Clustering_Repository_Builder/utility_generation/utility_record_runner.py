"""Core unit: run utility generation for one (dataset, view) pair.

Outputs (atomic) under ``<dataset_dir>/utility/<view_id>/``:

* ``clustopt_results.csv``
* ``metric_utilities.csv``
* ``utility_vector.json``
* ``utility_record_summary.json``
* ``timing_summary.json``
* ``run_log.json``
"""
from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from .clustopt_runner import run_clustopt
from .config import UtilityGenerationConfig
from .record_loader import LoadedDatasetForUtility
from .utility_output_writer import write_csv_atomic, write_json_atomic
from .view_builder import build_view


@dataclass
class RecordResult:
    """One-row payload for the subfamily-level summary CSV."""

    dataset_id: str
    dataset_dir: str
    family: Optional[str]
    subfamily: Optional[str]
    specific_subfamily: Optional[str]
    view_id: str
    status: str  # 'success' | 'failed' | 'skipped'
    metric_mode: str
    sample_size: int
    map_name: str
    n_configs: int = 0
    n_valid_configs: int = 0
    runtime_sec: float = 0.0
    fit_predict_sec: float = 0.0
    cvi_eval_sec: float = 0.0
    utility_compute_sec: float = 0.0
    n_metrics: int = 0
    top1_metric: Optional[str] = None
    top1_utility: float = 0.0
    top5_metrics: list[str] = field(default_factory=list)
    error_message: str = ""

    def as_summary_row(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "dataset_dir": self.dataset_dir,
            "family": self.family or "",
            "subfamily": self.subfamily or "",
            "specific_subfamily": self.specific_subfamily or "",
            "view_id": self.view_id,
            "status": self.status,
            "metric_mode": self.metric_mode,
            "sample_size": int(self.sample_size),
            "map_name": self.map_name,
            "n_configs": int(self.n_configs),
            "n_valid_configs": int(self.n_valid_configs),
            "runtime_sec": float(self.runtime_sec),
            "fit_predict_sec": float(self.fit_predict_sec),
            "cvi_eval_sec": float(self.cvi_eval_sec),
            "utility_compute_sec": float(self.utility_compute_sec),
            "n_metrics": int(self.n_metrics),
            "top1_metric": self.top1_metric or "",
            "top1_utility": float(self.top1_utility),
            "top5_metrics": ",".join(self.top5_metrics),
            "error_message": self.error_message,
        }


# ----------------------------------------------------------------- resume check

REQUIRED_VIEW_OUTPUTS = (
    "metric_utilities.csv",
    "utility_vector.json",
    "utility_record_summary.json",
)


def view_output_dir(dataset_dir: Path, view_id: str) -> Path:
    return Path(dataset_dir) / "utility" / view_id


def is_view_complete(dataset_dir: Path, view_id: str) -> bool:
    view_dir = view_output_dir(dataset_dir, view_id)
    for fname in REQUIRED_VIEW_OUTPUTS:
        if not (view_dir / fname).is_file():
            return False
    # Inspect summary for explicit success status.
    try:
        import json
        with (view_dir / "utility_record_summary.json").open("r", encoding="utf-8") as fh:
            summary = json.load(fh)
        return str(summary.get("status", "")).lower() == "success"
    except Exception:
        return False


def _existing_view_skipped_result(
    *,
    loaded: LoadedDatasetForUtility,
    view_id: str,
    cfg: UtilityGenerationConfig,
) -> RecordResult:
    return RecordResult(
        dataset_id=loaded.dataset_id,
        dataset_dir=str(loaded.dataset_dir),
        family=loaded.family_id,
        subfamily=loaded.subfamily_id,
        specific_subfamily=loaded.specific_subfamily,
        view_id=view_id,
        status="skipped",
        metric_mode=cfg.metric_mode,
        sample_size=cfg.fast_metric_sample_size if cfg.metric_mode == "fast" else 0,
        map_name=cfg.map_path_for_view(view_id).name,
        error_message="",
    )


# -------------------------------------------------------------------- core run

def run_view(
    *,
    loaded: LoadedDatasetForUtility,
    view_id: str,
    cfg: UtilityGenerationConfig,
) -> RecordResult:
    """Run the full flow for one view and write all outputs."""
    map_path = cfg.map_path_for_view(view_id)

    # Resume short-circuit.
    if cfg.resume and not cfg.overwrite and is_view_complete(loaded.dataset_dir, view_id):
        return _existing_view_skipped_result(loaded=loaded, view_id=view_id, cfg=cfg)

    view_dir = view_output_dir(loaded.dataset_dir, view_id)
    view_dir.mkdir(parents=True, exist_ok=True)

    started_at = datetime.now(timezone.utc).isoformat()
    sample_size = cfg.fast_metric_sample_size if cfg.metric_mode == "fast" else 0

    base_result = RecordResult(
        dataset_id=loaded.dataset_id,
        dataset_dir=str(loaded.dataset_dir),
        family=loaded.family_id,
        subfamily=loaded.subfamily_id,
        specific_subfamily=loaded.specific_subfamily,
        view_id=view_id,
        status="failed",
        metric_mode=cfg.metric_mode,
        sample_size=sample_size,
        map_name=map_path.name,
    )

    # 1) build the view inputs
    try:
        view = build_view(loaded, view_id)
    except Exception as exc:
        return _finalize_failure(
            base_result, view_dir, stage="view_build", error=exc, started_at=started_at, cfg=cfg
        )

    # 2) run ClustOpt — but reuse a previous clustopt_results.csv if a prior
    #    run got that far before crashing (e.g. on the post-processing print).
    #    This turns a 80–150 s retry into a sub-second utility-only retry.
    results_csv_path = view_dir / "clustopt_results.csv"
    reused_clustopt = False
    clustopt_out = None  # type: ignore[assignment]
    if not cfg.overwrite and _existing_clustopt_csv_is_usable(results_csv_path):
        reused_clustopt = True
    else:
        try:
            clustopt_out = run_clustopt(
                map_path=map_path,
                X_decision=view.X_decision,
                X_full=view.X_full,
                y_true=view.y_true,
                runtime_block=cfg.runtime_block(),
            )
        except Exception as exc:
            return _finalize_failure(
                base_result, view_dir, stage="clustopt_search", error=exc, started_at=started_at, cfg=cfg
            )

        # 3) save ClustOpt results CSV (atomic)
        try:
            write_csv_atomic(results_csv_path, clustopt_out.results_df)
        except Exception as exc:
            return _finalize_failure(
                base_result, view_dir, stage="save_results_csv", error=exc, started_at=started_at, cfg=cfg
            )

    # Normalize the ClustOpt-side stats so the rest of the flow doesn't care
    # whether we just ran ClustOpt or reused an existing clustopt_results.csv.
    if reused_clustopt:
        co_stats = _stats_from_existing_results_csv(results_csv_path)
    else:
        assert clustopt_out is not None
        co_stats = {
            "n_configs": int(clustopt_out.n_configs),
            "n_valid_configs": int(clustopt_out.n_valid_configs),
            "runtime_sec": float(clustopt_out.runtime_sec),
            "fit_predict_sec": float(clustopt_out.fit_predict_sec),
            "cvi_eval_sec": float(clustopt_out.cvi_eval_sec),
            "best_score": float(clustopt_out.best_score),
            "best_config": clustopt_out.best_config,
            "profiler_summary": clustopt_out.profiler_summary,
        }

    # 4) compute metric utilities (reuse existing implementation)
    from models.ClustOpt.external_evaluation.compute_metric_utility import (
        compute_metric_utilities,
    )

    metric_utils_path = view_dir / "metric_utilities.csv"
    metric_utils_tmp = view_dir / "metric_utilities.tmp.csv"
    utility_compute_t0 = time.perf_counter()
    try:
        utilities_df = compute_metric_utilities(
            csv_path=str(results_csv_path),
            output_path=str(metric_utils_tmp),
        )
    except Exception as exc:
        try:
            if metric_utils_tmp.exists():
                metric_utils_tmp.unlink()
        except Exception:
            pass
        return _finalize_failure(
            base_result, view_dir, stage="compute_metric_utilities", error=exc, started_at=started_at, cfg=cfg
        )
    utility_compute_sec = time.perf_counter() - utility_compute_t0

    # Atomically promote the metric_utilities.csv from .tmp.csv -> final.
    try:
        import os
        os.replace(metric_utils_tmp, metric_utils_path)
    except Exception as exc:
        return _finalize_failure(
            base_result, view_dir, stage="atomic_rename_metric_utilities", error=exc, started_at=started_at, cfg=cfg
        )

    # 5) build utility_vector.json
    try:
        utility_vector_payload = _build_utility_vector_payload(
            utilities_df=utilities_df,
            loaded=loaded,
            view_id=view_id,
            cfg=cfg,
            map_path=map_path,
            clustopt_n_configs=co_stats["n_configs"],
            clustopt_n_valid=co_stats["n_valid_configs"],
        )
        write_json_atomic(view_dir / "utility_vector.json", utility_vector_payload)
    except Exception as exc:
        return _finalize_failure(
            base_result, view_dir, stage="build_utility_vector", error=exc, started_at=started_at, cfg=cfg
        )

    # 6) timing summary
    timing_payload = {
        "runtime_sec": float(co_stats["runtime_sec"]),
        "fit_predict_sec": float(co_stats["fit_predict_sec"]),
        "cvi_eval_sec": float(co_stats["cvi_eval_sec"]),
        "utility_compute_sec": float(utility_compute_sec),
        "profiler_summary": co_stats["profiler_summary"],
        "clustopt_results_reused": bool(reused_clustopt),
    }
    write_json_atomic(view_dir / "timing_summary.json", timing_payload)

    # 7) run log + record summary
    top_metrics = _top_metrics(utilities_df, k=5)
    summary_payload = {
        "status": "success",
        "dataset_id": loaded.dataset_id,
        "view_id": view_id,
        "map_path": str(map_path),
        "map_name": map_path.name,
        "metric_mode": cfg.metric_mode,
        "sample_size": int(sample_size),
        "n_configs": int(co_stats["n_configs"]),
        "n_valid_configs": int(co_stats["n_valid_configs"]),
        "n_metrics": int(len(utilities_df)),
        "top1_metric": top_metrics[0][0] if top_metrics else None,
        "top1_utility": float(top_metrics[0][1]) if top_metrics else 0.0,
        "top5_metrics": [m for m, _ in top_metrics],
        "runtime_sec": float(co_stats["runtime_sec"]),
        "best_config": co_stats["best_config"],
        "best_aggregate_score": float(co_stats["best_score"]),
        "clustopt_results_reused": bool(reused_clustopt),
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json_atomic(view_dir / "utility_record_summary.json", summary_payload)

    run_log_payload = {
        "dataset_id": loaded.dataset_id,
        "view_id": view_id,
        "started_at": started_at,
        "finished_at": summary_payload["finished_at"],
        "map_path": str(map_path),
        "runtime_block": cfg.runtime_block(),
        "n_points": int(loaded.n_points),
        "n_features_in_csv": int(len(loaded.feature_names)),
        "cluster_count_from_generator": loaded.cluster_count_from_generator,
        "runtime_sec": float(co_stats["runtime_sec"]),
        "fit_predict_sec": float(co_stats["fit_predict_sec"]),
        "cvi_eval_sec": float(co_stats["cvi_eval_sec"]),
        "utility_compute_sec": float(utility_compute_sec),
        "clustopt_results_reused": bool(reused_clustopt),
    }
    write_json_atomic(view_dir / "run_log.json", run_log_payload)

    return RecordResult(
        dataset_id=loaded.dataset_id,
        dataset_dir=str(loaded.dataset_dir),
        family=loaded.family_id,
        subfamily=loaded.subfamily_id,
        specific_subfamily=loaded.specific_subfamily,
        view_id=view_id,
        status="success",
        metric_mode=cfg.metric_mode,
        sample_size=sample_size,
        map_name=map_path.name,
        n_configs=int(co_stats["n_configs"]),
        n_valid_configs=int(co_stats["n_valid_configs"]),
        runtime_sec=float(co_stats["runtime_sec"]),
        fit_predict_sec=float(co_stats["fit_predict_sec"]),
        cvi_eval_sec=float(co_stats["cvi_eval_sec"]),
        utility_compute_sec=float(utility_compute_sec),
        n_metrics=int(len(utilities_df)),
        top1_metric=top_metrics[0][0] if top_metrics else None,
        top1_utility=float(top_metrics[0][1]) if top_metrics else 0.0,
        top5_metrics=[m for m, _ in top_metrics],
    )


# ----------------------------------------------------------------------- utils

def _existing_clustopt_csv_is_usable(results_csv_path: Path) -> bool:
    """Return True if a previous run already wrote a valid clustopt_results.csv.

    We treat the CSV as usable when it can be read by pandas and has at least
    one row plus an ``ARI`` column (required by ``compute_metric_utilities``).
    """
    if not results_csv_path.is_file():
        return False
    try:
        df = pd.read_csv(results_csv_path, nrows=1)
    except Exception:
        return False
    if df.empty:
        return False
    if "ARI" not in df.columns:
        return False
    return True


def _stats_from_existing_results_csv(results_csv_path: Path) -> dict[str, Any]:
    """Recover the ClustOpt-side stats from an existing results CSV.

    Used when we skip the ClustOpt search because a previous (crashed) run
    already produced the same CSV. Timing fields are reported as zero so the
    summary makes it obvious this was a CSV-reuse path.
    """
    df = pd.read_csv(results_csv_path)
    n_configs = int(len(df))
    if "valid" in df.columns:
        n_valid = int(df["valid"].astype(bool).sum())
    else:
        n_valid = n_configs

    best_score = 0.0
    best_config: Optional[Dict[str, Any]] = None
    if not df.empty and "aggregate_score" in df.columns:
        finite = df.replace([float("inf"), float("-inf")], pd.NA).dropna(subset=["aggregate_score"])
        if not finite.empty:
            top = finite.sort_values("aggregate_score", ascending=False).iloc[0]
            try:
                best_score = float(top["aggregate_score"])
            except Exception:
                best_score = 0.0
            best_config = {
                "algorithm": top.get("algorithm"),
                "config_str": top.get("config_str"),
            }

    return {
        "n_configs": n_configs,
        "n_valid_configs": n_valid,
        "runtime_sec": 0.0,
        "fit_predict_sec": 0.0,
        "cvi_eval_sec": 0.0,
        "best_score": best_score,
        "best_config": best_config,
        "profiler_summary": {},
    }


def _top_metrics(utilities_df: pd.DataFrame, k: int = 5) -> list[tuple[str, float]]:
    if utilities_df is None or utilities_df.empty:
        return []
    if "metric" not in utilities_df.columns or "utility_score" not in utilities_df.columns:
        return []
    df = utilities_df.sort_values("utility_score", ascending=False).head(k)
    out: list[tuple[str, float]] = []
    for _, row in df.iterrows():
        try:
            out.append((str(row["metric"]), float(row["utility_score"])))
        except Exception:
            continue
    return out


def _build_utility_vector_payload(
    *,
    utilities_df: pd.DataFrame,
    loaded: LoadedDatasetForUtility,
    view_id: str,
    cfg: UtilityGenerationConfig,
    map_path: Path,
    clustopt_n_configs: int,
    clustopt_n_valid: int,
) -> dict[str, Any]:
    utility_keys: dict[str, float] = {}
    if utilities_df is not None and not utilities_df.empty:
        for _, row in utilities_df.iterrows():
            metric = str(row.get("metric", "")).strip()
            if not metric:
                continue
            try:
                score = float(row["utility_score"])
            except (TypeError, ValueError, KeyError):
                continue
            utility_keys[f"utility__{metric}"] = score

    top_metrics = _top_metrics(utilities_df, k=5)

    return {
        "dataset_id": loaded.dataset_id,
        "view_id": view_id,
        "map_path": str(map_path),
        "map_name": map_path.name,
        "metric_mode": cfg.metric_mode,
        "sample_size": int(cfg.fast_metric_sample_size if cfg.metric_mode == "fast" else 0),
        "n_configs": int(clustopt_n_configs),
        "n_valid_configs": int(clustopt_n_valid),
        "n_metrics": int(len(utility_keys)),
        "top_metrics": [{"metric": m, "utility_score": s} for m, s in top_metrics],
        "utilities": utility_keys,
    }


def _finalize_failure(
    base: RecordResult,
    view_dir: Path,
    *,
    stage: str,
    error: BaseException,
    started_at: str,
    cfg: UtilityGenerationConfig,
) -> RecordResult:
    """Persist a failure marker so resume mode knows this record didn't succeed."""
    err_message = f"{type(error).__name__}: {error}"
    tb = "".join(traceback.format_exception(type(error), error, error.__traceback__))

    payload = {
        "status": "failed",
        "dataset_id": base.dataset_id,
        "view_id": base.view_id,
        "stage": stage,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "traceback": tb,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "metric_mode": cfg.metric_mode,
        "map_path": str(cfg.map_path_for_view(base.view_id)),
    }
    try:
        view_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(view_dir / "utility_record_summary.json", payload)
    except Exception:
        # Never raise during failure-finalization.
        pass

    base.status = "failed"
    base.error_message = err_message
    return base
