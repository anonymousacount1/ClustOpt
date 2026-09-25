"""Thin wrapper around :class:`ClusteringOptimizationBuilder`.

This module **does not** reimplement any ClustOpt logic. It only:

* builds the pipeline from an existing utility-map JSON,
* injects the runtime config (``metric_mode``, sample size, random state),
* attaches a profiler so we can break the runtime down into
  ``fit_predict``/``cvi_evaluate`` slices for the summary CSV,
* runs ``search()`` and returns the results DataFrame.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class ClustOptRunResult:
    results_df: pd.DataFrame
    n_configs: int = 0
    n_valid_configs: int = 0
    best_score: float = 0.0
    best_config: Optional[Dict[str, Any]] = None
    runtime_sec: float = 0.0
    fit_predict_sec: float = 0.0
    cvi_eval_sec: float = 0.0
    profiler_summary: Dict[str, Dict[str, float]] = field(default_factory=dict)
    map_path: Optional[str] = None


def _count_valid(df: pd.DataFrame) -> int:
    if "valid" not in df.columns:
        return int(len(df))
    return int(df["valid"].astype(bool).sum())


def run_clustopt(
    *,
    map_path: Path,
    X_decision: np.ndarray,
    X_full: np.ndarray,
    y_true: np.ndarray,
    runtime_block: Dict[str, Any],
    attach_profiler: bool = True,
) -> ClustOptRunResult:
    """Run ClustOpt for a single (dataset, view) and return the results CSV.

    Notes:
        * The builder's ``runtime_config`` is overridden with our caller-supplied
          ``runtime_block`` before ``build_pipeline()`` so the search algorithm
          and its GlobalMetricContext see the right metric_mode/sample_size.
        * Ground truth is attached via ``set_ground_truth`` so ARI ends up in
          the results CSV (required by ``compute_metric_utilities``).
    """
    # Local imports to keep import-time cost low and to make import failures
    # surface inside the worker rather than at module load.
    from models.ClustOpt import ClusteringOptimizationBuilder
    from models.ClustOpt.profiling.profiler import Profiler

    builder = ClusteringOptimizationBuilder(map_path)
    # Override runtime block from caller — this completely replaces whatever the
    # map JSON ships with (typically nothing), so behaviour is deterministic.
    builder.runtime_config = dict(runtime_block)

    search_algorithm = builder.build_pipeline()
    search_algorithm.set_ground_truth(np.asarray(y_true))

    profiler = Profiler(enabled=bool(attach_profiler))
    if attach_profiler:
        search_algorithm.profiler = profiler

    t0 = time.perf_counter()
    search_algorithm.search(np.asarray(X_decision), np.asarray(X_full))
    elapsed = time.perf_counter() - t0

    df = search_algorithm.get_search_log_df()

    summary = profiler.summary_by_name() if attach_profiler else {}
    fit_predict_sec = float(summary.get("fit_predict", {}).get("total_sec", 0.0))
    cvi_eval_sec = float(summary.get("cvi_evaluate", {}).get("total_sec", 0.0))

    best_score = 0.0
    best_config: Optional[Dict[str, Any]] = None
    if not df.empty and "aggregate_score" in df.columns:
        finite = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["aggregate_score"])
        if not finite.empty:
            top = finite.sort_values("aggregate_score", ascending=False).iloc[0]
            best_score = float(top["aggregate_score"])
            best_config = {
                "algorithm": top.get("algorithm"),
                "config_str": top.get("config_str"),
            }

    return ClustOptRunResult(
        results_df=df,
        n_configs=int(len(df)),
        n_valid_configs=_count_valid(df),
        best_score=best_score,
        best_config=best_config,
        runtime_sec=float(elapsed),
        fit_predict_sec=fit_predict_sec,
        cvi_eval_sec=cvi_eval_sec,
        profiler_summary=summary,
        map_path=str(map_path),
    )
