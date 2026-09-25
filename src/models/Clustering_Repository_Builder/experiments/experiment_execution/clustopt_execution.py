"""Adapter around ClustOpt for a single (method, view) experiment run.

Responsibilities:

* build the ClustOpt pipeline from an experiment config,
* for dynamic CVI (``regressor_dynamic`` / ``oracle_dynamic``) resolve the
  metric weights for this dataset/view *before* the search algorithm is built
  (so the log captures the right per-metric columns),
* inject the runtime block (fast mode) and attach a profiler,
* run ``search(X_decision, X_full)`` -- partition space vs full 2-D eval space,
* recover the best configuration's predicted labels (by re-instantiating the
  best config and running ``fit_predict`` on the partition space),
* extract the selected metrics / weights actually used.

It never reimplements ClustOpt logic; it only orchestrates the existing builder
and search algorithm.
"""
from __future__ import annotations

import ast
import math
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_DYNAMIC_CVI_TYPES = {"regressor_dynamic", "oracle_dynamic", "knn_dynamic"}


@dataclass
class ExperimentRunOutcome:
    results_df: pd.DataFrame
    cvi_type: str
    selected_algorithm: Optional[str] = None
    selected_params: Optional[str] = None
    best_objective_score: Optional[float] = None
    n_trials_completed: int = 0
    runtime_sec: float = 0.0
    fit_predict_sec: float = 0.0
    cvi_eval_sec: float = 0.0
    profiler_summary: Dict[str, Any] = field(default_factory=dict)
    y_pred: Optional[np.ndarray] = None
    selected_metrics: List[str] = field(default_factory=list)
    selected_metric_weights: Dict[str, float] = field(default_factory=dict)
    top_metric: Optional[str] = None
    metric_weight_entropy: Optional[float] = None
    used_fallback: bool = False
    resolver_debug: Dict[str, Any] = field(default_factory=dict)
    # Search-budget accounting. ``n_trials_completed`` (== delivered trials) has
    # always been recorded; the rest make the fixed-50 protocol auditable:
    # a candidate that raises still consumes a trial, so
    # delivered == valid + failed.
    requested_trials: Optional[int] = None
    valid_trials: Optional[int] = None
    failed_trials: Optional[int] = None
    # Concrete Optuna sampler seed actually used (None => unseeded historical path).
    search_seed: Optional[int] = None


def _cvi_type_of(builder) -> str:
    cvi_cfg = builder.cvi_config
    if isinstance(cvi_cfg, dict):
        return str(cvi_cfg.get("type", "generic")).lower()
    return "generic"


def _weight_entropy(weights: Dict[str, float]) -> Optional[float]:
    vals = np.asarray([float(v) for v in weights.values()], dtype=float)
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size == 0:
        return None
    p = vals / vals.sum()
    entropy = float(-np.sum(p * np.log(p)))
    if vals.size > 1:
        return entropy / math.log(vals.size)  # normalized to [0, 1]
    return 0.0


def _extract_selected_metrics(evaluator) -> Tuple[List[str], Dict[str, float], Optional[str], Optional[float]]:
    """Return (metric_names, weights, top_metric, normalized_weight_entropy)."""
    # Dynamic evaluators expose resolved_weights (keyed by metric .name).
    weights: Dict[str, float] = {}
    resolved = getattr(evaluator, "resolved_weights", None)
    if resolved:
        weights = {str(k): float(v) for k, v in resolved.items()}
    else:
        ev_weights = getattr(evaluator, "weights", None)
        if ev_weights:
            weights = {str(k): float(v) for k, v in ev_weights.items()}
        else:
            names = getattr(evaluator, "metric_names", []) or []
            weights = {str(n): 1.0 for n in names}

    metric_names = list(weights.keys())
    top_metric = max(weights, key=weights.get) if weights else None
    return metric_names, weights, top_metric, _weight_entropy(weights)


def _best_row(df: pd.DataFrame):
    """Return the best (highest finite aggregate_score) row, or None."""
    if df is None or df.empty or "aggregate_score" not in df.columns:
        return None
    finite = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["aggregate_score"])
    if finite.empty:
        return None
    return finite.sort_values("aggregate_score", ascending=False).iloc[0]


def _predict_best_labels(search_space, best_row, X_decision: np.ndarray) -> Optional[np.ndarray]:
    """Re-instantiate the best config and fit_predict to recover labels."""
    try:
        config = ast.literal_eval(str(best_row["config_str"]))
    except (ValueError, SyntaxError):
        return None
    try:
        model = search_space.instantiate_from_config(config)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if hasattr(model, "fit_predict"):
                labels = model.fit_predict(np.asarray(X_decision))
            else:
                labels = model.fit(np.asarray(X_decision)).predict(np.asarray(X_decision))
        return np.asarray(labels).reshape(-1)
    except Exception:
        return None


def _requested_trials_of(builder) -> Optional[int]:
    params = getattr(builder, "search_algorithm_params", None)
    if not isinstance(params, dict):
        return None
    n = params.get("n_trials")
    return None if n is None else int(n)


def _count_valid_trials(df: pd.DataFrame) -> Optional[int]:
    """Trials whose candidate produced a usable partition (``valid`` truthy)."""
    if df is None or "valid" not in getattr(df, "columns", []):
        return None
    col = df["valid"]
    if col.dtype == bool:
        return int(col.sum())
    return int(col.astype(str).str.lower().isin(("true", "1")).sum())


def _count_failed_trials(df: pd.DataFrame) -> Optional[int]:
    """Trials that raised; these still consume a trial by design."""
    if df is None or "exception" not in getattr(df, "columns", []):
        return None
    col = df["exception"].astype(str).str.strip()
    return int(((col != "") & (col.str.lower() != "nan")).sum())


def _resolve_config_search_seed(builder, *, dataset_dir: Path, view_id: str):
    """Resolve ``search_algorithm.params.seed`` in place; return the seed or None.

    Mutates ``builder.search_algorithm_params`` so that:

    * ``base_seed`` (a policy key, not a constructor kwarg) is removed;
    * ``seed`` becomes a concrete ``int`` or is removed entirely.

    Absent ``seed`` => nothing changes => historical unseeded behaviour.
    """
    from .seeding import resolve_search_seed

    params = getattr(builder, "search_algorithm_params", None)
    if not isinstance(params, dict) or "seed" not in params:
        return None

    base_seed = params.pop("base_seed", None)
    seed = resolve_search_seed(
        params.get("seed"),
        base_seed=base_seed,
        dataset_id=Path(dataset_dir).name,
        view_id=view_id,
    )
    if seed is None:
        params.pop("seed", None)
    else:
        params["seed"] = int(seed)
    return seed


def run_experiment_clustopt(
    *,
    config_path: Path,
    dataset_dir: Path,
    view_id: str,
    X_decision: np.ndarray,
    X_full: np.ndarray,
    y_true: Optional[np.ndarray],
    runtime_block: Dict[str, Any],
    attach_profiler: bool = True,
) -> ExperimentRunOutcome:
    """Run one ClustOpt experiment and return a structured outcome.

    y_true may be None. It is used only to attach the external (ARI)
    evaluator, whose sole effect is an extra log column; the search objective is
    the aggregated CVI score in every case.
    """
    from models.ClustOpt import ClusteringOptimizationBuilder
    from models.ClustOpt.profiling.profiler import Profiler

    builder = ClusteringOptimizationBuilder(config_path)
    builder.runtime_config = dict(runtime_block)

    # Resolve the (optional) deterministic search seed BEFORE the search
    # algorithm is constructed. ``search_algorithm.params`` is splatted as
    # kwargs into the algorithm, so the policy keys ('auto' sentinel and
    # 'base_seed') must be resolved into a concrete ``seed`` int and the policy
    # key removed. Configs without a 'seed' key are untouched, so every
    # historical run keeps the unseeded Optuna path.
    resolved_search_seed = _resolve_config_search_seed(
        builder, dataset_dir=dataset_dir, view_id=view_id)

    cvi_type = _cvi_type_of(builder)

    # Build search space + evaluator, resolving dynamic CVI for this dataset/view
    # BEFORE constructing the search algorithm (so logging columns are correct).
    search_space = builder.build_search_space()
    evaluator = builder.build_evaluator()

    used_fallback = False
    resolver_debug: Dict[str, Any] = {}
    if cvi_type in _DYNAMIC_CVI_TYPES and hasattr(evaluator, "resolve"):
        evaluator.resolve(dataset_dir=str(dataset_dir), view_id=view_id)
        used_fallback = bool(getattr(evaluator, "_used_fallback", False))
        try:
            resolver_debug = evaluator.get_debug_info()
        except Exception:
            resolver_debug = {}

    search_algorithm = builder.build_search_algorithm(search_space, evaluator)
    search_algorithm.runtime_config = dict(runtime_block)
    # Ground truth is DIAGNOSTIC here: set_ground_truth only attaches an
    # evaluator that writes an ARI column into the search log, and the objective
    # never reads it. Passing None therefore runs an identical search with no
    # ARI column -- which is what a label-free external benchmark requires.
    if y_true is not None:
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

    metric_names, weights, top_metric, entropy = _extract_selected_metrics(evaluator)

    outcome = ExperimentRunOutcome(
        results_df=df,
        cvi_type=cvi_type,
        n_trials_completed=int(len(df)),
        runtime_sec=float(elapsed),
        fit_predict_sec=fit_predict_sec,
        cvi_eval_sec=cvi_eval_sec,
        profiler_summary=summary,
        selected_metrics=metric_names,
        selected_metric_weights=weights,
        top_metric=top_metric,
        metric_weight_entropy=entropy,
        used_fallback=used_fallback,
        resolver_debug=resolver_debug,
        requested_trials=_requested_trials_of(builder),
        valid_trials=_count_valid_trials(df),
        failed_trials=_count_failed_trials(df),
        search_seed=resolved_search_seed,
    )

    best = _best_row(df)
    if best is not None:
        outcome.selected_algorithm = best.get("algorithm")
        outcome.selected_params = str(best.get("config_str"))
        try:
            outcome.best_objective_score = float(best["aggregate_score"])
        except Exception:
            outcome.best_objective_score = None
        outcome.y_pred = _predict_best_labels(search_space, best, X_decision)

    return outcome
