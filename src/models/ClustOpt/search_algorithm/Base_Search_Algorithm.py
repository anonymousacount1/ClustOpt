"""Abstract base class shared by every concrete search algorithm.

It standardises:

* logger construction (with metric names and weights),
* optional ground-truth attachment for external metrics such as ARI,
* a single :meth:`evaluate_config` entry point so subclasses don't duplicate
  the fit / evaluate / log block,
* helpers for exporting search results and rebuilding the best models,
* a per-search :class:`GlobalMetricContext` and a per-config
  :class:`PartitionMetricContext`, both opt-in and silently ignored by
  metrics that don't accept them.
"""
from __future__ import annotations

import ast
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np

from models.ClustOpt.Search_Results_Logger import SearchResultsLogger

try:
    from models.ClustOpt.external_evaluation.external_metrics import (
        ExternalEvaluationConfig,
        ExternalEvaluator,
    )
except Exception:  # pragma: no cover - optional dependency at runtime
    ExternalEvaluator = None  # type: ignore[assignment]
    ExternalEvaluationConfig = None  # type: ignore[assignment]

try:
    from models.ClustOpt.contexts import GlobalMetricContext, PartitionMetricContext
except Exception:  # pragma: no cover
    GlobalMetricContext = None  # type: ignore[assignment]
    PartitionMetricContext = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)


_INVALID_SCORE = -np.inf


class BaseSearchAlgorithm(ABC):
    """Skeleton for clustering-search algorithms.

    Subclasses only need to implement :meth:`search_core`; they should call
    :meth:`evaluate_config` for each candidate so logging and ground-truth
    handling stay consistent across implementations.
    """

    def __init__(
        self,
        search_space,
        evaluator,
        algorithm_name: str,
        search_algorithm_params: Optional[Dict[str, Any]] = None,
    ):
        self.search_space = search_space
        self.evaluator = evaluator
        self.algorithm_name = algorithm_name
        self.logger = SearchResultsLogger(
            metrics=evaluator.metrics,
            weights=evaluator.weights,
            search_algorithm_name=algorithm_name,
            search_space=search_space.get_full_search_space(),
            search_algorithm_params=dict(search_algorithm_params or {}),
        )
        self.external_evaluator: Optional["ExternalEvaluator"] = None
        # Optional profiler; off by default. See models/ClustOpt/profiling.
        self.profiler: Optional[Any] = None
        # Per-search global metric context (built lazily inside ``search``).
        self._global_context: Optional[Any] = None
        # Optional runtime knobs ({metric_mode, fast_metric_sample_size,
        # fast_metric_random_state}). When empty the GlobalMetricContext
        # falls back to exact-mode defaults — fully backward-compatible.
        self.runtime_config: Dict[str, Any] = {}

    # ----------------------------------------------------------------- search

    def search(
        self,
        X_decision: np.ndarray,
        X_full: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        # Build the per-search global metric context once. Backward-compat:
        # if the contexts module is unavailable, we just leave it None.
        self._global_context = (
            GlobalMetricContext(
                X_decision, X_full, runtime_config=self.runtime_config or None,
            )
            if GlobalMetricContext is not None
            else None
        )
        if self.profiler is not None:
            self.profiler.event("global_context_built")
        try:
            return self.search_core(X_decision, X_full)
        finally:
            self._global_context = None

    @abstractmethod
    def search_core(
        self,
        X_decision: np.ndarray,
        X_full: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Run the actual search loop and return the best ``{config, score}``."""

    # ----------------------------------------------------- per-config helpers

    def evaluate_config(
        self,
        algo_name: str,
        config: Dict[str, Any],
        X_decision: np.ndarray,
        X_full: Optional[np.ndarray] = None,
    ) -> Tuple[float, Optional[np.ndarray]]:
        """Instantiate, fit, evaluate, and log a single configuration.

        Returns ``(aggregate_score, labels)``. On failure, ``labels`` is ``None``
        and the score is :data:`_INVALID_SCORE`; the failure is logged with the
        exception captured in the results row.
        """
        prof = self.profiler
        logger.debug("Testing configuration: %s", config)
        try:
            with _section(prof, "instantiate"):
                model = self.search_space.instantiate_from_config(config)

            with _section(prof, "fit_predict"):
                labels = self._fit_predict(model, X_decision)

            partition_ctx = self._build_partition_context(labels)

            with _section(prof, "cvi_evaluate"):
                raw_scores = self.evaluator.evaluate(
                    labels=labels,
                    X_decision=X_decision,
                    X_full=X_full,
                    context=self._global_context,
                    partition_context=partition_ctx,
                )
            with _section(prof, "cvi_normalize"):
                normalized = self.evaluator.normalize_scores(raw_scores, labels)
            with _section(prof, "cvi_aggregate"):
                score = float(self.evaluator.aggregate_score(normalized))
            extra_fields = self._compute_external_fields(labels)

            with _section(prof, "log"):
                self.logger.log_result(
                    algorithm=algo_name,
                    config=config,
                    raw_scores=raw_scores,
                    normalized_scores=normalized,
                    aggregate_score=score,
                    labels=labels,
                    extra_fields=extra_fields,
                )
            return score, labels

        except Exception as exc:
            logger.warning("Error evaluating config %s: %s", config, exc)
            self.logger.log_result(
                algorithm=algo_name,
                config=config,
                raw_scores={},
                normalized_scores={},
                aggregate_score=_INVALID_SCORE,
                labels=None,
                exception=str(exc),
                extra_fields=self._compute_external_fields(None),
            )
            return _INVALID_SCORE, None

    def _build_partition_context(self, labels: np.ndarray) -> Optional[Any]:
        if PartitionMetricContext is None or self._global_context is None:
            return None
        return PartitionMetricContext(
            labels=labels,
            global_context=self._global_context,
            noise_label=getattr(self.evaluator, "noise_label", -1),
        )

    @staticmethod
    def _fit_predict(model, X: np.ndarray) -> np.ndarray:
        if hasattr(model, "fit_predict"):
            return np.asarray(model.fit_predict(X))
        return np.asarray(model.fit(X).predict(X))

    # ------------------------------------------------------- ground-truth API

    def set_ground_truth(
        self,
        y_true: np.ndarray,
        ignore_noise: bool = False,
        noise_label: int = -1,
    ) -> None:
        if ExternalEvaluator is None or ExternalEvaluationConfig is None:
            raise ImportError(
                "ExternalEvaluator could not be imported. "
                "Make sure models/ClustOpt/external_evaluation/external_metrics.py exists."
            )
        cfg = ExternalEvaluationConfig(noise_label=noise_label, ignore_noise=ignore_noise)
        self.external_evaluator = ExternalEvaluator(y_true=y_true, config=cfg)

    def _compute_external_fields(self, labels: Optional[np.ndarray]) -> Dict[str, Any]:
        if self.external_evaluator is None:
            return {}
        if labels is None:
            return {"ARI": float("nan")}
        try:
            return self.external_evaluator.compute(labels)
        except Exception as exc:
            return {"ARI": float("nan"), "external_eval_error": str(exc)}

    # ----------------------------------------------------------- log / report

    def get_search_log_df(self):
        return self.logger.to_dataframe()

    def export_log_to_csv(self, path: str):
        return self.logger.to_csv(path)

    def generate_summary_report(self, path: str):
        return self.logger.generate_summary_report(path)

    def make_full_report(self, data_name: str, X: np.ndarray, plot_fn, output_root: str | None = None):
        models = self.get_best_models()
        return self.logger.make_full_report(data_name, X, models, plot_fn, output_root=output_root)

    def get_best_models(self) -> Dict[str, Any]:
        """Re-instantiate the best estimator per algorithm from the search log.

        The config string saved in the log row is parsed via
        :func:`ast.literal_eval` (not :func:`eval`), so untrusted log files
        cannot execute arbitrary Python.
        """
        df = self.logger.to_dataframe()
        best_models: Dict[str, Any] = {}

        if df.empty or "algorithm" not in df.columns:
            return best_models

        for algo_name in df["algorithm"].unique():
            algo_rows = df[df["algorithm"] == algo_name]
            best_row = algo_rows.sort_values("aggregate_score", ascending=False).iloc[0]
            try:
                config = ast.literal_eval(best_row["config_str"])
            except (ValueError, SyntaxError) as exc:
                logger.warning(
                    "Could not parse config_str for algorithm '%s': %s",
                    algo_name,
                    exc,
                )
                continue
            best_models[algo_name] = self.search_space.instantiate_from_config(config)

        return best_models


# ---------------------------------------------------------------- profiler glue
def _section(profiler, name: str):
    """Use the profiler's section context manager when one is attached."""
    if profiler is None:
        return _NullSection()
    return profiler.section(name)


class _NullSection:
    def __enter__(self):
        return self
    def __exit__(self, *exc):
        return False
