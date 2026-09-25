"""Dynamic CVI evaluators: ``regressor_dynamic`` and ``oracle_dynamic``.

Both types resolve a concrete ``{metric_name: weight}`` mapping *before* the
search runs (per dataset record/view) and then delegate every CVI computation
to a standard :class:`GenericCVIEvaluator`. This keeps the search algorithms,
the metric implementations, and the aggregation logic completely unchanged --
the only new behaviour is *how* the weight dictionary is produced.

Lifecycle expected from the (future) experiment runner::

    evaluator = builder.build_evaluator()          # a dynamic evaluator
    evaluator.resolve(feature_vector=feat)          # or dataset_dir + view_id
    search.run(...)                                 # uses evaluator normally

If :meth:`evaluate` is reached before :meth:`resolve`, the evaluator attempts a
best-effort lazy resolution from ``params`` (``dataset_dir`` / ``view_id`` when
present) and otherwise falls back to ``fallback_metrics`` (when
``fallback_on_error`` is true) or raises.

The generic CVI behaviour is untouched: the delegate is an ordinary
``GenericCVIEvaluator`` constructed from the resolved weights.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from models.ClustOpt.cluster_validity_indices.cvi_base import BaseCVI
from models.ClustOpt.cluster_validity_indices.cvi_implementations.cvi_generic import (
    GenericCVIEvaluator,
)
from models.ClustOpt.cluster_validity_indices.dynamic_metric_selection import (
    resolve_knn_weights,
    resolve_metric_key,
    resolve_oracle_weights,
    resolve_regressor_weights,
)
from models.ClustOpt.cluster_validity_indices.dynamic_metric_selection.resolvers import (
    ResolvedMetrics,
)

logger = logging.getLogger(__name__)

# Generic-CVI construction params that may travel with a dynamic config either
# in a dedicated ``params_generic`` block or inline in ``params``.
_GENERIC_PARAM_KEYS = ("remove_noise", "noise_label", "min_clusters")

_DEFAULT_FALLBACK_METRICS = {"silhouette": 1.0}


class _BaseDynamicCVI(BaseCVI):
    """Shared plumbing for dynamic CVI evaluators."""

    mode_name: str = "dynamic"

    def __init__(
        self,
        metrics=None,           # ignored: weights are resolved dynamically
        weights=None,           # ignored
        *,
        params_generic: Optional[Dict[str, Any]] = None,
        **params: Any,
    ) -> None:
        self.params: Dict[str, Any] = dict(params)

        # Separate generic-CVI params (params_generic wins over inline params).
        generic_kwargs: Dict[str, Any] = {}
        for key in _GENERIC_PARAM_KEYS:
            if key in self.params:
                generic_kwargs[key] = self.params.pop(key)
        if isinstance(params_generic, dict):
            for key in _GENERIC_PARAM_KEYS:
                if key in params_generic:
                    generic_kwargs[key] = params_generic[key]
        self.generic_params: Dict[str, Any] = generic_kwargs

        self.fallback_on_error: bool = bool(self.params.get("fallback_on_error", True))
        self.fallback_metrics: Dict[str, float] = dict(
            self.params.get("fallback_metrics", _DEFAULT_FALLBACK_METRICS)
        )

        self._delegate: Optional[GenericCVIEvaluator] = None
        self._resolved: Optional[ResolvedMetrics] = None
        self._used_fallback: bool = False

    # ----------------------------------------------------- resolution hooks

    def _resolve_weights(self, **kwargs) -> ResolvedMetrics:  # pragma: no cover
        """Subclasses return a :class:`ResolvedMetrics` for the given context."""
        raise NotImplementedError

    def resolve(self, **kwargs) -> "_BaseDynamicCVI":
        """Resolve the metric weights for the current dataset/view context.

        Accepted kwargs (subclass dependent): ``feature_vector`` /
        ``utility_vector``, ``dataset_dir``, ``view_id``. On any error, falls
        back to ``fallback_metrics`` when ``fallback_on_error`` is true, else
        re-raises.
        """
        try:
            resolved = self._resolve_weights(**kwargs)
            self._delegate = self._build_delegate(resolved.weights)
            self._resolved = resolved
            self._used_fallback = False
        except Exception as exc:  # noqa: BLE001 - we deliberately gate on a flag
            if not self.fallback_on_error:
                raise
            logger.error(
                "[%s] dynamic metric resolution failed (%s); using fallback "
                "metrics %s.",
                self.mode_name, exc, self.fallback_metrics,
            )
            self._delegate = self._build_fallback_delegate()
            self._resolved = ResolvedMetrics(
                weights=dict(self.fallback_metrics),
                selected_metrics=list(self.fallback_metrics.keys()),
                debug={"mode": self.mode_name, "fallback": True, "error": str(exc)},
            )
            self._used_fallback = True
        return self

    # --------------------------------------------------------- delegate build

    def _build_delegate(self, weights_by_name: Dict[str, float]) -> GenericCVIEvaluator:
        """Build a GenericCVIEvaluator from ``{metric_name: weight}``.

        Metric names (``.name`` style) are mapped to registry keys; the
        generic evaluator looks weights up by ``.name`` so we keep the dict
        keyed by name. Unknown metrics are skipped with a warning.
        """
        if not weights_by_name:
            raise ValueError("Dynamic resolution produced an empty weight set.")

        metric_keys = []
        usable_weights: Dict[str, float] = {}
        skipped = []
        for name, weight in weights_by_name.items():
            key = resolve_metric_key(name)
            if key is None:
                skipped.append(name)
                continue
            metric_keys.append(key)
            usable_weights[name] = float(weight)

        if skipped:
            logger.warning(
                "[%s] skipping %d unknown metric(s): %s",
                self.mode_name, len(skipped), skipped,
            )
        if not metric_keys:
            raise ValueError(
                "No resolved metrics map to known registry keys "
                f"(got names: {list(weights_by_name.keys())})."
            )

        return GenericCVIEvaluator(
            metrics=metric_keys,
            weights=usable_weights,
            **self.generic_params,
        )

    def _build_fallback_delegate(self) -> GenericCVIEvaluator:
        fb = dict(self.fallback_metrics) or dict(_DEFAULT_FALLBACK_METRICS)
        # Fallback keys are config-facing registry keys; map any ``.name``
        # aliases through the same resolver for robustness.
        metric_keys = []
        weights: Dict[str, float] = {}
        for name, weight in fb.items():
            key = resolve_metric_key(name) or name
            metric_keys.append(key)
            weights[name] = float(weight)
        return GenericCVIEvaluator(
            metrics=metric_keys, weights=weights, **self.generic_params
        )

    # ------------------------------------------------------------- delegation

    def _ensure_delegate(self) -> GenericCVIEvaluator:
        if self._delegate is not None:
            return self._delegate
        # Best-effort lazy resolution from params (e.g. dataset_dir/view_id).
        lazy_kwargs = {
            k: self.params[k]
            for k in ("dataset_dir", "view_id")
            if k in self.params
        }
        self.resolve(**lazy_kwargs)
        if self._delegate is None:  # pragma: no cover - resolve guarantees this
            raise RuntimeError(
                f"{self.mode_name} CVI is not resolved; call resolve(...) first."
            )
        return self._delegate

    def evaluate(self, labels, X_decision, X_full=None, **kwargs):
        return self._ensure_delegate().evaluate(
            labels, X_decision, X_full, **kwargs
        )

    def normalize_scores(self, raw_scores, labels):
        return self._ensure_delegate().normalize_scores(raw_scores, labels)

    def aggregate_score(self, normalized_scores):
        return self._ensure_delegate().aggregate_score(normalized_scores)

    # ------------------------------------------------- generic-evaluator proxy

    # The search framework (SearchResultsLogger / partition context) reads
    # ``metrics``, ``weights`` and ``noise_label`` off the evaluator at
    # construction time. Proxy them to the resolved delegate so a dynamic
    # evaluator is a drop-in replacement for a GenericCVIEvaluator. Before
    # resolution these are empty/default, so constructing the search algorithm
    # never crashes -- resolve(...) before building the algorithm to capture
    # the correct per-metric columns in the log.

    @property
    def metrics(self):
        return self._delegate.metrics if self._delegate is not None else []

    @property
    def metric_names(self):
        return self._delegate.metric_names if self._delegate is not None else []

    @property
    def weights(self) -> Dict[str, float]:
        return dict(self._delegate.weights) if self._delegate is not None else {}

    @property
    def noise_label(self) -> int:
        if self._delegate is not None:
            return self._delegate.noise_label
        return int(self.generic_params.get("noise_label", -1))

    # ------------------------------------------------------------------ debug

    @property
    def resolved_weights(self) -> Dict[str, float]:
        return dict(self._resolved.weights) if self._resolved else {}

    def get_debug_info(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "mode": self.mode_name,
            "used_fallback": self._used_fallback,
            "generic_params": dict(self.generic_params),
        }
        if self._resolved is not None:
            info.update(self._resolved.debug)
        return info

    def save_debug_info(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as fh:
            json.dump(self.get_debug_info(), fh, indent=2, default=str)


class RegressorDynamicCVIEvaluator(_BaseDynamicCVI):
    """CVI weights resolved from a trained MLP utility regressor."""

    mode_name = "regressor_dynamic"

    def _resolve_weights(
        self,
        *,
        feature_vector=None,
        dataset_dir=None,
        view_id=None,
    ) -> ResolvedMetrics:
        return resolve_regressor_weights(
            params=self.params,
            feature_vector=feature_vector,
            dataset_dir=dataset_dir if dataset_dir is not None
            else self.params.get("dataset_dir"),
            view_id=view_id if view_id is not None
            else self.params.get("view_id"),
        )


class KNNDynamicCVIEvaluator(_BaseDynamicCVI):
    """CVI weights resolved from the frozen Phase-E1 KNN utility predictor.

    Deployable analogue of :class:`RegressorDynamicCVIEvaluator`: only the
    utility source differs (frozen view-specific KNN instead of the MLP). The
    KNN is view-specific, so ``view_id`` is required at resolve time.
    """

    mode_name = "knn_dynamic"

    def _resolve_weights(
        self,
        *,
        feature_vector=None,
        dataset_dir=None,
        view_id=None,
    ) -> ResolvedMetrics:
        return resolve_knn_weights(
            params=self.params,
            feature_vector=feature_vector,
            dataset_dir=dataset_dir if dataset_dir is not None
            else self.params.get("dataset_dir"),
            view_id=view_id if view_id is not None
            else self.params.get("view_id"),
        )


class OracleDynamicCVIEvaluator(_BaseDynamicCVI):
    """CVI weights resolved from the *true* (oracle) utility vector.

    Experiment-only upper bound -- never use for real inference.
    """

    mode_name = "oracle_dynamic"

    def _resolve_weights(
        self,
        *,
        utility_vector=None,
        dataset_dir=None,
        view_id=None,
    ) -> ResolvedMetrics:
        return resolve_oracle_weights(
            params=self.params,
            utility_vector=utility_vector,
            dataset_dir=dataset_dir if dataset_dir is not None
            else self.params.get("dataset_dir"),
            view_id=view_id if view_id is not None
            else self.params.get("view_id"),
        )
