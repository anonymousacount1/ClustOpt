"""High-level resolvers: utility vector -> ``{metric_name: weight}`` + debug.

Both resolvers return a :class:`ResolvedMetrics` carrying the final weights
(keyed by metric ``.name``) plus debug metadata (selected metrics, raw utility
scores, weighting scheme, etc.). The CVI evaluators map the metric names to
registry keys before constructing the underlying generic evaluator.

The resolvers support two input modes for each source:

* a **preloaded** vector passed directly by the caller (no file I/O), or
* **on-demand** loading from the dataset folder.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .dynamic_k import (
    DYNAMIC_TOPK_POLICY,
    is_dynamic_top_k,
    select_dynamic_top_k,
)
from .knn_predictor_provider import get_knn_provider
from .loaders import load_feature_vector, load_oracle_utilities
from .regressor_predictor import get_regressor_predictor
from .weighting import compute_weights, select_top_k

logger = logging.getLogger(__name__)


@dataclass
class ResolvedMetrics:
    """Outcome of a dynamic metric resolution."""

    weights: Dict[str, float] = field(default_factory=dict)  # keyed by metric .name
    selected_metrics: List[str] = field(default_factory=list)
    raw_utilities: Dict[str, float] = field(default_factory=dict)
    debug: Dict = field(default_factory=dict)


def resolve_regressor_weights(
    *,
    params: Dict,
    feature_vector=None,
    dataset_dir=None,
    view_id: Optional[str] = None,
) -> ResolvedMetrics:
    """Resolve weights from the MLP utility regressor.

    ``params`` is the ``cvi.params`` block. A preloaded ``feature_vector``
    (1-D array or ``{feature_name: value}`` dict) short-circuits file I/O;
    otherwise the feature row is read from the dataset folder.
    """
    model_run_dir = params.get("model_run_dir")
    if not model_run_dir:
        raise ValueError("regressor_dynamic requires params.model_run_dir.")

    checkpoint_policy = params.get("checkpoint_policy", "fold_ensemble")
    fold_index = int(params.get("fold_index", 0))
    top_k_cfg = params.get("top_k", 5)
    weighting = params.get("weighting", "softmax")
    softmax_temperature = float(params.get("softmax_temperature", 0.5))
    metric_name_prefix = params.get("metric_name_prefix", "utility__")
    feature_source = params.get("feature_source", "features/features_records.csv")
    view_aware = bool(params.get("view_aware", True))
    # Phase-D: K can be sized from predicted (default) or real/oracle utilities.
    k_source = str(params.get("k_source", "predicted")).lower()

    predictor = get_regressor_predictor(
        model_run_dir, checkpoint_policy, fold_index
    )

    if feature_vector is None:
        if dataset_dir is None or view_id is None:
            raise ValueError(
                "regressor_dynamic needs either a preloaded feature_vector or "
                "both dataset_dir and view_id to load features on demand."
            )
        feature_vector = load_feature_vector(
            dataset_dir=dataset_dir,
            view_id=view_id,
            feature_columns=predictor.feature_columns,
            feature_source=feature_source,
            view_aware=view_aware,
        )

    # Predicted utility vector, keyed by bare metric name. Always the *ranking*
    # source for regressor methods.
    raw_utilities = predictor.predict(feature_vector)

    dynamic = is_dynamic_top_k(top_k_cfg)
    real_utility_path = None
    selected_k = None
    k_rel_raw = None
    if dynamic:
        if k_source == "real":
            # realK: rank by predicted utility, size K from the real/oracle vector.
            real_source = params.get(
                "real_utility_source", "utility/<view_id>/utility_vector.json")
            real_utilities = load_oracle_utilities(
                dataset_dir=dataset_dir, view_id=view_id,
                utility_source=real_source, metric_name_prefix=metric_name_prefix)
            real_utility_path = str(real_source).replace("<view_id>", str(view_id))
            k_src_vec = real_utilities
        else:
            k_source = "predicted"
            k_src_vec = raw_utilities
        selected, selected_k, k_rel_raw = select_dynamic_top_k(
            ranking_utilities=raw_utilities, k_source_utilities=k_src_vec)
    else:
        selected = select_top_k(raw_utilities, int(top_k_cfg))

    weights = compute_weights(selected, weighting, softmax_temperature)

    debug = {
        "mode": "regressor_dynamic",
        "model_run_dir": str(model_run_dir),
        "checkpoint_policy": checkpoint_policy,
        "fold_index": fold_index if checkpoint_policy == "best_single_fold" else None,
        "loaded_folds": list(getattr(predictor, "_fold_names", [])),
        "top_k": top_k_cfg,
        "dynamic_top_k": dynamic,
        "dynamic_topk_policy": DYNAMIC_TOPK_POLICY if dynamic else None,
        "dynamic_k_source": (k_source if dynamic else "none"),
        "selected_k_metrics_count": (selected_k if dynamic else len(selected)),
        "k_rel_raw": k_rel_raw,
        "real_utility_path": real_utility_path,
        "weighting": weighting,
        "softmax_temperature": softmax_temperature,
        "metric_name_prefix": metric_name_prefix,
        "view_id": view_id,
        "selected_metrics": [name for name, _ in selected],
        "selected_utilities": {name: score for name, score in selected},
        "final_weights": dict(weights),
    }
    return ResolvedMetrics(
        weights=weights,
        selected_metrics=[name for name, _ in selected],
        raw_utilities=raw_utilities,
        debug=debug,
    )


def resolve_knn_weights(
    *,
    params: Dict,
    feature_vector=None,
    dataset_dir=None,
    view_id: Optional[str] = None,
) -> ResolvedMetrics:
    """Resolve weights from the frozen Phase-E1 KNN utility predictor.

    Identical in every respect to :func:`resolve_regressor_weights` except that
    the utility vector comes from the frozen view-specific KNN model instead of
    the MLP regressor. Ranking, fixed/dynamic Top-K, predicted/real K sizing, and
    raw/softmax weighting are all reused unchanged.
    """
    knn_model_dir = params.get("knn_model_dir") or params.get("model_run_dir")
    if not knn_model_dir:
        raise ValueError("knn_dynamic requires params.knn_model_dir.")

    top_k_cfg = params.get("top_k", 5)
    weighting = params.get("weighting", "softmax")
    softmax_temperature = float(params.get("softmax_temperature", 0.5))
    metric_name_prefix = params.get("metric_name_prefix", "utility__")
    feature_source = params.get("feature_source", "features/features_records.csv")
    view_aware = bool(params.get("view_aware", True))
    k_source = str(params.get("k_source", "predicted")).lower()

    provider = get_knn_provider(knn_model_dir)

    if view_id is None:
        raise ValueError("knn_dynamic requires a view_id (the KNN is view-specific).")

    if feature_vector is None:
        if dataset_dir is None:
            raise ValueError(
                "knn_dynamic needs either a preloaded feature_vector or both "
                "dataset_dir and view_id to load features on demand.")
        feature_vector = load_feature_vector(
            dataset_dir=dataset_dir, view_id=view_id,
            feature_columns=provider.feature_columns,
            feature_source=feature_source, view_aware=view_aware)

    cache_key = None
    if dataset_dir is not None:
        from pathlib import Path as _P
        cache_key = (str(_P(dataset_dir).resolve()), str(view_id))

    # Predicted utility vector (bare metric names). Always the ranking source.
    raw_utilities, neighbor_debug = provider.predict(
        feature_vector, view_id, cache_key=cache_key, want_neighbor_debug=True)

    dynamic = is_dynamic_top_k(top_k_cfg)
    real_utility_path = None
    selected_k = None
    k_rel_raw = None
    if dynamic:
        if k_source == "real":
            real_source = params.get(
                "real_utility_source", "utility/<view_id>/utility_vector.json")
            real_utilities = load_oracle_utilities(
                dataset_dir=dataset_dir, view_id=view_id,
                utility_source=real_source, metric_name_prefix=metric_name_prefix)
            real_utility_path = str(real_source).replace("<view_id>", str(view_id))
            k_src_vec = real_utilities
        else:
            k_source = "predicted"
            k_src_vec = raw_utilities
        selected, selected_k, k_rel_raw = select_dynamic_top_k(
            ranking_utilities=raw_utilities, k_source_utilities=k_src_vec)
    else:
        selected = select_top_k(raw_utilities, int(top_k_cfg))

    # Record whether the raw-normalized weighting had to fall back to uniform.
    used_weight_fallback = False
    if str(weighting).lower() == "normalized_positive":
        pos_sum = sum(max(float(s), 0.0) for _, s in selected)
        used_weight_fallback = pos_sum <= 0.0

    weights = compute_weights(selected, weighting, softmax_temperature)

    debug = {
        "mode": "knn_dynamic",
        "utility_source": "knn",
        "metric_ranking_source": "knn",
        "top_k": top_k_cfg,
        "dynamic_top_k": dynamic,
        "dynamic_topk_policy": DYNAMIC_TOPK_POLICY if dynamic else None,
        "dynamic_k_source": (k_source if dynamic else "none"),
        "selected_k_metrics_count": (selected_k if dynamic else len(selected)),
        "k_rel_raw": k_rel_raw,
        "real_utility_path": real_utility_path,
        "weighting": weighting,
        "softmax_temperature": softmax_temperature,
        "metric_name_prefix": metric_name_prefix,
        "view_id": view_id,
        "used_weight_fallback": used_weight_fallback,
        "selected_metrics": [name for name, _ in selected],
        "selected_utilities": {name: score for name, score in selected},
        "final_weights": dict(weights),
        "neighbor_debug": neighbor_debug,
        **provider.config_summary(),
    }
    return ResolvedMetrics(
        weights=weights,
        selected_metrics=[name for name, _ in selected],
        raw_utilities=raw_utilities,
        debug=debug,
    )


def resolve_oracle_weights(
    *,
    params: Dict,
    utility_vector=None,
    dataset_dir=None,
    view_id: Optional[str] = None,
) -> ResolvedMetrics:
    """Resolve weights from the *true* (oracle) utility vector.

    ``utility_vector`` may be preloaded as ``{metric_name: utility}`` (with or
    without the ``utility__`` prefix); otherwise it is read from the dataset
    folder. This is an experiment-only upper bound and must never be used for
    real inference.

    Supported ``params`` keys (beyond the shared ones):

    ``candidate_metrics``
        Optional sequence of metric names restricting the candidate pool
        *before* Top-K selection. ``None`` / absent (default) reproduces the
        historical behaviour of ranking over the entire utility vector. Used by
        the Stage-1 ``HO`` arm to select the true Top-5 among the Head-14
        inventory only. Raises if any listed metric is absent from the vector.
    """
    top_k_cfg = params.get("top_k", 5)
    weighting = params.get("weighting", "softmax")
    softmax_temperature = float(params.get("softmax_temperature", 0.5))
    metric_name_prefix = params.get("metric_name_prefix", "utility__")
    utility_source = params.get(
        "utility_source", "utility/<view_id>/utility_vector.json"
    )

    if utility_vector is None:
        if dataset_dir is None or view_id is None:
            raise ValueError(
                "oracle_dynamic needs either a preloaded utility_vector or "
                "both dataset_dir and view_id to load utilities on demand."
            )
        raw_utilities = load_oracle_utilities(
            dataset_dir=dataset_dir,
            view_id=view_id,
            utility_source=utility_source,
            metric_name_prefix=metric_name_prefix,
        )
    else:
        prefix = metric_name_prefix or ""
        raw_utilities = {
            (k[len(prefix):] if prefix and k.startswith(prefix) else k): float(v)
            for k, v in dict(utility_vector).items()
        }

    # Optional restriction of the candidate pool BEFORE Top-K selection.
    # ``candidate_metrics=None`` (default) preserves the historical behaviour
    # exactly: Top-K is chosen over the whole utility vector.  When a list is
    # supplied (e.g. the Head-14 inventory for the Stage-1 HO arm), the true
    # utility vector is filtered to those metrics first, so Top-K can only ever
    # come from the restricted inventory.
    candidate_metrics = params.get("candidate_metrics")
    all_utilities = dict(raw_utilities)
    restricted_to = None
    missing_candidates: list = []
    if candidate_metrics:
        restricted_to = [str(m) for m in candidate_metrics]
        missing_candidates = [m for m in restricted_to if m not in raw_utilities]
        if missing_candidates:
            raise ValueError(
                f"oracle_dynamic candidate_metrics lists {len(missing_candidates)} "
                f"metric(s) absent from the utility vector: {missing_candidates}."
            )
        raw_utilities = {m: raw_utilities[m] for m in restricted_to}

    # For oracle methods the real utility vector is both the ranking and the
    # K source (real_top_dynamic_*).
    dynamic = is_dynamic_top_k(top_k_cfg)
    selected_k = None
    k_rel_raw = None
    if dynamic:
        selected, selected_k, k_rel_raw = select_dynamic_top_k(
            ranking_utilities=raw_utilities, k_source_utilities=raw_utilities)
    else:
        selected = select_top_k(raw_utilities, int(top_k_cfg))

    weights = compute_weights(selected, weighting, softmax_temperature)

    debug = {
        "mode": "oracle_dynamic",
        "top_k": top_k_cfg,
        "dynamic_top_k": dynamic,
        "dynamic_topk_policy": DYNAMIC_TOPK_POLICY if dynamic else None,
        "dynamic_k_source": ("real" if dynamic else "none"),
        "selected_k_metrics_count": (selected_k if dynamic else len(selected)),
        "k_rel_raw": k_rel_raw,
        "weighting": weighting,
        "softmax_temperature": softmax_temperature,
        "metric_name_prefix": metric_name_prefix,
        "utility_source": utility_source,
        "real_utility_path": str(utility_source).replace("<view_id>", str(view_id)),
        "view_id": view_id,
        # Candidate-pool restriction (None => historical unrestricted behaviour).
        # Recorded in full so the selection can be replayed offline.
        "candidate_metrics": restricted_to,
        "candidate_metric_count": (
            len(restricted_to) if restricted_to is not None else len(all_utilities)
        ),
        "candidate_pool_utilities": (
            {m: all_utilities[m] for m in restricted_to}
            if restricted_to is not None else None
        ),
        "selected_metrics": [name for name, _ in selected],
        "selected_utilities": {name: score for name, score in selected},
        "final_weights": dict(weights),
    }
    return ResolvedMetrics(
        weights=weights,
        selected_metrics=[name for name, _ in selected],
        raw_utilities=raw_utilities,
        debug=debug,
    )
