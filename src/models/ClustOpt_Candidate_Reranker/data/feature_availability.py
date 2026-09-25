"""What an ONLINE ClustOpt search can supply per visited candidate.

Every verdict below is traced to production source, not to inspection of any run
outcome. Split 1 is never opened -- neither its files nor its schemas are read.

The decisive chain:

1. ``DynamicCVI._build_delegate`` constructs
   ``GenericCVIEvaluator(metrics=metric_keys, weights=usable_weights)`` where
   ``metric_keys`` is exactly the Top-K set resolved from the utility vector.
2. ``GenericCVIEvaluator.evaluate`` computes ``for m in self.metrics`` only.
3. ``normalize_scores`` and ``aggregate_score`` iterate the same ``self.metrics``.

So an online trial computes K of 60 CVIs, K in {1, 3, 5, 10} or the Dynamic-K
value. The other 50-59 are never evaluated. Recovering them post-hoc requires the
partition labels, which no online artifact persists -- i.e. re-clustering.

``method_runner.write_run_outputs`` persists per method x view:
``clustopt_results.csv`` (the full per-trial search log from
``Search_Results_Logger``), ``best_result.json``, ``external_metrics.json``,
``selected_metrics.json``, ``timing_summary.json``, ``config_snapshot.json``,
``run_log.json``. The search log carries one row per delivered trial with
``algorithm``, ``config_str``, ``valid``, ``n_clusters_wo_noise``, ``noise_ratio``,
``n_labels_unique``, ``exception``, ``aggregate_score``, ``ARI`` and the raw/_norm
columns of the metrics that were actually computed.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

STATUSES: Tuple[str, ...] = (
    "ONLINE_ALREADY_AVAILABLE",
    "ONLINE_CHEAP_TO_COMPUTE",
    "ONLINE_EXPENSIVE_TO_COMPUTE",
    "OFFLINE_ONLY",
    "FORBIDDEN_GROUND_TRUTH",
)

# Proven from source. ``evidence`` names the module that settles each verdict.
AVAILABILITY: Tuple[Dict[str, Any], ...] = (
    {"field": "candidate algorithm", "status": "ONLINE_ALREADY_AVAILABLE",
     "evidence": "Search_Results_Logger.log_result writes 'algorithm' per trial",
     "note": "one row per delivered trial in the online clustopt_results.csv"},
    {"field": "candidate hyperparameters", "status": "ONLINE_ALREADY_AVAILABLE",
     "evidence": "Search_Results_Logger.log_result writes 'config_str' per trial",
     "note": "dict literal, parsed by candidate_schema.parse_config"},
    {"field": "candidate ID / config identity", "status": "ONLINE_ALREADY_AVAILABLE",
     "evidence": "algorithm|config_str is the identity used offline too",
     "note": "identical construction to the fixed32 candidate_key"},
    {"field": "candidate cluster labels", "status": "OFFLINE_ONLY",
     "evidence": "Search_Results_Logger._label_stats consumes labels and stores "
                 "only derived counts; no labels column is written",
     "note": "labels exist in memory during the trial but are never persisted"},
    {"field": "observed K (n_clusters_wo_noise)", "status": "ONLINE_ALREADY_AVAILABLE",
     "evidence": "Search_Results_Logger._label_stats -> n_clusters_wo_noise",
     "note": "label-free descriptor, persisted per trial"},
    {"field": "noise fraction", "status": "ONLINE_ALREADY_AVAILABLE",
     "evidence": "Search_Results_Logger._label_stats -> noise_ratio",
     "note": "mean(labels == -1)"},
    {"field": "n_labels_unique", "status": "ONLINE_ALREADY_AVAILABLE",
     "evidence": "Search_Results_Logger._label_stats -> n_labels_unique", "note": ""},
    {"field": "valid flag", "status": "ONLINE_ALREADY_AVAILABLE",
     "evidence": "Search_Results_Logger._label_stats -> valid "
                 "(len(unique_wo_noise) > 1)", "note": ""},
    {"field": "further partition descriptors "
              "(cluster-size entropy / imbalance / min / max / variance)",
     "status": "ONLINE_EXPENSIVE_TO_COMPUTE",
     "evidence": "derivable from labels only; labels are not persisted",
     "note": "cheap IF computed inside the trial while labels are live; "
             "expensive post-hoc because it needs re-clustering"},
    {"field": "SELECTED candidate CVI values", "status": "ONLINE_ALREADY_AVAILABLE",
     "evidence": "GenericCVIEvaluator.evaluate returns {m.name: value} for the "
                 "selected metrics; logged as '<m> (w=..)' and '<m>_norm'",
     "note": "exactly K of 60"},
    {"field": "UNSELECTED candidate CVI values", "status": "OFFLINE_ONLY",
     "evidence": "GenericCVIEvaluator.evaluate loops 'for m in self.metrics', "
                 "and self.metrics is the Top-K set from DynamicCVI",
     "note": "never computed online; post-hoc recovery needs the labels, "
             "hence re-clustering"},
    {"field": "objective J", "status": "ONLINE_ALREADY_AVAILABLE",
     "evidence": "GenericCVIEvaluator.aggregate_score -> 'aggregate_score' column",
     "note": "online divides by total weight; offline weights already sum to 1"},
    {"field": "selected metric IDs", "status": "ONLINE_ALREADY_AVAILABLE",
     "evidence": "method_runner writes selected_metrics.json "
                 "(selected_metrics, top_metric, resolver_debug)", "note": ""},
    {"field": "metric weights", "status": "ONLINE_ALREADY_AVAILABLE",
     "evidence": "selected_metrics.json -> selected_metric_weights", "note": ""},
    {"field": "predicted utilities (60-dim)", "status": "ONLINE_ALREADY_AVAILABLE",
     "evidence": "resolve_regressor_weights / resolve_oracle_weights debug block "
                 "recorded in selected_metrics.json resolver_debug",
     "note": "also reconstructable by re-running the predictor on stored "
             "meta-features, no clustering involved"},
    {"field": "utility source (MLP / KNN)", "status": "ONLINE_ALREADY_AVAILABLE",
     "evidence": "method identity + config_snapshot.json", "note": ""},
    {"field": "Top-K / Dynamic-K regime and actual K",
     "status": "ONLINE_ALREADY_AVAILABLE",
     "evidence": "selected_metrics.json (len(selected_metrics)) and "
                 "config_snapshot.json; dynamic_k_rel_raw in Phase-D metadata",
     "note": ""},
    {"field": "trial number / order", "status": "ONLINE_ALREADY_AVAILABLE",
     "evidence": "row order of the per-trial search log", "note": ""},
    {"field": "runtime", "status": "ONLINE_ALREADY_AVAILABLE",
     "evidence": "timing_summary.json (runtime_sec, fit_predict_sec, "
                 "cvi_eval_sec)",
     "note": "per method x view, not per trial"},
    {"field": "search-history descriptors "
              "(delivered / valid / failed trials, seed)",
     "status": "ONLINE_ALREADY_AVAILABLE",
     "evidence": "best_result.json budget block written by method_runner",
     "note": ""},
    {"field": "ARI of a visited candidate", "status": "FORBIDDEN_GROUND_TRUTH",
     "evidence": "Base_Search_Algorithm._compute_external_fields requires y_true",
     "note": "available offline for training only; never an input at deployment"},
    {"field": "true K", "status": "FORBIDDEN_GROUND_TRUTH",
     "evidence": "method_runner reads dataset_meta['cluster_count']",
     "note": "ground truth; excluded from every feature block"},
    {"field": "250 dataset/view meta-features", "status": "ONLINE_ALREADY_AVAILABLE",
     "evidence": "features/features_records.csv, computed before search and "
                 "consumed by the utility predictor",
     "note": "label-free by the Stage-2A-3A audit"},
)


def availability_rows() -> List[Dict[str, Any]]:
    rows = [dict(r) for r in AVAILABILITY]
    bad = [r for r in rows if r["status"] not in STATUSES]
    if bad:
        raise ValueError(f"unknown status in availability table: {bad[:2]}")
    return rows


def deployable_fields() -> List[str]:
    """Fields a deployed reranker may legitimately read."""
    return [r["field"] for r in AVAILABILITY
            if r["status"] in ("ONLINE_ALREADY_AVAILABLE",
                               "ONLINE_CHEAP_TO_COMPUTE")]


def rich_only_fields() -> List[str]:
    return [r["field"] for r in AVAILABILITY if r["status"] == "OFFLINE_ONLY"]


def forbidden_fields() -> List[str]:
    return [r["field"] for r in AVAILABILITY
            if r["status"] == "FORBIDDEN_GROUND_TRUTH"]


ONLINE_CVI_VERDICT: Dict[str, Any] = {
    "question": "Does an online learned-policy trial compute only its selected "
                "Top-K CVIs, all 60, or some other subset?",
    "answer": "A -- only the selected Top-K.",
    "chain": [
        "DynamicCVI._build_delegate -> GenericCVIEvaluator(metrics=metric_keys)",
        "metric_keys == the Top-K resolved from the utility vector",
        "GenericCVIEvaluator.evaluate: 'for m in self.metrics' only",
        "normalize_scores / aggregate_score iterate the same self.metrics",
    ],
    "consequence": "Masking is the true deployment state, not an experimental "
                   "observability restriction. The offline fixed32 repository "
                   "carries all 60 because it was produced by the full60 "
                   "utility-generation inventory, not by a learned policy.",
    "cost_of_recovering_unselected_cvis": "Requires the partition labels, which "
                                          "no online artifact persists; i.e. "
                                          "re-running the clustering.",
}

TRACE_SUFFICIENCY: Tuple[Dict[str, Any], ...] = (
    {"question": "Can every visited trial be joined to dataset/view/policy?",
     "answer": "YES",
     "evidence": "artifacts live at <dataset>/experiments/<split_dir>/<method>/"
                 "<view>/; run_log.json repeats dataset_id, view_id, method_name"},
    {"question": "Is algorithm/configuration persisted per trial?", "answer": "YES",
     "evidence": "search-log columns algorithm, config_str"},
    {"question": "Are cluster labels persisted?", "answer": "NO",
     "evidence": "Search_Results_Logger stores only label-derived counts"},
    {"question": "Are selected CVI values persisted?", "answer": "YES",
     "evidence": "'<m> (w=..)' and '<m>_norm' columns for the computed metrics"},
    {"question": "Is objective J persisted?", "answer": "YES",
     "evidence": "aggregate_score column; best_objective_score in best_result.json"},
    {"question": "Are metric weights persisted?", "answer": "YES",
     "evidence": "selected_metrics.json -> selected_metric_weights"},
    {"question": "Can masks be reconstructed?", "answer": "YES",
     "evidence": "selected_metrics list + canonical 60-metric order"},
    {"question": "Is ARI available offline after search?", "answer": "YES",
     "evidence": "ARI column + external_metrics.json; training-time only"},
    {"question": "Are unselected CVIs reconstructable post-hoc?", "answer": "NO",
     "evidence": "needs labels; labels are not persisted"},
    {"question": "Would missing candidate descriptors require re-clustering?",
     "answer": "YES for unselected CVIs and for any new partition descriptor "
               "beyond the four persisted ones",
     "evidence": "both are functions of the labels"},
    {"question": "Could all required DEPLOYABLE reranker inputs be built from "
                 "historical-style traces?",
     "answer": "YES",
     "evidence": "every FEATURE block of the deployable protocol maps to an "
                 "ONLINE_ALREADY_AVAILABLE field"},
)
