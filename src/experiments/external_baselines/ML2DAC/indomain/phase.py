"""The in-domain ML2DAC application phase for a single view (record)."""
from __future__ import annotations

import json
import time
from typing import Dict, List, Optional

import numpy as np

from . import search as S
from .mkr import CLUSTOPT_CVIS, CVI_DIRECTION, ML2DAC_ONLY_CVIS, InDomainMKR


def _external_metrics(y_true: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import (
        adjusted_mutual_info_score, adjusted_rand_score,
        fowlkes_mallows_score, normalized_mutual_info_score,
    )
    return {
        "ari": float(adjusted_rand_score(y_true, labels)),
        "nmi": float(normalized_mutual_info_score(y_true, labels)),
        "ami": float(adjusted_mutual_info_score(y_true, labels)),
        "fmi": float(fowlkes_mallows_score(y_true, labels)),
    }


def _selected_cvi_value(engine, labels: np.ndarray, cvi: str) -> float:
    if cvi in CLUSTOPT_CVIS:
        return float(engine._clustopt_raw(labels, [cvi]).get(cvi, float("nan")))
    if cvi in ML2DAC_ONLY_CVIS:
        return float(engine._ml2dac(labels).get(cvi, float("nan")))
    return float("nan")


def _oriented(cvi: str, value: float) -> float:
    if value != value:  # NaN
        return -np.inf
    return value if CVI_DIRECTION.get(cvi) == "maximize" else -value


def run_view(
    *,
    view,                      # unified_mkr ViewRecord (X_decision, X_full, y_true, ...)
    mkr: InDomainMKR,
    engine,                    # ViewMetricEngine for this view
    search_space,              # ClusteringSearchSpace (knows catalogue algorithms)
    n_warmstarts: int,
    n_optimizer: int,
    k_range,
    seed: int,
    constrain_space: Optional[dict] = None,  # ClustOpt grid for *_same_search_space
) -> Dict[str, object]:
    from experiments.external_baselines.Unified_MKR.unified_mkr.clustering import run_configuration
    from experiments.external_baselines.Unified_MKR.unified_mkr.configuration import ConfigSpec

    t0 = time.perf_counter()
    rid = view.record_id
    if not mkr.has_record(rid):
        return {"status": "failed", "failure_reason": f"no meta-features for {rid}",
                "record_id": rid, "view_type": view.view_type,
                "runtime_seconds": time.perf_counter() - t0}

    feats = mkr.features(rid)
    selected_cvi = mkr.predict_cvi(feats)
    nn_record = mkr.nearest_neighbor(feats)
    ranked = mkr.warmstart_configs(nn_record)
    warmstarts, skipped = S.select_warmstarts(ranked, k_range, n_warmstarts)

    # "same search space" variants: project the selected warmstarts onto the
    # ClustOpt grid (algorithm + rank order preserved) and sample the optimizer
    # from that grid instead of ML2DAC's native ranges. The meta-ranking above
    # (predicted CVI, nearest neighbour, warmstart ranking + K filter) is
    # untouched; only the candidate configuration values change.
    if constrain_space is not None:
        projected: List[dict] = []
        for w in warmstarts:
            snapped = S.snap_to_space(w["algorithm"], w["hyperparameters"],
                                      constrain_space, k_range)
            if snapped is None:
                skipped += 1  # warmstart algorithm not in the ClustOpt space
                continue
            w2 = dict(w)
            w2["hyperparameters"] = snapped
            w2["hyperparameters_json"] = json.dumps(snapped, sort_keys=True)
            projected.append(w2)
        warmstarts = projected

    ws_algorithms = sorted({w["algorithm"] for w in warmstarts}) or \
        sorted({r["algorithm"] for r in ranked})

    # search configs over the warmstart algorithms, K-restricted
    avoid = {w["algorithm"] + "|" + w["hyperparameters_json"] for w in warmstarts}
    if constrain_space is not None:
        opt_specs = S.sample_optimizer_configs_constrained(
            ws_algorithms, constrain_space, n_optimizer, k_range, seed, avoid=avoid)
    else:
        opt_specs = S.sample_optimizer_configs(ws_algorithms, n_optimizer, k_range, seed, avoid=avoid)

    # evaluate live: warmstarts first, then optimizer
    plan = [("warmstart", w["algorithm"], w["hyperparameters"]) for w in warmstarts] + \
           [("optimizer", o["algorithm"], o["hyperparameters"]) for o in opt_specs]

    history: List[dict] = []
    best = None  # (oriented_cvi, record dict)
    y_true = np.asarray(view.y_true)
    for source, algo, params in plan:
        spec = ConfigSpec(algorithm=algo, params=dict(params), original_config_id="")
        res = run_configuration(search_space, spec, view.X_decision)
        entry = {
            "source": source, "algorithm": algo, "hyperparameters": params,
            "valid_result": bool(res.valid_result), "predicted_k": int(res.predicted_k),
            "n_clusters_found": int(res.n_clusters_found),
            "failure_reason": res.failure_reason or None,
            "selected_cvi_value": None, "ari": None, "nmi": None, "ami": None, "fmi": None,
        }
        if res.valid_result and res.labels is not None and res.predicted_k >= 1:
            cvi_val = _selected_cvi_value(engine, res.labels, selected_cvi)
            ext = _external_metrics(y_true, res.labels)
            entry.update({"selected_cvi_value": cvi_val, **ext})
            oriented = _oriented(selected_cvi, cvi_val)
            if best is None or oriented > best[0]:
                best = (oriented, entry, res.labels)
        history.append(entry)

    runtime = time.perf_counter() - t0
    if best is None:
        return {
            "status": "failed",
            "failure_reason": "no valid clustering among evaluated configs",
            "record_id": rid, "view_type": view.view_type,
            "selected_cvi": selected_cvi, "nearest_neighbor_record_id": nn_record,
            "warmstart_configs_used": len(warmstarts),
            "warmstarts_skipped_due_to_k_range": skipped,
            "n_configs_evaluated": len(history), "config_history": history,
            "runtime_seconds": runtime,
        }

    oriented, inc, inc_labels = best
    return {
        "status": "success",
        "record_id": rid, "view_type": view.view_type,
        "selected_cvi": selected_cvi,
        "nearest_neighbor_record_id": nn_record,
        "warmstart_iterations": n_warmstarts,
        "optimizer_iterations": n_optimizer,
        "total_iterations": n_warmstarts + n_optimizer,
        "warmstart_configs_used": len(warmstarts),
        "warmstarts_skipped_due_to_k_range": skipped,
        "n_configs_evaluated": len(history),
        "incumbent_config": {"algorithm": inc["algorithm"], "hyperparameters": inc["hyperparameters"]},
        "algorithm": inc["algorithm"],
        "hyperparameters": inc["hyperparameters"],
        "predicted_k": inc["predicted_k"],
        "n_clusters_found": inc["n_clusters_found"],
        "incumbent_selected_cvi_value": inc["selected_cvi_value"],
        "ari": inc["ari"], "nmi": inc["nmi"], "ami": inc["ami"], "fmi": inc["fmi"],
        "true_k": view.true_k,
        "config_history": history,
        "labels": np.asarray(inc_labels).astype(int),
        "runtime_seconds": runtime,
        "failure_reason": None,
    }
