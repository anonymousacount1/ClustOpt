"""Run the in-domain ML2DAC phase for one dataset and write outputs in the SAME
on-disk format as the original ML2DAC adapter (result_converter), so in-domain
results are directly comparable file-for-file.

Per-dataset layout (variant dir ``ML2DAC_Original_InDomain``)::

    experiments/split_01/ML2DAC_Original_InDomain/
        1d_x/ 1d_y/ 2d/
            result.json  summary_metrics.json  raw_result.json
            ml2dac_artifact_report.json  labels.npy  run_log.txt
        dataset_summary.json
        README_ML2DAC_RUN.md
"""
from __future__ import annotations

import shutil
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import numpy as np

from ..adapters.result_converter import (
    build_dataset_summary, evaluate_record, record_output_dir, write_record_outputs,
)
from ..utils.dataset_loader import load_dataset
from ..utils.logging_utils import RunLogger, configure_process
from ..utils.path_utils import ensure_dir, read_json, split_dir_name, write_json_atomic, write_text_atomic

VARIANT_NAME = "ML2DAC_Original_InDomain"  # default; overridable per run
RECORD_TO_VIEW = {"1d_x": "x_only", "1d_y": "y_only", "2d": "xy_2d"}
CVI_ABBREV = {"calinski_harabasz": "CH", "davies_bouldin": "DBI", "silhouette": "SIL",
              "dbcv": "DBCV", "dunn_index": "DI", "coggins_jain_index": "CJI", "cop": "COP"}
ML2DAC_CVI_IDS = set(CVI_ABBREV)               # the 7 original ML2DAC CVIs
ML2DAC_CVI_ABBREVS = set(CVI_ABBREV.values())  # CH/DBI/SIL/DBCV/DI/CJI/COP


def cvi_source(metric_id: str) -> str:
    """Whether a selected CVI is one of the 7 ML2DAC CVIs or a ClustOpt metric."""
    return "ml2dac_cvi" if metric_id in ML2DAC_CVI_IDS else "clustopt_metric"


def variant_dir_for(dataset_dir: Path, split_id: int, variant: str = VARIANT_NAME) -> Path:
    return Path(dataset_dir) / "experiments" / split_dir_name(split_id) / variant


def _mkr_source(mkr) -> str:
    return "Unified_MKR/derived/ml2dac_split1/" + Path(mkr.variant_dir).name


def _artifact_status(mkr, params, variant: str) -> Dict[str, Any]:
    return {
        "has_pretrained_artifacts": True,
        "can_run_full_ml2dac": True,
        "missing_required_artifacts": [],
        "mkr_path": str(mkr.variant_dir),
        "mf_set_name": "statistical+info-theory+general (in-domain, train splits 2..16)",
        "variant": variant,
        "mkr_source": _mkr_source(mkr),
        "heldout_test_split": 1,
        "train_splits": mkr.train_split_ids,
        "cvi_candidate_count": len(mkr.candidate_cvis),
    }


def _details(vr, partition, evaluation, params) -> Dict[str, Any]:
    return {
        "mode": "in_domain",
        "selected_cvi_source": cvi_source(vr.get("selected_cvi")) if vr.get("selected_cvi") else None,
        "use_meta_cvi_selection": True,
        "use_warmstarting": True,
        "use_algorithm_reduction": True,
        "n_optimizer_loops": params["n_optimizer"],
        "n_warmstarts": params["n_warmstarts"],
        "total_iterations": params["n_warmstarts"] + params["n_optimizer"],
        "n_similar_datasets": 1,
        "constrain_search_space": bool(params.get("constrain_search_space", False)),
        "search_space_source": ("clustopt_phaseE2_map"
                                if params.get("constrain_search_space") else "ml2dac_native"),
        "k_range": list(params["k_range"]),
        "n_points": int(partition.shape[0]),
        "partition_dim": int(partition.shape[1]),
        "evaluation_dim": int(np.asarray(evaluation).shape[1]),
        "similar_dataset": [vr.get("nearest_neighbor_record_id")],
        "warmstart_configs_used": vr.get("warmstart_configs_used"),
        "warmstarts_skipped_due_to_k_range": vr.get("warmstarts_skipped_due_to_k_range"),
        "n_configs_evaluated": vr.get("n_configs_evaluated"),
        "note": ("in-domain ML2DAC-compatible execution phase (py3.12, lightweight seeded "
                 "random search; NOT the original SMAC ApplicationPhase). All configs run "
                 "live; incumbent selected by the predicted CVI; ground truth used only for "
                 "external evaluation."),
    }


def _raw_result(vr, mkr, variant: str) -> Dict[str, Any]:
    return {
        "variant": variant,
        "mkr_source": _mkr_source(mkr),
        "selected_cvi": CVI_ABBREV.get(vr.get("selected_cvi"), vr.get("selected_cvi")),
        "selected_cvi_metric_id": vr.get("selected_cvi"),
        "selected_cvi_source": cvi_source(vr.get("selected_cvi")) if vr.get("selected_cvi") else None,
        "nearest_neighbor_record_id": vr.get("nearest_neighbor_record_id"),
        "n_warmstarts": vr.get("warmstart_iterations"),
        "warmstart_configs_used": vr.get("warmstart_configs_used"),
        "warmstarts_skipped_due_to_k_range": vr.get("warmstarts_skipped_due_to_k_range"),
        "n_optimizer_iterations": vr.get("optimizer_iterations"),
        "n_configs_evaluated": vr.get("n_configs_evaluated"),
        "incumbent": vr.get("incumbent_config"),
        "incumbent_selected_cvi_value": vr.get("incumbent_selected_cvi_value"),
        "config_history": vr.get("config_history"),
    }


def _adapter_result(vr, partition, evaluation, mkr, params, variant: str) -> Dict[str, Any]:
    art = _artifact_status(mkr, params, variant)
    if vr.get("status") != "success":
        return {
            "method": "ML2DAC", "status": "failed", "labels": None,
            "runtime_sec": float(vr.get("runtime_seconds", 0.0) or 0.0),
            "selected_k": None, "selected_cvi": None, "selected_algorithm": None,
            "selected_algorithms": None, "warmstart_count": vr.get("warmstart_configs_used"),
            "best_configuration": None, "details": _details(vr, partition, evaluation, params),
            "artifact_status": art, "raw_ml2dac_result": _raw_result(vr, mkr, variant),
            "failure_kind": None, "error": vr.get("failure_reason"), "error_type": None,
            "error_message": vr.get("failure_reason"), "stage": "in_domain_phase", "traceback": None,
        }
    ws_algos = sorted({h["algorithm"] for h in (vr.get("config_history") or [])
                       if h.get("source") == "warmstart"})
    best_cfg = {"algorithm": vr["algorithm"], **(vr.get("hyperparameters") or {})}
    return {
        "method": "ML2DAC", "status": "success", "labels": vr.get("labels"),
        "runtime_sec": float(vr.get("runtime_seconds", 0.0) or 0.0),
        "selected_k": vr.get("predicted_k"),
        "selected_cvi": CVI_ABBREV.get(vr.get("selected_cvi"), vr.get("selected_cvi")),
        "selected_cvi_source": cvi_source(vr.get("selected_cvi")),
        "selected_algorithm": vr.get("algorithm"),
        "selected_algorithms": ws_algos,
        "warmstart_count": vr.get("warmstart_configs_used"),
        "best_configuration": best_cfg,
        "details": _details(vr, partition, evaluation, params),
        "artifact_status": art, "raw_ml2dac_result": _raw_result(vr, mkr, variant),
        "failure_kind": None, "error": None, "error_type": None,
        "error_message": None, "stage": None, "traceback": None,
    }


def _readme(dataset_meta, summary, variant: str, mkr_source: str) -> str:
    best = summary.get("best_by_ari", {})
    lines = [
        f"# {variant} run", "",
        f"- Dataset: **{dataset_meta.get('dataset_id')}**",
        f"- Family / Subfamily: {dataset_meta.get('family')} / {dataset_meta.get('subfamily')}",
        f"- Difficulty: {dataset_meta.get('difficulty')}",
        f"- True K (external eval only): {dataset_meta.get('true_k')}",
        f"- Variant: **{variant}** "
        f"(MKR={mkr_source}, train splits 2..16, test split 1)",
        "", "## Records", "", "| record | status |", "| --- | --- |",
    ]
    for rtype, status in summary.get("record_statuses", {}).items():
        lines.append(f"| {rtype} | {status} |")
    lines += [
        "", "## Best record (by ARI)", "",
        f"- record_type: **{best.get('best_record_type')}**",
        f"- ARI: {best.get('best_ari')}",
        f"- NMI: {best.get('best_nmi')}  AMI: {best.get('best_ami')}",
        f"- selected_k: {best.get('selected_k')} (exact_k_match={best.get('exact_k_match')})",
        f"- selected_cvi: {best.get('selected_cvi')}  selected_algorithm: {best.get('selected_algorithm')}",
        f"- warmstart_count: {best.get('warmstart_count')}", "",
        f"Total runtime over all records: {summary.get('total_runtime_sec', 0.0):.2f}s", "",
        "## Output files (per record)", "",
        "- `result.json` -- full result + metrics + provenance + artifact status",
        "- `labels.npy` -- predicted labels (success only)",
        "- `summary_metrics.json` -- comparable metric subset (+ selected_cvi/algorithm)",
        "- `raw_result.json` -- in-domain meta-learning info + 50-config search history",
        "- `run_log.txt` -- human-readable per-record log",
        "- `ml2dac_artifact_report.json` -- in-domain MKR artifact report", "",
        "_25 warmstart + 25 live search configs (K-based algorithms restricted to 2..5); "
        "ground truth used only for external evaluation._",
    ]
    return "\n".join(lines) + "\n"


def process_dataset_indomain(
    dataset_dir: str | Path,
    *,
    split_id: int,
    record_types: List[str],
    repo_root: str | Path,
    mkr,
    config_root: str,
    n_warmstarts: int = 25,
    n_optimizer: int = 25,
    k_range=(2, 5),
    seed: int = 1234,
    overwrite: bool = False,
    echo: bool = False,
    variant: str = VARIANT_NAME,
    constrain_search_space: bool = False,
) -> Dict[str, Any]:
    configure_process()
    from .phase import run_view
    from experiments.external_baselines.Unified_MKR.unified_mkr.metric_runner import ViewMetricEngine
    from experiments.external_baselines.Unified_MKR.unified_mkr.configuration import resolve_view_configurations
    from models.ClustOpt.search_space.Clustering_Search_Space import ClusteringSearchSpace

    # "*_same_search_space" variants constrain the candidate clustering configs to
    # the ClustOpt phaseE2 search map (build_search_space), per view. The ML2DAC
    # meta-ranking is unchanged; only the config values the model can pick change.
    clustopt_space_for = None
    if constrain_search_space:
        from models.ClustOpt.configs.experiments.generate_experiment_configs import (
            build_search_space as _build_clustopt_space,
        )
        clustopt_space_for = _build_clustopt_space

    t0 = time.perf_counter()
    dataset_dir = Path(dataset_dir)
    params = {"n_warmstarts": n_warmstarts, "n_optimizer": n_optimizer, "k_range": list(k_range),
              "constrain_search_space": bool(constrain_search_space)}
    art = _artifact_status(mkr, params, variant)

    try:
        dataset = load_dataset(dataset_dir, true_k_override=None)
    except Exception as exc:  # noqa: BLE001
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        return {"dataset_id": dataset_dir.name, "dataset_dir": str(dataset_dir),
                "fatal_error": f"{type(exc).__name__}: {exc}", "traceback": tb,
                "record_statuses": {}, "best_ari": None,
                "total_runtime_sec": time.perf_counter() - t0, "artifact_status": art}

    dataset_meta = {"dataset_id": dataset.dataset_id, "family": dataset.family,
                    "subfamily": dataset.subfamily, "difficulty": dataset.difficulty,
                    "true_k": dataset.true_k}
    vdir = variant_dir_for(dataset_dir, split_id, variant)
    if overwrite and vdir.exists():
        shutil.rmtree(vdir, ignore_errors=True)
    ensure_dir(vdir)

    record_payloads: Dict[str, Dict[str, Any]] = {}
    for record_type in record_types:
        out_dir = record_output_dir(vdir, record_type)
        if not overwrite and (out_dir / "result.json").exists():
            prev = read_json(out_dir / "result.json")
            if prev and str(prev.get("status", "")).lower() in ("success", "failed"):
                prev["status"] = "skipped" if prev.get("status") == "success" else prev.get("status")
                record_payloads[record_type] = prev
                continue
        logger = RunLogger(prefix=f"ML2DAC-InDomain {dataset.dataset_id}/{record_type}", echo=echo)
        try:
            view_type = RECORD_TO_VIEW[record_type]
            record_id = f"{dataset.dataset_id}__{view_type}"
            partition, evaluation = dataset.get_record(record_type)
            logger.log(f"start record_id={record_id} partition={partition.shape} evaluation={evaluation.shape}")
            engine = ViewMetricEngine(config_root, np.asarray(partition, dtype=float),
                                      np.asarray(evaluation, dtype=float), repo_root=str(repo_root))
            ss = ClusteringSearchSpace.from_dict(resolve_view_configurations(config_root, view_type).search_space_config)
            view = SimpleNamespace(record_id=record_id, view_type=view_type,
                                   X_decision=np.asarray(partition, dtype=float),
                                   X_full=np.asarray(evaluation, dtype=float),
                                   y_true=np.asarray(dataset.y_true), true_k=dataset.true_k)
            constrain_space = clustopt_space_for(view_type) if clustopt_space_for else None
            vr = run_view(view=view, mkr=mkr, engine=engine, search_space=ss,
                          n_warmstarts=n_warmstarts, n_optimizer=n_optimizer,
                          k_range=tuple(k_range), seed=seed + (hash(view_type) % 1000),
                          constrain_space=constrain_space)
            adapter_result = _adapter_result(vr, partition, evaluation, mkr, params, variant)
            logger.log(f"status={adapter_result['status']} selected_cvi={adapter_result.get('selected_cvi')} "
                       f"algo={adapter_result.get('selected_algorithm')} k={adapter_result.get('selected_k')} "
                       f"nn={vr.get('nearest_neighbor_record_id')} "
                       f"warmstarts_used={vr.get('warmstart_configs_used')} "
                       f"skipped_k={vr.get('warmstarts_skipped_due_to_k_range')} "
                       f"configs={vr.get('n_configs_evaluated')}")
            metrics = evaluate_record(adapter_result, dataset.y_true, dataset.true_k)
            logger.log(f"metrics ari={metrics.get('ari')} nmi={metrics.get('nmi')} "
                       f"selected_k={metrics.get('selected_k')} true_k={metrics.get('true_k')}")
            payload = write_record_outputs(
                out_dir, adapter_result=adapter_result, metrics=metrics,
                dataset_meta=dataset_meta, record_type=record_type, split_id=split_id,
                config=_run_config(params, mkr, variant), run_log_text=logger.text)
            record_payloads[record_type] = payload
        except Exception as exc:  # noqa: BLE001
            tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            logger.log(f"FATAL record error: {type(exc).__name__}: {exc}")
            payload = {"method": "ML2DAC", "record_type": record_type, "split_id": int(split_id),
                       "status": "failed", "dataset_id": dataset.dataset_id,
                       "error_type": type(exc).__name__, "error_message": str(exc),
                       "traceback": tb, "metrics": {}}
            try:
                ensure_dir(out_dir)
                write_json_atomic(out_dir / "result.json", payload)
                write_text_atomic(out_dir / "run_log.txt", logger.text)
            except Exception:
                pass
            record_payloads[record_type] = payload

    total_runtime = time.perf_counter() - t0
    summary = build_dataset_summary(dataset_meta, record_payloads, split_id=split_id,
                                    total_runtime_sec=total_runtime, config=_run_config(params, mkr, variant),
                                    artifact_status=art)
    try:
        write_json_atomic(vdir / "dataset_summary.json", summary)
        write_text_atomic(vdir / "README_ML2DAC_RUN.md", _readme(dataset_meta, summary, variant, _mkr_source(mkr)))
    except Exception:
        pass

    b = summary.get("best_by_ari", {})
    sel_cvi = b.get("selected_cvi")
    return {
        "dataset_id": dataset.dataset_id, "dataset_dir": str(dataset_dir), "fatal_error": None,
        "record_statuses": {r: p.get("status") for r, p in record_payloads.items()},
        "best_ari": b.get("best_ari"), "best_record_type": b.get("best_record_type"),
        "best_nmi": b.get("best_nmi"), "best_ami": b.get("best_ami"),
        "selected_k": b.get("selected_k"), "selected_cvi": sel_cvi,
        "selected_cvi_source": (("ml2dac_cvi" if sel_cvi in ML2DAC_CVI_ABBREVS else "clustopt_metric")
                                if sel_cvi else None),
        "selected_algorithm": b.get("selected_algorithm"), "exact_k_match": b.get("exact_k_match"),
        "total_runtime_sec": total_runtime, "true_k": dataset.true_k,
        "family": dataset.family, "subfamily": dataset.subfamily, "difficulty": dataset.difficulty,
        "artifact_status": art, "split_id": int(split_id), "variant": variant,
    }


def _run_config(params, mkr, variant: str) -> Dict[str, Any]:
    return {
        "method": "ML2DAC", "variant": variant,
        "mode": "in_domain",
        "mkr_source": _mkr_source(mkr),
        "heldout_test_split": 1, "train_splits": list(range(2, 17)),
        "use_meta_cvi_selection": True, "use_warmstarting": True, "use_algorithm_reduction": True,
        "optimizer": "in_domain_lightweight_random_search",
        "constrain_search_space": bool(params.get("constrain_search_space", False)),
        "search_space_source": ("clustopt_phaseE2_map"
                                if params.get("constrain_search_space") else "ml2dac_native"),
        "n_warmstarts": params["n_warmstarts"], "n_optimizer_loops": params["n_optimizer"],
        "total_iterations": params["n_warmstarts"] + params["n_optimizer"],
        "k_range": list(params["k_range"]),
        "cvi_candidate_count": len(mkr.candidate_cvis), "random_state": 1234,
        "execution_phase_note": "in-domain ML2DAC-compatible (py3.12); NOT original SMAC ApplicationPhase",
    }
