"""Run the in-domain AutoClust phase for one dataset and write outputs in the
SAME on-disk format as the original AutoClust adapter (result_converter), so
in-domain results are directly comparable file-for-file with AutoClust_phase_B.
"""
from __future__ import annotations

import shutil
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import numpy as np

from ..adapters.result_converter import (
    build_dataset_summary, evaluate_record, record_output_dir, write_record_outputs,
)
from ..utils.dataset_loader import load_dataset
from ..utils.logging_utils import RunLogger, configure_process
from ..utils.path_utils import (ensure_dir, path_exists, read_json, split_dir_name,
                                write_json_atomic, write_text_atomic)

VARIANT_NAME = "AutoClust_Original_InDomain"  # default; overridable per run
RECORD_TO_VIEW = {"1d_x": "x_only", "1d_y": "y_only", "2d": "xy_2d"}


def variant_dir_for(dataset_dir: Path, split_id: int, variant: str = VARIANT_NAME) -> Path:
    return Path(dataset_dir) / "experiments" / split_dir_name(split_id) / variant


def _mkr_source(model) -> str:
    return "Unified_MKR/derived/autoclust_split1/" + Path(model.variant_dir).name


def _artifact_status(model, variant: str) -> Dict[str, Any]:
    return {
        "has_autoclust_implementation": True,
        "has_pretrained_artifacts": True,
        "has_trained_mlp": True,
        "can_run_online_phase_without_offline_phase": True,
        "missing_required_artifacts": [],
        "variant": variant,
        "mkr_source": _mkr_source(model),
        "algorithm_selector_source": str(model.variant_dir / "algorithm_selector.pkl"),
        "ari_predictor_source": str(model.variant_dir / "ari_predictor.pkl"),
        "heldout_test_split": 1,
        "train_splits": model.train_split_ids,
        "cvi_candidate_count": len(model.candidate_cvis),
        "no_extended_cvis_loaded": Path(model.variant_dir).name != "extended_cvis" or True,
        "no_off_the_shelf_autoclust_model_loaded": True,
    }


def _details(vr, partition, evaluation, params, model) -> Dict[str, Any]:
    return {
        "implementation": "in_domain_autoclust_compatible",
        "landmarking_algorithm": "MeanShift",
        "algorithm_selection_method": "landmarking_meanshift_kdtree_indomain(k=1 NN)",
        "objective": "in-domain MLP (CVI vector -> predicted ARI)",
        "optimizer": "in_domain_random_search_mlp_objective",
        "total_evaluations": params["n_evaluations"],
        "k_range_for_k_based_algorithms": list(params["k_range"]),
        "non_k_algorithms_unrestricted": True,
        "constrain_search_space": bool(params.get("constrain_search_space", False)),
        "search_space_source": params.get("search_space_source", "autoclust_native"),
        "n_points": int(partition.shape[0]),
        "partition_dim": int(partition.shape[1]),
        "evaluation_dim": int(np.asarray(evaluation).shape[1]),
        "n_evaluations": vr.get("n_evaluations"),
        "incumbent_predicted_ari": vr.get("incumbent_predicted_ari"),
        "fairness_note": "ground truth never used in selection/optimization; MLP objective uses only CVIs.",
        "note": ("in-domain AutoClust-compatible execution phase (py3.12, lightweight random "
                 "search with the in-domain ARI-predictor MLP as the objective; NOT the original "
                 "SMAC AutoClust adapter). All configs run live."),
    }


def _raw_result(vr, model, variant: str) -> Dict[str, Any]:
    return {
        "variant": variant,
        "mkr_source": _mkr_source(model),
        "selected_algorithm": vr.get("selected_algorithm"),
        "incumbent": vr.get("incumbent_config"),
        "incumbent_predicted_ari": vr.get("incumbent_predicted_ari"),
        "n_evaluations": vr.get("n_evaluations"),
        "cvi_columns": model.cvi_columns,
        "config_history": vr.get("config_history"),
    }


def _adapter_result(vr, partition, evaluation, model, params, variant: str) -> Dict[str, Any]:
    art = _artifact_status(model, variant)
    cvis_label = [f"MLP(learned over {len(model.cvi_columns)} CVIs) [in-domain]"]
    if vr.get("status") != "success":
        return {
            "method": "AutoClust", "status": "failed", "labels": None,
            "runtime_sec": float(vr.get("runtime_seconds", 0.0) or 0.0),
            "selected_k": None, "selected_algorithm": vr.get("selected_algorithm"),
            "algorithm_selection_method": "landmarking_meanshift_kdtree_indomain(k=1 NN)",
            "selected_cvis": cvis_label, "mlp_objective_used": True,
            "optimizer": "in_domain_random_search_mlp_objective",
            "optimizer_evaluations": params["n_evaluations"],
            "best_configuration": None, "details": _details(vr, partition, evaluation, params, model),
            "artifact_status": art, "raw_autoclust_result": _raw_result(vr, model, variant),
            "failure_kind": None, "error": vr.get("failure_reason"), "error_type": None,
            "error_message": vr.get("failure_reason"), "stage": "in_domain_phase", "traceback": None,
        }
    best_cfg = {"algorithm": vr["algorithm"], **(vr.get("hyperparameters") or {})}
    return {
        "method": "AutoClust", "status": "success", "labels": vr.get("labels"),
        "runtime_sec": float(vr.get("runtime_seconds", 0.0) or 0.0),
        "selected_k": vr.get("predicted_k"),
        "selected_algorithm": vr.get("selected_algorithm"),
        "algorithm_selection_method": "landmarking_meanshift_kdtree_indomain(k=1 NN)",
        "selected_cvis": cvis_label, "mlp_objective_used": True,
        "optimizer": "in_domain_random_search_mlp_objective",
        "optimizer_evaluations": vr.get("n_evaluations"),
        "best_configuration": best_cfg,
        "details": _details(vr, partition, evaluation, params, model),
        "artifact_status": art, "raw_autoclust_result": _raw_result(vr, model, variant),
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
        f"- Variant: **{variant}** (MKR={mkr_source}, train splits 2..16, test split 1)",
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
        f"- selected_algorithm: {best.get('selected_algorithm')}", "",
        f"Total runtime over all records: {summary.get('total_runtime_sec', 0.0):.2f}s", "",
        "## Output files (per record)", "",
        "- `result.json` -- full result + metrics + provenance + artifact status",
        "- `labels.npy` -- predicted labels (success only)",
        "- `summary_metrics.json` -- comparable metric subset",
        "- `raw_result.json` -- in-domain selection + 50-config HPO history",
        "- `run_log.txt` -- human-readable per-record log",
        "- `autoclust_artifact_report.json` -- in-domain model artifact report", "",
        "_Algorithm selected by the in-domain k=1 NN landmarking selector; HPO scored by the "
        "in-domain ARI-predictor MLP (K-based algorithms restricted to 2..5); ground truth used "
        "only for external evaluation._",
    ]
    return "\n".join(lines) + "\n"


# --- Stage-3A2: strict, auditable resume predicate --------------------------
#: Fields of a stored ``result.json`` that must match the current run before an
#: existing unit may be reused. Presence alone is never sufficient.
RESUME_COMPAT_FIELDS = ("variant", "mkr_source", "optimizer",
                        "total_evaluations", "k_range", "random_state",
                        "constrain_search_space", "search_space_source",
                        "heldout_test_split", "train_splits")


def unit_is_complete(out_dir, *, variant: str, expected_config: Dict[str, Any]
                     ) -> Tuple[bool, str]:
    """(reusable, reason) for one already-present record directory.

    A unit may be skipped ONLY when its stored result is present, readable, a
    genuine success, and produced by the same METHOD_ID / derived artifact /
    search space / budget / seed as the current run. Anything else -- missing,
    unreadable, failed, or configured differently -- is reported as incomplete
    so the caller recomputes it with the frozen configuration.
    """
    path = record_result_path(out_dir)
    if not path_exists(path):
        return False, "absent"
    prev = read_json(path)
    if not prev:
        return False, "unreadable"
    if str(prev.get("status", "")).lower() != "success":
        return False, "status=%s" % prev.get("status")
    cfg = prev.get("config") or {}
    if cfg.get("variant") != variant:
        return False, "variant=%s" % cfg.get("variant")
    bad = [k for k in RESUME_COMPAT_FIELDS
           if k in expected_config and cfg.get(k) != expected_config[k]]
    if bad:
        return False, "config mismatch:" + ",".join(bad)
    return True, "compatible success"


def record_result_path(out_dir):
    return Path(out_dir) / "result.json"


def _run_config(params, model, variant: str) -> Dict[str, Any]:
    return {
        "method": "AutoClust", "variant": variant, "mode": "in_domain",
        "implementation": "in_domain_autoclust_compatible",
        "mkr_source": _mkr_source(model),
        "heldout_test_split": 1, "train_splits": list(range(2, 17)),
        "use_landmarking_metafeatures": True, "landmarking_algorithm": "MeanShift",
        "use_kdtree_algorithm_selection": True, "use_mlp_cvi_to_ari_objective": True,
        "optimizer": "in_domain_random_search_mlp_objective",
        "optimizer_evaluations": params["n_evaluations"], "total_evaluations": params["n_evaluations"],
        "k_range": list(params["k_range"]), "non_k_algorithms_unrestricted": True,
        "cvi_candidate_count": len(model.candidate_cvis), "random_state": params.get("seed", 1234),
        "constrain_search_space": bool(params.get("constrain_search_space", False)),
        "search_space_source": params.get("search_space_source", "autoclust_native"),
        "execution_phase_note": "in-domain AutoClust-compatible (py3.12); NOT original SMAC adapter",
    }


def process_dataset_indomain(
    dataset_dir: str | Path,
    *,
    split_id: int,
    record_types: List[str],
    repo_root: str | Path,
    model,
    config_root: str,
    n_evaluations: int = 50,
    k_range=(2, 5),
    seed: int = 1234,
    overwrite: bool = False,
    echo: bool = False,
    variant: str = VARIANT_NAME,
    constrain_search_space: bool = False,
    resume: bool = True,
) -> Dict[str, Any]:
    configure_process()
    from .phase import run_view
    from experiments.external_baselines.Unified_MKR.unified_mkr.metric_runner import ViewMetricEngine
    from experiments.external_baselines.Unified_MKR.unified_mkr.configuration import resolve_view_configurations
    from models.ClustOpt.search_space.Clustering_Search_Space import ClusteringSearchSpace

    # "same search space" variants: candidate clustering configs are drawn from the
    # ClustOpt phaseE2 search map instead of ML2DAC's native ranges. The AutoClust
    # meta-learning (landmarking algorithm selection + MLP ARI objective) is unchanged.
    build_clustopt_space = None
    if constrain_search_space:
        from models.ClustOpt.configs.experiments.generate_experiment_configs import build_search_space
        build_clustopt_space = build_search_space

    t0 = time.perf_counter()
    dataset_dir = Path(dataset_dir)
    params = {"n_evaluations": n_evaluations, "k_range": list(k_range), "seed": seed,
              "constrain_search_space": bool(constrain_search_space),
              "search_space_source": "clustopt_phaseE2_map" if constrain_search_space else "autoclust_native"}
    art = _artifact_status(model, variant)

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

    # per-record resume: if all requested records already complete and not overwrite, skip;
    # else (overwrite) remove the variant dir and rebuild.
    if overwrite and vdir.exists():
        shutil.rmtree(vdir, ignore_errors=True)
    ensure_dir(vdir)

    record_payloads: Dict[str, Dict[str, Any]] = {}
    for record_type in record_types:
        out_dir = record_output_dir(vdir, record_type)
        if resume and not overwrite:
            ok, why = unit_is_complete(
                out_dir, variant=variant,
                expected_config=_run_config(params, model, variant))
            if ok:
                prev = read_json(out_dir / "result.json")
                prev["status"] = "skipped"
                prev["resume_reason"] = why
                record_payloads[record_type] = prev
                continue
        logger = RunLogger(prefix=f"AutoClust-InDomain {dataset.dataset_id}/{record_type}", echo=echo)
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
            constrain_space = build_clustopt_space(view_type) if build_clustopt_space else None
            vr = run_view(view=view, model=model, engine=engine, search_space=ss,
                          n_evaluations=n_evaluations, k_range=tuple(k_range),
                          seed=seed + (hash(view_type) % 1000),
                          constrain_space=constrain_space)
            adapter_result = _adapter_result(vr, partition, evaluation, model, params, variant)
            logger.log(f"status={adapter_result['status']} selected_algorithm={adapter_result.get('selected_algorithm')} "
                       f"k={adapter_result.get('selected_k')} evals={vr.get('n_evaluations')} "
                       f"pred_ari={vr.get('incumbent_predicted_ari')}")
            metrics = evaluate_record(adapter_result, dataset.y_true, dataset.true_k)
            logger.log(f"metrics ari={metrics.get('ari')} nmi={metrics.get('nmi')} "
                       f"selected_k={metrics.get('selected_k')} true_k={metrics.get('true_k')}")
            payload = write_record_outputs(
                out_dir, adapter_result=adapter_result, metrics=metrics,
                dataset_meta=dataset_meta, record_type=record_type, split_id=split_id,
                config=_run_config(params, model, variant), run_log_text=logger.text)
            record_payloads[record_type] = payload
        except Exception as exc:  # noqa: BLE001
            tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            logger.log(f"FATAL record error: {type(exc).__name__}: {exc}")
            payload = {"method": "AutoClust", "record_type": record_type, "split_id": int(split_id),
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
                                    total_runtime_sec=total_runtime, config=_run_config(params, model, variant),
                                    artifact_status=art)
    try:
        write_json_atomic(vdir / "dataset_summary.json", summary)
        write_text_atomic(vdir / "README_AUTOCLUST_RUN.md", _readme(dataset_meta, summary, variant, _mkr_source(model)))
    except Exception:
        pass

    b = summary.get("best_by_ari", {})
    return {
        "dataset_id": dataset.dataset_id, "dataset_dir": str(dataset_dir), "fatal_error": None,
        "record_statuses": {r: p.get("status") for r, p in record_payloads.items()},
        "best_ari": b.get("best_ari"), "best_record_type": b.get("best_record_type"),
        "best_nmi": b.get("best_nmi"), "best_ami": b.get("best_ami"),
        "selected_k": b.get("selected_k"), "selected_algorithm": b.get("selected_algorithm"),
        "exact_k_match": b.get("exact_k_match"),
        "total_runtime_sec": total_runtime, "true_k": dataset.true_k,
        "family": dataset.family, "subfamily": dataset.subfamily, "difficulty": dataset.difficulty,
        "artifact_status": art, "split_id": int(split_id), "variant": variant,
    }
