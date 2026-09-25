"""Run AutoML4Clust on a single dataset folder (all requested record types).

Also exposes :func:`process_single_dataset` -- the picklable unit of work reused
by the subfamily/split multiprocessing runner. Per the project robustness
standard, a failure in any one record is captured into ``result.json`` (with
traceback) and never stops the remaining records; a dataset-level fatal error is
captured and returned, never raised.

CLI example (Windows ``^`` line continuation)::

    python -m experiments.external_baselines.AutoML4Clust.runners.run_automl4clust_dataset ^
      --dataset-folder "<dataset_folder>" ^
      --split-id 1 ^
      --config "experiments/external_baselines/AutoML4Clust/configs/automl4clust_smac_2d.json" ^
      --record-types 1d_x 1d_y 2d ^
      --repo-root "." ^
      --overwrite
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

# Allow running both as a module (-m) and, defensively, as a script.
from experiments.external_baselines.AutoML4Clust.adapters.automl4clust_adapter import (
    AutoML4ClustAdapter,
)
from experiments.external_baselines.AutoML4Clust.adapters.result_converter import (
    build_dataset_summary,
    evaluate_record,
    record_is_complete,
    record_output_dir,
    write_record_outputs,
)
from experiments.external_baselines.AutoML4Clust.utils.dataset_loader import load_dataset
from experiments.external_baselines.AutoML4Clust.utils.logging_utils import (
    RunLogger,
    configure_process,
)
from experiments.external_baselines.AutoML4Clust.utils.path_utils import (
    RECORD_TYPES,
    ensure_dir,
    read_json,
    split_dir_name,
    write_json_atomic,
    write_text_atomic,
)

configure_process()


def load_config(config_path: str | Path) -> Dict[str, Any]:
    payload = read_json(config_path)
    if payload is None:
        raise FileNotFoundError(f"Could not read config JSON: {config_path}")
    return payload


def _clean_config_for_record(config: Dict[str, Any], record_type: str) -> Dict[str, Any]:
    """Config copy embedded into result.json: drop docs + pin the record type."""
    clean = {k: v for k, v in config.items() if not str(k).startswith("_")}
    clean["record_types"] = [record_type]
    return clean


def automl_dir_for(dataset_dir: Path, split_id: int) -> Path:
    return Path(dataset_dir) / "experiments" / split_dir_name(split_id) / "AutoML4Clust"


def _readme_text(dataset_meta: Dict[str, Any], summary: Dict[str, Any]) -> str:
    best = summary.get("best_by_ari", {})
    lines = [
        "# AutoML4Clust run",
        "",
        f"- Dataset: **{dataset_meta.get('dataset_id')}**",
        f"- Family / Subfamily: {dataset_meta.get('family')} / {dataset_meta.get('subfamily')}",
        f"- Difficulty: {dataset_meta.get('difficulty')}",
        f"- True K (external eval only): {dataset_meta.get('true_k')}",
        f"- Optimizer / Metric: {summary.get('optimizer')} / {summary.get('metric')}",
        "",
        "## Records",
        "",
        "| record | status |",
        "| --- | --- |",
    ]
    for rtype, status in summary.get("record_statuses", {}).items():
        lines.append(f"| {rtype} | {status} |")
    lines += [
        "",
        "## Best record (by ARI)",
        "",
        f"- record_type: **{best.get('best_record_type')}**",
        f"- ARI: {best.get('best_ari')}",
        f"- NMI: {best.get('best_nmi')}",
        f"- AMI: {best.get('best_ami')}",
        f"- selected_k: {best.get('selected_k')} (exact_k_match={best.get('exact_k_match')})",
        "",
        f"Total runtime over all records: {summary.get('total_runtime_sec'):.2f}s",
        "",
        "## Output files (per record)",
        "",
        "- `result.json` -- full result + metrics + provenance + any error/traceback",
        "- `labels.npy` -- predicted labels (success only)",
        "- `summary_metrics.json` -- comparable metric subset",
        "- `raw_result.json` -- AutoML4Clust optimizer history",
        "- `run_log.txt` -- human-readable per-record log",
        "",
        "_AutoML4Clust never receives true labels, true K, ClustOpt metric "
        "utilities, or family metadata; these are used only for external "
        "evaluation after prediction._",
    ]
    return "\n".join(lines) + "\n"


def process_single_dataset(
    dataset_dir: str | Path,
    *,
    split_id: int,
    config: Dict[str, Any],
    record_types: List[str],
    repo_root: str | Path,
    overwrite: bool = False,
    true_k_override: Optional[int] = None,
    echo: bool = True,
) -> Dict[str, Any]:
    """Run all requested record types for one dataset; persist all outputs.

    Returns a compact dataset-level result dict (also used for run-level
    aggregation): dataset_id, dataset_dir, per-record statuses, best ARI,
    total runtime, and any fatal_error.
    """
    configure_process()
    t0 = time.perf_counter()
    dataset_dir = Path(dataset_dir)

    try:
        dataset = load_dataset(dataset_dir, true_k_override=true_k_override)
    except Exception as exc:  # dataset-level fatal -> recorded, not raised
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        return {
            "dataset_id": dataset_dir.name,
            "dataset_dir": str(dataset_dir),
            "fatal_error": f"{type(exc).__name__}: {exc}",
            "traceback": tb,
            "record_statuses": {},
            "best_ari": None,
            "total_runtime_sec": time.perf_counter() - t0,
        }

    dataset_meta = {
        "dataset_id": dataset.dataset_id,
        "family": dataset.family,
        "subfamily": dataset.subfamily,
        "difficulty": dataset.difficulty,
        "true_k": dataset.true_k,
    }
    automl_dir = automl_dir_for(dataset_dir, split_id)
    ensure_dir(automl_dir)

    record_payloads: Dict[str, Dict[str, Any]] = {}
    for record_type in record_types:
        out_dir = record_output_dir(automl_dir, record_type)

        # Resume / skip.
        if not overwrite and record_is_complete(out_dir):
            existing = read_json(out_dir / "result.json") or {"status": "success"}
            existing["status"] = "skipped" if existing.get("status") == "success" else existing.get("status")
            # Preserve the original metrics but mark as skipped for this run.
            existing.setdefault("metrics", {})
            record_payloads[record_type] = existing
            if echo:
                print(f"[SKIP] AutoML4Clust results already exist for dataset "
                      f"{dataset.dataset_id} record {record_type}", flush=True)
            continue

        logger = RunLogger(prefix=f"AutoML4Clust {dataset.dataset_id}/{record_type}", echo=echo)
        try:
            logger.log(f"start (true_k={dataset.true_k}, n_points={dataset.n_points})")
            partition, evaluation = dataset.get_record(record_type)
            logger.log(f"partition_space shape={partition.shape}, "
                       f"evaluation_space shape={evaluation.shape}")

            # Keep SMAC's internal scratch OUT of the dataset folder (only the
            # meaningful outputs belong there). Use a unique temp dir per
            # dataset+record+pid so concurrent workers never collide.
            work_dir = (
                Path(tempfile.gettempdir()) / "automl4clust_smac"
                / f"{dataset.dataset_id}_{record_type}_{os.getpid()}"
            )
            adapter = AutoML4ClustAdapter(config, repo_root, work_dir=work_dir)
            adapter_result = adapter.fit_predict(
                partition, evaluation, record_type=record_type,
            )
            logger.log(f"adapter status={adapter_result.get('status')} "
                       f"selected_k={adapter_result.get('selected_k')} "
                       f"runtime={adapter_result.get('runtime_sec')}")
            if adapter_result.get("status") != "success":
                logger.log(f"error: {adapter_result.get('error')}")

            metrics = evaluate_record(adapter_result, dataset.y_true, dataset.true_k)
            logger.log(f"metrics: ari={metrics.get('ari')} nmi={metrics.get('nmi')} "
                       f"ami={metrics.get('ami')} selected_k={metrics.get('selected_k')} "
                       f"true_k={metrics.get('true_k')}")

            payload = write_record_outputs(
                out_dir,
                adapter_result=adapter_result,
                metrics=metrics,
                dataset_meta=dataset_meta,
                record_type=record_type,
                split_id=split_id,
                config=_clean_config_for_record(config, record_type),
                run_log_text=logger.text,
            )
            record_payloads[record_type] = payload
        except Exception as exc:  # per-record safety net
            tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            logger.log(f"FATAL record error: {type(exc).__name__}: {exc}")
            payload = {
                "method": "AutoML4Clust",
                "record_type": record_type,
                "split_id": int(split_id),
                "status": "failed",
                "dataset_id": dataset.dataset_id,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": tb,
                "metrics": {},
            }
            try:
                ensure_dir(out_dir)
                write_json_atomic(out_dir / "result.json", payload)
                write_text_atomic(out_dir / "run_log.txt", logger.text)
            except Exception:
                pass
            record_payloads[record_type] = payload

    # Dataset-level summary + README (always written).
    total_runtime = time.perf_counter() - t0
    summary = build_dataset_summary(
        dataset_meta, record_payloads,
        split_id=split_id, total_runtime_sec=total_runtime, config=config,
    )
    try:
        write_json_atomic(automl_dir / "dataset_summary.json", summary)
        write_text_atomic(automl_dir / "README_AUTOML4CLUST_RUN.md",
                          _readme_text(dataset_meta, summary))
    except Exception:
        pass

    return {
        "dataset_id": dataset.dataset_id,
        "dataset_dir": str(dataset_dir),
        "fatal_error": None,
        "record_statuses": {r: p.get("status") for r, p in record_payloads.items()},
        "best_ari": summary.get("best_by_ari", {}).get("best_ari"),
        "best_record_type": summary.get("best_by_ari", {}).get("best_record_type"),
        "total_runtime_sec": total_runtime,
        "true_k": dataset.true_k,
        "family": dataset.family,
        "subfamily": dataset.subfamily,
        "difficulty": dataset.difficulty,
    }


# --------------------------------------------------------------------------- CLI
def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run AutoML4Clust on a single dataset folder (all record types).",
    )
    p.add_argument("--dataset-folder", required=True, type=str)
    p.add_argument("--split-id", required=True, type=int)
    p.add_argument("--config", required=True, type=str)
    p.add_argument("--record-types", nargs="+", default=list(RECORD_TYPES),
                   choices=list(RECORD_TYPES))
    p.add_argument("--repo-root", required=True, type=str)
    p.add_argument("--true-k", type=int, default=None,
                   help="Optional true K override (external eval only).")
    p.add_argument("--overwrite", action="store_true")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    config = load_config(args.config)
    print("=" * 70, flush=True)
    print("AutoML4Clust -- single dataset", flush=True)
    print("=" * 70, flush=True)
    print(f"Dataset:      {args.dataset_folder}", flush=True)
    print(f"Split id:     {args.split_id}", flush=True)
    print(f"Record types: {args.record_types}", flush=True)
    print(f"Config:       {args.config}", flush=True)
    print(f"Overwrite:    {args.overwrite}", flush=True)
    print(flush=True)

    result = process_single_dataset(
        args.dataset_folder,
        split_id=args.split_id,
        config=config,
        record_types=args.record_types,
        repo_root=args.repo_root,
        overwrite=args.overwrite,
        true_k_override=args.true_k,
    )

    print(flush=True)
    print("=" * 70, flush=True)
    if result.get("fatal_error"):
        print(f"FATAL: {result['fatal_error']}", flush=True)
        return 1
    print(f"Done: {result['dataset_id']}", flush=True)
    print(f"Record statuses: {result['record_statuses']}", flush=True)
    print(f"Best ARI: {result['best_ari']} (record={result.get('best_record_type')})", flush=True)
    print(f"Total runtime: {result['total_runtime_sec']:.2f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
