r"""Subfamily-level Unified Raw MKR builder.

Builds one unified raw repository for a single subfamily: for every dataset
view (x_only / y_only / xy_2d) it evaluates exactly 32 clustering
configurations (constrained-HDBSCAN replaced), computes the de-duplicated union
of ML2DAC / AutoClust / ClustOpt metrics, extracts ML2DAC + AutoClust
meta-features, and writes the mandatory subfamily tables + a validation report.

Example
-------
    python experiments/external_baselines/Unified_MKR/runners/run_subfamily_unified_mkr.py ^
      --subfamily-root "...\arcs_circles_rings\subfamilies\annuli_variable_thickness" ^
      --output-root    "...\Unified_MKR\outputs" ^
      --n-jobs 8 --resume
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

# --- Make the repo importable before importing the package -----------------
_THIS = Path(__file__).resolve()
for _p in [_THIS, *_THIS.parents]:
    if (_p / "models" / "ClustOpt").is_dir():
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
        break

from experiments.external_baselines.Unified_MKR.unified_mkr import io_utils  # noqa: E402
from experiments.external_baselines.Unified_MKR.unified_mkr.builder import (  # noqa: E402
    BuildConfig,
    DatasetResult,
    build_configurations_table,
    build_metric_registry_table,
    process_dataset,
)
from experiments.external_baselines.Unified_MKR.unified_mkr.dataset_scanner import (  # noqa: E402
    SplitInfo,
    discover_dataset_dirs,
    load_split_assignments,
    split_info_for,
)
from experiments.external_baselines.Unified_MKR.unified_mkr.logging_utils import (  # noqa: E402
    configure_process,
    log,
)
from experiments.external_baselines.Unified_MKR.unified_mkr.storage import (  # noqa: E402
    UnifiedMKRStorage,
)
from experiments.external_baselines.Unified_MKR.unified_mkr.validation import (  # noqa: E402
    run_validation,
)

DEFAULT_CONFIG_ROOT = "models/ClustOpt/configs/utilities_maps"
DEFAULT_SPLIT_CSV = (
    "results_analysis/clustering_repository/analyzed_data/"
    "experiment_splits/dataset_split_assignments.csv"
)


def _derive_family_subfamily(subfamily_root: Path) -> tuple[str, str]:
    subfamily = subfamily_root.name
    parent = subfamily_root.parent
    if parent.name == "subfamilies":
        family = parent.parent.name
    else:
        family = parent.name
    return family, subfamily


def _fmt_hms(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def _worker_entry(args):
    dataset_dir, split, cfg = args
    io_utils.ensure_repo_on_path(cfg.repo_root)
    configure_process()
    return process_dataset(dataset_dir, split, cfg)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Subfamily-level Unified Raw MKR builder.")
    p.add_argument("--subfamily-root", required=True, type=str)
    p.add_argument("--output-root", required=True, type=str)
    p.add_argument("--config-root", type=str, default=None,
                   help="ClustOpt utilities_maps dir (default: repo standard).")
    p.add_argument("--split-assignment-csv", type=str, default=None,
                   help="dataset_split_assignments.csv (default: repo standard).")
    p.add_argument("--repo-root", type=str, default=None)
    p.add_argument("--n-jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--overwrite", action="store_true",
                   help="Recompute datasets even if already complete.")
    p.add_argument("--resume", action="store_true",
                   help="Skip datasets already marked complete (default behaviour).")
    p.add_argument("--metafeatures", choices=("auto", "on", "off"), default="auto")
    p.add_argument("--reuse-precomputed", choices=("auto", "off"), default="auto",
                   help="Reuse ClustOpt metrics from utility/<view>/clustopt_results.csv "
                        "for matching configs (default: auto/on).")
    p.add_argument("--ml2dac-mf-python", type=str, default=None,
                   help="Interpreter with pymfe for the ML2DAC meta-feature pass "
                        "(default: auto-detect .mfenv).")
    p.add_argument("--skip-ml2dac-mf-pass", action="store_true",
                   help="Do not auto-run the ML2DAC pymfe meta-feature pass.")
    p.add_argument("--limit-datasets", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    return p


def run(args: argparse.Namespace) -> int:
    configure_process()
    repo_root = Path(args.repo_root).resolve() if args.repo_root else io_utils.find_repo_root()
    io_utils.ensure_repo_on_path(repo_root)

    subfamily_root = Path(args.subfamily_root).resolve()
    config_root = Path(args.config_root).resolve() if args.config_root else (repo_root / DEFAULT_CONFIG_ROOT)
    split_csv = Path(args.split_assignment_csv).resolve() if args.split_assignment_csv else (repo_root / DEFAULT_SPLIT_CSV)

    family, subfamily = _derive_family_subfamily(subfamily_root)
    out_dir = Path(args.output_root).resolve() / family / subfamily
    storage = UnifiedMKRStorage(out_dir)

    log(f"[unified-mkr] subfamily      : {family}/{subfamily}")
    log(f"[unified-mkr] subfamily_root : {subfamily_root}")
    log(f"[unified-mkr] output_dir     : {out_dir}")
    log(f"[unified-mkr] config_root    : {config_root}")

    dataset_dirs = discover_dataset_dirs(subfamily_root)
    if args.limit_datasets:
        dataset_dirs = dataset_dirs[: args.limit_datasets]
    log(f"[unified-mkr] datasets found : {len(dataset_dirs)}")

    try:
        assignments = load_split_assignments(split_csv)
    except FileNotFoundError as exc:
        log(f"[unified-mkr] WARNING: {exc}. Proceeding with split_id=None.")
        assignments = {}

    created_at = datetime.now().isoformat(timespec="seconds")
    cfg = BuildConfig(
        out_dir=str(out_dir),
        config_root=str(config_root),
        repo_root=str(repo_root),
        created_at=created_at,
        family=family,
        subfamily=subfamily,
        metafeatures_mode=args.metafeatures,
        overwrite=args.overwrite,
        reuse_precomputed=(args.reuse_precomputed == "auto"),
    )

    if args.dry_run:
        log("[unified-mkr] dry-run: not executing. Sample configs/registry only.")
        configs = build_configurations_table(str(config_root))
        registry = build_metric_registry_table(str(config_root))
        log(f"  configurations rows: {len(configs)} | metric registry rows: {len(registry)}")
        log(f"  metrics by source: {registry['implementation_source'].value_counts().to_dict()}")
        return 0

    io_utils.ensure_dir(out_dir)
    io_utils.ensure_dir(storage.logs_dir)

    work = [
        (str(d), split_info_for(d.name, assignments), cfg)
        for d in dataset_dirs
    ]

    results: list[DatasetResult] = []
    t_start = time.perf_counter()

    def _track(idx: int, res: DatasetResult) -> None:
        elapsed = time.perf_counter() - t_start
        eta = (elapsed / idx) * (len(work) - idx) if idx else 0.0
        tag = "skip" if res.skipped else res.status
        log(
            f"[dataset {idx:>5}/{len(work)}] {res.dataset_id} "
            f"({res.runtime_sec:.1f}s, {tag}) "
            f"evals={res.n_evaluations} valid={res.n_valid} fail={res.n_failures} "
            f"reuse={res.metrics_reused}/{res.metrics_reused + res.metrics_computed} "
            f"mf=ml2dac:{res.ml2dac_mf_status}/autoclust:{res.autoclust_mf_status} "
            f"| elapsed={_fmt_hms(elapsed)} ETA={_fmt_hms(eta)}"
        )

    if args.n_jobs <= 1 or len(work) <= 1:
        for i, w in enumerate(work, start=1):
            results.append(_worker_entry(w))
            _track(i, results[-1])
    else:
        with ProcessPoolExecutor(max_workers=args.n_jobs) as pool:
            futures = {pool.submit(_worker_entry, w): i for i, w in enumerate(work, start=1)}
            done = 0
            for fut in as_completed(futures):
                done += 1
                try:
                    res = fut.result()
                except Exception as exc:  # noqa: BLE001
                    idx = futures[fut]
                    res = DatasetResult(
                        dataset_id=Path(work[idx - 1][0]).name,
                        status="fatal",
                        error=f"{type(exc).__name__}: {exc}"[:300],
                    )
                results.append(res)
                _track(done, res)

    # --- Aggregate + validate ---------------------------------------------
    log("[unified-mkr] aggregating subfamily tables ...")
    configurations = build_configurations_table(str(config_root))
    metric_registry = build_metric_registry_table(str(config_root))
    dataset_ids = [Path(d).name for d in dataset_dirs]

    build_config_payload = {
        "created_at": created_at,
        "family": family,
        "subfamily": subfamily,
        "subfamily_root": str(subfamily_root),
        "config_root": str(config_root),
        "split_assignment_csv": str(split_csv),
        "n_jobs": args.n_jobs,
        "overwrite": args.overwrite,
        "metafeatures_mode": args.metafeatures,
        "n_datasets": len(dataset_dirs),
        "n_datasets_fatal": sum(1 for r in results if r.status == "fatal"),
        "n_datasets_skipped": sum(1 for r in results if r.skipped),
    }
    counts = storage.aggregate(
        dataset_ids,
        configurations=configurations,
        metric_registry=metric_registry,
        build_config=build_config_payload,
    )
    log(f"[unified-mkr] aggregated rows: {counts}")

    reports = run_validation(
        storage,
        configurations=configurations,
        metric_registry=metric_registry,
        dataset_results=[r.__dict__ for r in results],
    )
    log("[unified-mkr] validation summary:")
    log(f"  configs_per_view_ok       : {reports['integrity'].get('configs_per_view_ok')}")
    log(f"  no_constrained_in_resolved: {reports['integrity'].get('no_constrained_in_resolved')}")
    log(f"  overlapping_metrics_once  : {reports['integrity'].get('overlapping_metrics_computed_once')}")
    log(f"  all_valid_have_labels     : {reports['integrity'].get('all_valid_have_labels')}")
    log(f"  all_records_32_configs    : {reports['coverage'].get('all_records_have_32_configs')}")
    log(f"  datasets_with_3_views     : {reports['coverage'].get('datasets_with_3_views')}/{reports['coverage'].get('n_datasets')}")
    log(f"  cache_hit_rate            : {reports['cache'].get('cache_hit_rate'):.3f}")
    log(f"  number_of_failures        : {reports['cache'].get('number_of_failures')}")

    # --- ML2DAC (pymfe) meta-feature pass (env-isolated, auto) ------------
    if not args.skip_ml2dac_mf_pass and args.metafeatures != "off":
        from experiments.external_baselines.Unified_MKR.unified_mkr.pymfe_pass import (
            find_mf_python,
            run_pymfe_pass,
        )

        mf_python = (
            Path(args.ml2dac_mf_python)
            if args.ml2dac_mf_python
            else find_mf_python(repo_root)
        )
        if mf_python and Path(mf_python).is_file():
            log(f"[unified-mkr] ML2DAC meta-feature pass via {mf_python}")
            try:
                res = run_pymfe_pass(out_dir, mf_python, overwrite=args.overwrite)
                log(f"[unified-mkr] ML2DAC meta-feature pass: {res}")
            except Exception as exc:  # noqa: BLE001
                log(f"[unified-mkr] WARNING: ML2DAC meta-feature pass failed: {exc}")
        else:
            log("[unified-mkr] NOTE: no pymfe interpreter (.mfenv) found; ML2DAC "
                "meta-features left unavailable. Create one with setup/create_mf_env "
                "or pass --ml2dac-mf-python.")

    log(f"[unified-mkr] DONE in {_fmt_hms(time.perf_counter() - t_start)} | output: {out_dir}")
    return 0


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
