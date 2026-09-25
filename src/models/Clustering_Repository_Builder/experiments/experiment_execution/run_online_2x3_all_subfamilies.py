"""Stage-1E orchestrator: ONLINE fixed-50 2x3 sweep across all subfamilies.

Unlike Stage-1C (which scored a frozen 32-candidate set), this runs the REAL
ClustOpt pipeline: each of the six policies steers its own 50-trial Optuna
search over the identical clustering search space.

Execution contract:

    for subfamily in canonical_order:          # STRICTLY SERIAL
        revalidate the on-disk completion barrier
        if complete: skip
        else: run this subfamily's Split-1 datasets with `workers` processes
              wait for all workers
              build dataset + subfamily summaries
              validate the barrier
              persist progress and print the update
              only then advance

Execution is delegated to :func:`run_subfamily_experiments.run` -- the existing,
unmodified online runner -- so dataset discovery, the worker pool, method_runner,
ClustOpt/Optuna execution, persistence, resume, summary_writer and Best-View all
come from the established framework. This module only sequences subfamilies,
enforces the hardened resume, and reports progress; it mirrors
``run_phaseD_all_subfamilies`` and its offline sibling
``run_offline_2x3_all_subfamilies``.

Isolation: results land in ``experiments/split_01_online_fixed50/`` via the
existing ``split_dir_suffix`` mechanism, so historical ``split_01`` and the
Stage-1C ``split_01_offline_fixed32`` trees are untouched and cannot be confused
with these.

Run from the repo root::

    .venv_clustopt/Scripts/python.exe -m models.Clustering_Repository_Builder\
.experiments.experiment_execution.run_online_2x3_all_subfamilies \
        --repo-root . --split-id 1 --workers 8
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from models.Clustering_Repository_Builder.utility_generation.worker_setup import (
    configure_process,
)

from .config import ExperimentExecutionConfig, METHOD_NAMES_STAGE1_2X3, VIEW_IDS
from .method_runner import is_run_complete, run_output_dir
from .output_writer import ensure_dir, path_exists, read_json, write_json_atomic
from .run_offline_2x3_all_subfamilies import (
    ARM_ORDER, _fmt, _split1_dataset_dirs, discover_subfamilies,
)
from .run_subfamily_experiments import run as run_subfamily_online
from .summary_writer import summary_dir

configure_process()

EXECUTION_MODE = "online_fixed50"
ONLINE_SPLIT_DIR_SUFFIX = "online_fixed50"

# Files method_runner writes for a successful ClustOpt run.
ONLINE_REQUIRED_FILES = (
    "clustopt_results.csv", "best_result.json", "external_metrics.json",
    "selected_metrics.json", "timing_summary.json", "run_log.json",
    "status.json", "config_snapshot.json",
)


def _thread_provenance():
    """Effective thread configuration of THIS process, recorded in the manifest."""
    try:
        from models.Clustering_Repository_Builder.utility_generation.worker_setup import (
            thread_configuration,
        )
        return thread_configuration()
    except Exception as exc:                                   # pragma: no cover
        return f"<unavailable: {exc}>"


def online_run_is_complete(dataset_dir: Path, split_dir: str, method: str,
                           view: str) -> bool:
    """Complete iff status==success AND every expected output file exists.

    Long-path safe (paths reach ~300 chars and this machine has
    LongPathsEnabled=0), so ``path_exists`` is used, never ``Path.is_file()``.
    """
    out = run_output_dir(dataset_dir, split_dir, method, view)
    if not is_run_complete(out):
        return False
    return all(path_exists(out / f) for f in ONLINE_REQUIRED_FILES)


def subfamily_is_complete(cfg: ExperimentExecutionConfig, dataset_dirs,
                          methods) -> tuple[bool, Dict[str, Any]]:
    """HARD completion barrier for one online subfamily."""
    split_dir = cfg.split_dir_name()
    missing: List[str] = []
    for d in dataset_dirs:
        for view in VIEW_IDS:
            for m in methods:
                if not online_run_is_complete(Path(d), split_dir, m, view):
                    missing.append(f"{Path(d).name}/{view}/{m}")
    sd = summary_dir(cfg)
    summaries_ok = path_exists(sd / "execution_summary.csv")
    return (not missing and summaries_ok), {
        "expected_runs": len(dataset_dirs) * len(VIEW_IDS) * len(methods),
        "missing_runs": len(missing), "missing_examples": missing[:5],
        "summaries_present": summaries_ok, "datasets": len(dataset_dirs),
    }


def _load_online_records(cfg: ExperimentExecutionConfig, dataset_dirs):
    """Per-run online results for the interim report (from best_result/status)."""
    import pandas as pd
    split_dir = cfg.split_dir_name()
    rows = []
    for d in dataset_dirs:
        for m in METHOD_NAMES_STAGE1_2X3:
            for v in VIEW_IDS:
                out = run_output_dir(Path(d), split_dir, m, v)
                st = read_json(out / "status.json") or {}
                br = read_json(out / "best_result.json") or {}
                if not st:
                    continue
                rows.append({
                    "dataset_id": Path(d).name, "view_id": v, "method_name": m,
                    "status": st.get("status"), "ari": st.get("ari"),
                    "selected_k": st.get("selected_k"), "true_k": st.get("true_k"),
                    "runtime_sec": st.get("runtime_sec"),
                    "requested_trials": br.get("requested_trials"),
                    "delivered_trials": br.get("delivered_trials",
                                               br.get("n_trials_completed")),
                    "valid_trials": br.get("valid_trials"),
                    "failed_trials": br.get("failed_trials"),
                    "search_seed": br.get("search_seed"),
                    "k_correct": br.get("k_correct"),
                })
    return pd.DataFrame(rows)


def _emit(cfg, index, total, family, subfamily, dataset_dirs, status, skipped,
          cumulative, subfamilies, progress, t0):
    import pandas as pd
    df = _load_online_records(cfg, dataset_dirs)
    stats = {}
    if not df.empty:
        ok = df[df["status"] == "success"].copy()
        for c in ("ari", "valid_trials", "failed_trials", "delivered_trials"):
            ok[c] = pd.to_numeric(ok[c], errors="coerce")
        ok["exact_k"] = (pd.to_numeric(ok["selected_k"], errors="coerce")
                         == pd.to_numeric(ok["true_k"], errors="coerce")).astype(float)
        for code, method in ARM_ORDER:
            g = ok[ok["method_name"] == method]
            if g.empty:
                continue
            bv = g.loc[g.groupby("dataset_id")["ari"].idxmax(), "ari"]
            stats[code] = {
                "bv_mean": float(bv.mean()), "bv_median": float(bv.median()),
                "valid": float(g["valid_trials"].mean()),
                "failed": float(g["failed_trials"].mean()),
                "exact_k": float(g["exact_k"].mean()),
                "_sum": float(bv.sum()), "_n": int(bv.size),
                "_vsum": float(g["valid_trials"].sum()), "_vn": int(g["valid_trials"].notna().sum()),
                "_fsum": float(g["failed_trials"].sum()),
                "_eksum": float(g["exact_k"].sum()), "_ekn": int(g["exact_k"].size),
            }
    for code, s in stats.items():
        c = cumulative.setdefault(code, {"s": 0.0, "n": 0, "vs": 0.0, "vn": 0,
                                         "fs": 0.0, "eks": 0.0, "ekn": 0})
        c["s"] += s["_sum"]; c["n"] += s["_n"]
        c["vs"] += s["_vsum"]; c["vn"] += s["_vn"]; c["fs"] += s["_fsum"]
        c["eks"] += s["_eksum"]; c["ekn"] += s["_ekn"]

    n_done = sum(1 for v in progress["completed"].values() if v)
    nxt = f"{subfamilies[index].parent.parent.name}/{subfamilies[index].name}" \
        if index < len(subfamilies) else "(none - sweep complete)"

    L = ["=" * 64, f"STAGE 1E ONLINE PROGRESS - SUBFAMILY {index} / {total}", "",
         f"Family:              {family}", f"Subfamily:           {subfamily}", ""]
    if skipped:
        L.append("Action:              SKIPPED (complete, barrier revalidated)")
    if status:
        L += [f"Datasets:            {status.get('datasets', len(dataset_dirs))}",
              f"Expected online runs:{len(dataset_dirs) * 3 * 6}",
              f"Completed:           {status.get('n_success', '?')}",
              f"Failures:            {status.get('n_failed', '?')}",
              f"Runtime:             {status.get('runtime_sec', '?')}s",
              f"Completion barrier:  {status.get('barrier', '?')}"]
    for label, key in (("Mean Best-View ARI", "bv_mean"),
                       ("Median Best-View ARI", "bv_median"),
                       ("Mean valid trials", "valid"),
                       ("Mean failed trials", "failed"),
                       ("Exact-K", "exact_k")):
        L += ["", f"{label}:", "  " + "  ".join(
            f"{c}:{_fmt(stats.get(c, {}).get(key))}" for c, _ in ARM_ORDER)]
    g = lambda c: stats.get(c, {}).get("bv_mean")  # noqa: E731

    def d(a, b):
        x, y = g(a), g(b)
        return "  n/a " if (x is None or y is None) else f"{x - y:+.4f}"
    L += ["", "Simple local deltas:",
          f"  HP-HU:{d('HP','HU')}  HO-HU:{d('HO','HU')}  FU-HU:{d('FU','HU')}  "
          f"FP-HP:{d('FP','HP')}  FO-HO:{d('FO','HO')}"]
    tot_ds = sum(v.get("completed_datasets") or 0
                 for v in progress.get("subfamilies", {}).values())
    L += ["", "Cumulative:",
          f"  subfamilies {n_done}/{total}   datasets {tot_ds}/1055   "
          f"elapsed {(time.time() - t0) / 60:.1f} min"]
    if cumulative:
        L.append("  arm      N     meanBV-ARI   meanValid   meanFailed   exactK")
        for code, _ in ARM_ORDER:
            c = cumulative.get(code)
            if not c or not c["n"]:
                continue
            L.append(f"  {code:6} {c['n']:5d}   {c['s']/c['n']:10.4f}   "
                     f"{(c['vs']/c['vn']) if c['vn'] else float('nan'):9.2f}   "
                     f"{(c['fs']/c['vn']) if c['vn'] else float('nan'):10.2f}   "
                     f"{(c['eks']/c['ekn']) if c['ekn'] else float('nan'):6.4f}")
    L += ["", f"Next subfamily:      {nxt}", "=" * 64]
    print("\n".join(L), flush=True)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", required=True)
    p.add_argument("--split-id", type=int, default=1)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--split-dir-suffix", default=ONLINE_SPLIT_DIR_SUFFIX)
    p.add_argument("--limit-subfamilies", type=int, default=None)
    p.add_argument("--limit-datasets", type=int, default=None)
    p.add_argument("--only-subfamily", default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--stop-on-incomplete", action="store_true")
    args = p.parse_args(argv)

    repo = Path(args.repo_root).resolve()
    analyzed = repo / "results_analysis" / "clustering_repository" / "analyzed_data"
    split_csv = analyzed / "experiment_splits" / "dataset_split_assignments.csv"

    subfamilies = discover_subfamilies(analyzed)
    if args.only_subfamily:
        want = args.only_subfamily.replace("\\", "/")
        subfamilies = [s for s in subfamilies
                       if f"{s.parent.parent.name}/{s.name}" == want]
    if args.limit_subfamilies:
        subfamilies = subfamilies[: int(args.limit_subfamilies)]

    methods = list(METHOD_NAMES_STAGE1_2X3)
    split_dir = f"split_{int(args.split_id):02d}_{args.split_dir_suffix}"
    out_root = ensure_dir(repo / "results_analysis" / "clustering_repository"
                          / "stage1_2x3_aggregation" / EXECUTION_MODE)
    progress_path = out_root / f"{split_dir}_progress.json"
    progress: Dict[str, Any] = read_json(progress_path) or {"completed": {}}

    write_json_atomic(out_root / "experiment_manifest.json", {
        "execution_mode": EXECUTION_MODE,
        "stage": "Stage-1E online fixed-50 2x3",
        "source_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                                        capture_output=True, text=True).stdout.strip(),
        "environment_lockfile": "environment/stage1_clustopt_requirements.lock.txt",
        "workers": int(args.workers), "split_id": int(args.split_id),
        "split_dir": split_dir, "methods": methods,
        "n_subfamilies": len(subfamilies),
        "subfamily_execution_order": [f"{s.parent.parent.name}/{s.name}"
                                      for s in subfamilies],
        "search_contract": {"n_trials": 50, "timeout": None, "patience": None,
                            "seed": "auto", "base_seed": 42},
        "resource_protocol": {
            "worker_processes": int(args.workers),
            "threads_per_worker": 1,
            "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
            "BLIS_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
            "note": ("Corrected after an aborted uncapped attempt caused ~64 "
                     "threads on 8 cores (~10x CPU starvation). Speed-only "
                     "change: seeded sampler, pinned clustering random_state, "
                     "no timeout/patience => results are time-independent."),
            "aborted_attempt_archived_as":
                "split_01_online_fixed50_ABORTED_OVERSUBSCRIPTION_ATTEMPT",
            "thread_configuration": _thread_provenance(),
        },
        "execution_source": "run_subfamily_experiments.run (unmodified online runner)",
        "best_view_source": "summary_writer._best_view_summary",
        "statistics_source": "paper_analysis/stats.py",
        "started_at": datetime.now(timezone.utc).isoformat(),
    })

    print(f"[stage1e] execution_mode={EXECUTION_MODE} | {len(subfamilies)} subfamilies "
          f"| split={args.split_id} | 6 arms x 3 views x 50 trials | "
          f"workers={args.workers}", flush=True)
    print("[stage1e] SERIAL by subfamily; workers INSIDE each", flush=True)

    t0 = time.time()
    cumulative: Dict[str, Any] = {}
    n_ok = n_bad = 0
    for i, sub in enumerate(subfamilies, 1):
        family = sub.parent.parent.name
        key = f"{family}/{sub.name}"
        cfg = ExperimentExecutionConfig(
            repo_root=repo, subfamily_dir=sub, split_assignments=split_csv,
            split_id=int(args.split_id), split_dir_suffix=str(args.split_dir_suffix),
            method_set="stage1_2x3", methods=tuple(methods),
            max_workers=int(args.workers), resume=not args.overwrite,
            overwrite=bool(args.overwrite), limit_datasets=args.limit_datasets,
        )
        dirs = _split1_dataset_dirs(cfg)

        # Hardened resume: the progress flag is a hint; the barrier decides.
        if not args.overwrite and progress["completed"].get(key):
            still_ok, det = subfamily_is_complete(cfg, dirs, methods)
            if still_ok:
                print(f"[{i}/{len(subfamilies)}] {key} -- complete (barrier "
                      f"revalidated: {det['expected_runs']} runs), skipping", flush=True)
                n_ok += 1
                _emit(cfg, i, len(subfamilies), family, sub.name, dirs, None, True,
                      cumulative, subfamilies, progress, t0)
                continue
            print(f"[{i}/{len(subfamilies)}] {key} -- stale complete flag "
                  f"(missing_runs={det['missing_runs']}); recovering", flush=True)
            progress["completed"][key] = False

        print(f"\n[{i}/{len(subfamilies)}] {key} -- {len(dirs)} datasets x 3 views "
              f"x 6 arms = {len(dirs)*18} runs ...", flush=True)
        ts = time.time()
        try:
            rc = run_subfamily_online(cfg)
        except Exception as exc:
            print(f"  [ERROR] {type(exc).__name__}: {exc}", flush=True)
            n_bad += 1
            if args.stop_on_incomplete:
                return 2
            continue

        ok, det = subfamily_is_complete(cfg, dirs, methods)
        elapsed = time.time() - ts
        status = {"datasets": len(dirs), "runtime_sec": round(elapsed, 1),
                  "barrier": "PASS" if ok else "FAIL",
                  "n_success": det["expected_runs"] - det["missing_runs"],
                  "n_failed": det["missing_runs"], "rc": rc}
        if not ok:
            print(f"  [INCOMPLETE] missing_runs={det['missing_runs']} "
                  f"e.g. {det['missing_examples']}", flush=True)

        progress["completed"][key] = bool(ok)
        progress.setdefault("subfamilies", {})[key] = {
            "canonical_index": i, "family": family, "subfamily": sub.name,
            "expected_datasets": len(dirs),
            "completed_datasets": len(dirs) if ok else None,
            "expected_runs": det["expected_runs"], "missing_runs": det["missing_runs"],
            "completion_barrier": status["barrier"], "runtime_sec": status["runtime_sec"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        write_json_atomic(progress_path, progress)
        n_ok += ok; n_bad += (not ok)
        _emit(cfg, i, len(subfamilies), family, sub.name, dirs, status, False,
              cumulative, subfamilies, progress, t0)
        if not ok and args.stop_on_incomplete:
            return 2

    print(f"\n[stage1e] {n_ok} complete / {n_bad} incomplete in "
          f"{(time.time()-t0)/60:.1f} min", flush=True)
    return 0 if n_bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
