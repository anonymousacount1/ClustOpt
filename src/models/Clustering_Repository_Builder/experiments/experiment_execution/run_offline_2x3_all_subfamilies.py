"""Stage-1C orchestrator: offline 2x3 ablation across all subfamilies.

Execution contract (Stage-1C, section 25-29):

    for subfamily in canonical_order:          # STRICTLY SERIAL
        run its Split-1 datasets with `workers` parallel processes
        wait for every worker
        validate the subfamily completion barrier
        finalize subfamily aggregation
        mark complete
        only then advance

The serial guarantee is structural, not conventional: this module contains a
plain ``for`` loop and each iteration calls
:func:`run_offline_subfamily.run`, which creates its ``ProcessPoolExecutor``
inside the call and exits the ``with`` block before returning. No executor,
future or task list ever spans two subfamilies -- there is no global pool
anywhere in the offline path.

This mirrors :mod:`run_phaseD_all_subfamilies`, the established repository
pattern for a serial subfamily sweep with a per-subfamily worker pool.

Run from the repo root::

    .venv_clustopt/Scripts/python.exe -m models.Clustering_Repository_Builder.experiments\
.experiment_execution.run_offline_2x3_all_subfamilies \
        --repo-root . --split-id 1 --workers 8 \
        --head14-model-run-dir results_analysis/mlp/<ts>__metric_utility_mlp_head14_split1_holdout
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from models.Clustering_Repository_Builder.utility_generation.worker_setup import (
    configure_process,
)

from .config import ExperimentExecutionConfig
from .offline_method_runner import EXECUTION_MODE
from .output_writer import ensure_dir, path_exists, read_json, write_json_atomic
from .run_offline_subfamily import (
    FULL60_MODEL, OFFLINE_SPLIT_DIR_SUFFIX, build_arms, metric_inventories,
    run as run_offline_subfamily, subfamily_is_complete,
)
from .summary_writer import summary_dir

configure_process()


def discover_subfamilies(analyzed_data: Path) -> List[Path]:
    """Canonical deterministic order: sorted(family) then sorted(subfamily).

    Identical to :func:`run_phaseD_all_subfamilies.discover_subfamilies`, so the
    offline sweep visits subfamilies in the same repository-native order as the
    historical online sweeps. Ordering cannot affect scientific results: each
    (dataset, view, arm) is scored independently from immutable stored data.
    """
    out: List[Path] = []
    for fam in sorted(analyzed_data.iterdir()):
        sub_root = fam / "subfamilies"
        if not sub_root.is_dir():
            continue
        for sub in sorted(sub_root.iterdir()):
            if sub.is_dir():
                out.append(sub)
    return out


ARM_ORDER = [
    ("HU", "stage1_head14_uniform_fixed50"),
    ("HP", "stage1_head14_mlp_top5_raw_fixed50"),
    ("HO", "stage1_head14_oracle_top5_raw_fixed50"),
    ("FU", "stage1_full60_uniform_fixed50"),
    ("FP", "stage1_full60_mlp_top5_raw_fixed50"),
    ("FO", "stage1_full60_oracle_top5_raw_fixed50"),
]


def _split1_dataset_dirs(cfg: ExperimentExecutionConfig) -> List[Path]:
    """The Split-1 dataset dirs of one subfamily (same path the runner uses)."""
    from .dataset_discovery import discover_datasets
    from .split_filter import filter_dataset_dirs_by_split, load_split_assignments
    valid, _ = discover_datasets(cfg.subfamily_dir, limit=None)
    dirs = filter_dataset_dirs_by_split(
        [d.dataset_dir for d in valid],
        load_split_assignments(cfg.split_assignments), cfg.split_id)
    if cfg.limit_datasets:
        dirs = dirs[: int(cfg.limit_datasets)]
    return dirs


def load_offline_records(cfg: ExperimentExecutionConfig, dataset_dirs=None):
    """Load every offline_result.json of one subfamily into a DataFrame.

    The per-run JSON -- not execution_summary.csv -- is the authoritative record
    source for Stage-1C: it carries the offline-specific fields (ari_regret,
    ceiling_hit, n_tied_at_max, objective_J, candidate fingerprint) that the
    shared online summary schema does not define. Reading them here keeps
    summary_writer (and therefore the online path) completely untouched.

    Columns are named so the frame can be fed straight to
    ``summary_writer._best_view_summary``.
    """
    import pandas as pd
    if dataset_dirs is None:
        dataset_dirs = _split1_dataset_dirs(cfg)
    split_dir = cfg.split_dir_name()
    rows = []
    for d in dataset_dirs:
        base = Path(d) / "experiments" / split_dir
        for _code, method in ARM_ORDER:
            for view in ("x_only", "y_only", "xy_2d"):
                payload = read_json(base / method / view / "offline_result.json")
                if not payload:
                    continue
                rows.append({
                    "dataset_id": payload.get("dataset_id"),
                    "family": payload.get("family"),
                    "subfamily": payload.get("subfamily"),
                    "difficulty": payload.get("difficulty"),
                    "view_id": payload.get("view_id"),
                    "method_name": payload.get("method_id"),
                    "arm": _code,
                    "status": payload.get("status"),
                    "metric_inventory": payload.get("metric_inventory"),
                    "utility_strategy": payload.get("utility_strategy"),
                    "deployable": payload.get("deployable"),
                    "predictor": payload.get("predictor"),
                    "ari": payload.get("ari"),
                    "observed_K": payload.get("observed_K"),
                    "true_K": payload.get("true_K"),
                    "exact_K": payload.get("exact_K"),
                    "objective_J": payload.get("objective_J"),
                    "max_objective_J": payload.get("max_objective_J"),
                    "n_tied_at_max": payload.get("n_tied_at_max"),
                    "tie_flag": payload.get("tie_flag"),
                    "best_possible_ari_in_candidate_set":
                        payload.get("best_possible_ari_in_candidate_set"),
                    "ari_regret": payload.get("ari_regret"),
                    "ceiling_hit": payload.get("ceiling_hit"),
                    "tie_ari_min": payload.get("tie_ari_min"),
                    "tie_ari_mean": payload.get("tie_ari_mean"),
                    "tie_ari_max": payload.get("tie_ari_max"),
                    "candidate_count": payload.get("candidate_count"),
                    "valid_candidate_count": payload.get("valid_candidate_count"),
                    "candidate_set_fingerprint": payload.get("candidate_set_fingerprint"),
                    "selected_metrics": "|".join(payload.get("selected_metrics") or []),
                    "selected_algorithm": payload.get("selected_algorithm"),
                    "execution_mode": payload.get("execution_mode"),
                })
    return pd.DataFrame(rows)


def _subfamily_frames(cfg: ExperimentExecutionConfig):
    """(records_df, best_view_df) for one completed subfamily, or (None, None)."""
    from .summary_writer import _best_view_summary
    df = load_offline_records(cfg)
    if df is None or df.empty:
        return None, None
    return df, _best_view_summary(df)


def _arm_stats(df, bv):
    """Per-arm descriptive numbers for the interim progress report."""
    import numpy as np
    import pandas as pd
    out = {}
    if df is None or df.empty:
        return out
    ok = df[df["status"] == "success"].copy()
    for col in ("ari", "ari_regret", "objective_J", "n_tied_at_max", "exact_K"):
        if col in ok.columns:
            ok[col] = pd.to_numeric(ok[col], errors="coerce")
    for code, method in ARM_ORDER:
        g = ok[ok["method_name"] == method]
        if g.empty:
            continue
        b = bv[bv["method_name"] == method] if bv is not None and not bv.empty else None
        # Best-View ARI per dataset via the SAME first-wins idxmax convention.
        bvals = g.loc[g.groupby("dataset_id")["ari"].idxmax(), "ari"]
        out[code] = {
            "n_datasets": int(g["dataset_id"].nunique()),
            "bv_mean_ari": float(bvals.mean()),
            "bv_median_ari": float(bvals.median()),
            "mean_regret": (float(g["ari_regret"].mean())
                            if "ari_regret" in g and g["ari_regret"].notna().any() else None),
            "exact_k_rate": (float(g["exact_K"].mean())
                             if "exact_K" in g and g["exact_K"].notna().any() else None),
            "tie_rate": (float((g["n_tied_at_max"] > 1).mean())
                         if "n_tied_at_max" in g and g["n_tied_at_max"].notna().any() else None),
            "_bv_sum": float(bvals.sum()), "_bv_n": int(bvals.size),
            "_regret_sum": (float(g["ari_regret"].sum())
                            if "ari_regret" in g and g["ari_regret"].notna().any() else 0.0),
            "_regret_n": (int(g["ari_regret"].notna().sum())
                          if "ari_regret" in g else 0),
            "_ek_sum": (float(g["exact_K"].sum())
                        if "exact_K" in g and g["exact_K"].notna().any() else 0.0),
            "_ek_n": (int(g["exact_K"].notna().sum()) if "exact_K" in g else 0),
        }
    return out


def _fmt(v, nd=4):
    return "  n/a " if v is None else f"{v:.{nd}f}"


def _emit_progress_report(*, repo, cfg, index, total, family, subfamily, arms,
                          status, skipped, cumulative, subfamilies, progress, t0):
    """Print the interim per-subfamily report and update the cumulative table."""
    df, bv = _subfamily_frames(cfg)
    stats = _arm_stats(df, bv)

    # accumulate
    for code, s in stats.items():
        c = cumulative.setdefault(code, {"bv_sum": 0.0, "bv_n": 0, "reg_sum": 0.0,
                                         "reg_n": 0, "ek_sum": 0.0, "ek_n": 0})
        c["bv_sum"] += s["_bv_sum"]; c["bv_n"] += s["_bv_n"]
        c["reg_sum"] += s["_regret_sum"]; c["reg_n"] += s["_regret_n"]
        c["ek_sum"] += s["_ek_sum"]; c["ek_n"] += s["_ek_n"]

    n_done = sum(1 for v in progress["completed"].values() if v)
    ds_done = cumulative.get("HU", {}).get("bv_n", 0)
    nxt = None
    for s in subfamilies[index:]:
        nxt = f"{s.parent.parent.name}/{s.name}"
        break

    L = []
    L.append("=" * 64)
    L.append(f"STAGE 1C PROGRESS - SUBFAMILY {index} / {total}")
    L.append(f"Family:              {family}")
    L.append(f"Subfamily:           {subfamily}")
    if skipped:
        L.append("Action:              SKIPPED (already complete, barrier revalidated)")
    if status:
        L.append(f"Datasets:            {status.get('datasets')}")
        L.append(f"Dataset/view records:{status.get('datasets', 0) * 3}")
        L.append(f"Arm-records:         {status.get('records')}")
        L.append(f"Failures:            {status.get('n_failed')}")
        L.append(f"Runtime:             {status.get('runtime_sec')}s")
        L.append(f"Completion barrier:  {'PASS' if status.get('status') == 'complete' else 'FAIL'}")
    if stats:
        for label, key, nd in (("Best-View mean ARI", "bv_mean_ari", 4),
                               ("Best-View median ARI", "bv_median_ari", 4),
                               ("Mean candidate regret", "mean_regret", 4),
                               ("Exact-K rate", "exact_k_rate", 4),
                               ("Tie rate", "tie_rate", 4)):
            L.append("")
            L.append(f"{label}:")
            L.append("  " + "  ".join(
                f"{c}:{_fmt(stats.get(c, {}).get(key), nd)}" for c, _m in ARM_ORDER))
        g = lambda c: stats.get(c, {}).get("bv_mean_ari")  # noqa: E731
        def d(a, b):
            x, y = g(a), g(b)
            return "  n/a " if (x is None or y is None) else f"{x - y:+.4f}"
        L.append("")
        L.append("Simple local deltas (Best-View mean ARI):")
        L.append(f"  HP-HU:{d('HP','HU')}  HO-HU:{d('HO','HU')}  FU-HU:{d('FU','HU')}  "
                 f"FP-HP:{d('FP','HP')}  FO-HO:{d('FO','HO')}")
    L.append("")
    L.append("Cumulative (descriptive, all completed datasets so far):")
    L.append(f"  subfamilies {n_done}/{total}   datasets {ds_done}/1055   "
             f"elapsed {(time.time() - t0) / 60:.1f} min")
    if cumulative:
        L.append("  arm      N     meanBV-ARI   meanRegret   exactK")
        for code, _m in ARM_ORDER:
            c = cumulative.get(code)
            if not c or not c["bv_n"]:
                continue
            L.append(f"  {code:6} {c['bv_n']:5d}   {c['bv_sum']/c['bv_n']:10.4f}   "
                     f"{(c['reg_sum']/c['reg_n']) if c['reg_n'] else float('nan'):10.4f}   "
                     f"{(c['ek_sum']/c['ek_n']) if c['ek_n'] else float('nan'):6.4f}")
    L.append("")
    L.append(f"Next subfamily:      {nxt or '(none - sweep complete)'}")
    L.append("=" * 64)
    print("\n".join(L), flush=True)


def _progress_path(repo: Path, split_dir: str) -> Path:
    return (repo / "results_analysis" / "clustering_repository"
            / "stage1_2x3_aggregation" / EXECUTION_MODE / f"{split_dir}_progress.json")


def _git_commit(repo: Path) -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                              capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception:
        return "<unavailable>"


def build_manifest(*, repo: Path, args, subfamilies: List[Path], arms,
                   head14: List[str], full60: List[str]) -> Dict[str, Any]:
    import hashlib

    def sha(p: Path) -> str:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                h.update(chunk)
        return h.hexdigest()

    analyzed = repo / "results_analysis" / "clustering_repository" / "analyzed_data"
    split_csv = analyzed / "experiment_splits" / "dataset_split_assignments.csv"
    families = sorted({s.parent.parent.name for s in subfamilies})

    def model_hashes(rel: Optional[str]) -> Dict[str, str]:
        if not rel:
            return {}
        base = repo / rel
        out = {}
        for f in ["folds/fold_0/best_model.pt", "folds/fold_0/x_scaler.pkl",
                  "artifacts/target_columns.json"]:
            p = base / f
            if p.is_file():
                out[f] = sha(p)
        return out

    return {
        "execution_mode": EXECUTION_MODE,
        "stage": "Stage-1C offline fixed-candidate 2x3 ablation",
        "source_commit": _git_commit(repo),
        "environment_lockfile": "environment/stage1_clustopt_requirements.lock.txt",
        "python": sys.version.split()[0],
        "workers": int(args.workers),
        "split_id": int(args.split_id),
        "split_dir": f"split_{int(args.split_id):02d}_{args.split_dir_suffix}",
        "n_families": len(families),
        "n_subfamilies": len(subfamilies),
        "families": families,
        "subfamily_execution_order": [f"{s.parent.parent.name}/{s.name}"
                                      for s in subfamilies],
        "split_assignments_sha256": sha(split_csv) if split_csv.is_file() else None,
        "head14_model_run_dir": args.head14_model_run_dir,
        "head14_model_hashes": model_hashes(args.head14_model_run_dir),
        "full60_model_run_dir": args.full60_model_run_dir,
        "full60_model_hashes": model_hashes(args.full60_model_run_dir),
        "metric_partition": {"head14": list(head14), "n_head14": len(head14),
                             "n_full60": len(full60)},
        "arms": [{"method_id": a.method_id, "inventory": a.inventory,
                  "strategy": a.strategy, "deployable": a.deployable,
                  "model_run_dir": a.model_run_dir} for a in arms],
        "candidate_source": "<dataset>/utility/<view>/clustopt_results.csv",
        "candidate_contract": {"candidates_per_record": 32,
                               "note": "_fixed50 is the POLICY identity / future "
                                       "online protocol; Stage-1C runs no Optuna."},
        "implementation": {
            "objective": "experiment_execution/offline_candidate_execution.py "
                         "(weights from models/ClustOpt production resolvers)",
            "execution": "experiment_execution/run_offline_subfamily.py",
            "worker": "experiment_execution/offline_dataset_worker.py "
                      "(one dataset per worker)",
            "persistence": "experiment_execution/offline_method_runner.py",
            "aggregation": "experiment_execution/summary_writer.py "
                           "(SAME module as the online runs)",
            "best_view": "summary_writer._best_view_summary "
                         "(groupby(method_name, dataset_id).ari.idxmax on "
                         "status=='success')",
            "statistics": "Clustering_Repository_Builder/experiments/paper_analysis/"
                          "stats.py (paired bootstrap + Wilcoxon + Holm)",
        },
    }


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", required=True)
    p.add_argument("--split-id", type=int, default=1)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--split-dir-suffix", default=OFFLINE_SPLIT_DIR_SUFFIX)
    p.add_argument("--head14-model-run-dir", required=True)
    p.add_argument("--full60-model-run-dir", default=FULL60_MODEL)
    p.add_argument("--limit-subfamilies", type=int, default=None,
                   help="SMOKE ONLY: process the first N subfamilies.")
    p.add_argument("--limit-datasets", type=int, default=None,
                   help="SMOKE ONLY: cap datasets per subfamily.")
    p.add_argument("--only-subfamily", default=None,
                   help="SMOKE ONLY: restrict to one '<family>/<subfamily>'.")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--stop-on-incomplete", action="store_true",
                   help="Abort the sweep if a subfamily fails its completion "
                        "barrier instead of recording and continuing.")
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

    arms = build_arms(args.head14_model_run_dir, args.full60_model_run_dir)
    head14, full60 = metric_inventories()

    split_dir = f"split_{int(args.split_id):02d}_{args.split_dir_suffix}"
    out_root = ensure_dir(repo / "results_analysis" / "clustering_repository"
                          / "stage1_2x3_aggregation" / EXECUTION_MODE)
    manifest = build_manifest(repo=repo, args=args, subfamilies=subfamilies,
                              arms=arms, head14=head14, full60=full60)
    write_json_atomic(out_root / "experiment_manifest.json", manifest)

    progress_path = _progress_path(repo, split_dir)
    progress: Dict[str, Any] = read_json(progress_path) or {"completed": {}}

    print(f"[stage1c] execution_mode={EXECUTION_MODE} | {len(subfamilies)} subfamilies "
          f"| split={args.split_id} | 6 arms x 3 views | workers={args.workers}",
          flush=True)
    print(f"[stage1c] SERIAL by subfamily; {args.workers} workers INSIDE each",
          flush=True)

    t0 = time.time()
    cumulative: Dict[str, Any] = {}
    n_complete = n_incomplete = 0
    for i, sub in enumerate(subfamilies, 1):
        family = sub.parent.parent.name
        key = f"{family}/{sub.name}"
        cfg = ExperimentExecutionConfig(
            repo_root=repo, subfamily_dir=sub, split_assignments=split_csv,
            split_id=int(args.split_id), split_dir_suffix=str(args.split_dir_suffix),
            method_set="stage1_2x3", max_workers=int(args.workers),
            resume=not args.overwrite, overwrite=bool(args.overwrite),
            limit_datasets=args.limit_datasets,
        )

        # Resume hardening: the progress JSON is a HINT, never the source of
        # truth. A subfamily is skipped only if the on-disk completion barrier
        # still passes right now. If outputs were deleted or corrupted after the
        # flag was written, the stale flag is dropped and the subfamily is
        # reprocessed through the normal resume path (which itself skips only
        # individually-complete runs, so nothing valid is recomputed).
        if not args.overwrite and progress["completed"].get(key):
            ds_dirs = _split1_dataset_dirs(cfg)
            still_ok, detail = subfamily_is_complete(cfg, ds_dirs, arms)
            if still_ok:
                print(f"[{i}/{len(subfamilies)}] {key} -- complete (barrier "
                      f"revalidated on disk: {detail['expected_runs']} runs), skipping",
                      flush=True)
                n_complete += 1
                _emit_progress_report(
                    repo=repo, cfg=cfg, index=i, total=len(subfamilies),
                    family=family, subfamily=sub.name, arms=arms,
                    status=None, skipped=True, cumulative=cumulative,
                    subfamilies=subfamilies, progress=progress, t0=t0)
                continue
            print(f"[{i}/{len(subfamilies)}] {key} -- progress said complete but the "
                  f"on-disk barrier FAILED (missing_runs={detail['missing_runs']}); "
                  f"stale flag cleared, recovering", flush=True)
            progress["completed"][key] = False
            progress.setdefault("recovered", []).append(
                {"subfamily": key, "missing_runs": detail["missing_runs"]})

        print(f"\n[{i}/{len(subfamilies)}] {key} ...", flush=True)
        ts = time.perf_counter()
        try:
            status = run_offline_subfamily(cfg, arms=arms, head14=head14, full60=full60)
        except Exception as exc:                       # infrastructure failure
            print(f"  [ERROR] {type(exc).__name__}: {exc}", flush=True)
            n_incomplete += 1
            if args.stop_on_incomplete:
                return 2
            continue

        ok = status["status"] in ("complete", "empty")
        print(f"  status={status['status']} datasets={status['datasets']} "
              f"records={status['records']} success={status.get('n_success')} "
              f"skipped={status.get('n_skipped')} failed={status.get('n_failed')} "
              f"runtime={time.perf_counter()-ts:.1f}s", flush=True)
        if not ok:
            c = status.get("completion", {})
            print(f"  [INCOMPLETE] missing_runs={c.get('missing_runs')} "
                  f"e.g. {c.get('missing_examples')}", flush=True)

        # HARD BARRIER: only a complete subfamily is recorded and advanced past.
        progress["completed"][key] = bool(ok)
        progress["last_subfamily"] = key
        progress.setdefault("subfamilies", {})[key] = {
            "canonical_index": i,
            "family": family,
            "subfamily": sub.name,
            "expected_datasets": status.get("datasets"),
            "completed_datasets": status.get("datasets") if ok else None,
            "failed_records": status.get("n_failed"),
            "records": status.get("records"),
            "completion_barrier": "PASS" if ok else "FAIL",
            "aggregation_complete": bool(status.get("completion", {})
                                         .get("summaries_present")),
            "runtime_sec": status.get("runtime_sec"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        progress["cumulative_completed_datasets"] = sum(
            (v.get("completed_datasets") or 0)
            for v in progress.get("subfamilies", {}).values())
        ensure_dir(progress_path.parent)
        write_json_atomic(progress_path, progress)

        # Interim descriptive report (NOT a scientific conclusion).
        _emit_progress_report(
            repo=repo, cfg=cfg, index=i, total=len(subfamilies), family=family,
            subfamily=sub.name, arms=arms, status=status, skipped=False,
            cumulative=cumulative, subfamilies=subfamilies, progress=progress, t0=t0)

        n_complete += ok
        n_incomplete += (not ok)
        if not ok and args.stop_on_incomplete:
            print("  [STOP] --stop-on-incomplete set; aborting sweep.", flush=True)
            return 2

    print(f"\n[stage1c] {n_complete} complete / {n_incomplete} incomplete "
          f"in {(time.time()-t0)/60:.1f} min", flush=True)
    print(f"[stage1c] manifest -> {out_root / 'experiment_manifest.json'}", flush=True)
    return 0 if n_incomplete == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
