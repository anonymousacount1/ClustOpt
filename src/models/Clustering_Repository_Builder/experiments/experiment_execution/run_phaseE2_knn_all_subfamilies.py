"""Phase-E2 orchestrator: run the 12 KNN methods across all 86 subfamilies.

A *thin* wrapper over the existing per-subfamily runner (``run_subfamily``),
identical in spirit to the Phase-D orchestrator but selecting the ``phaseE2_knn``
method set and writing to an isolated output suffix. It runs subfamilies
**sequentially** (each with its own 8-worker dataset pool), streams a per-
subfamily provisional report, and persists an overall progress file after every
subfamily so the sweep can resume after interruption.

The underlying pipeline is unchanged; the frozen Phase-E1 KNN is loaded read-only
by the ``knn_dynamic`` CVI. Resume is automatic (per-run ``status.json`` skip).

Run from the repo root::

    python -m models.Clustering_Repository_Builder.experiments.experiment_execution.run_phaseE2_knn_all_subfamilies \
        --repo-root . --split-id 1 --max-workers 8
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import pandas as pd

from models.Clustering_Repository_Builder.utility_generation.worker_setup import (
    configure_process,
)

from .config import ExperimentExecutionConfig, METHODS_PHASE_E2_KNN
from .run_phaseD_all_subfamilies import discover_subfamilies
from .run_subfamily_experiments import run as run_subfamily
from .summary_writer import summary_dir

configure_process()

PHASE_E2_SUFFIX = "phaseE2_knn_50trials_no_earlystop"
PHASE_E2_METHODS = tuple(m.name for m in METHODS_PHASE_E2_KNN)


def _progress_dir(repo: Path) -> Path:
    return repo / "results_analysis" / "metric_utility_knn" / "phase_e2" / "progress"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _long(p: Path) -> str:
    """Extended-length path so pandas can read past Windows MAX_PATH (260)."""
    import os
    ap = os.path.abspath(str(p))
    if os.name == "nt" and not ap.startswith("\\\\?\\"):
        return "\\\\?\\" + ap
    return ap


def _read_summary_csv(cfg: ExperimentExecutionConfig):
    """Long-path-safe read of a subfamily execution_summary.csv (or None)."""
    csv = summary_dir(cfg) / "execution_summary.csv"
    try:
        return pd.read_csv(_long(csv))
    except Exception:
        return None


def _subfamily_counts(cfg: ExperimentExecutionConfig) -> Dict[str, int]:
    df = _read_summary_csv(cfg)
    if df is None or df.empty:
        return {"success": 0, "failed": 0, "skipped": 0, "datasets": 0}
    vc = df["status"].value_counts().to_dict() if "status" in df.columns else {}
    return {
        "success": int(vc.get("success", 0)),
        "failed": int(vc.get("failed", 0)),
        "skipped": int(vc.get("skipped", 0)),
        "datasets": int(df["dataset_id"].nunique()) if "dataset_id" in df.columns else 0,
    }


def _subfamily_report(cfg: ExperimentExecutionConfig) -> str:
    """Long-path-safe provisional rollup (best-view mean ARI + dyn-K dist)."""
    df = _read_summary_csv(cfg)
    if df is None:
        return "  (no execution_summary.csv readable)"
    if df.empty:
        return "  (empty summary)"
    lines = []
    n_ds = df["dataset_id"].nunique()
    counts = df["status"].value_counts().to_dict()
    lines.append("  datasets=%d runs: %s" % (
        n_ds, " ".join(f"{k}={v}" for k, v in sorted(counts.items()))))
    ok = df[df["status"] == "success"].copy()
    if not ok.empty:
        ok["ari"] = pd.to_numeric(ok["ari"], errors="coerce")
        best = ok.loc[ok.groupby(["method_name", "dataset_id"])["ari"].idxmax()]
        bv = (best.groupby("method_name")
              .agg(best_view_mean_ari=("ari", "mean"), n=("dataset_id", "nunique"))
              .sort_values("best_view_mean_ari", ascending=False))
        lines.append("  best-view mean ARI by method:")
        for name, r in bv.iterrows():
            lines.append(f"    {name:<42} {r['best_view_mean_ari']:.4f}  (n={int(r['n'])})")
        if "selected_metric_count" in ok.columns:
            dyn = ok[ok["method_name"].str.contains("dynamic")]
            if not dyn.empty:
                dist = dyn["selected_metric_count"].value_counts().sort_index().to_dict()
                lines.append(f"  dynamic selected_metric_count dist: {dist}")
    return "\n".join(lines)


def _write_progress(progress_dir: Path, state: Dict) -> None:
    progress_dir.mkdir(parents=True, exist_ok=True)
    tmp = progress_dir / "phase_e2_progress.json.tmp"
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2)
    tmp.replace(progress_dir / "phase_e2_progress.json")

    cum = state["cumulative_run_counts"]
    md = [
        "# Phase E2 — KNN full-run progress",
        "",
        f"- Last update: {state['last_update']}",
        f"- Subfamilies completed: **{len(state['completed_subfamilies'])} / {state['total_subfamilies']}**",
        f"- Failed subfamilies: {len(state['failed_subfamilies'])}",
        f"- Last completed: {state['last_completed_subfamily']}",
        f"- Cumulative runs: success={cum['success']} failed={cum['failed']} skipped={cum['skipped']}",
        f"- Datasets processed (cumulative): {state['cumulative_datasets']}",
        f"- Output suffix: `{state['split_dir_suffix']}`",
        f"- Elapsed: {state['elapsed_min']:.1f} min",
        "",
        "## Remaining subfamilies",
        "",
        *[f"- {s}" for s in state["remaining_subfamilies"][:200]],
    ]
    (progress_dir / "phase_e2_progress.md").write_text("\n".join(md), encoding="utf-8")


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", required=True)
    p.add_argument("--split-id", type=int, default=1)
    p.add_argument("--max-workers", type=int, default=8)
    p.add_argument("--split-dir-suffix", default=PHASE_E2_SUFFIX)
    p.add_argument("--limit-subfamilies", type=int, default=None,
                   help="process only the first N subfamilies (smoke/debug).")
    p.add_argument("--limit-datasets", type=int, default=None,
                   help="cap datasets per subfamily (smoke/debug).")
    p.add_argument("--methods", nargs="+", default=list(PHASE_E2_METHODS))
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)

    repo = Path(args.repo_root).resolve()
    analyzed = repo / "results_analysis" / "clustering_repository" / "analyzed_data"
    split_csv = analyzed / "experiment_splits" / "dataset_split_assignments.csv"
    progress_dir = _progress_dir(repo)

    subfamilies = discover_subfamilies(analyzed)
    if args.limit_subfamilies:
        subfamilies = subfamilies[: args.limit_subfamilies]
    all_names = [f"{s.parent.parent.name}/{s.name}" for s in subfamilies]
    print(f"[phaseE2] {len(subfamilies)} subfamilies | split={args.split_id} | "
          f"{len(args.methods)} KNN methods x 3 views | workers={args.max_workers} | "
          f"suffix={args.split_dir_suffix}", flush=True)

    t0 = time.perf_counter()
    completed: List[str] = []
    failed: List[str] = []
    cum = {"success": 0, "failed": 0, "skipped": 0}
    cum_datasets = 0

    for i, sub in enumerate(subfamilies, 1):
        family = sub.parent.parent.name
        name = f"{family}/{sub.name}"
        cfg = ExperimentExecutionConfig(
            repo_root=repo, subfamily_dir=sub, split_assignments=split_csv,
            split_id=int(args.split_id), split_dir_suffix=str(args.split_dir_suffix),
            method_set="phaseE2_knn", methods=tuple(args.methods),
            max_workers=int(args.max_workers), resume=not args.overwrite,
            overwrite=bool(args.overwrite), limit_datasets=args.limit_datasets,
        )
        ts = time.perf_counter()
        print(f"\n[{i}/{len(subfamilies)}] {name} ...", flush=True)
        try:
            rc = run_subfamily(cfg)
            elapsed = time.perf_counter() - ts
            print(f"  rc={rc} runtime={elapsed:.0f}s", flush=True)
            print(_subfamily_report(cfg), flush=True)
            c = _subfamily_counts(cfg)
            cum["success"] += c["success"]
            cum["failed"] += c["failed"]
            cum["skipped"] += c["skipped"]
            cum_datasets += c["datasets"]
            completed.append(name)
        except Exception as exc:  # never let one subfamily stop the sweep
            print(f"  [ERROR] {type(exc).__name__}: {exc}", flush=True)
            failed.append(name)

        state = {
            "phase": "ClustOpt_PhaseE2_KNN_Ablation",
            "split_id": int(args.split_id),
            "split_dir_suffix": str(args.split_dir_suffix),
            "methods": list(args.methods),
            "total_subfamilies": len(subfamilies),
            "completed_subfamilies": completed,
            "failed_subfamilies": failed,
            "remaining_subfamilies": [n for n in all_names
                                      if n not in completed and n not in failed],
            "last_completed_subfamily": completed[-1] if completed else None,
            "cumulative_run_counts": cum,
            "cumulative_datasets": cum_datasets,
            "elapsed_min": (time.perf_counter() - t0) / 60.0,
            "last_update": _now(),
        }
        try:
            _write_progress(progress_dir, state)
        except Exception as exc:
            print(f"  (could not persist progress: {exc})", flush=True)

    print(f"\n[phaseE2] swept {len(completed)}/{len(subfamilies)} subfamilies in "
          f"{(time.perf_counter()-t0)/60:.1f} min "
          f"(failed={len(failed)})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
