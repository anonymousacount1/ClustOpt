"""Stage 2C-4A: execute the RAW/SOFTMAX online twins of the frozen sample.

The sample is a sparse set of (split, dataset, view) rows spread over all 15
development splits, so the subfamily-serial sweepers do not fit. This module is
a *scheduler*, not an executor: every search is run by the unmodified
:func:`method_runner.run_method_view`, through the unmodified ClustOpt/Optuna
path, with the unmodified dataset loader and view builder. Worker-side dataset
and view caching mirrors ``record_worker`` so a dataset is loaded once per
process regardless of how many of its runs land there.

Resume is at the level of a single search (``status.json`` success plus every
expected output file). A *scientific row* counts as complete only when BOTH of
its twins are complete, which is what the inventory and the execution summary
report; a half-finished pair is never treated as data.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from models.Clustering_Repository_Builder.experiments.experiment_execution.config import (  # noqa: E402,E501
    ExperimentExecutionConfig, MethodSpec,
)
from models.Clustering_Repository_Builder.experiments.experiment_execution.method_runner import (  # noqa: E402,E501
    RunRecord, is_run_complete, run_method_view,
)
from models.Clustering_Repository_Builder.experiments.experiment_execution.output_writer import (  # noqa: E402,E501
    _ext, ensure_dir, path_exists, read_json, write_csv_atomic, write_json_atomic,
)
from models.Clustering_Repository_Builder.experiments.experiment_execution.record_worker import (  # noqa: E402,E501
    write_dataset_summaries,
)
from models.Clustering_Repository_Builder.experiments.experiment_execution.summary_writer import (  # noqa: E402,E501
    write_execution_summaries,
)
from models.Clustering_Repository_Builder.experiments.experiment_execution.view_builder import (  # noqa: E402,E501
    build_view, load_dataset_for_utility,
)
from models.Clustering_Repository_Builder.utility_generation.worker_setup import (  # noqa: E402,E501
    configure_process,
)
from models.ClustOpt_Policy_Predictor.stage2c import stage2c4_common as C  # noqa: E402,E501

REQUIRED_FILES: Tuple[str, ...] = (
    "clustopt_results.csv", "best_result.json", "external_metrics.json",
    "selected_metrics.json", "timing_summary.json", "run_log.json",
    "status.json", "config_snapshot.json", "phaseD_metadata.json",
)
CONFIG_ROOT_REL = "models/ClustOpt/configs/experiments"
_MAX_CACHED = 2

_CFG: Dict[str, Any] = {}
_DS: "OrderedDict[str, Any]" = OrderedDict()
_VW: "OrderedDict[Tuple[str, str], Any]" = OrderedDict()


# --------------------------------------------------------------- completion
def run_is_complete(run_dir: Path) -> bool:
    if not is_run_complete(run_dir):
        return False
    return all(path_exists(run_dir / f) for f in REQUIRED_FILES)


# ------------------------------------------------------------------- worker
def _init(payload: Dict[str, Any]) -> None:
    configure_process()
    # Log volume only: 1,080 x 50 Optuna trial lines is unusable in a terminal.
    # Verbosity has no effect on the sampler, the trials or the results.
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except Exception:                                              # noqa: BLE001
        pass
    _CFG.update(payload)


def _loaded(d: str):
    got = _DS.get(d)
    if got is None:
        got = load_dataset_for_utility(Path(d))
        _DS[d] = got
        while len(_DS) > _MAX_CACHED:
            old, _ = _DS.popitem(last=False)
            for k in [k for k in _VW if k[0] == old]:
                _VW.pop(k, None)
    _DS.move_to_end(d)
    return got


def _view(loaded, d: str, v: str):
    key = (d, v)
    got = _VW.get(key)
    if got is None:
        got = build_view(loaded, v)
        _VW[key] = got
    _VW.move_to_end(key)
    return got


def _run_one(task: Dict[str, Any]) -> Dict[str, Any]:
    """One (dataset, view, twin) search. Never raises."""
    t0 = time.perf_counter()
    repo = Path(_CFG["repo_root"])
    dd = task["dataset_dir"]
    out: Dict[str, Any] = dict(task)
    try:
        loaded = _loaded(dd)
        view = _view(loaded, dd, task["view_id"])
        method = MethodSpec(task["method_name"], task["utility_source"],
                            "%s/%s" % (C.CONFIG_SUBDIR, task["method_name"]))
        cfg = ExperimentExecutionConfig(
            repo_root=repo, subfamily_dir=Path(dd).parent,
            split_assignments=repo / C.SPLIT_CSV_REL,
            split_id=int(task["split_id"]),
            split_dir_suffix=C.SPLIT_DIR_SUFFIX,
            methods=(task["method_name"],), views=(task["view_id"],),
            resume=True, overwrite=False)
        rec = run_method_view(
            loaded=loaded, view=view, method=method,
            config_path=(repo / CONFIG_ROOT_REL / C.CONFIG_SUBDIR
                         / task["method_name"] / ("%s.json" % task["view_id"])),
            dataset_meta={"family": task["family_id"],
                          "subfamily": task["subfamily_id"],
                          "difficulty": None,
                          "cluster_count": loaded.cluster_count_from_generator},
            cfg=cfg)
        out.update({
            "status": rec.status, "ari": rec.ari,
            "runtime_sec": rec.runtime_sec, "selected_k": rec.selected_k,
            "true_k": rec.true_k, "top_metric": rec.top_metric,
            "failure_reason": rec.failure_reason,
            "extra": json.dumps(rec.extra or {}),
            "wall_sec": time.perf_counter() - t0})
    except Exception as exc:                                       # noqa: BLE001
        out.update({"status": "failed", "ari": None,
                    "runtime_sec": 0.0, "selected_k": None, "true_k": None,
                    "top_metric": None,
                    "failure_reason": "%s: %s" % (type(exc).__name__, exc),
                    "extra": "{}", "wall_sec": time.perf_counter() - t0})
    return out


# --------------------------------------------------------------------- main
def _records(frame: pd.DataFrame) -> List[RunRecord]:
    return [RunRecord(
        dataset_id=r["dataset_id"], dataset_dir=r["dataset_dir"],
        split_id=int(r["split_id"]), family=r["family_id"],
        subfamily=r["subfamily_id"], difficulty=None, cluster_count=None,
        view_id=r["view_id"], method_name=r["method_name"],
        method_group=r["utility_source"], status=str(r["status"]),
        failure_reason=str(r.get("failure_reason") or ""),
        runtime_sec=float(r.get("runtime_sec") or 0.0),
        ari=(None if pd.isna(r.get("ari")) else float(r["ari"])),
        selected_k=(None if pd.isna(r.get("selected_k"))
                    else int(r["selected_k"])),
        true_k=(None if pd.isna(r.get("true_k")) else int(r["true_k"])),
        top_metric=r.get("top_metric"),
        extra=json.loads(r.get("extra") or "{}"))
        for _, r in frame.iterrows()]


def write_summaries(repo: Path, done: pd.DataFrame) -> int:
    """Per-dataset and per-subfamily summaries, in the existing conventions."""
    n = 0
    for (fam, sub), g in done.groupby(["family_id", "subfamily_id"]):
        sub_dir = (repo / C.ANALYZED_REL / fam / "subfamilies" / sub)
        for s, gs in g.groupby("split_id"):
            cfg = ExperimentExecutionConfig(
                repo_root=repo, subfamily_dir=sub_dir,
                split_assignments=repo / C.SPLIT_CSV_REL, split_id=int(s),
                split_dir_suffix=C.SPLIT_DIR_SUFFIX,
                methods=tuple(sorted(gs["method_name"].unique())),
                views=C.VIEWS, resume=True)
            recs = _records(gs)
            write_dataset_summaries(cfg, recs)
            write_execution_summaries(
                cfg=cfg, records=recs,
                tracker_summary={"stage": C.STAGE, "n_records": len(recs),
                                 "n_success": int((gs["status"] == "success").sum()),
                                 "n_failed": int((gs["status"] == "failed").sum())},
                family_id=fam, subfamily_id=sub)
            n += 1
    return n


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", default=".")
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--rows", nargs="*", default=None,
                   help="restrict to these row_ids (dataset_id|view_id)")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--label", default="batch")
    args = p.parse_args(argv)
    repo = Path(args.repo_root).resolve()
    out = ensure_dir(repo / C.OUT_REL)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                            capture_output=True, text=True).stdout.strip()
    print("=" * 78)
    print("STAGE 2C-4A  ONLINE RAW/SOFTMAX BATCH  [%s]" % args.label)
    print("=" * 78, flush=True)

    C.load_sample(repo)                       # hash gate on the FULL sample
    INV = pd.read_csv(_ext(out / "online_pair_inventory.csv"))
    INV = INV[INV["rank"] <= int(args.n)].reset_index(drop=True)
    if args.rows:
        INV = INV[INV["row_id"].isin(set(args.rows))].reset_index(drop=True)
    if args.limit:
        INV = INV.head(int(args.limit)).reset_index(drop=True)
    if INV.empty:
        raise SystemExit("no searches selected")

    INV["complete"] = [run_is_complete(Path(r)) for r in INV["run_dir"]]
    todo = INV[~INV["complete"]].copy()
    print("  %d searches in scope | %d already complete | %d to run | "
          "%d workers" % (len(INV), int(INV["complete"].sum()), len(todo),
                          args.workers), flush=True)

    tasks = todo.to_dict("records")
    results: List[Dict[str, Any]] = []
    t0 = time.time()
    if tasks:
        payload = {"repo_root": str(repo)}
        with ProcessPoolExecutor(max_workers=int(args.workers),
                                 initializer=_init,
                                 initargs=(payload,)) as ex:
            futs = {ex.submit(_run_one, t): t for t in tasks}
            for i, f in enumerate(as_completed(futs), 1):
                r = f.result()
                results.append(r)
                el = time.time() - t0
                if i % 10 == 0 or i == len(tasks) or i <= 6:
                    print("   [%4d/%4d] %s %s/%s %s -> %s ari=%s %.0fs | "
                          "elapsed %.2fh ETA %.2fh"
                          % (i, len(tasks), r["dataset_id"][:20], r["view_id"],
                             r["weighting_arm"], r["regime_id"], r["status"],
                             ("%.4f" % r["ari"]) if r.get("ari") is not None
                             and not pd.isna(r.get("ari")) else "na",
                             r.get("runtime_sec") or 0.0, el / 3600.0,
                             el / i * (len(tasks) - i) / 3600.0), flush=True)
    wall = time.time() - t0

    # ---- reconcile the whole scope from disk -------------------------------
    rows: List[Dict[str, Any]] = []
    for _, r in INV.iterrows():
        rd = Path(r["run_dir"])
        st = read_json(rd / "status.json") or {}
        br = read_json(rd / "best_result.json") or {}
        sm = read_json(rd / "selected_metrics.json") or {}
        tm = read_json(rd / "timing_summary.json") or {}
        pd_meta = read_json(rd / "phaseD_metadata.json") or {}
        rows.append({
            **{k: r[k] for k in ("split_id", "rank", "dataset_id", "view_id",
                                 "family_id", "subfamily_id", "regime_id",
                                 "utility_source", "k_mode", "weighting_arm",
                                 "method_name", "dataset_dir", "run_dir",
                                 "search_seed", "row_id")},
            "status": st.get("status", "missing"),
            "ari": st.get("ari"), "selected_k": st.get("selected_k"),
            "true_k": st.get("true_k"),
            "runtime_sec": tm.get("runtime_sec"),
            "cvi_eval_sec": tm.get("cvi_eval_sec"),
            "fit_predict_sec": tm.get("fit_predict_sec"),
            "requested_trials": br.get("requested_trials"),
            "delivered_trials": br.get("delivered_trials",
                                       br.get("n_trials_completed")),
            "valid_trials": br.get("valid_trials"),
            "failed_trials": br.get("failed_trials"),
            "config_search_seed": br.get("search_seed"),
            "selected_metric_count": len(sm.get("selected_metrics") or []),
            "selected_metrics": "|".join(sm.get("selected_metrics") or []),
            "weighting_mode_recorded": pd_meta.get("weighting_mode"),
            "real_utility_path": pd_meta.get("real_utility_path"),
            "used_fallback": sm.get("used_fallback"),
            "top_metric": sm.get("top_metric"),
            "failure_reason": (read_json(rd / "status.json") or {}).get(
                "error_message", ""),
            "extra": "{}"})
    DONE = pd.DataFrame(rows)
    write_csv_atomic(out / ("online_pair_execution_records_%s.csv" % args.label),
                     DONE)

    ok = DONE[DONE["status"] == "success"]
    pairs = (ok.groupby(["row_id"])["weighting_arm"].nunique() == 2)
    n_sub = write_summaries(repo, DONE[DONE["status"].isin(("success",
                                                            "failed"))])

    summary = {
        "stage": C.STAGE, "label": args.label, "n_per_split": int(args.n),
        "workers": int(args.workers),
        "searches_in_scope": int(len(INV)),
        "searches_run_this_invocation": int(len(results)),
        "searches_successful": int(len(ok)),
        "searches_failed": int((DONE["status"] == "failed").sum()),
        "searches_missing": int((DONE["status"] == "missing").sum()),
        "scientific_rows_in_scope": int(INV["row_id"].nunique()),
        "complete_pairs": int(pairs.sum()),
        "incomplete_pairs": int(INV["row_id"].nunique() - pairs.sum()),
        "wall_clock_sec": wall,
        "wall_clock_hours": wall / 3600.0,
        "cpu_hours_search": float(ok["runtime_sec"].sum()) / 3600.0,
        "search_runtime": {
            "mean": float(ok["runtime_sec"].mean()) if len(ok) else None,
            "median": float(ok["runtime_sec"].median()) if len(ok) else None,
            "p95": float(ok["runtime_sec"].quantile(0.95)) if len(ok) else None,
            "max": float(ok["runtime_sec"].max()) if len(ok) else None},
        "delivered_trials": {
            "mean": float(pd.to_numeric(ok["delivered_trials"],
                                        errors="coerce").mean()) if len(ok) else None,
            "min": (int(pd.to_numeric(ok["delivered_trials"],
                                      errors="coerce").min()) if len(ok) else None),
            "n_below_requested": int((pd.to_numeric(ok["delivered_trials"],
                                                    errors="coerce")
                                      < C.N_TRIALS).sum()) if len(ok) else 0},
        "timeout_incidence": (float((pd.to_numeric(ok["delivered_trials"],
                                                   errors="coerce")
                                     < C.N_TRIALS).mean()) if len(ok) else None),
        "subfamily_summaries_written": int(n_sub),
        "split_1_used": False, "source_commit": commit,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    prev = read_json(out / "online_pair_execution_summary.json") or {}
    hist = list(prev.get("invocations") or [])
    hist.append({k: summary[k] for k in
                 ("label", "n_per_split", "workers", "searches_run_this_invocation",
                  "wall_clock_hours", "updated_at")})
    summary["invocations"] = hist
    summary["cumulative_wall_clock_hours"] = float(
        sum(h["wall_clock_hours"] for h in hist))
    write_json_atomic(out / "online_pair_execution_summary.json", summary)

    print("\n  %d/%d searches successful | %d complete pairs / %d rows"
          % (len(ok), len(INV), int(pairs.sum()), INV["row_id"].nunique()),
          flush=True)
    if len(ok):
        print("  search runtime mean %.1fs median %.1fs p95 %.1fs max %.1fs"
              % (summary["search_runtime"]["mean"],
                 summary["search_runtime"]["median"],
                 summary["search_runtime"]["p95"],
                 summary["search_runtime"]["max"]), flush=True)
        print("  delivered trials mean %.1f (min %s, %d below %d)"
              % (summary["delivered_trials"]["mean"],
                 summary["delivered_trials"]["min"],
                 summary["delivered_trials"]["n_below_requested"], C.N_TRIALS),
              flush=True)
    print("  wall clock %.2f h | %d subfamily summaries" % (wall / 3600.0, n_sub),
          flush=True)
    return 0 if summary["searches_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
