"""Stage-1G phase 1: re-collect the historical Split-1 run frame from raw artifacts.

FORENSIC / READ-ONLY. Runs no clustering, no Optuna, no training, and writes
nothing into any experiment directory.

Deliberately does NOT read the cached
``paper_analysis_v2/13_raw_and_reproducibility/canonical_run_frame.parquet``:
the Stage-1G brief requires recomputation from the persisted per-run artifacts
rather than from any prior summary.

Reads, per (dataset x method x view), exactly the fields the audit needs:

    ClustOpt schema   status.json, external_metrics.json, best_result.json,
                      timing_summary.json
    external schema   result.json (falling back to summary_metrics.json)

Method identity, taxonomy and on-disk names come from the repository's own
``paper_analysis.method_registry``; nothing is derived by parsing names.

Sharded by family across processes (JSON parsing is CPU-bound, so processes
rather than threads), each shard written as soon as it completes, so a killed
run resumes instead of restarting. Long-path safe.

Run:
    .venv_clustopt/Scripts/python.exe scripts/stage1/collect_stage1g_run_frame.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from models.Clustering_Repository_Builder.utility_generation.worker_setup import (  # noqa: E402
    configure_process,
)

configure_process()

from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402

from models.Clustering_Repository_Builder.experiments.experiment_execution.output_writer import (  # noqa: E402
    _ext, path_exists,
)
from models.Clustering_Repository_Builder.experiments.paper_analysis.method_registry import (  # noqa: E402
    CLUSTOPT_VIEW_DIRS, EXTERNAL_VIEW_DIRS, METHODS, VIEW_IDS,
)

OUT = Path(REPO) / "results_analysis/clustering_repository/stage1_2x3_aggregation/policy_vbs_audit"
SHARDS = OUT / "_shards"
RUN_FRAME = OUT / "stage1g_run_frame_split01.csv"
ANALYZED = Path(REPO) / "results_analysis/clustering_repository/analyzed_data"


def _read_json(p: Path):
    if not path_exists(p):
        return None
    try:
        with open(_ext(p), "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def _collect_dataset(ds_dir: Path, family: str, subfamily: str, specs) -> list:
    """All (method x view) rows for one dataset."""
    split = ds_dir / "experiments" / "split_01"
    rows = []
    for spec in specs:
        vdirs = (CLUSTOPT_VIEW_DIRS if spec.schema == "clustopt" else EXTERNAL_VIEW_DIRS)
        for view in VIEW_IDS:
            run = split / spec.on_disk_name / vdirs[view]
            row = {
                "dataset_id": ds_dir.name, "family": family, "subfamily": subfamily,
                "view_id": view, "method_name": spec.method_name,
                "framework": spec.framework, "method_family": spec.method_family,
                "utility_source": spec.utility_source,
                "weighting_mode": spec.weighting_mode,
                "k_policy": spec.k_policy,
                "fixed_metric_count": spec.fixed_metric_count,
                "dynamic_k_source": spec.dynamic_k_source,
                "uses_real_utility": bool(spec.uses_real_utility),
                "is_oracle_assisted": bool(spec.is_oracle_assisted),
                "is_external": bool(spec.is_external),
                "is_paper_method": bool(spec.is_paper_method),
                "same_search_space": bool(spec.same_search_space),
                "status": "missing", "ari": None, "selected_k": None,
                "true_k": None, "k_correct": None,
                "n_trials_completed": None, "runtime_sec": None,
            }
            if spec.schema == "clustopt":
                st = _read_json(run / "status.json")
                if not st:
                    rows.append(row); continue
                row["status"] = str(st.get("status", "")).lower() or "missing"
                if row["status"] != "success":
                    rows.append(row); continue
                ext = _read_json(run / "external_metrics.json") or {}
                best = _read_json(run / "best_result.json") or {}
                tim = _read_json(run / "timing_summary.json") or {}
                row.update({
                    "ari": ext.get("ari", st.get("ari")),
                    "selected_k": best.get("selected_k", st.get("selected_k")),
                    "true_k": best.get("true_k", st.get("true_k")),
                    "k_correct": best.get("k_correct"),
                    "n_trials_completed": best.get("n_trials_completed"),
                    "runtime_sec": tim.get("runtime_sec", st.get("runtime_sec")),
                })
            else:
                res = _read_json(run / "result.json") or _read_json(
                    run / "summary_metrics.json")
                if not res:
                    rows.append(row); continue
                row["status"] = str(res.get("status", "")).lower() or "missing"
                if row["status"] != "success":
                    rows.append(row); continue
                met = res.get("metrics") if isinstance(res.get("metrics"), dict) else res
                row.update({
                    "ari": met.get("ari"),
                    "selected_k": res.get("selected_k", met.get("selected_k")),
                    "true_k": met.get("true_k"),
                    "k_correct": met.get("k_correct", met.get("exact_k_match")),
                    "n_trials_completed": res.get("optimizer_evaluations"),
                    "runtime_sec": res.get("runtime_sec", met.get("runtime_sec")),
                })
            rows.append(row)
    return rows


def collect_shard(payload: tuple) -> tuple:
    """One SUBFAMILY -> shard CSV. Runs in a worker process.

    Sharding at subfamily rather than family granularity keeps each unit of work
    small (~10-20 datasets), so a shard is written every ~30-60 s and an
    interrupted collection resumes with almost no lost work.
    """
    configure_process()
    key, family, subfamily, dataset_dirs = payload
    shard = SHARDS / f"{key}.csv"
    if path_exists(shard):                      # resume: shard already complete
        return key, -1
    specs = list(METHODS)
    rows = []
    for d in dataset_dirs:
        rows.extend(_collect_dataset(Path(d), family, subfamily, specs))
    pd.DataFrame(rows).to_csv(_ext(shard), index=False)
    return key, len(rows)


def _split1_shards() -> list:
    """[(shard_key, family_id, subfamily_id, [dataset_dir, ...]), ...] for split 1.

    Uses the split CSV's own ``dataset_dir`` (absolute) and ``family_id`` /
    ``subfamily_id`` (on-disk folder names). The ``family`` column is a display
    label -- it contains spaces and slashes -- and must never be used as a path
    component.
    """
    split_csv = ANALYZED / "experiment_splits/dataset_split_assignments.csv"
    a = pd.read_csv(_ext(split_csv))
    a = a[a["split_id"].astype("Int64") == 1]
    groups: dict = {}
    for _, r in a.iterrows():
        fam, sub = str(r["family_id"]), str(r["subfamily_id"])
        groups.setdefault((fam, sub), []).append(str(r["dataset_dir"]))
    return [(f"{fam}__{sub}", fam, sub, dirs)
            for (fam, sub), dirs in sorted(groups.items())]


def main() -> int:
    print("=" * 78, flush=True)
    print("STAGE-1G PHASE 1  --  RE-COLLECT HISTORICAL SPLIT-1 RUN FRAME", flush=True)
    print("=" * 78, flush=True)
    os.makedirs(_ext(SHARDS), exist_ok=True)

    payloads = _split1_shards()
    n_ds = sum(len(p[3]) for p in payloads)
    print(f"  subfamily shards {len(payloads)} | split-1 datasets {n_ds} | "
          f"methods {len(METHODS)} | views {len(VIEW_IDS)}", flush=True)
    print(f"  expected rows: {n_ds * len(METHODS) * len(VIEW_IDS):,}", flush=True)

    t0 = time.time()
    payloads = sorted(payloads, key=lambda p: -len(p[3]))
    done = n_new = 0
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(collect_shard, p): p[0] for p in payloads}
        for fut in as_completed(futs):
            key, n = fut.result()
            done += 1
            if n >= 0:
                n_new += 1
                print(f"    [{done}/{len(payloads)}] {key:52} {n:,} rows "
                      f"({time.time()-t0:.0f}s)", flush=True)
    print(f"  {n_new} shards collected this run, "
          f"{done - n_new} already cached", flush=True)

    frames = [pd.read_csv(_ext(SHARDS / f"{p[0]}.csv")) for p in payloads]
    frame = pd.concat(frames, ignore_index=True)
    frame = frame.sort_values(["family", "subfamily", "dataset_id",
                               "method_name", "view_id"])
    frame.to_csv(_ext(RUN_FRAME), index=False)

    print(f"\n  run frame -> {RUN_FRAME.name}  ({len(frame):,} rows)", flush=True)
    print(f"  datasets {frame['dataset_id'].nunique()} | "
          f"methods {frame['method_name'].nunique()} | "
          f"success {int((frame['status'] == 'success').sum()):,}", flush=True)
    print(f"  elapsed {(time.time()-t0)/60:.1f} min", flush=True)
    g = (frame.groupby(["framework", "method_family"])
         .agg(rows=("method_name", "size"), methods=("method_name", "nunique"),
              success=("status", lambda s: int((s == "success").sum()))).reset_index())
    print("\n" + g.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
