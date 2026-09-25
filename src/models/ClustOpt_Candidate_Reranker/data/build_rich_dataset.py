"""Table B -- the RICH candidate base, extracted from the fixed-32 repository.

1,537,248 rows = 16,013 datasets x 3 views x 32 candidates. No clustering is
invoked and no candidate partition is regenerated: this reads the persisted
``clustopt_results.csv`` slates and normalises them.

Normalisation that matters: the 32-candidate configuration grid is FIXED PER VIEW
and identical across datasets (three distinct slate fingerprints, one per view).
Configuration is therefore stored once as a 96-row catalogue keyed by
(view_id, candidate_index) instead of being repeated 1.5 M times. The extractor
verifies that invariant on every slate rather than trusting the sample that
suggested it.

Work is sharded by subfamily and each shard is written before the next starts, so
an interrupted run resumes from the first missing shard.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from models.Clustering_Repository_Builder.experiments.experiment_execution.output_writer import (  # noqa: E402,E501
    _ext, ensure_dir, path_exists, write_json_atomic,
)
from models.ClustOpt_Candidate_Reranker.data import candidate_schema as CS  # noqa: E402,E501
from models.ClustOpt_Candidate_Reranker.protocol import target_protocols as TP  # noqa: E402,E501

OUT_REL = ("results_analysis/clustopt_candidate_reranker/"
           "stage2b0_data_protocol_audit")
SHARD_DIR = "_candidate_shards"
SPLIT_CSV_REL = ("results_analysis/clustering_repository/analyzed_data/"
                 "experiment_splits/dataset_split_assignments.csv")
GZ = {"index": False, "compression": "gzip"}

_STATE: Dict[str, Any] = {}


def _init_worker(payload: Dict[str, Any]) -> None:
    from models.Clustering_Repository_Builder.utility_generation import worker_setup
    try:
        worker_setup.configure_process(thread_limit=1)
    except Exception:                                               # noqa: BLE001
        pass
    _STATE.update(payload)


def extract_slate(dataset_dir: str, dataset_id: str, split_id: int,
                  view_id: str, metric_names: Sequence[str]) -> Dict[str, Any]:
    """One (dataset, view) slate -> aligned candidate arrays + diagnostics."""
    path = CS.candidate_csv_path(Path(dataset_dir), view_id)
    df = pd.read_csv(_ext(path))
    CS.validate_slate(df, metric_names, path)

    algs = df["algorithm"].astype(str).tolist()
    cfgs = df["config_str"].astype(str).tolist()
    valid = df["valid"].map(CS.is_valid_flag).to_numpy(bool)
    ari = pd.to_numeric(df["ARI"], errors="coerce").to_numpy(np.float64)

    ncols = CS.norm_columns(metric_names)
    cvi = df[ncols].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)
    cvi_valid = np.isfinite(cvi).astype(np.int8)
    cvi = np.where(np.isfinite(cvi), cvi, np.nan)

    tgt = TP.candidate_targets(ari, valid)
    TP.validate_slate_targets(tgt, valid)

    part = {
        "part__n_clusters_wo_noise": pd.to_numeric(
            df["n_clusters_wo_noise"], errors="coerce").to_numpy(np.float64),
        "part__noise_ratio": pd.to_numeric(
            df["noise_ratio"], errors="coerce").to_numpy(np.float64),
        "part__n_labels_unique": pd.to_numeric(
            df["n_labels_unique"], errors="coerce").to_numpy(np.float64),
        "part__valid": valid.astype(np.int8),
        "part__has_exception": df["exception"].notna().to_numpy().astype(np.int8),
    }
    return {
        "dataset_id": dataset_id, "split_id": int(split_id), "view_id": view_id,
        "algorithms": algs, "configs": cfgs,
        "fingerprint": CS.slate_fingerprint(algs, cfgs),
        "cvi": cvi, "cvi_valid": cvi_valid, "part": part, "targets": tgt,
        "aggregate_score": pd.to_numeric(
            df["aggregate_score"], errors="coerce").to_numpy(np.float64),
    }


def _slate_diagnostics(s: Dict[str, Any]) -> Dict[str, Any]:
    """Per-slate target diagnostics (spec section 27)."""
    a = np.asarray(s["targets"]["candidate_ari"], dtype=np.float64)
    v = np.asarray(s["part"]["part__valid"], dtype=bool)
    av = a[v]
    best = float(np.max(av)) if av.size else float("nan")
    distinct = np.unique(np.round(av, 12))[::-1] if av.size else np.array([])
    second = float(distinct[1]) if distinct.size > 1 else float("nan")
    n_best = int(np.isclose(av, best, atol=TP.TIE_TOL).sum()) if av.size else 0
    algs = np.asarray(s["algorithms"])
    best_alg = ""
    if av.size:
        idx = np.flatnonzero(v)[np.isclose(av, best, atol=TP.TIE_TOL)]
        best_alg = str(algs[idx[0]]) if idx.size else ""
    out = {
        "dataset_id": s["dataset_id"], "split_id": s["split_id"],
        "view_id": s["view_id"],
        "n_candidates": int(len(a)), "n_valid": int(v.sum()),
        "best_ari": best, "second_distinct_ari": second,
        "top_gap": (best - second) if np.isfinite(second) else np.nan,
        "n_exact_best": n_best,
        "all_tied": int(bool(av.size and np.isclose(av.max(), av.min(),
                                                    atol=TP.TIE_TOL))),
        "ari_mean": float(np.mean(av)) if av.size else np.nan,
        "ari_median": float(np.median(av)) if av.size else np.nan,
        "ari_sd": float(np.std(av)) if av.size else np.nan,
        "ari_min": float(np.min(av)) if av.size else np.nan,
        "ari_range": float(np.max(av) - np.min(av)) if av.size else np.nan,
        "best_algorithm": best_alg,
        "candidate_oracle_ari_valid": best,
        "candidate_oracle_ari_all": float(np.nanmax(a)) if a.size else np.nan,
    }
    for thr, lab in ((0.001, "001"), (0.005, "005"), (0.01, "01")):
        out[f"n_within_{lab}"] = (int((best - av <= thr).sum()) if av.size else 0)
    return out


def _run_subfamily(task: Dict[str, Any]) -> Dict[str, Any]:
    """Extract every slate of one subfamily and write its shard."""
    metric_names: List[str] = _STATE["metric_names"]
    out_dir: Path = Path(_STATE["shard_root"])
    key = task["shard_key"]

    ident_rows: List[Dict[str, Any]] = []
    cvis, cvals, parts, targs, diags = [], [], [], [], []
    fingerprints: Dict[str, set] = {}
    cat_rows: Dict[Tuple[str, int], Dict[str, Any]] = {}

    for rec in task["datasets"]:
        for view in CS.VIEW_IDS:
            s = extract_slate(rec["dataset_dir"], rec["dataset_id"],
                              rec["split_id"], view, metric_names)
            fingerprints.setdefault(view, set()).add(s["fingerprint"])
            n = len(s["algorithms"])
            for i in range(n):
                ident_rows.append({
                    "dataset_id": s["dataset_id"], "split_id": s["split_id"],
                    "view_id": view, "candidate_index": i,
                    "candidate_key": CS.candidate_key(s["algorithms"][i],
                                                      s["configs"][i]),
                })
                if (view, i) not in cat_rows:
                    cat_rows[(view, i)] = {
                        "view_id": view, "candidate_index": i,
                        **CS.configuration_row(s["algorithms"][i], s["configs"][i]),
                    }
            cvis.append(s["cvi"].astype(np.float32))
            cvals.append(s["cvi_valid"])
            parts.append(pd.DataFrame(s["part"]))
            t = s["targets"]
            targs.append(pd.DataFrame({
                "candidate_ari": t["candidate_ari"],
                "slate_regret": t["slate_regret"],
                "slate_oracle_ari": t["slate_oracle_ari_valid"],
                "slate_oracle_ari_all": t["slate_oracle_ari_all"],
                "rank_group": t["rank_group"], "n_at_rank": t["n_at_rank"],
                "is_slate_best": t["is_slate_best"],
                "aggregate_score_stored": s["aggregate_score"],
            }))
            diags.append(_slate_diagnostics(s))

    ident = pd.DataFrame(ident_rows)
    ncols = [f"cvi__{m}" for m in metric_names]
    vcols = [f"cvi_valid__{m}" for m in metric_names]
    cvi_df = pd.DataFrame(np.vstack(cvis), columns=ncols)
    val_df = pd.DataFrame(np.vstack(cvals), columns=vcols)
    part_df = pd.concat(parts, ignore_index=True)
    tgt_df = pd.concat(targs, ignore_index=True)
    diag_df = pd.DataFrame(diags)

    d = ensure_dir(out_dir / key)
    ident.to_csv(_ext(d / "identifiers.csv.gz"), **GZ)
    cvi_df.to_csv(_ext(d / "cvi.csv.gz"), **GZ)
    val_df.to_csv(_ext(d / "cvi_valid.csv.gz"), **GZ)
    part_df.to_csv(_ext(d / "partition.csv.gz"), **GZ)
    tgt_df.to_csv(_ext(d / "targets.csv.gz"), **GZ)
    diag_df.to_csv(_ext(d / "slate_diagnostics.csv.gz"), **GZ)
    pd.DataFrame(list(cat_rows.values())).to_csv(
        _ext(d / "config_catalogue.csv.gz"), **GZ)
    manifest = {
        "shard_key": key, "status": "complete",
        "n_datasets": len(task["datasets"]), "n_rows": int(len(ident)),
        "n_slates": int(len(diag_df)),
        "view_fingerprints": {k: sorted(v) for k, v in fingerprints.items()},
    }
    write_json_atomic(d / "manifest.json", manifest)
    return manifest


def shard_is_complete(shard_root: Path, key: str) -> bool:
    m = shard_root / key / "manifest.json"
    if not path_exists(m):
        return False
    try:
        return json.loads(Path(_ext(m)).read_text("utf-8")).get("status") == "complete"
    except Exception:                                               # noqa: BLE001
        return False


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", default=".")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit-shards", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)

    repo = Path(args.repo_root).resolve()
    out_root = ensure_dir(repo / OUT_REL)
    shard_root = ensure_dir(out_root / SHARD_DIR)
    metric_names = CS.load_metric_names(repo)

    assign = pd.read_csv(_ext(repo / SPLIT_CSV_REL))
    assign = assign[assign["split_id"] != 1]
    if assign["split_id"].min() < 2 or assign["split_id"].max() > 16:
        raise SystemExit("split range outside 2..16")

    groups = list(assign.groupby(["family_id", "subfamily_id"], sort=True))
    tasks = []
    for (fam, sub), grp in groups:
        tasks.append({
            "shard_key": f"{fam}__{sub}",
            "datasets": [{"dataset_dir": r["dataset_dir"],
                          "dataset_id": r["dataset_id"],
                          "split_id": int(r["split_id"])}
                         for _, r in grp.iterrows()],
        })
    if args.limit_shards:
        tasks = tasks[: int(args.limit_shards)]

    todo = [t for t in tasks
            if args.overwrite or not shard_is_complete(shard_root, t["shard_key"])]
    print("=" * 78)
    print("STAGE 2B-0  --  RICH CANDIDATE EXTRACTION (Table B)")
    print("=" * 78)
    print(f"  {len(tasks)} subfamily shards | {len(todo)} to build | "
          f"{len(assign)} datasets | {args.workers} workers x 1 thread",
          flush=True)
    if not todo:
        print("  all shards already complete", flush=True)
        return 0

    payload = {"metric_names": metric_names, "shard_root": str(shard_root)}
    t0, done, rows = time.time(), 0, 0
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=int(args.workers),
                                 initializer=_init_worker,
                                 initargs=(payload,)) as pool:
            for m in pool.map(_run_subfamily, todo):
                done += 1
                rows += m["n_rows"]
                print(f"   [{done}/{len(todo)}] {m['shard_key']:<52} "
                      f"{m['n_rows']:>7} rows  ({time.time()-t0:.0f}s)", flush=True)
    else:
        _init_worker(payload)
        for t in todo:
            m = _run_subfamily(t)
            done += 1
            rows += m["n_rows"]
            print(f"   [{done}/{len(todo)}] {m['shard_key']:<52} "
                  f"{m['n_rows']:>7} rows", flush=True)

    print(f"\n  {done} shards, {rows} candidate rows, {time.time()-t0:.0f}s",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
