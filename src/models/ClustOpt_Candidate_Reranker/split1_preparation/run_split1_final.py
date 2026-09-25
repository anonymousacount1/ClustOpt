"""Stage 2B-5B: final Split-1 online candidate-reranker evaluation (WET).

Orchestration follows the repository experiment-execution conventions: one
subfamily at a time, 8 dataset workers inside the active subfamily, dry-run plan,
checkpointing, per-dataset ``result.json``, and subfamily summaries written in the
existing ``experiment_execution_summaries`` format under a namespaced experiment
id.

Scientific discipline, unchanged and frozen:
  * candidates are the ACTUAL eligible visited candidates -- never extrapolated,
    synthesised, re-run or dropped;
  * the reranked candidate id is frozen BEFORE its ARI is read;
  * RAW and SOFTMAX share a reranker but never a slate;
  * OPTICS/Spectral and any other online-only family are genuine out-of-domain
    candidates under the frozen 347-dim representation -- counted, never removed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from models.Clustering_Repository_Builder.experiments.experiment_execution.output_writer import (  # noqa: E402,E501
    _ext, ensure_dir, path_exists, read_json, write_csv_atomic, write_json_atomic,
)
from models.ClustOpt_Candidate_Reranker.data import candidate_schema as CS  # noqa: E402,E501
from models.ClustOpt_Candidate_Reranker.split1_preparation import (  # noqa: E402,E501
    online_feature_builder as OFB,
)
from models.ClustOpt_Candidate_Reranker.training import feature_builder as FB  # noqa: E402,E501
from models.ClustOpt_Candidate_Reranker.training import model_registry as MR  # noqa: E402,E501
from models.ClustOpt_Candidate_Reranker.training.final_models import (  # noqa: E402,E501
    policy_mapping as PM,
)

EXPERIMENT_ID = "split_01_candidate_reranker_online_final"
PREP_REL = ("results_analysis/clustopt_candidate_reranker/"
            "stage2b5a_final_models_and_split1_preparation")
GLOBAL_ROOT_REL = ("results_analysis/clustering_repository/"
                   "candidate_reranker_split1_online_final")
SPLIT_CSV_REL = ("results_analysis/clustering_repository/analyzed_data/"
                 "experiment_splits/dataset_split_assignments.csv")
MLP_SPLIT1_PRED = ("results_analysis/mlp/20260615_231715__metric_utility_mlp_"
                   "split1_holdout/artifacts/all_fold_test_predictions.csv")
KNN_SPLIT1_CTX = f"{PREP_REL}/split1_knn_utility_context.csv.gz"
ONLINE_DIR = "split_01"
VIEWS: Tuple[str, ...] = ("x_only", "y_only", "xy_2d")
SPLIT1 = 1
TIE_TOL = 1e-12
GZ = {"index": False, "compression": "gzip"}
RESULT_SCHEMA_VERSION = "stage2b5.result.v1"

_S: Dict[str, Any] = {}


# --------------------------------------------------------------------- worker
def _init_worker(payload: Dict[str, Any]) -> None:
    import joblib
    _S.update(payload)
    prep = Path(payload["prep_root"])
    models, load_ms = {}, {}
    for rid in PM.RERANKER_IDS:
        t0 = time.perf_counter()
        models[rid] = joblib.load(_ext(prep / "final_models" / rid / "model.joblib"))
        load_ms[rid] = (time.perf_counter() - t0) * 1000.0
    _S["models"] = models
    _S["model_load_ms"] = load_ms


def _original_index(log: pd.DataFrame, br: Dict[str, Any]) -> Tuple[int, str]:
    """Production-selected candidate, matched on the authoritative record.

    ``best_result.json`` is ground truth. J ties exist, so the row is matched on
    objective score AND algorithm (then config) rather than a bare ``idxmax``.
    """
    agg = pd.to_numeric(log["aggregate_score"], errors="coerce").to_numpy(float)
    best = br.get("best_objective_score")
    alg = str(br.get("selected_algorithm"))
    if best is None or not np.isfinite(agg).any():
        return int(np.nanargmax(agg)), "argmax_fallback"
    cand = np.flatnonzero(np.abs(agg - float(best)) <= 1e-9)
    if cand.size == 0:
        return int(np.nanargmax(agg)), "argmax_score_mismatch"
    same = [i for i in cand if str(log["algorithm"].iloc[i]) == alg]
    if len(same) == 1:
        return int(same[0]), "exact_score_and_algorithm"
    if same:
        params = json.dumps(br.get("selected_params"), sort_keys=True)
        for i in same:
            try:
                import ast
                if json.dumps(ast.literal_eval(str(log["config_str"].iloc[i])),
                              sort_keys=True) == params:
                    return int(i), "exact_score_algorithm_params"
            except Exception:                                    # noqa: BLE001
                pass
        return int(same[0]), "score_and_algorithm_tie_first"
    return int(cand[0]), "score_only_tie_first"


def _evaluate_dataset(task: Dict[str, Any]) -> Dict[str, Any]:
    """All 20 policies x 3 views for one dataset."""
    metric_names: List[str] = _S["metric_names"]
    mapping: List[Dict[str, Any]] = _S["mapping"]
    models = _S["models"]
    util: Dict[Tuple[str, str], np.ndarray] = _S["utility"]
    dsid = task["dataset_id"]
    droot = Path(task["dataset_dir"])
    seen_fams = set(FB.ALGORITHM_BASES)

    rows: List[Dict[str, Any]] = []
    for pol in mapping:
        rid = pol["reranker_id"]
        for v in VIEWS:
            rd = droot / "experiments" / ONLINE_DIR / pol["online_run_dir"] / v
            log = pd.read_csv(_ext(rd / CS.CANDIDATE_CSV))
            br = read_json(rd / "best_result.json") or {}
            sm = read_json(rd / "selected_metrics.json") or {}
            tim = read_json(rd / "timing_summary.json") or {}
            elig = log["valid"].map(CS.is_valid_flag).to_numpy(bool)
            ari = pd.to_numeric(log["ARI"], errors="coerce").to_numpy(float)
            base = log["algorithm"].astype(str).map(
                lambda s: CS.parse_config(s, "{}")["algorithm_base"]).to_numpy()
            unseen = ~np.isin(base, list(seen_fams))

            t0 = time.perf_counter()
            X, ncom, diag = OFB.build_online_X(
                trial_log=log, view_id=v, metric_names=metric_names,
                utility_vector=util[(dsid, v)],
                source=pol["utility_source"], k_mode=pol["k_mode"],
                actual_k=int(len(sm.get("selected_metrics") or [])) or 1)
            build_ms = (time.perf_counter() - t0) * 1000.0

            t1 = time.perf_counter()
            pred = np.asarray(models[rid].predict(X), dtype=np.float64)
            predict_ms = (time.perf_counter() - t1) * 1000.0

            t2 = time.perf_counter()
            idx = np.flatnonzero(elig)
            vsel = pred[idx]
            tied = idx[np.abs(vsel - vsel.min()) <= TIE_TOL]
            chosen = int(tied[0])
            argmin_ms = (time.perf_counter() - t2) * 1000.0
            if not elig[chosen]:
                raise AssertionError("ineligible candidate selected")

            # ---- identity frozen; ARI read only now ------------------------
            oi, omode = _original_index(log, br)
            orig_ari = float(ari[oi])
            rr_ari = float(ari[chosen])
            eligible_ari = ari[elig]
            vo_pos = int(idx[int(np.nanargmax(eligible_ari))])
            vo_ari = float(np.nanmax(eligible_ari))
            seen_elig = elig & ~unseen
            sf_vo = (float(np.nanmax(ari[seen_elig])) if seen_elig.any()
                     else float("nan"))
            head = vo_ari - orig_ari
            gain = rr_ari - orig_ari
            rec = (gain / head) if head > 1e-9 else float("nan")

            r: Dict[str, Any] = {
                "dataset_id": dsid, "split_id": SPLIT1,
                "family": task["family"], "subfamily": task["subfamily"],
                "view_id": v, "policy_id": pol["policy_id"],
                "reranker_id": rid, "utility_source": pol["utility_source"],
                "k_mode": pol["k_mode"],
                "weighting_mode": pol["weighting_mode"],
                "n_trial_records": int(len(log)),
                "n_eligible_candidates": int(elig.sum()),
                "n_failed_trials": int(((log["exception"].astype(str).str.strip()
                                         != "")
                                        & (log["exception"].astype(str).str.lower()
                                           != "nan")).sum()),
                "original_selected_index": oi,
                "original_selected_algorithm": str(log["algorithm"].iloc[oi]),
                "original_selected_ari": orig_ari,
                "original_match_mode": omode,
                "reranked_selected_index": chosen,
                "reranked_selected_algorithm": str(log["algorithm"].iloc[chosen]),
                "reranked_selected_ari": rr_ari,
                "visited_oracle_index": vo_pos,
                "visited_oracle_ari": vo_ari,
                "seen_family_visited_oracle_ari": sf_vo,
                "gain": gain, "headroom": head, "recovery": rec,
                "recovery_defined": bool(head > 1e-9),
                "reranker_feature_build_time_ms": build_ms,
                "reranker_predict_time_ms": predict_ms,
                "reranker_argmin_time_ms": argmin_ms,
                "reranker_total_incremental_time_ms": build_ms + predict_ms
                + argmin_ms,
                "original_runtime_sec": tim.get("runtime_sec"),
                "n_observed_metrics": diag["n_observed_metrics"],
                "n_unique_configs": diag["n_unique_configs"],
                # ---- algorithm domain shift ---------------------------------
                "n_unseen_family_candidates": int(unseen.sum()),
                "n_unseen_family_eligible": int((unseen & elig).sum()),
                "unseen_fraction": float(unseen.mean()),
                "unseen_fraction_eligible": float(
                    (unseen & elig).sum() / max(int(elig.sum()), 1)),
                "original_selected_unseen": bool(unseen[oi]),
                "reranked_selected_unseen": bool(unseen[chosen]),
                "oracle_selected_unseen": bool(unseen[vo_pos]),
            }
            vc = pd.Series(base).value_counts()
            for fam, n in vc.items():
                r[f"algcnt__{fam}"] = int(n)
            rows.append(r)
    return {"dataset_id": dsid, "rows": rows}


# ----------------------------------------------------------------- per-dataset
def dataset_result_path(dataset_dir: str) -> Path:
    return (Path(dataset_dir) / "experiments" / EXPERIMENT_ID / "result.json")


def dataset_is_complete(dataset_dir: str, schema_hash: str) -> bool:
    m = read_json(dataset_result_path(dataset_dir))
    if not m:
        return False
    return (m.get("schema_version") == RESULT_SCHEMA_VERSION
            and m.get("provenance", {}).get("feature_schema_hash") == schema_hash
            and m.get("n_policies") == PM.N_POLICIES
            and m.get("n_views") == len(VIEWS)
            and m.get("status") == "complete")


def write_dataset_result(dataset_dir: str, dsid: str, rows: List[Dict[str, Any]],
                         hashes: Dict[str, Any], commit: str) -> None:
    df = pd.DataFrame(rows)
    pol_block: Dict[str, Any] = {}
    for pid, g in df.groupby("policy_id", sort=False):
        views = {}
        for r in g.itertuples(index=False):
            views[r.view_id] = {
                "n_trial_records": int(r.n_trial_records),
                "n_eligible_candidates": int(r.n_eligible_candidates),
                "requested_trial_budget": 50,
                "timeout_bound_first": bool(r.n_trial_records < 50),
                "original_selected_index": int(r.original_selected_index),
                "original_selected_ari": float(r.original_selected_ari),
                "reranked_selected_index": int(r.reranked_selected_index),
                "reranked_selected_ari": float(r.reranked_selected_ari),
                "visited_oracle_index": int(r.visited_oracle_index),
                "visited_oracle_ari": float(r.visited_oracle_ari),
                "seen_family_visited_oracle_ari": (
                    None if not np.isfinite(r.seen_family_visited_oracle_ari)
                    else float(r.seen_family_visited_oracle_ari)),
                "gain": float(r.gain), "headroom": float(r.headroom),
                "recovery": (None if not r.recovery_defined
                             else float(r.recovery)),
                "n_unseen_family_candidates": int(r.n_unseen_family_candidates),
                "unseen_fraction": float(r.unseen_fraction),
                "reranker_feature_build_time_ms": float(
                    r.reranker_feature_build_time_ms),
                "reranker_predict_time_ms": float(r.reranker_predict_time_ms),
                "reranker_argmin_time_ms": float(r.reranker_argmin_time_ms),
                "reranker_total_incremental_time_ms": float(
                    r.reranker_total_incremental_time_ms),
                "original_runtime_sec": (None if pd.isna(r.original_runtime_sec)
                                         else float(r.original_runtime_sec)),
            }
        ob = g.loc[g["original_selected_ari"].idxmax()]
        rb = g.loc[g["reranked_selected_ari"].idxmax()]
        vb = g.loc[g["visited_oracle_ari"].idxmax()]
        obv, rbv, vbv = (float(ob["original_selected_ari"]),
                         float(rb["reranked_selected_ari"]),
                         float(vb["visited_oracle_ari"]))
        hb = vbv - obv
        pol_block[pid] = {
            "utility_source": g["utility_source"].iloc[0],
            "k_mode": g["k_mode"].iloc[0],
            "weighting_mode": g["weighting_mode"].iloc[0],
            "reranker_id": g["reranker_id"].iloc[0],
            "views": views,
            "original_bestview_ari": obv,
            "reranked_bestview_ari": rbv,
            "visited_oracle_bestview_ari": vbv,
            "bestview_gain": rbv - obv,
            "bestview_headroom": hb,
            "bestview_recovery": (None if hb <= 1e-9 else (rbv - obv) / hb),
            "winning_original_view": str(ob["view_id"]),
            "winning_reranked_view": str(rb["view_id"]),
            "winning_oracle_view": str(vb["view_id"]),
            "three_view_reranker_time_ms": float(
                g["reranker_total_incremental_time_ms"].sum()),
        }
    target = dataset_result_path(dataset_dir)
    ensure_dir(target.parent)
    write_json_atomic(target, {
        "schema_version": RESULT_SCHEMA_VERSION, "status": "complete",
        "experiment_id": EXPERIMENT_ID, "dataset_id": dsid,
        "family": rows[0]["family"], "subfamily": rows[0]["subfamily"],
        "split_id": SPLIT1, "n_policies": PM.N_POLICIES, "n_views": len(VIEWS),
        "policies": pol_block, "provenance": {**hashes, "source_commit": commit},
    })


# ---------------------------------------------------------------------- main
def load_utility_context(repo: Path, metric_names: List[str]
                         ) -> Dict[Tuple[str, str], np.ndarray]:
    """120-dim final Split-1 context: 60 MLP then 60 KNN, canonical order."""
    mlp = pd.read_csv(_ext(repo / MLP_SPLIT1_PRED),
                      usecols=["dataset_id", "view_mode"]
                      + [f"pred_{m}" for m in metric_names])
    knn = pd.read_csv(_ext(repo / KNN_SPLIT1_CTX))
    if len(mlp) != 3165 or len(knn) != 3165:
        raise SystemExit(f"utility context rows: mlp={len(mlp)} knn={len(knn)}")
    m = mlp.rename(columns={"view_mode": "view_id"}).merge(
        knn, on=["dataset_id", "view_id"], how="inner")
    if len(m) != 3165:
        raise SystemExit(f"MLP/KNN utility join produced {len(m)} rows")
    mc = [f"pred_{x}" for x in metric_names]
    kc = [f"knn_utility__{x}" for x in metric_names]
    blk = np.hstack([m[mc].to_numpy(np.float32), m[kc].to_numpy(np.float32)])
    if not np.isfinite(blk).all():
        raise SystemExit("non-finite value in the Split-1 utility context")
    return {(d, v): blk[i] for i, (d, v) in
            enumerate(zip(m["dataset_id"], m["view_id"]))}


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", default=".")
    p.add_argument("--mode", choices=["dry", "wet"], default="dry")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit-subfamilies", type=int, default=None)
    p.add_argument("--limit-datasets", type=int, default=None)
    args = p.parse_args(argv)

    repo = Path(args.repo_root).resolve()
    prep = repo / PREP_REL
    groot = ensure_dir(repo / GLOBAL_ROOT_REL)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                            capture_output=True, text=True).stdout.strip()
    metric_names = CS.load_metric_names(repo)
    mapping = PM.build_mapping()
    reg = read_json(prep / "final_reranker_registry.json") or {}
    hashes = {
        "experiment_id": EXPERIMENT_ID,
        "policy_mapping_hash": PM.mapping_hash(mapping),
        "registry_semantics_hash": PM.registry_semantics_hash(),
        "feature_schema_hash": "C2_LOCAL_UTIL/347",
        "model_config_hash": MR.config_hash(),
        "metric_order_hash": CS.metric_order_hash(metric_names),
        "knn_context_sha": (read_json(
            prep / "split1_knn_utility_context_manifest.json") or {}
        ).get("file_sha256_16"),
    }
    if hashes["policy_mapping_hash"] != "f8c3acc507d9a5e6":
        raise SystemExit("policy mapping hash drifted")
    if hashes["registry_semantics_hash"] != "beaccd4a0cb61c46":
        raise SystemExit("registry semantics hash drifted")

    print("=" * 78)
    print(f"STAGE 2B-5B  --  {args.mode.upper()} RUN  --  {EXPERIMENT_ID}")
    print("=" * 78, flush=True)
    print(f"  mapping {hashes['policy_mapping_hash']} | registry "
          f"{hashes['registry_semantics_hash']} | model "
          f"{hashes['model_config_hash']} | knn ctx {hashes['knn_context_sha']}",
          flush=True)

    util = load_utility_context(repo, metric_names)
    print(f"  Split-1 utility context: {len(util)} dataset/view vectors "
          f"(60 MLP + 60 KNN)", flush=True)

    assign = pd.read_csv(_ext(repo / SPLIT_CSV_REL))
    s1 = assign[assign["split_id"] == SPLIT1]
    groups = list(s1.groupby(["family_id", "subfamily_id"], sort=True))
    if args.limit_subfamilies:
        groups = groups[: int(args.limit_subfamilies)]
    n_sub = len(groups)

    payload = {"metric_names": metric_names, "mapping": mapping,
               "prep_root": str(prep), "utility": util}

    if args.mode == "dry":
        return _dry(repo, groot, groups, s1, mapping, util, hashes, commit,
                    args, payload)

    ensure_dir(groot / "subfamily_progress")
    t_all = time.time()
    all_rows: List[Dict[str, Any]] = []
    done_ds = 0
    for si, ((fam, sub), g) in enumerate(groups, 1):
        recs = [{"dataset_dir": r["dataset_dir"], "dataset_id": r["dataset_id"],
                 "family": fam, "subfamily": sub} for _, r in g.iterrows()]
        if args.limit_datasets:
            recs = recs[: int(args.limit_datasets)]
        todo = [r for r in recs
                if not dataset_is_complete(r["dataset_dir"],
                                           hashes["feature_schema_hash"])]
        print("=" * 64)
        print(f"[START] Subfamily {si}/{n_sub}")
        print(f"family={fam}")
        print(f"subfamily={sub}")
        print(f"datasets={len(recs)} (pending {len(todo)})")
        print(f"workers={args.workers}")
        print("=" * 64, flush=True)
        t_sub = time.time()
        rows: List[Dict[str, Any]] = []
        fails: List[Dict[str, Any]] = []
        if todo:
            with ProcessPoolExecutor(max_workers=args.workers,
                                     initializer=_init_worker,
                                     initargs=(payload,)) as pool:
                for k, res in enumerate(pool.map(_evaluate_dataset, todo), 1):
                    write_dataset_result(
                        next(r["dataset_dir"] for r in todo
                             if r["dataset_id"] == res["dataset_id"]),
                        res["dataset_id"], res["rows"], hashes, commit)
                    rows.extend(res["rows"])
                    if k % 20 == 0 or k == len(todo):
                        print(f"   {k}/{len(todo)} datasets", flush=True)
        for r in recs:
            if dataset_is_complete(r["dataset_dir"], hashes["feature_schema_hash"]):
                continue
            fails.append({"dataset_id": r["dataset_id"], "reason": "incomplete"})
        if not rows:                      # fully resumed subfamily
            rows = _reload_subfamily_rows(recs)
        sdf = pd.DataFrame(rows)
        spath = _write_subfamily_summary(g, fam, sub, sdf, fails, commit,
                                         hashes, args.workers)
        all_rows.extend(rows)
        done_ds += len(recs)
        mean_rr = float(sdf.groupby(["dataset_id", "policy_id"])
                        ["reranked_selected_ari"].max().mean())
        print("=" * 64)
        print(f"[DONE] Subfamily {si}/{n_sub}")
        print(f"family={fam}")
        print(f"subfamily={sub}")
        print(f"datasets={len(recs)}/{len(recs)}")
        print(f"runs={len(sdf)}/{len(recs)*PM.N_POLICIES*len(VIEWS)}")
        print(f"failures={len(fails)}")
        print(f"elapsed={time.time()-t_sub:.0f}s")
        print(f"mean_reranked_bestview={mean_rr:.5f}")
        print(f"summary_path={spath}")
        print("=" * 64, flush=True)
        if si < n_sub:
            print(f"[NEXT] Starting subfamily {si+1}/{n_sub}", flush=True)

    df = pd.DataFrame(all_rows)
    df.to_csv(_ext(groot / "per_run_reranker_results.csv.gz"), **GZ)
    write_json_atomic(groot / "experiment_manifest.json", {
        "experiment_id": EXPERIMENT_ID, "mode": "wet",
        "n_subfamilies": n_sub, "n_datasets": done_ds, "n_runs": int(len(df)),
        "hashes": hashes, "workers": args.workers,
        "runtime_sec": time.time() - t_all, "source_commit": commit})
    print(f"\n  WET RUN COMPLETE: {done_ds} datasets, {len(df)} runs, "
          f"{time.time()-t_all:.0f}s", flush=True)
    return 0


def _reload_subfamily_rows(recs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rebuild run rows from already-written per-dataset result.json files."""
    rows: List[Dict[str, Any]] = []
    for r in recs:
        m = read_json(dataset_result_path(r["dataset_dir"])) or {}
        for pid, pb in (m.get("policies") or {}).items():
            for v, vb in pb["views"].items():
                rows.append({
                    "dataset_id": m["dataset_id"], "split_id": SPLIT1,
                    "family": m["family"], "subfamily": m["subfamily"],
                    "view_id": v, "policy_id": pid,
                    "reranker_id": pb["reranker_id"],
                    "utility_source": pb["utility_source"],
                    "k_mode": pb["k_mode"],
                    "weighting_mode": pb["weighting_mode"],
                    "n_trial_records": vb["n_trial_records"],
                    "n_eligible_candidates": vb["n_eligible_candidates"],
                    "original_selected_ari": vb["original_selected_ari"],
                    "reranked_selected_ari": vb["reranked_selected_ari"],
                    "visited_oracle_ari": vb["visited_oracle_ari"],
                    "seen_family_visited_oracle_ari":
                        vb["seen_family_visited_oracle_ari"],
                    "gain": vb["gain"], "headroom": vb["headroom"],
                    "recovery": vb["recovery"],
                    "recovery_defined": vb["recovery"] is not None,
                    "n_unseen_family_candidates":
                        vb["n_unseen_family_candidates"],
                    "unseen_fraction": vb["unseen_fraction"],
                    "reranker_feature_build_time_ms":
                        vb["reranker_feature_build_time_ms"],
                    "reranker_predict_time_ms": vb["reranker_predict_time_ms"],
                    "reranker_argmin_time_ms": vb["reranker_argmin_time_ms"],
                    "reranker_total_incremental_time_ms":
                        vb["reranker_total_incremental_time_ms"],
                    "original_runtime_sec": vb["original_runtime_sec"],
                })
    return rows


def _write_subfamily_summary(g, fam, sub, sdf, fails, commit, hashes,
                             workers) -> str:
    """Repository-format summaries, namespaced by the experiment id."""
    sroot = Path(g.iloc[0]["dataset_dir"]).parent
    out = ensure_dir(sroot / "experiment_execution_summaries" / EXPERIMENT_ID)
    write_csv_atomic(out / "execution_summary.csv", sdf)
    bv = sdf.groupby(["dataset_id", "policy_id"]).agg(
        original_bestview=("original_selected_ari", "max"),
        reranked_bestview=("reranked_selected_ari", "max"),
        oracle_bestview=("visited_oracle_ari", "max")).reset_index()
    per_pol = bv.groupby("policy_id").agg(
        n_datasets=("original_bestview", "size"),
        original_bestview_mean=("original_bestview", "mean"),
        original_bestview_median=("original_bestview", "median"),
        reranked_bestview_mean=("reranked_bestview", "mean"),
        reranked_bestview_median=("reranked_bestview", "median"),
        oracle_bestview_mean=("oracle_bestview", "mean")).reset_index()
    per_pol["mean_gain"] = (per_pol["reranked_bestview_mean"]
                            - per_pol["original_bestview_mean"])
    write_csv_atomic(out / "execution_summary_best_view.csv", per_pol)
    av = sdf.groupby(["policy_id", "view_id"]).agg(
        mean_original=("original_selected_ari", "mean"),
        mean_reranked=("reranked_selected_ari", "mean"),
        mean_oracle=("visited_oracle_ari", "mean"),
        mean_eligible=("n_eligible_candidates", "mean"),
        mean_reranker_ms=("reranker_total_incremental_time_ms", "mean")
    ).reset_index()
    write_csv_atomic(out / "execution_summary_all_views.csv", av)
    write_csv_atomic(out / "experiment_failures.csv",
                     pd.DataFrame(fails or [{"dataset_id": "", "reason": ""}]))
    summary = {
        "n_datasets": int(bv["dataset_id"].nunique()),
        "n_runs": int(len(sdf)), "n_failures": len(fails),
        "mean_original_bestview": float(bv["original_bestview"].mean()),
        "mean_reranked_bestview": float(bv["reranked_bestview"].mean()),
        "mean_oracle_bestview": float(bv["oracle_bestview"].mean()),
        "mean_gain": float((bv["reranked_bestview"]
                            - bv["original_bestview"]).mean()),
        "best_original_policy": per_pol.loc[
            per_pol["original_bestview_mean"].idxmax(), "policy_id"],
        "best_reranked_policy": per_pol.loc[
            per_pol["reranked_bestview_mean"].idxmax(), "policy_id"],
        "reranked_policy_vbs": float(
            bv.groupby("dataset_id")["reranked_bestview"].max().mean()),
        "hybrid_visited_oracle": float(
            bv.groupby("dataset_id")["oracle_bestview"].max().mean()),
        "mean_eligible_candidates": float(sdf["n_eligible_candidates"].mean()),
        "mean_reranker_ms": float(
            sdf["reranker_total_incremental_time_ms"].mean()),
        "p95_reranker_ms": float(
            sdf["reranker_total_incremental_time_ms"].quantile(0.95)),
        "timeout_fraction": float((sdf["n_trial_records"] < 50).mean()),
        "mean_unseen_fraction": float(sdf["unseen_fraction"].mean())
        if "unseen_fraction" in sdf else None,
    }
    write_json_atomic(out / "execution_summary.json", {
        "config": {"experiment_id": EXPERIMENT_ID, "split_id": SPLIT1,
                   "workers": workers, **hashes},
        "family_id": fam, "subfamily_id": sub, "summary": summary,
        "created_at": time.time()})
    write_json_atomic(out / "checkpoint.json", {
        "config": {"experiment_id": EXPERIMENT_ID, "split_id": SPLIT1},
        "summary": summary, "n_records": int(len(sdf)),
        "updated_at": time.time()})
    return str(out)


def _dry(repo, groot, groups, s1, mapping, util, hashes, commit, args, payload):
    """Validate discovery, policy resolution, features, orchestration, memory.

    Deliberately includes runs that contain unseen-family (OPTICS/Spectral)
    candidates, so feature construction is exercised on the out-of-domain cases
    before the wet run rather than meeting them for the first time at scale.
    """
    from models.ClustOpt_Candidate_Reranker.training.run_masking_feasibility import (
        _memory_probes,
    )
    mem, sysmem = _memory_probes()
    t0 = time.time()
    problems: List[Dict[str, Any]] = []
    plan: List[Dict[str, Any]] = []
    n_runs = 0
    for (fam, sub), g in groups:
        plan.append({"family_id": fam, "subfamily_id": sub,
                     "n_datasets": int(len(g)),
                     "planned_runs": int(len(g) * PM.N_POLICIES * len(VIEWS))})
        n_runs += len(g) * PM.N_POLICIES * len(VIEWS)
    # ---- policy path resolution on a sample, incl. missing-path detection ---
    sample = s1.head(args.limit_datasets or 24)
    for _, r in sample.iterrows():
        for pol in mapping:
            for v in VIEWS:
                rd = (Path(r["dataset_dir"]) / "experiments" / ONLINE_DIR
                      / pol["online_run_dir"] / v / CS.CANDIDATE_CSV)
                if not path_exists(rd):
                    problems.append({"dataset_id": r["dataset_id"],
                                     "policy_id": pol["policy_id"],
                                     "view_id": v, "issue": "missing trial log"})
                if (r["dataset_id"], v) not in util:
                    problems.append({"dataset_id": r["dataset_id"],
                                     "policy_id": pol["policy_id"],
                                     "view_id": v,
                                     "issue": "missing utility context"})
    # ---- feature construction, forcing unseen-family coverage ---------------
    metric_names = payload["metric_names"]
    seen = set(FB.ALGORITHM_BASES)
    probes, n_unseen_probes = [], 0
    for _, r in sample.head(8).iterrows():
        for pol in mapping[:4] + mapping[-4:]:
            for v in VIEWS:
                rd = (Path(r["dataset_dir"]) / "experiments" / ONLINE_DIR
                      / pol["online_run_dir"] / v)
                log = pd.read_csv(_ext(rd / CS.CANDIDATE_CSV))
                sm = read_json(rd / "selected_metrics.json") or {}
                X, ncom, diag = OFB.build_online_X(
                    trial_log=log, view_id=v, metric_names=metric_names,
                    utility_vector=util[(r["dataset_id"], v)],
                    source=pol["utility_source"], k_mode=pol["k_mode"],
                    actual_k=int(len(sm.get("selected_metrics") or [])) or 1)
                OFB.assert_no_hidden_exposure(X, ncom, metric_names, log)
                if X.shape[1] != OFB.EXPECTED_DIM:
                    problems.append({"dataset_id": r["dataset_id"],
                                     "policy_id": pol["policy_id"],
                                     "view_id": v, "issue": "bad feature dim"})
                if diag["n_unseen_algorithm_candidates"] > 0:
                    n_unseen_probes += 1
                probes.append({"n_candidates": diag["n_candidates"],
                               "n_unseen": diag["n_unseen_algorithm_candidates"],
                               "n_observed_metrics": diag["n_observed_metrics"]})
    if n_unseen_probes == 0:
        problems.append({"issue": "dry run exercised no unseen-family case"})

    # ---- 8-worker orchestration + memory probe on a real subfamily ---------
    (fam, sub), g = groups[0]
    recs = [{"dataset_dir": r["dataset_dir"], "dataset_id": r["dataset_id"],
             "family": fam, "subfamily": sub} for _, r in g.iterrows()]
    before = mem()
    tw = time.time()
    with ProcessPoolExecutor(max_workers=args.workers,
                             initializer=_init_worker,
                             initargs=(payload,)) as pool:
        got = list(pool.map(_evaluate_dataset, recs))
    worker_sec = time.time() - tw
    peak_parent = mem()
    total, avail = sysmem()
    n_probe_runs = sum(len(x["rows"]) for x in got)
    per_ds = worker_sec / max(len(recs), 1)
    est_hours = per_ds * int(s1["dataset_id"].nunique()) / 3600

    pdf = pd.DataFrame(probes)
    payload_out = {
        "experiment_id": EXPERIMENT_ID, "mode": "dry",
        "n_subfamilies": len(groups),
        "n_datasets": int(s1["dataset_id"].nunique()),
        "n_policies": PM.N_POLICIES, "n_views": len(VIEWS),
        "total_planned_runs": n_runs,
        "policy_paths_checked": int(len(sample) * PM.N_POLICIES * len(VIEWS)),
        "feature_probes": int(len(probes)),
        "feature_probes_with_unseen_family": n_unseen_probes,
        "probe_mean_unseen_per_run": float(pdf["n_unseen"].mean()),
        "probe_max_unseen_per_run": int(pdf["n_unseen"].max()),
        "workers": args.workers,
        "worker_probe_subfamily": f"{fam}/{sub}",
        "worker_probe_datasets": len(recs),
        "worker_probe_runs": n_probe_runs,
        "worker_probe_sec": round(worker_sec, 1),
        "sec_per_dataset": round(per_ds, 2),
        "estimated_wet_hours": round(est_hours, 2),
        "parent_rss_before_mb": round(before, 1),
        "parent_rss_after_mb": round(peak_parent, 1),
        "system_total_mb": round(total, 1),
        "system_available_mb": round(avail, 1),
        "workers_safe": bool(avail > 1500),
        "problems": problems[:50], "n_problems": len(problems),
        "plan": plan, "hashes": hashes, "source_commit": commit,
        "note": "DRY RUN -- no per-dataset result.json written, no scientific "
                "outcome reported",
    }
    write_json_atomic(groot / "dry_run_plan.json", payload_out)
    print(f"  planned runs        : {n_runs}")
    print(f"  policy paths checked: {payload_out['policy_paths_checked']} "
          f"(problems {len(problems)})")
    print(f"  feature probes      : {len(probes)} "
          f"({n_unseen_probes} contained unseen-family candidates, "
          f"mean {pdf['n_unseen'].mean():.1f} per run)")
    print(f"  8-worker probe      : {len(recs)} datasets, {n_probe_runs} runs, "
          f"{worker_sec:.0f}s -> {per_ds:.2f}s/dataset")
    print(f"  memory              : parent {peak_parent:.0f}MB, "
          f"available {avail:.0f}MB, safe={payload_out['workers_safe']}")
    print(f"  estimated wet run   : {est_hours:.2f} h")
    if problems:
        print(f"  DRY RUN PROBLEMS: {len(problems)} -- see dry_run_plan.json",
              flush=True)
        return 2
    print(f"\n  DRY RUN PASS ({time.time()-t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
