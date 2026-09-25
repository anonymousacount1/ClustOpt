"""Stage 2B-4: three dedicated C2 rerankers x 15 outer folds = 45 fits.

Each model sees the FULL outer-training slate population under a single regime --
the opposite of Stage 2B-3, where each slate carried one of ten regimes. Holding
context, model, target and evaluation regime fixed, the only difference from the
unified run is dedicated vs mixed-regime training, which is what makes the
decomposition exact.

The design matrix is built with the Stage-2B3 builder using a CONSTANT per-slate
regime code, so the schema is byte-identical to Stage-2B2 ``C2_LOCAL_UTIL``
(347 columns, same names and order). MLP_TOP10 is not retrained: Stage 2B-2
already provides its exact dedicated C2 control.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from models.Clustering_Repository_Builder.experiments.experiment_execution.output_writer import (  # noqa: E402,E501
    _ext, ensure_dir, path_exists, read_json, write_csv_atomic, write_json_atomic,
)
from models.ClustOpt_Candidate_Reranker.data import candidate_eligibility as CE  # noqa: E402,E501
from models.ClustOpt_Candidate_Reranker.training import evaluate_fold as EV  # noqa: E402,E501
from models.ClustOpt_Candidate_Reranker.training import feature_builder as FB  # noqa: E402,E501
from models.ClustOpt_Candidate_Reranker.training import model_registry as MR  # noqa: E402,E501
from models.ClustOpt_Candidate_Reranker.training import train_fold as TF  # noqa: E402,E501
from models.ClustOpt_Candidate_Reranker.training.context_ablation.run_context_ablation import (  # noqa: E402,E501
    WorkerLock,
)
from models.ClustOpt_Candidate_Reranker.training.run_masking_feasibility import (  # noqa: E402,E501
    CandidateStore, load_fold_utilities, resolve_masks, _memory_probes,
)
from models.ClustOpt_Candidate_Reranker.training.unified_masking import (  # noqa: E402,E501
    unified_feature_builder as UFB,
)

B1_REL = ("results_analysis/clustopt_candidate_reranker/"
          "stage2b1_masking_feasibility")
OUT_REL = ("results_analysis/clustopt_candidate_reranker/"
           "stage2b4_same_context_controls")
OUTER_SPLITS: Tuple[int, ...] = tuple(range(2, 17))
GZ = {"index": False, "compression": "gzip"}
LOCK_NAME = "stage2b4_worker.pid"

# Exactly three new controls. MLP_TOP10's dedicated C2 already exists.
NEW_REGIMES: Tuple[Dict[str, str], ...] = (
    {"condition_id": "KNN_TOP1", "source": "knn", "k_mode": "top1",
     "reason": "largest Stage-2B3 pipeline gain over dedicated LOCAL"},
    {"condition_id": "MLP_TOP5", "source": "mlp", "k_mode": "top5",
     "reason": "historical ClustOpt operating point"},
    {"condition_id": "KNN_TOP10", "source": "knn", "k_mode": "top10",
     "reason": "first configuration to cross Recovery_RAW >= 0.45"},
)
FROZEN_CONTROL = "MLP_TOP10"


def unit_dir(out: Path, cid: str, s: int) -> Path:
    return out / "outer_fold_predictions" / cid / f"fold_{s:02d}"


def unit_is_complete(out: Path, cid: str, s: int, uhash: str) -> bool:
    m = read_json(unit_dir(out, cid, s) / "unit_manifest.json")
    return bool(m and m.get("status") == "complete"
                and m.get("unit_hash") == uhash
                and m.get("leakage_checks_pass"))


def _fold_arrays(repo: Path, store: CandidateStore, s: int):
    fu = load_fold_utilities(repo, s, store.metric_names)
    tr_slates = np.flatnonzero(store.slate_split != s)
    te_slates = np.flatnonzero(store.slate_split == s)
    key = store.slate_key.to_numpy()
    tr_pos = pd.Index(key[tr_slates]).get_indexer(fu["train_key"])
    te_pos = pd.Index(key[te_slates]).get_indexer(fu["test_key"])
    if (tr_pos < 0).any() or (te_pos < 0).any():
        raise AssertionError(f"fold {s}: utility identifiers misaligned")
    tr_order, te_order = np.argsort(tr_pos), np.argsort(te_pos)
    util_tr = np.hstack([fu["train_util"]["mlp"][tr_order],
                         fu["train_util"]["knn"][tr_order]]).astype(np.float32)
    util_te = np.hstack([fu["test_util"]["mlp"][te_order],
                         fu["test_util"]["knn"][te_order]]).astype(np.float32)
    tr_rows_all = store.rows_for_slates(tr_slates)
    tr_rows = tr_rows_all[store.eligible[tr_rows_all]]
    te_rows = store.rows_for_slates(te_slates)
    tr_slate_pos = pd.Series(np.arange(len(tr_slates)), index=tr_slates
                             ).reindex(store.slate_of_row[tr_rows]
                                       ).to_numpy().astype(np.int64)
    te_slate_pos = np.repeat(np.arange(len(te_slates)), 32)
    return {"fu": fu, "tr_slates": tr_slates, "te_slates": te_slates,
            "tr_order": tr_order, "te_order": te_order,
            "util_tr": util_tr, "util_te": util_te,
            "tr_rows": tr_rows, "te_rows": te_rows,
            "tr_slate_pos": tr_slate_pos, "te_slate_pos": te_slate_pos}


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", default=".")
    p.add_argument("--preflight", action="store_true")
    p.add_argument("--conditions", nargs="*", default=None)
    p.add_argument("--splits", type=int, nargs="*", default=None)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)

    repo = Path(args.repo_root).resolve()
    out = ensure_dir(repo / OUT_REL)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                            capture_output=True, text=True).stdout.strip()

    print("=" * 78)
    print("STAGE 2B-4  --  SAME-CONTEXT DEDICATED CONTROLS")
    print("=" * 78, flush=True)
    print(f"  context C2 LOCAL_UTIL ({UFB.EXPECTED_DIM} dims) | "
          f"model {MR.config_hash()} | target T2_SLATE_REGRET | "
          f"{len(NEW_REGIMES)} new regimes ({FROZEN_CONTROL} NOT retrained)",
          flush=True)

    with WorkerLock(out / LOCK_NAME):
        store = CandidateStore(repo)
        regs = list(NEW_REGIMES)
        if args.conditions:
            regs = [r for r in regs if r["condition_id"] in args.conditions]
        splits = args.splits or list(OUTER_SPLITS)
        if 1 in splits:
            raise SystemExit("Split 1 is forbidden")

        write_csv_atomic(out / "configuration_registry.csv", pd.DataFrame([
            {**r, "input_dim": UFB.EXPECTED_DIM, "context": UFB.CONDITION,
             "training": "dedicated single-regime, full slate population",
             "model_family": MR.MODEL_FAMILY,
             "model_config_hash": MR.config_hash(),
             "target": "T2_SLATE_REGRET"} for r in NEW_REGIMES]))

        if args.preflight:
            return _preflight(repo, store, out)

        t_all, n_done, n_skip = time.time(), 0, 0
        for s in splits:
            A = _fold_arrays(repo, store, s)
            fu = A["fu"]
            y_tr = TF.target_regret(store.ari[A["tr_rows"]],
                                    store.oracle_row[A["tr_rows"]])
            te_starts = np.cumsum(
                np.r_[0, store.slate_size[A["te_slates"]]])[:-1]
            te_sizes = store.slate_size[A["te_slates"]]
            ari_te = store.ari[A["te_rows"]]
            orc_te = store.oracle_row[A["te_rows"]]
            elig_te = store.eligible[A["te_rows"]]
            ds_te = store.dataset_id[A["te_rows"]][te_starts]
            vw_te = store.view_id[A["te_rows"]][te_starts]
            orc_sl = orc_te[te_starts]

            for reg in regs:
                cid, sx, km = reg["condition_id"], reg["source"], reg["k_mode"]
                uhash = MR.unit_hash(f"DEDICATED_C2__{cid}", s, UFB.CONDITION,
                                     CE.definition_hash(),
                                     f"{fu['train_source']} / {fu['test_source']}")
                if not args.overwrite and unit_is_complete(out, cid, s, uhash):
                    n_skip += 1
                    continue
                t0 = time.time()
                n_tr, n_te = len(A["tr_slates"]), len(A["te_slates"])
                mtr = resolve_masks(fu["train_util"][sx][A["tr_order"]],
                                    store.metric_names, km)
                mte = resolve_masks(fu["test_util"][sx][A["te_order"]],
                                    store.metric_names, km)
                sc_tr = np.full(n_tr, FB.SOURCES.index(sx), np.int64)
                kc_tr = np.full(n_tr, FB.K_MODES.index(km), np.int64)
                sc_te = np.full(n_te, FB.SOURCES.index(sx), np.int64)
                kc_te = np.full(n_te, FB.K_MODES.index(km), np.int64)

                X_tr, n_common = UFB.build_unified_X(
                    view_codes=store.view_code[A["tr_rows"]],
                    cat_rows=store.cat_row[A["tr_rows"]],
                    cat_matrix=store.cat_matrix,
                    partition=store.partition[A["tr_rows"]],
                    cvi=store.cvi[A["tr_rows"]],
                    cvi_valid=store.cvi_valid[A["tr_rows"]],
                    observability=mtr["mask"], actual_k=mtr["actual_k"],
                    source_code=sc_tr, kmode_code=kc_tr,
                    slate_pos=A["tr_slate_pos"], util=A["util_tr"])
                UFB.assert_regime_identity(X_tr, A["tr_slate_pos"], sc_tr,
                                           kc_tr, n_common)
                X_te, _ = UFB.build_unified_X(
                    view_codes=store.view_code[A["te_rows"]],
                    cat_rows=store.cat_row[A["te_rows"]],
                    cat_matrix=store.cat_matrix,
                    partition=store.partition[A["te_rows"]],
                    cvi=store.cvi[A["te_rows"]],
                    cvi_valid=store.cvi_valid[A["te_rows"]],
                    observability=mte["mask"], actual_k=mte["actual_k"],
                    source_code=sc_te, kmode_code=kc_te,
                    slate_pos=A["te_slate_pos"], util=A["util_te"])
                UFB.assert_no_hidden_leakage(X_te, mte["mask"],
                                             A["te_slate_pos"], n_common)
                res = TF.run_unit(condition={"kind": "DEPLOYABLE"}, X_tr=X_tr,
                                  y_tr=y_tr, X_te=X_te, eligible_te=elig_te,
                                  slate_starts=te_starts, slate_sizes=te_sizes,
                                  ari_te=ari_te, oracle_te=orc_te)
                n_train_rows = int(X_tr.shape[0])
                del X_tr, X_te

                m = EV.slate_metrics(res["selected_ari"], orc_sl, vw_te, ds_te)
                b1 = pd.read_csv(_ext(repo / B1_REL / "outer_fold_predictions"
                                      / cid / f"fold_{s:02d}"
                                      / "slate_selections.csv.gz"))
                if not np.array_equal(b1["dataset_id"].to_numpy(), ds_te) or \
                   not np.array_equal(b1["view_id"].to_numpy(), vw_te):
                    raise AssertionError(f"{cid}: frozen slate order differs")
                if not np.allclose(b1["candidate_oracle_ari"].to_numpy(),
                                   orc_sl, atol=1e-12):
                    raise AssertionError(f"{cid}: frozen oracle differs")
                for lab in ("raw", "softmax"):
                    bm = EV.slate_metrics(b1[f"b_{lab}_ari"].to_numpy(float),
                                          orc_sl, vw_te, ds_te)
                    m[f"b_{lab}_view"] = bm["mean_selected_ari"]
                    m[f"b_{lab}_bestview"] = bm["mean_bestview_selected_ari"]
                    m[f"recovery_{lab}"] = EV.recovery(
                        m["mean_bestview_selected_ari"],
                        bm["mean_bestview_selected_ari"],
                        m["mean_bestview_candidate_oracle"])
                    m[f"beats_{lab}"] = bool(m["mean_bestview_selected_ari"]
                                             > bm["mean_bestview_selected_ari"])
                dl = read_json(repo / B1_REL / "outer_fold_predictions" / cid
                               / f"fold_{s:02d}" / "unit_manifest.json")
                m["dedicated_local_bestview"] = float(
                    dl["metrics"]["mean_bestview_selected_ari"])

                d = ensure_dir(unit_dir(out, cid, s))
                pd.DataFrame({
                    "dataset_id": ds_te, "split_id": s, "view_id": vw_te,
                    "selected_candidate_index":
                        store.candidate_index[A["te_rows"]][res["chosen"]],
                    "selected_ari": res["selected_ari"],
                    "candidate_oracle_ari": orc_sl,
                    "oracle_regret": orc_sl - res["selected_ari"],
                    "b_raw_ari": b1["b_raw_ari"].to_numpy(),
                    "b_softmax_ari": b1["b_softmax_ari"].to_numpy(),
                    "dedicated_local_selected_ari": b1["selected_ari"].to_numpy(),
                }).to_csv(_ext(d / "slate_selections.csv.gz"), **GZ)
                write_json_atomic(d / "unit_manifest.json", {
                    "status": "complete", "unit_hash": uhash,
                    "condition_id": cid, "outer_split": int(s),
                    "utility_source": sx, "k_mode": km,
                    "training": "dedicated single-regime, full slate population",
                    "context": UFB.CONDITION, "n_features": UFB.EXPECTED_DIM,
                    "n_train_rows": n_train_rows,
                    "n_train_slates": int(n_tr),
                    "eligibility_hash": CE.definition_hash(),
                    "utility_source_train": fu["train_source"],
                    "utility_source_test": fu["test_source"],
                    "model_config_hash": MR.config_hash(),
                    "target": "T2_SLATE_REGRET",
                    "fit_info": res["fit_info"], "metrics": m,
                    "leakage_checks_pass": True, "split_1_used": False,
                    "source_commit": commit,
                })
                n_done += 1
                print(f"   [{cid:<10} fold {s:02d}] rows={n_train_rows} "
                      f"BV={m['mean_bestview_selected_ari']:.4f} "
                      f"rec={m['recovery_raw']:+.4f} "
                      f"(local {m['dedicated_local_bestview']:.4f}) "
                      f"({time.time()-t0:.0f}s)", flush=True)
            del A

        print(f"\n  {n_done} units executed, {n_skip} skipped, "
              f"{time.time()-t_all:.0f}s", flush=True)
    return 0


def _preflight(repo: Path, store: CandidateStore, out: Path) -> int:
    """Resource-only probe on ONE outer-training population per regime."""
    import gc
    mem, sysmem = _memory_probes()
    s = 2
    A = _fold_arrays(repo, store, s)
    fu = A["fu"]
    y = TF.target_regret(store.ari[A["tr_rows"]], store.oracle_row[A["tr_rows"]])
    rows = []
    for reg in NEW_REGIMES:
        cid, sx, km = reg["condition_id"], reg["source"], reg["k_mode"]
        gc.collect()
        before = mem()
        n_tr = len(A["tr_slates"])
        mtr = resolve_masks(fu["train_util"][sx][A["tr_order"]],
                            store.metric_names, km)
        t0 = time.time()
        X, n_common = UFB.build_unified_X(
            view_codes=store.view_code[A["tr_rows"]],
            cat_rows=store.cat_row[A["tr_rows"]], cat_matrix=store.cat_matrix,
            partition=store.partition[A["tr_rows"]],
            cvi=store.cvi[A["tr_rows"]],
            cvi_valid=store.cvi_valid[A["tr_rows"]],
            observability=mtr["mask"], actual_k=mtr["actual_k"],
            source_code=np.full(n_tr, FB.SOURCES.index(sx), np.int64),
            kmode_code=np.full(n_tr, FB.K_MODES.index(km), np.int64),
            slate_pos=A["tr_slate_pos"], util=A["util_tr"])
        build_sec = time.time() - t0
        after = mem()
        t1 = time.time()
        model = MR.build_model()
        model.fit(X, y)
        fit_sec = time.time() - t1
        peak = max(after, mem())
        rows.append({"condition_id": cid, "n_rows": int(len(X)),
                     "n_features": int(X.shape[1]), "dtype": str(X.dtype),
                     "input_matrix_mb": round(X.nbytes / 2 ** 20, 1),
                     "rss_before_mb": round(before, 1),
                     "rss_after_build_mb": round(after, 1),
                     "rss_peak_mb": round(peak, 1),
                     "build_sec": round(build_sec, 1),
                     "fit_sec": round(fit_sec, 1)})
        print(f"   {cid:<10} {X.shape} X={X.nbytes/2**20:.0f}MB "
              f"peak={peak:.0f}MB fit={fit_sec:.0f}s", flush=True)
        del X, model
        gc.collect()

    total, avail = sysmem()
    worst = max(r["rss_peak_mb"] for r in rows)
    payload = {"model": MR.MODEL_FAMILY, "config": MR.HGBR_CONFIG,
               "context": UFB.CONDITION, "probes": rows,
               "system_total_mb": round(total, 1),
               "system_available_mb": round(avail, 1),
               "worst_peak_rss_mb": worst,
               "safe": bool(avail - worst > 300),
               "recommended_concurrency": 1, "float32_used": True,
               "features_reduced": "none", "model_changed": "none",
               "estimated_total_hours": round(
                   sum(r["fit_sec"] for r in rows) / len(rows) * 45 / 3600, 2)}
    write_json_atomic(out / "resource_preflight.json", payload)
    print(f"\n  worst peak {worst:.0f} MB | available {avail:.0f} MB | "
          f"safe={payload['safe']} | est {payload['estimated_total_hours']} h",
          flush=True)
    return 0 if payload["safe"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
