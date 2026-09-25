"""Stage 2B-3 orchestrator: 15 unified models, each evaluated under all 10 regimes.

One fit per outer fold on a balanced mixed-regime training population whose row
count matches a dedicated specialist. At test time the SAME model scores ten
counterfactual versions of every held-out slate -- that is evaluation, not
training, so it does not inflate the training budget.
"""
from __future__ import annotations

import argparse
import json
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
from models.ClustOpt_Candidate_Reranker.protocol import masking_regimes as MR_REG  # noqa: E402,E501
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
from . import regime_assignment as RA
from . import unified_feature_builder as UFB

B1_REL = ("results_analysis/clustopt_candidate_reranker/"
          "stage2b1_masking_feasibility")
OUT_REL = ("results_analysis/clustopt_candidate_reranker/"
           "stage2b3_unified_masking")
OUTER_SPLITS: Tuple[int, ...] = tuple(range(2, 17))
GZ = {"index": False, "compression": "gzip"}
LOCK_NAME = "stage2b3_worker.pid"
FULL_PRED_REGIMES = {"MLP_TOP5", "MLP_TOP10"}


def fold_dir(out: Path, s: int) -> Path:
    return out / "outer_fold_predictions" / f"fold_{s:02d}"


def fold_is_complete(out: Path, s: int, uhash: str) -> bool:
    m = read_json(fold_dir(out, s) / "unit_manifest.json")
    return bool(m and m.get("status") == "complete"
                and m.get("unit_hash") == uhash
                and m.get("leakage_checks_pass")
                and int(m.get("n_regimes_evaluated", 0)) == RA.N_REGIMES)


def _b1_baselines(repo: Path, s: int, cond: str) -> pd.DataFrame:
    d = (repo / B1_REL / "outer_fold_predictions" / cond / f"fold_{s:02d}"
         / "slate_selections.csv.gz")
    return pd.read_csv(_ext(d))


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", default=".")
    p.add_argument("--preflight", action="store_true")
    p.add_argument("--splits", type=int, nargs="*", default=None)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)

    repo = Path(args.repo_root).resolve()
    out = ensure_dir(repo / OUT_REL)
    ensure_dir(out / "regime_assignment_manifests")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                            capture_output=True, text=True).stdout.strip()

    print("=" * 78)
    print("STAGE 2B-3  --  UNIFIED MASKING-AWARE CANDIDATE RERANKER")
    print("=" * 78, flush=True)
    print(f"  context C2 LOCAL_UTIL ({UFB.EXPECTED_DIM} dims) | "
          f"model {MR.config_hash()} | target T2_SLATE_REGRET | "
          f"{RA.N_REGIMES} regimes", flush=True)

    with WorkerLock(out / LOCK_NAME):
        store = CandidateStore(repo)
        splits = args.splits or list(OUTER_SPLITS)
        if 1 in splits:
            raise SystemExit("Split 1 is forbidden")
        if args.preflight:
            return _preflight(repo, store, out)

        t_all, n_done, n_skip = time.time(), 0, 0
        bal_rows: List[Dict[str, Any]] = []
        for s in splits:
            fu = load_fold_utilities(repo, s, store.metric_names)
            tr_slates = np.flatnonzero(store.slate_split != s)
            te_slates = np.flatnonzero(store.slate_split == s)
            key = store.slate_key.to_numpy()
            tr_pos = pd.Index(key[tr_slates]).get_indexer(fu["train_key"])
            te_pos = pd.Index(key[te_slates]).get_indexer(fu["test_key"])
            if (tr_pos < 0).any() or (te_pos < 0).any():
                raise AssertionError(f"fold {s}: utility identifiers misaligned")
            tr_order, te_order = np.argsort(tr_pos), np.argsort(te_pos)

            # ---- deterministic balanced assignment ------------------------
            asg = RA.assign(s, store.slate_dataset[tr_slates],
                            store.slate_view[tr_slates])
            RA.validate(asg, len(tr_slates))
            ahash = RA.manifest_hash(asg)
            rep = RA.balance_report(asg)
            asg.to_csv(_ext(out / "regime_assignment_manifests"
                            / f"fold_{s:02d}_regime_assignment.csv.gz"), **GZ)
            bal_rows.append({"outer_split": s, "assignment_hash": ahash,
                             **{f"n_{r}": rep["counts"][r]
                                for r in RA.REGIME_ORDER},
                             "overall_imbalance": rep["overall_imbalance"],
                             "max_within_view_imbalance":
                                 rep["max_within_view_imbalance"],
                             "n_slates": rep["n_slates"]})

            uhash = MR.unit_hash(f"UNIFIED_C2__{ahash}", s,
                                 UFB.CONDITION, CE.definition_hash(),
                                 f"{fu['train_source']} / {fu['test_source']}")
            if not args.overwrite and fold_is_complete(out, s, uhash):
                n_skip += 1
                continue

            # ---- per-slate regime masks for TRAINING ----------------------
            t0 = time.time()
            n_tr = len(tr_slates)
            mask_tr = np.zeros((n_tr, 60), dtype=np.int8)
            ak_tr = np.zeros(n_tr, dtype=np.int16)
            src_tr = np.zeros(n_tr, dtype=np.int64)
            km_tr = np.zeros(n_tr, dtype=np.int64)
            util_by_src = {sx: fu["train_util"][sx][tr_order]
                           for sx in MR_REG.SOURCES}
            reg_of = asg["assigned_regime"].to_numpy()
            for rid in RA.REGIME_ORDER:
                sel = np.flatnonzero(reg_of == rid)
                if sel.size == 0:
                    continue
                sx, km = rid.split("__")
                r = resolve_masks(util_by_src[sx][sel], store.metric_names, km)
                mask_tr[sel] = r["mask"]
                ak_tr[sel] = r["actual_k"]
                src_tr[sel] = FB.SOURCES.index(sx)
                km_tr[sel] = FB.K_MODES.index(km)

            util_tr = np.hstack([fu["train_util"]["mlp"][tr_order],
                                 fu["train_util"]["knn"][tr_order]]
                                ).astype(np.float32)
            util_te = np.hstack([fu["test_util"]["mlp"][te_order],
                                 fu["test_util"]["knn"][te_order]]
                                ).astype(np.float32)

            tr_rows_all = store.rows_for_slates(tr_slates)
            tr_rows = tr_rows_all[store.eligible[tr_rows_all]]
            tr_slate_pos = pd.Series(np.arange(n_tr), index=tr_slates
                                     ).reindex(store.slate_of_row[tr_rows]
                                               ).to_numpy().astype(np.int64)
            y_tr = TF.target_regret(store.ari[tr_rows],
                                    store.oracle_row[tr_rows])
            X_tr, n_common = UFB.build_unified_X(
                view_codes=store.view_code[tr_rows],
                cat_rows=store.cat_row[tr_rows], cat_matrix=store.cat_matrix,
                partition=store.partition[tr_rows], cvi=store.cvi[tr_rows],
                cvi_valid=store.cvi_valid[tr_rows], observability=mask_tr,
                actual_k=ak_tr, source_code=src_tr, kmode_code=km_tr,
                slate_pos=tr_slate_pos, util=util_tr)
            UFB.assert_regime_identity(X_tr, tr_slate_pos, src_tr, km_tr,
                                       n_common)
            model = MR.build_model()
            tf0 = time.time()
            model.fit(X_tr, y_tr)
            fit_sec = time.time() - tf0
            n_train_rows = int(len(X_tr))
            del X_tr

            # ---- evaluate the SAME model under all 10 regimes -------------
            te_rows = store.rows_for_slates(te_slates)
            te_slate_pos = np.repeat(np.arange(len(te_slates)), 32)
            te_starts = np.cumsum(np.r_[0, store.slate_size[te_slates]])[:-1]
            te_sizes = store.slate_size[te_slates]
            ari_te, orc_te = store.ari[te_rows], store.oracle_row[te_rows]
            elig_te = store.eligible[te_rows]
            ds_te = store.dataset_id[te_rows][te_starts]
            vw_te = store.view_id[te_rows][te_starts]
            orc_sl = orc_te[te_starts]

            reg_metrics: Dict[str, Any] = {}
            for rid in RA.REGIME_ORDER:
                sx, km = rid.split("__")
                cond_id = RA.regime_condition_id(rid)
                r = resolve_masks(fu["test_util"][sx][te_order],
                                  store.metric_names, km)
                n_te = len(te_slates)
                X_te, _ = UFB.build_unified_X(
                    view_codes=store.view_code[te_rows],
                    cat_rows=store.cat_row[te_rows],
                    cat_matrix=store.cat_matrix,
                    partition=store.partition[te_rows], cvi=store.cvi[te_rows],
                    cvi_valid=store.cvi_valid[te_rows],
                    observability=r["mask"], actual_k=r["actual_k"],
                    source_code=np.full(n_te, FB.SOURCES.index(sx), np.int64),
                    kmode_code=np.full(n_te, FB.K_MODES.index(km), np.int64),
                    slate_pos=te_slate_pos, util=util_te)
                UFB.assert_no_hidden_leakage(X_te, r["mask"], te_slate_pos,
                                             n_common)
                pred = np.asarray(model.predict(X_te), dtype=np.float64)
                del X_te
                chosen = EV.select_per_slate(pred, elig_te, te_starts, te_sizes,
                                             minimise=True)
                if not elig_te[chosen].all():
                    raise AssertionError("an ineligible candidate was selected")
                sel_ari = ari_te[chosen]                 # ARI read AFTER choice
                m = EV.slate_metrics(sel_ari, orc_sl, vw_te, ds_te)
                b1 = _b1_baselines(repo, s, cond_id)
                if not np.array_equal(b1["dataset_id"].to_numpy(), ds_te) or \
                   not np.array_equal(b1["view_id"].to_numpy(), vw_te):
                    raise AssertionError(f"{cond_id}: baseline slate order "
                                         f"differs from this fold")
                if not np.allclose(b1["candidate_oracle_ari"].to_numpy(),
                                   orc_sl, atol=1e-12):
                    raise AssertionError(f"{cond_id}: baseline oracle differs")
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
                m["dedicated_local_bestview"] = float(
                    read_json(repo / B1_REL / "outer_fold_predictions"
                              / cond_id / f"fold_{s:02d}"
                              / "unit_manifest.json")["metrics"]
                    ["mean_bestview_selected_ari"])
                reg_metrics[cond_id] = m

                dd = ensure_dir(fold_dir(out, s) / cond_id)
                pd.DataFrame({
                    "dataset_id": ds_te, "split_id": s, "view_id": vw_te,
                    "selected_candidate_index":
                        store.candidate_index[te_rows][chosen],
                    "selected_ari": sel_ari, "candidate_oracle_ari": orc_sl,
                    "oracle_regret": orc_sl - sel_ari,
                    "b_raw_ari": b1["b_raw_ari"].to_numpy(),
                    "b_softmax_ari": b1["b_softmax_ari"].to_numpy(),
                    "dedicated_local_selected_ari":
                        b1["selected_ari"].to_numpy(),
                }).to_csv(_ext(dd / "slate_selections.csv.gz"), **GZ)
                if cond_id in FULL_PRED_REGIMES:
                    pd.DataFrame({
                        "dataset_id": store.dataset_id[te_rows], "split_id": s,
                        "view_id": store.view_id[te_rows],
                        "candidate_index": store.candidate_index[te_rows],
                        "eligible": elig_te.astype(np.int8),
                        "predicted_regret": pred,
                        "candidate_ari": ari_te,
                        "candidate_oracle_ari": orc_te,
                    }).to_csv(_ext(dd / "candidate_predictions.csv.gz"), **GZ)

            write_json_atomic(fold_dir(out, s) / "unit_manifest.json", {
                "status": "complete", "unit_hash": uhash, "outer_split": int(s),
                "assignment_hash": ahash, "balance": rep,
                "context": UFB.CONDITION, "n_features": UFB.EXPECTED_DIM,
                "eligibility_hash": CE.definition_hash(),
                "utility_source_train": fu["train_source"],
                "utility_source_test": fu["test_source"],
                "model_config_hash": MR.config_hash(),
                "target": "T2_SLATE_REGRET",
                "n_train_rows": n_train_rows, "n_train_slates": int(n_tr),
                "fit_sec": fit_sec, "n_regimes_evaluated": RA.N_REGIMES,
                "metrics": reg_metrics, "leakage_checks_pass": True,
                "split_1_used": False, "source_commit": commit,
            })
            n_done += 1
            top10 = reg_metrics["MLP_TOP10"]
            top5 = reg_metrics["MLP_TOP5"]
            print(f"   [fold {s:02d}] rows={n_train_rows} fit={fit_sec:.0f}s | "
                  f"T10 BV={top10['mean_bestview_selected_ari']:.4f} "
                  f"rec={top10['recovery_raw']:+.4f} | "
                  f"T5 BV={top5['mean_bestview_selected_ari']:.4f} "
                  f"rec={top5['recovery_raw']:+.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            del model

        if bal_rows:
            write_csv_atomic(out / "regime_assignment_summary.csv",
                             pd.DataFrame(bal_rows))
            write_json_atomic(out / "regime_assignment_hashes.json", {
                "seed": RA.SEED, "namespace": RA.NAMESPACE,
                "regime_order": list(RA.REGIME_ORDER),
                "per_fold": {int(r["outer_split"]): r["assignment_hash"]
                             for r in bal_rows},
                "target_independence": RA.target_independence_statement(),
            })
        print(f"\n  {n_done} unified folds fitted, {n_skip} skipped, "
              f"{time.time()-t_all:.0f}s", flush=True)
    return 0


def _preflight(repo: Path, store: CandidateStore, out: Path) -> int:
    """Resource-only probe on ONE outer-training population."""
    import gc
    mem, sysmem = _memory_probes()
    s = 2
    fu = load_fold_utilities(repo, s, store.metric_names)
    tr_slates = np.flatnonzero(store.slate_split != s)
    key = store.slate_key.to_numpy()
    tr_order = np.argsort(pd.Index(key[tr_slates]).get_indexer(fu["train_key"]))
    asg = RA.assign(s, store.slate_dataset[tr_slates],
                    store.slate_view[tr_slates])
    RA.validate(asg, len(tr_slates))
    rep = RA.balance_report(asg)

    n_tr = len(tr_slates)
    mask_tr = np.zeros((n_tr, 60), dtype=np.int8)
    ak = np.zeros(n_tr, np.int16)
    sc = np.zeros(n_tr, np.int64)
    kc = np.zeros(n_tr, np.int64)
    ubs = {sx: fu["train_util"][sx][tr_order] for sx in MR_REG.SOURCES}
    reg_of = asg["assigned_regime"].to_numpy()
    rows_per_regime: Dict[str, int] = {}
    for rid in RA.REGIME_ORDER:
        sel = np.flatnonzero(reg_of == rid)
        sx, km = rid.split("__")
        r = resolve_masks(ubs[sx][sel], store.metric_names, km)
        mask_tr[sel] = r["mask"]
        ak[sel] = r["actual_k"]
        sc[sel] = FB.SOURCES.index(sx)
        kc[sel] = FB.K_MODES.index(km)
    util_tr = np.hstack([fu["train_util"]["mlp"][tr_order],
                         fu["train_util"]["knn"][tr_order]]).astype(np.float32)
    tr_rows_all = store.rows_for_slates(tr_slates)
    tr_rows = tr_rows_all[store.eligible[tr_rows_all]]
    slate_pos = pd.Series(np.arange(n_tr), index=tr_slates
                          ).reindex(store.slate_of_row[tr_rows]
                                    ).to_numpy().astype(np.int64)
    for rid in RA.REGIME_ORDER:
        sel = set(np.flatnonzero(reg_of == rid).tolist())
        rows_per_regime[rid] = int(np.isin(slate_pos, list(sel)).sum())
    y = TF.target_regret(store.ari[tr_rows], store.oracle_row[tr_rows])

    gc.collect()
    before = mem()
    t0 = time.time()
    X, n_common = UFB.build_unified_X(
        view_codes=store.view_code[tr_rows], cat_rows=store.cat_row[tr_rows],
        cat_matrix=store.cat_matrix, partition=store.partition[tr_rows],
        cvi=store.cvi[tr_rows], cvi_valid=store.cvi_valid[tr_rows],
        observability=mask_tr, actual_k=ak, source_code=sc, kmode_code=kc,
        slate_pos=slate_pos, util=util_tr)
    build_sec = time.time() - t0
    UFB.assert_regime_identity(X, slate_pos, sc, kc, n_common)
    after = mem()
    t1 = time.time()
    model = MR.build_model()
    model.fit(X, y)
    fit_sec = time.time() - t1
    peak = max(after, mem())
    t2 = time.time()
    model.predict(X[:100000])
    pred_est = (time.time() - t2) * (len(X) / 100000)
    total, avail = sysmem()

    b2 = read_json(repo / "results_analysis/clustopt_candidate_reranker/"
                   "stage2b2_context_ablation/resource_preflight.json") or {}
    dedicated_rows = next((p["n_rows"] for p in b2.get("probes", [])
                           if p["condition_id"] == "C2_LOCAL_UTIL"), None)
    payload = {
        "model": MR.MODEL_FAMILY, "config": MR.HGBR_CONFIG,
        "context": UFB.CONDITION, "n_features": int(X.shape[1]),
        "dtype": str(X.dtype),
        "n_train_slates": int(n_tr), "n_train_rows": int(len(X)),
        "dedicated_c2_train_rows": dedicated_rows,
        "row_budget_ratio": (round(len(X) / dedicated_rows, 4)
                             if dedicated_rows else None),
        "no_10x_expansion": bool(dedicated_rows
                                 and abs(len(X) - dedicated_rows) == 0),
        "regime_slate_counts": rep["counts"],
        "regime_row_counts": rows_per_regime,
        "per_view_counts": rep["per_view_counts"],
        "overall_imbalance": rep["overall_imbalance"],
        "max_within_view_imbalance": rep["max_within_view_imbalance"],
        "input_matrix_mb": round(X.nbytes / 2 ** 20, 1),
        "rss_before_mb": round(before, 1),
        "rss_after_build_mb": round(after, 1),
        "rss_peak_mb": round(peak, 1),
        "build_sec": round(build_sec, 1), "fit_sec": round(fit_sec, 1),
        "predict_full_sec_est": round(pred_est, 1),
        "predict_all_10_regimes_sec_est": round(pred_est * 10 * 0.093, 1),
        "system_total_mb": round(total, 1), "system_available_mb": round(avail, 1),
        "safe": bool(avail - peak > 300),
        "recommended_concurrency": 1,
        "estimated_total_hours": round((fit_sec + 10 * 25) * 15 / 3600, 2),
        "float32_used": True, "regimes_reduced": "none",
        "features_reduced": "none",
    }
    write_json_atomic(out / "resource_preflight.json", payload)
    print(f"   slates={n_tr} rows={len(X)} (dedicated C2 {dedicated_rows}) "
          f"dims={X.shape[1]} X={X.nbytes/2**20:.0f}MB peak={peak:.0f}MB "
          f"fit={fit_sec:.0f}s", flush=True)
    print(f"   regime slate balance: overall imbalance "
          f"{rep['overall_imbalance']}, within-view "
          f"{rep['max_within_view_imbalance']}", flush=True)
    print(f"   available {avail:.0f} MB | safe={payload['safe']} | "
          f"est {payload['estimated_total_hours']} h", flush=True)
    del X, model
    gc.collect()
    if not payload["safe"]:
        print("\n  RESOURCE BLOCKER: regimes and features were NOT reduced.",
              flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
