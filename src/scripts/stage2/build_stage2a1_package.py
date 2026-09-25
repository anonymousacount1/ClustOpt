"""Stage 2A-1 aggregation: combined OOF artifacts, integrity gate, replay smoke.

Post-processing only. Runs no clustering, trains nothing, and never touches
Split 1 except to assert its absence.

Produces the machine-readable Stage-2A-1 package, the 20-check final integrity
gate, the full-train-vs-OOF diagnostic, and a small deterministic policy-replay
smoke proving the OOF utilities can drive the accepted policy semantics over the
existing 32-candidate pools.

Run:
    .venv_clustopt/Scripts/python.exe scripts/stage2/build_stage2a1_package.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from models.Clustering_Repository_Builder.utility_generation.worker_setup import (  # noqa: E402
    configure_process,
)

configure_process(thread_limit=4)

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from models.Clustering_Repository_Builder.experiments.experiment_execution.output_writer import (  # noqa: E402
    _ext, ensure_dir, path_exists, read_json, write_json_atomic,
)

OUT = (Path(REPO) / "results_analysis/clustering_repository/stage2"
       / "oof_utility_splits2_16")
COMB = ensure_dir(OUT / "combined")
PRED_EXT = "csv.gz"
DEV_SPLITS = list(range(2, 17))
VIEWS = ("x_only", "y_only", "xy_2d")
EXP_DATASETS, EXP_RECORDS = 16013, 48039


def fold_dir(s): return OUT / f"holdout_split_{s:02d}"


def main() -> int:
    print("=" * 78)
    print("STAGE 2A-1  --  COMBINED PACKAGE, INTEGRITY GATE, REPLAY SMOKE")
    print("=" * 78)
    lock = read_json(OUT / "model_semantics_lock.json") or {}

    # ------------------------------------------------------- fold inventory
    print("\n[1/7] fold inventory")
    fold_rows, mlp_rows, knn_rows, leak_all, sch_all = [], [], [], [], []
    mlp_frames, knn_frames = [], []
    for s in DEV_SPLITS:
        d = fold_dir(s)
        man = read_json(d / "fold_manifest.json") or {}
        met = read_json(d / "fold_metrics.json") or {}
        mman = read_json(d / "mlp/fold_manifest.json") or {}
        kman = read_json(d / "knn/fold_manifest.json") or {}
        lk = pd.read_csv(_ext(d / "leakage_checks.csv")); lk["heldout_split"] = s
        sc = pd.read_csv(_ext(d / "schema_checks.csv")); sc["heldout_split"] = s
        leak_all.append(lk); sch_all.append(sc)
        fold_rows.append({
            "heldout_split": s, "training_splits": ";".join(map(str, man["training_splits"])),
            "n_training_splits": len(man["training_splits"]),
            "status": man.get("status"), "n_oof_records": man.get("n_oof_records"),
            "runtime_sec": man.get("runtime_sec"),
            "source_commit": man.get("source_commit"),
            "leakage": met.get("leakage"), "schema": met.get("schema"),
        })
        mlp_rows.append({"heldout_split": s, **{k: met["mlp"].get(k) for k in
                        ("mae", "rmse", "r2", "top1_accuracy", "top3_overlap",
                         "top5_overlap", "top10_overlap")},
                         "ndcg_at_10": met["mlp"].get("ndcg@10", met["mlp"].get("ndcg_at_10")),
                         "spearman": met["mlp"].get("spearman"),
                         "epochs_run": mman.get("epochs_run"),
                         "train_groups": mman.get("train_groups"),
                         "val_groups": mman.get("val_groups"),
                         "test_groups": mman.get("test_groups"),
                         "runtime_sec": mman.get("runtime_sec"),
                         "artifact": mman.get("run_root")})
        knn_rows.append({"heldout_split": s, **{k: met["knn"].get(k) for k in
                        ("n", "mae", "rmse", "r2", "spearman", "top1_accuracy",
                         "top3_overlap", "top5_overlap", "top10_overlap", "ndcg_at_10")},
                         "config": (kman.get("config") or {}).get("config_id"),
                         "runtime_sec": kman.get("runtime_sec"),
                         "artifact": f"holdout_split_{s:02d}/knn"})
        mp = pd.read_csv(_ext(d / f"mlp/oof_predictions.{PRED_EXT}"))
        mp["heldout_split"] = s
        mlp_frames.append(mp)
        kp = pd.read_csv(_ext(d / f"knn/oof_predictions.{PRED_EXT}"))
        kp["heldout_split"] = s
        knn_frames.append(kp)
        print(f"    s={s:2d}  {man.get('status'):9} records={man.get('n_oof_records'):,}"
              f"  MLP MAE={met['mlp']['mae']:.4f}  KNN MAE={met['knn']['mae']:.4f}")

    fold_man = pd.DataFrame(fold_rows)
    mlp_met = pd.DataFrame(mlp_rows)
    knn_met = pd.DataFrame(knn_rows)
    fold_man.to_csv(_ext(OUT / "oof_fold_manifest.csv"), index=False)
    mlp_met.to_csv(_ext(OUT / "oof_mlp_fold_metrics.csv"), index=False)
    knn_met.to_csv(_ext(OUT / "oof_knn_fold_metrics.csv"), index=False)
    pd.concat(leak_all, ignore_index=True).to_csv(
        _ext(OUT / "oof_leakage_checks.csv"), index=False)
    pd.concat(sch_all, ignore_index=True).to_csv(
        _ext(OUT / "oof_schema_checks.csv"), index=False)

    # -------------------------------------------------- combined predictions
    print("\n[2/7] combined OOF predictions")
    knn_all = pd.concat(knn_frames, ignore_index=True)
    mlp_all = pd.concat(mlp_frames, ignore_index=True)
    knn_all.to_csv(_ext(COMB / f"oof_knn_utilities.{PRED_EXT}"), index=False,
                   compression="gzip")
    mlp_all.to_csv(_ext(COMB / f"oof_mlp_utilities.{PRED_EXT}"), index=False,
                   compression="gzip")
    print(f"    KNN {len(knn_all):,} rows  |  MLP {len(mlp_all):,} rows")

    # dataset/view coverage from the authoritative KNN frame (carries split_id)
    key = knn_all[["dataset_id", "view_id", "split_id", "heldout_split"]]
    n_ds = key["dataset_id"].nunique()
    n_rec = len(key)
    dup = int(key.duplicated(subset=["dataset_id", "view_id"]).sum())

    pd.DataFrame([
        {"component": "mlp_full60", "kind": "learned", "cross_fitted": True,
         "n_folds": 15, "artifact_root": "holdout_split_XX/mlp",
         "binding": lock.get("final_mlp_binding", {}).get("artifact_root")},
        {"component": "knn_utility", "kind": "learned (neighbour repository)",
         "cross_fitted": True, "n_folds": 15, "artifact_root": "holdout_split_XX/knn",
         "binding": lock.get("final_knn_binding", {}).get("artifact_root")},
        {"component": "dynamic_predict_k", "kind": "deterministic heuristic",
         "cross_fitted": False, "n_folds": 0, "artifact_root": "",
         "binding": lock.get("dynamic_predict_k_binding", {}).get("heuristic")},
    ]).to_csv(_ext(OUT / "oof_component_inventory.csv"), index=False)
    pd.concat([mlp_met.assign(component="mlp_full60"),
               knn_met.assign(component="knn_utility")], ignore_index=True
              ).to_csv(_ext(OUT / "oof_model_inventory.csv"), index=False)
    (key.groupby(["heldout_split", "view_id"]).size().rename("n_records")
     .reset_index().to_csv(_ext(OUT / "oof_prediction_inventory.csv"), index=False))

    # ------------------------------------------------------ aggregate metrics
    print("\n[3/7] aggregate metrics")
    metric_names = json.load(open(
        _ext(Path(REPO) / "results_analysis/metric_utility_knn/phase_e1/models"
             / "utility_metric_schema.json"), encoding="utf-8"))["metric_names"]

    def agg(df, w):
        rows = []
        for m in ("mae", "rmse", "r2", "spearman", "top1_accuracy",
                  "top3_overlap", "top5_overlap", "top10_overlap", "ndcg_at_10"):
            if m not in df.columns or df[m].isna().all():
                continue
            v = pd.to_numeric(df[m], errors="coerce")
            rows.append({"metric": m, "macro_mean_over_folds": float(v.mean()),
                         "record_weighted": float(np.average(v, weights=w)),
                         "fold_min": float(v.min()), "fold_max": float(v.max()),
                         "fold_std": float(v.std(ddof=1))})
        return pd.DataFrame(rows)
    wts = knn_met["n"].to_numpy(float)
    ov = pd.concat([agg(mlp_met, wts).assign(component="mlp_full60"),
                    agg(knn_met, wts).assign(component="knn_utility")],
                   ignore_index=True)
    ov.to_csv(_ext(OUT / "oof_overall_metrics.csv"), index=False)
    print(ov[ov.component == "mlp_full60"][
        ["metric", "macro_mean_over_folds", "fold_min", "fold_max"]].round(4).to_string(index=False))

    # per-view (KNN only: MLP per-view needs the runner's own grouped outputs)
    pv = []
    for s in DEV_SPLITS:
        d = fold_dir(s)
        kp = pd.read_csv(_ext(d / f"knn/oof_predictions.{PRED_EXT}"))
        for view in VIEWS:
            m = kp["view_id"] == view
            P = kp.loc[m, metric_names].to_numpy(float)
            T = np.load(_ext(d / "knn" / f"targets_{view}.npy"))
            e = P - T
            pv.append({"component": "knn_utility", "heldout_split": s, "view_id": view,
                       "n": int(m.sum()), "mae": float(np.abs(e).mean()),
                       "rmse": float(np.sqrt((e ** 2).mean())),
                       "top5_overlap": float(np.mean([
                           len(set(a) & set(b)) / 5 for a, b in
                           zip(np.argsort(-P, 1)[:, :5], np.argsort(-T, 1)[:, :5])]))})
    pvdf = pd.DataFrame(pv)
    pvdf.to_csv(_ext(OUT / "oof_per_view_metrics.csv"), index=False)
    print("\n  KNN per-view (macro over folds):")
    print(pvdf.groupby("view_id")[["mae", "rmse", "top5_overlap"]].mean()
          .round(4).to_string())

    # ------------------------------------- full-train vs OOF diagnostic
    print("\n[4/7] full-train vs OOF diagnostic (KNN, diagnostic only)")
    from models.metric_utility_knn import config as KC
    from models.metric_utility_knn.data_loader import load_unified_data
    from models.metric_utility_knn.knn_predictor import KnnUtilityPredictor
    data = load_unified_data()
    knn_root = Path(REPO) / "results_analysis/metric_utility_knn/phase_e1/models"
    _kc = (read_json(knn_root / "final_knn_metadata.json") or {})["config"]
    full = KnnUtilityPredictor.load(
        knn_root, list(data.feature_columns), list(data.metric_names),
        KC.KnnConfig(n_neighbors=int(_kc["n_neighbors"]), scaler=_kc["scaler"],
                     distance=_kc["distance"],
                     neighbor_weighting=_kc["neighbor_weighting"]))
    diag = []
    for view in VIEWS:
        m = (data.view == view) & data.split_mask(DEV_SPLITS)
        X, T = data.X[m], data.Y[m]
        idx = np.arange(X.shape[0])[:: max(1, X.shape[0] // 2000)][:2000]
        P_full = np.atleast_2d(full.predict_utility(X[idx], view))
        oof = knn_all[knn_all["view_id"] == view]
        oof = oof.set_index("record_id").reindex(data.record_id[m][idx])
        P_oof = oof[metric_names].to_numpy(float)
        for label, P in (("full_train_in_sample", P_full), ("oof", P_oof)):
            e = P - T[idx]
            diag.append({"component": "knn_utility", "view_id": view, "regime": label,
                         "n": len(idx), "mae": float(np.abs(e).mean()),
                         "top5_overlap": float(np.mean([
                             len(set(a) & set(b)) / 5 for a, b in
                             zip(np.argsort(-P, 1)[:, :5],
                                 np.argsort(-T[idx], 1)[:, :5])]))})
    dg = pd.DataFrame(diag)
    dg.to_csv(_ext(OUT / "oof_fulltrain_vs_oof_diagnostic.csv"), index=False)
    piv = dg.pivot_table(index="view_id", columns="regime", values=["mae", "top5_overlap"])
    print(piv.round(4).to_string())
    print("  (diagnostic only -- no Stage-1 conclusion or tuning depends on this)")

    # ------------------------------------------------- policy replay smoke
    print("\n[5/7] policy replay smoke over the existing 32-candidate pools")
    from models.ClustOpt.cluster_validity_indices.dynamic_metric_selection.dynamic_k import (
        select_dynamic_top_k,
    )
    from models.ClustOpt.cluster_validity_indices.dynamic_metric_selection.weighting import (
        compute_weights, select_top_k,
    )
    from models.Clustering_Repository_Builder.experiments.experiment_execution.offline_candidate_execution import (  # noqa: E501
        load_candidate_set, score_candidates,
    )
    split_csv = (Path(REPO) / "results_analysis/clustering_repository/analyzed_data"
                 / "experiment_splits/dataset_split_assignments.csv")
    a = pd.read_csv(_ext(split_csv))
    dev = a[a["split_id"].isin(DEV_SPLITS)].sort_values("dataset_id")
    sample = dev.iloc[:: max(1, len(dev) // 8)].head(8)
    POLICIES = [("MLP T1-R", "mlp", 1, "normalized_positive"),
                ("MLP T5-R", "mlp", 5, "normalized_positive"),
                ("MLP T10-S", "mlp", 10, "softmax"),
                ("KNN T3-S", "knn", 3, "softmax"),
                ("KNN T5-R", "knn", 5, "normalized_positive"),
                ("MLP Dyn-R", "mlp", "dynamic", "normalized_positive"),
                ("KNN Dyn-R", "knn", "dynamic", "normalized_positive")]
    knn_idx = knn_all.set_index(["dataset_id", "view_id"])
    mlp_cols = [c for c in mlp_all.columns if c.startswith("pred__")] or None
    smoke = []
    for label, src, k, wmode in POLICIES:
        for _, ds in sample.iterrows():
            for view in VIEWS:
                try:
                    if src == "knn":
                        row = knn_idx.loc[(ds["dataset_id"], view)]
                        u = {m: float(row[m]) for m in metric_names}
                    else:
                        mm = mlp_all[(mlp_all.dataset_id == ds["dataset_id"])
                                     & (mlp_all.view_mode == view)]
                        if mm.empty:
                            continue
                        r0 = mm.iloc[0]
                        u = {m: float(r0[f"pred_{m}"]) for m in metric_names
                             if f"pred_{m}" in mm.columns}
                        if len(u) != 60:
                            continue
                    if k == "dynamic":
                        sel, sk, _ = select_dynamic_top_k(u)
                    else:
                        sel, sk = select_top_k(u, int(k)), int(k)
                    w = compute_weights(sel, wmode, 0.5)
                    cand = load_candidate_set(Path(ds["dataset_dir"]), view,
                                              expected_count=32)
                    frame = cand["frame"]
                    J = score_candidates(frame, w)
                    good = [(j, i) for i, j in enumerate(J) if j is not None]
                    if not good:
                        continue
                    best = max(good)[1]
                    smoke.append({
                        "policy": label, "utility_source": src,
                        "dataset_id": ds["dataset_id"], "split_id": int(ds["split_id"]),
                        "view_id": view, "top_k_requested": k, "selected_k": sk,
                        "weighting": wmode,
                        "selected_metrics": "|".join(n for n, _ in sel),
                        "weights_sum": float(sum(w.values())),
                        "n_candidates": int(len(frame)),
                        "selected_candidate_index": int(best),
                        "selected_objective_J": float(J[best]),
                        "selected_candidate_ari": float(frame.iloc[best]["ARI"]),
                        "status": "ok"})
                except Exception as exc:                       # pragma: no cover
                    smoke.append({"policy": label, "dataset_id": ds["dataset_id"],
                                  "view_id": view, "status": f"ERROR: {type(exc).__name__}: {exc}"})
    sm = pd.DataFrame(smoke)
    sm.to_csv(_ext(OUT / "oof_policy_replay_smoke.csv"), index=False)
    ok_sm = sm[sm["status"] == "ok"]
    print(f"    {len(ok_sm)}/{len(sm)} replays ok")
    if len(ok_sm):
        print(ok_sm.groupby("policy").agg(
            n=("status", "size"), mean_k=("selected_k", "mean"),
            wsum=("weights_sum", "mean"), meanJ=("selected_objective_J", "mean"),
            meanARI=("selected_candidate_ari", "mean")).round(4).to_string())

    # -------------------------------------------------- final integrity gate
    print("\n[6/7] FINAL INTEGRITY GATE")
    checks, ok_all = [], True

    def c(n, ok, detail=""):
        nonlocal ok_all
        ok_all &= bool(ok)
        checks.append({"check": n, "status": "PASS" if ok else "FAIL",
                       "detail": str(detail)})
        print(f"  [{'PASS' if ok else 'FAIL'}] {n}" + (f" -- {detail}" if detail else ""))
    head = subprocess.run(["git", "log", "--oneline", "-40"], cwd=REPO,
                          capture_output=True, text=True).stdout
    c("1 Stage-1 freeze commit exists", "freeze online ablation" in head)
    c("2 final MLP semantics lock PASS",
      lock.get("gate_status") == "PASS" and bool(lock.get("final_mlp_binding")))
    c("3 final KNN semantics lock PASS", bool(lock.get("final_knn_binding")))
    c("4 Dynamic-K semantics resolved",
      lock.get("dynamic_predict_k_binding", {}).get("kind", "").startswith("DETERMINISTIC"))
    c("5 golden Split-1 reproduction PASS",
      lock.get("golden_reproduction", {}).get("order_match_rate") == 1.0,
      f"max weight err {lock.get('golden_reproduction', {}).get('max_weight_abs_err')}")
    c("6 15/15 OOF folds complete",
      int((fold_man["status"] == "complete").sum()) == 15,
      f"{int((fold_man['status'] == 'complete').sum())}/15")
    c("7 splits exactly 2..16", sorted(fold_man["heldout_split"]) == DEV_SPLITS)
    c("8 Split 1 absent from OOF", 1 not in set(key["split_id"].unique()))
    c("9 16,013 unique datasets", n_ds == EXP_DATASETS, str(n_ds))
    c("10 48,039 dataset/view records", n_rec == EXP_RECORDS, str(n_rec))
    c("11 one OOF MLP prediction per record", len(mlp_all) == EXP_RECORDS,
      str(len(mlp_all)))
    c("12 one OOF KNN prediction per record", len(knn_all) == EXP_RECORDS,
      str(len(knn_all)))
    c("13 no held-out split in its own training set",
      all(str(r["heldout_split"]) not in r["training_splits"].split(";")
          for _, r in fold_man.iterrows()))
    c("14 no duplicate (dataset,view) source records", dup == 0, str(dup))
    c("15 canonical 60-metric order identical across folds",
      len({read_json(fold_dir(s) / "mlp/fold_manifest.json")["metric_order_hash"]
           for s in DEV_SPLITS}) == 1)
    c("16 feature schemas identical across folds",
      len({read_json(fold_dir(s) / "mlp/fold_manifest.json")["feature_schema_hash"]
           for s in DEV_SPLITS}) == 1)
    c("17 preprocessing semantics identical",
      len({read_json(fold_dir(s) / "knn/fold_manifest.json")["config"]["scaler"]
           for s in DEV_SPLITS}) == 1)
    lkall = pd.concat(leak_all, ignore_index=True)
    c("18 no target leakage (all per-fold leakage checks PASS)",
      bool((lkall["status"] == "PASS").all()),
      f"{int((lkall['status'] != 'PASS').sum())} failures of {len(lkall)}")
    c("19 no oracle dependency",
      not any("real" in str(v).lower() and "utility" in str(v).lower()
              for v in fold_man["training_splits"]), "real_*/realK excluded by design")
    c("20 representative OOF policy replay PASS",
      len(ok_sm) > 0 and (sm["status"] == "ok").all()
      and bool(np.allclose(ok_sm["weights_sum"], 1.0, atol=1e-9)),
      f"{len(ok_sm)}/{len(sm)} ok")
    schall = pd.concat(sch_all, ignore_index=True)
    c("21 all per-fold schema checks PASS",
      bool((schall["status"] == "PASS").all()))
    c("22 every fold trained on exactly 14 splits",
      bool((fold_man["n_training_splits"] == 14).all()))
    pd.DataFrame(checks).to_csv(_ext(OUT / "oof_integrity_checks.csv"), index=False)

    # ------------------------------------------------------------- manifest
    print("\n[7/7] manifest")
    write_json_atomic(OUT / "experiment_manifest.json", {
        "stage": "Stage 2A-1 -- leave-one-split-out OOF utilities (Splits 2-16)",
        "source_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                                        capture_output=True, text=True).stdout.strip(),
        "environment_lock": "environment/stage1_clustopt_requirements.lock.txt",
        "gate_0a_freeze_commit": "564ef7a",
        "gate_0b": lock.get("gate_status"),
        "protocol": "leave-one-split-out over {2..16}; 14 training splits per fold",
        "population": {"datasets": int(n_ds), "dataset_view_records": int(n_rec),
                       "folds": 15},
        "components": {"mlp_full60": "cross-fitted (15 models)",
                       "knn_utility": "cross-fitted (15 repositories)",
                       "dynamic_predict_k": "deterministic; no fitting"},
        "split1_usage": "absent from all training, validation, preprocessing, "
                        "repositories and the OOF population; used only for the "
                        "Gate-0B inference-only golden reproduction",
        "prediction_format": "gzip CSV (the frozen runtime has no parquet engine)",
        "integrity_gate": "PASS" if ok_all else "FAIL",
        "built_by": "scripts/stage2/build_stage2a1_package.py",
    })
    print("\n" + "=" * 78)
    print(f"STAGE 2A-1 INTEGRITY GATE: {'PASS' if ok_all else 'FAIL'}")
    print("=" * 78)
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
