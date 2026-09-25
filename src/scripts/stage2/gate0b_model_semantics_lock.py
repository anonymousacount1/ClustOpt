"""Stage-2 Gate 0B: lock the final learned-utility semantics and prove them.

Inference-only. Trains nothing, writes nothing into any experiment directory,
and never uses Split 1 for fitting -- Split 1 appears here solely as the
golden-reproduction target against already-frozen historical artifacts.

What it does:

  1. Resolves the FINAL bindings for the Full-60 MLP, the KNN utility predictor
     and Dynamic Predict-K, from the accepted historical policy configs rather
     than from file names.
  2. Hashes the canonical 60-metric order, the 250-feature schema and the
     preprocessing artifacts.
  3. GOLDEN REPRODUCTION: replays the production resolver over a deterministic
     sample of Split-1 dataset/views and compares the resulting Top-K metric
     sets and weights against the persisted ``selected_metrics.json`` written by
     the accepted historical runs.
  4. Emits ``model_semantics_lock.json``.

Run:
    .venv_clustopt/Scripts/python.exe scripts/stage2/gate0b_model_semantics_lock.py
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from models.Clustering_Repository_Builder.utility_generation.worker_setup import (  # noqa: E402
    configure_process,
)

configure_process()

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from models.Clustering_Repository_Builder.experiments.experiment_execution.output_writer import (  # noqa: E402
    _ext, ensure_dir, path_exists, read_json, write_json_atomic,
)

OUT = (Path(REPO) / "results_analysis/clustering_repository/stage2"
       / "oof_utility_splits2_16")
LOCK = OUT / "model_semantics_lock.json"

MLP_RUN_DIR = "results_analysis/mlp/20260615_231715__metric_utility_mlp_split1_holdout"
KNN_MODEL_DIR = "results_analysis/metric_utility_knn/phase_e1/models"

# Accepted historical policy configs -> the binding evidence.
POLICY_CONFIGS = {
    "regressor_top1_raw": "phaseD/regressor_top1_raw",
    "regressor_top5_raw": "phaseD/regressor_top5_raw",
    "regressor_top10_softmax_t05": "phaseD/regressor_top10_softmax_t05",
    "regressor_top_dynamic_predictK_raw": "phaseD/regressor_top_dynamic_predictK_raw",
    "knn_top3_softmax_t05": "phaseE2_knn/knn_top3_softmax_t05",
    "knn_top5_raw": "phaseE2_knn/knn_top5_raw",
    "knn_top_dynamic_predictK_raw": "phaseE2_knn/knn_top_dynamic_predictK_raw",
}
# Policies replayed against persisted Split-1 artifacts.
GOLDEN_POLICIES = ("regressor_top5_raw", "regressor_top1_raw",
                   "regressor_top10_softmax_t05", "knn_top3_softmax_t05",
                   "knn_top5_raw", "regressor_top_dynamic_predictK_raw",
                   "knn_top_dynamic_predictK_raw")
VIEWS = ("x_only", "y_only", "xy_2d")
N_GOLDEN_DATASETS = 12          # deterministic sample, not chosen by performance
TOL = 1e-9


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(_ext(p), "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_obj(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def config_path(rel: str, view: str) -> Path:
    return Path(REPO) / "models/ClustOpt/configs/experiments" / rel / f"{view}.json"


def main() -> int:
    ensure_dir(OUT)
    print("=" * 78)
    print("STAGE-2 GATE 0B  --  FINAL MODEL SEMANTICS LOCK")
    print("=" * 78)
    lock: dict = {"gate": "0B", "inference_only": True}
    ok_all = True

    def chk(name, ok, detail=""):
        nonlocal ok_all
        ok_all &= bool(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
        return bool(ok)

    # ------------------------------------------------------------- bindings
    print("\n" + "-" * 78)
    print("0B.1  BINDING EVIDENCE FROM ACCEPTED HISTORICAL POLICY CONFIGS")
    print("-" * 78)
    bindings = {}
    for pol, rel in POLICY_CONFIGS.items():
        per_view = {}
        for v in VIEWS:
            cfg = read_json(config_path(rel, v))
            if not cfg:
                continue
            cvi = cfg.get("cvi", {})
            per_view[v] = {"type": cvi.get("type"), "params": cvi.get("params", {})}
        bindings[pol] = per_view
        v0 = per_view.get("xy_2d", {})
        p0 = v0.get("params", {})
        src = p0.get("model_run_dir") or p0.get("knn_model_dir")
        print(f"  {pol:38} type={v0.get('type'):18} top_k={str(p0.get('top_k')):8} "
              f"weighting={p0.get('weighting')}")
        print(f"  {'':38} artifact={src}")
    lock["policy_bindings"] = bindings

    mlp_dirs = {b[v]["params"].get("model_run_dir")
                for b in bindings.values() for v in b
                if b[v]["type"] == "regressor_dynamic"}
    knn_dirs = {b[v]["params"].get("knn_model_dir")
                for b in bindings.values() for v in b
                if b[v]["type"] == "knn_dynamic"}
    chk("all MLP policies bind to exactly one artifact",
        mlp_dirs == {MLP_RUN_DIR}, str(sorted(mlp_dirs)))
    chk("all KNN policies bind to exactly one artifact",
        knn_dirs == {KNN_MODEL_DIR}, str(sorted(knn_dirs)))

    # ---------------------------------------------------------------- MLP
    print("\n" + "-" * 78)
    print("0B.2  FINAL FULL-60 MLP BINDING")
    print("-" * 78)
    mlp_root = Path(REPO) / MLP_RUN_DIR
    mcfg = read_json(mlp_root / "config.json") or {}
    feat_cols = read_json(mlp_root / "artifacts/feature_columns.json") or []
    targ_cols = read_json(mlp_root / "artifacts/target_columns.json") or []
    folds = sorted(p.name for p in (mlp_root / "folds").iterdir() if p.is_dir())
    split_summary = pd.read_csv(_ext(mlp_root / "artifacts/cv_split_summary.csv"))
    mlp_binding = {
        "artifact_root": MLP_RUN_DIR,
        "training_entry_point": "models/metric_utility_mlp/run_train_split_holdout.py",
        "model_class": "models/metric_utility_mlp -- type=" + str(mcfg.get("model", {}).get("type")),
        "inference_path": ("models/ClustOpt/cluster_validity_indices/"
                           "dynamic_metric_selection/resolvers.resolve_regressor_weights"
                           " -> regressor_predictor_provider.get_regressor_predictor"),
        "model": mcfg.get("model"), "loss": mcfg.get("loss"),
        "training": mcfg.get("training"),
        "data": {k: mcfg.get("data", {}).get(k) for k in
                 ("n_feature_columns", "n_target_columns", "target_prefix",
                  "split_group_column", "csv_path")},
        "checkpoint_policy": "fold_ensemble",
        "folds_present": folds,
        "checkpoint_selection": "best_model.pt (lowest inner-validation loss)",
        "feature_scaler": "folds/<fold>/x_scaler.pkl",
        "n_features": len(feat_cols), "n_targets": len(targ_cols),
        "feature_schema_hash": sha256_obj(feat_cols),
        "target_schema_hash": sha256_obj(targ_cols),
        "metric_order_hash": sha256_obj(
            [c[len("utility__"):] if c.startswith("utility__") else c for c in targ_cols]),
        "cv_split_summary": split_summary.to_dict("records"),
        "checkpoint_sha256": {f: sha256_file(mlp_root / "folds" / f / "best_model.pt")
                              for f in folds},
        "scaler_sha256": {f: sha256_file(mlp_root / "folds" / f / "x_scaler.pkl")
                          for f in folds},
    }
    lock["final_mlp_binding"] = mlp_binding
    print(f"  artifact       {MLP_RUN_DIR}")
    print(f"  architecture   {mcfg['model']['type']} "
          f"hidden={mcfg['model']['hidden_dims']} act={mcfg['model']['activation']} "
          f"out={mcfg['model']['output_activation']}")
    print(f"  loss           {mcfg['loss']['mode']} mse={mcfg['loss']['mse_weight']} "
          f"rank={mcfg['loss']['pairwise_rank_weight']} "
          f"topk_mse={mcfg['loss']['topk_mse_weight']} topk={mcfg['loss']['topk']}")
    print(f"  optim          {mcfg['training']['optimizer']} "
          f"lr={mcfg['training']['learning_rate']} bs={mcfg['training']['batch_size']} "
          f"epochs={mcfg['training']['epochs']} "
          f"es_patience={mcfg['training']['early_stopping_patience']} "
          f"seed={mcfg['training']['seed']}")
    print(f"  folds present  {folds}   checkpoint_policy=fold_ensemble")
    r = split_summary.iloc[0]
    print(f"  fold_0 split   train_groups={int(r.train_groups)} "
          f"val_groups={int(r.val_groups)} test_groups={int(r.test_groups)} "
          f"leaks={int(r.leak_train_val)}/{int(r.leak_train_test)}/{int(r.leak_val_test)}")
    chk("MLP: 250 features / 60 targets",
        len(feat_cols) == 250 and len(targ_cols) == 60,
        f"{len(feat_cols)}/{len(targ_cols)}")
    chk("MLP: train+val groups == 16,013 (Splits 2-16)",
        int(r.train_groups) + int(r.val_groups) == 16013,
        f"{int(r.train_groups)}+{int(r.val_groups)}")
    chk("MLP: test groups == 1,055 (Split 1 held out)", int(r.test_groups) == 1055)
    chk("MLP: zero intra-fold leakage",
        int(r.leak_train_val) == 0 and int(r.leak_train_test) == 0
        and int(r.leak_val_test) == 0)
    chk("MLP: no target column is also a feature",
        not (set(feat_cols) & set(targ_cols)))

    # ---------------------------------------------------------------- KNN
    print("\n" + "-" * 78)
    print("0B.3  FINAL KNN BINDING")
    print("-" * 78)
    knn_root = Path(REPO) / KNN_MODEL_DIR
    kmeta = read_json(knn_root / "final_knn_metadata.json") or {}
    ksel = read_json(Path(REPO) / "results_analysis/metric_utility_knn/phase_e1"
                     / "selected_configuration.json") or {}
    kfeat = read_json(knn_root / "feature_schema.json") or {}
    kmetr = read_json(knn_root / "utility_metric_schema.json") or {}
    knn_binding = {
        "artifact_root": KNN_MODEL_DIR,
        "source_package": "models/metric_utility_knn",
        "training_entry_point": "models/metric_utility_knn/run_phase_e1.py",
        "predictor": "models/metric_utility_knn/knn_predictor.py",
        "final_model_builder": "models/metric_utility_knn/final_model_builder.py",
        "inference_path": ("models/ClustOpt/cluster_validity_indices/"
                           "dynamic_metric_selection/resolvers.resolve_knn_weights"
                           " -> knn_predictor_provider"),
        "representation": ("neighbour REPOSITORY, not a fitted sklearn estimator: "
                           "per-view training_features.npy / training_utilities.npy "
                           "/ training_record_ids.npy + preprocessor.pkl"),
        "config": kmeta.get("config"),
        "selection_rule": ksel.get("selection", {}).get("selection_rule"),
        "one_se_candidates": ksel.get("selection", {}).get("candidate_config_ids"),
        "dynamic_topk_heuristic": kmeta.get("dynamic_topk_heuristic"),
        "train_splits": kmeta.get("train_splits"),
        "locked_holdout_split": kmeta.get("locked_holdout_split"),
        "split1_in_training": kmeta.get("split1_in_training"),
        "per_view_train_counts": kmeta.get("per_view_train_counts"),
        "n_features": kmeta.get("n_features"), "n_metrics": kmeta.get("n_metrics"),
        "artifact_sha256": kmeta.get("artifact_sha256"),
        "feature_schema_hash": sha256_obj(kfeat),
        "metric_schema_hash": sha256_obj(kmetr),
    }
    lock["final_knn_binding"] = knn_binding
    c = kmeta.get("config", {})
    print(f"  artifact       {KNN_MODEL_DIR}")
    print(f"  config         {c.get('config_id')}  n_neighbors={c.get('n_neighbors')} "
          f"scaler={c.get('scaler')} distance={c.get('distance')} "
          f"weighting={c.get('neighbor_weighting')}")
    print(f"  representation neighbour repository (npy + preprocessor.pkl), per view")
    print(f"  train_splits   {kmeta.get('train_splits')}")
    print(f"  split1 in train: {kmeta.get('split1_in_training')}   "
          f"per-view train count {list((kmeta.get('per_view_train_counts') or {}).values())[:1]}")
    chk("KNN: Split 1 excluded from the repository",
        kmeta.get("split1_in_training") is False)
    chk("KNN: train splits are exactly 2..16",
        list(kmeta.get("train_splits") or []) == list(range(2, 17)))
    chk("KNN: per-view repository holds 16,013 records",
        set((kmeta.get("per_view_train_counts") or {}).values()) == {16013},
        str(kmeta.get("per_view_train_counts")))
    chk("KNN: 250 features / 60 metrics",
        kmeta.get("n_features") == 250 and kmeta.get("n_metrics") == 60)

    # -------------------------------------------------------- Dynamic K
    print("\n" + "-" * 78)
    print("0B.5  DYNAMIC PREDICT-K")
    print("-" * 78)
    from models.ClustOpt.dynamic_topk_selection.heuristics.selected_dynamic_topk import (
        ALPHA, HEURISTIC_NAME, MAX_K, MIN_K, SOFT_CAP_BASE,
    )
    dyn = {
        "kind": "DETERMINISTIC -- not a learned model",
        "heuristic": HEURISTIC_NAME,
        "source": ("models/ClustOpt/dynamic_topk_selection/heuristics/"
                   "selected_dynamic_topk.py"),
        "formula": ("clean u to [0,1]; k_rel = #{u >= ALPHA*max(u)}; "
                    "K = k_rel if k_rel <= SOFT_CAP_BASE else ceil((k_rel+SOFT_CAP_BASE)/2); "
                    "K = clip(K, MIN_K, MAX_K)"),
        "ALPHA": ALPHA, "SOFT_CAP_BASE": SOFT_CAP_BASE,
        "MIN_K": MIN_K, "MAX_K": MAX_K,
        "k_source_for_deployable_policies": "predicted",
        "oof_treatment": ("no cross-fitting required; it consumes the fold's own OOF "
                          "utility vector"),
        "excluded": "dynamic_realK (oracle-assisted) and all real_* utility oracles",
    }
    lock["dynamic_predict_k_binding"] = dyn
    print(f"  {dyn['kind']}")
    print(f"  heuristic  {HEURISTIC_NAME}  ALPHA={ALPHA} SOFT_CAP_BASE={SOFT_CAP_BASE} "
          f"MIN_K={MIN_K} MAX_K={MAX_K}")
    print(f"  formula    {dyn['formula']}")
    chk("Dynamic-K heuristic matches the KNN metadata",
        kmeta.get("dynamic_topk_heuristic") == HEURISTIC_NAME)

    # ------------------------------------------------- weighting semantics
    lock["weighting_semantics"] = {
        "source": ("models/ClustOpt/cluster_validity_indices/"
                   "dynamic_metric_selection/weighting.py"),
        "RAW (normalized_positive)": "clip negatives to 0, normalise to sum 1; "
                                     "all-non-positive -> uniform",
        "SOFTMAX (softmax_t05)": "softmax(scores / 0.5), max-shifted for stability",
        "TopK": "sort by (-score, name); non-finite -> -inf; deterministic name tiebreak",
    }

    # ---------------------------------------------- golden reproduction
    print("\n" + "-" * 78)
    print("0B.6  GOLDEN REPRODUCTION vs PERSISTED SPLIT-1 ARTIFACTS (inference only)")
    print("-" * 78)
    from models.ClustOpt.cluster_validity_indices.dynamic_metric_selection.resolvers import (
        resolve_knn_weights, resolve_regressor_weights,
    )
    split_csv = (Path(REPO) / "results_analysis/clustering_repository/analyzed_data"
                 / "experiment_splits/dataset_split_assignments.csv")
    a = pd.read_csv(_ext(split_csv))
    s1 = a[a["split_id"].astype("Int64") == 1].sort_values("dataset_id")
    sample = s1.iloc[:: max(1, len(s1) // N_GOLDEN_DATASETS)].head(N_GOLDEN_DATASETS)
    print(f"  deterministic sample: {len(sample)} Split-1 datasets "
          f"(every {max(1, len(s1)//N_GOLDEN_DATASETS)}th by dataset_id)")

    rows = []
    for pol in GOLDEN_POLICIES:
        for _, ds in sample.iterrows():
            dsdir = Path(ds["dataset_dir"])
            for v in VIEWS:
                cfg = read_json(config_path(POLICY_CONFIGS[pol], v))
                params = cfg["cvi"]["params"]
                kind = cfg["cvi"]["type"]
                persisted = read_json(dsdir / "experiments" / "split_01" / pol / v
                                      / "selected_metrics.json")
                if not persisted:
                    continue
                fn = (resolve_regressor_weights if kind == "regressor_dynamic"
                      else resolve_knn_weights)
                res = fn(params=params, dataset_dir=dsdir, view_id=v)
                exp_m = list(persisted.get("selected_metrics") or [])
                exp_w = dict(persisted.get("selected_metric_weights") or {})
                got_m = list(res.selected_metrics)
                got_w = dict(res.weights)
                w_err = max((abs(got_w.get(k, np.nan) - exp_w.get(k, np.nan))
                             for k in set(exp_w) | set(got_w)), default=np.nan)
                rows.append({
                    "policy": pol, "dataset_id": ds["dataset_id"], "view_id": v,
                    "kind": kind,
                    "metrics_match": got_m == exp_m,
                    "metric_set_match": set(got_m) == set(exp_m),
                    "n_expected": len(exp_m), "n_got": len(got_m),
                    "max_weight_abs_err": float(w_err),
                    "weights_match": bool(np.isfinite(w_err) and w_err <= 1e-6),
                    "expected_top1": exp_m[0] if exp_m else None,
                    "got_top1": got_m[0] if got_m else None,
                })
    g = pd.DataFrame(rows)
    g.to_csv(_ext(OUT / "gate0b_golden_reproduction.csv"), index=False)
    print(f"\n  {'policy':38} {'n':>4} {'order':>7} {'set':>7} {'weights':>8} {'maxWerr':>10}")
    for pol, grp in g.groupby("policy"):
        print(f"  {pol:38} {len(grp):4d} "
              f"{grp['metrics_match'].mean()*100:6.1f}% "
              f"{grp['metric_set_match'].mean()*100:6.1f}% "
              f"{grp['weights_match'].mean()*100:7.1f}% "
              f"{grp['max_weight_abs_err'].max():10.2e}")
    chk("GOLDEN: Top-K metric ORDER reproduced on every sampled run",
        bool(g["metrics_match"].all()),
        f"{int((~g['metrics_match']).sum())} mismatches of {len(g)}")
    chk("GOLDEN: weights reproduced within 1e-6",
        bool(g["weights_match"].all()),
        f"max err {g['max_weight_abs_err'].max():.2e}")
    lock["golden_reproduction"] = {
        "n_checks": int(len(g)), "n_datasets": int(len(sample)),
        "policies": list(GOLDEN_POLICIES),
        "order_match_rate": float(g["metrics_match"].mean()),
        "weight_match_rate": float(g["weights_match"].mean()),
        "max_weight_abs_err": float(g["max_weight_abs_err"].max()),
        "split1_used_for": "inference-only verification; never for fitting",
    }

    # ---------------------------------------------- rejected implementations
    lock["excluded_implementations"] = {
        "real_* (10 policies)": "utility oracle -- consumes true utility vector",
        "*_dynamic_realK_* (4 policies)": "oracle-assisted -- K sized from the true "
                                          "utility vector",
        "head14 MLP (20260823_195830__metric_utility_mlp_head14_split1_holdout)":
            "Stage-1B artifact for the 2x3 ablation only; no accepted historical "
            "policy binds to it (all regressor_* configs bind to 20260615_231715)",
        "knn phase_e2 artifacts": "phase_e2 is the ClustOpt integration output, not a "
                                  "predictor; the predictor artifact is phase_e1/models",
        "KNN configuration search candidates": "the Phase-E1 search evaluated a grid of "
                                               "(n_neighbors x scaler x distance x "
                                               "neighbour weighting); all rejected by the "
                                               "one-SE rule except "
                                               "n015__standard__euclidean__inverse_distance",
    }
    lock["source_commit"] = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
        text=True).stdout.strip()
    lock["environment_lock"] = "environment/stage1_clustopt_requirements.lock.txt"
    lock["gate_status"] = "PASS" if ok_all else "FAIL"
    write_json_atomic(LOCK, lock)

    print("\n" + "=" * 78)
    print(f"GATE 0B: {'PASS' if ok_all else 'FAIL'}   ->  {LOCK}")
    print("=" * 78)
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
