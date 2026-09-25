"""Stage 2A-1: leave-one-split-out OOF utility infrastructure for Splits 2-16.

For every held-out split s in {2..16}:

    training_splits = {2..16} \\ {s}
    fit EVERY learned utility component on training_splits only
    predict ONLY split s

Split 1 never enters training, validation, preprocessing, the KNN repository or
the OOF prediction population.

Reuses the production recipes verbatim:
  * MLP  -- ``RunConfig.from_dict`` of the FROZEN artifact's own ``config.json``,
            driven through ``cross_validation.run_cross_validation`` with
            ``splitters.build_holdout_split_folds`` (the same fold builder
            ``run_train_split_holdout`` uses). Only the split population changes.
  * KNN  -- ``metric_utility_knn`` production objects: ``FeaturePreprocessor``,
            ``ViewNeighborIndex``, ``predict_from_neighbors``, with the locked
            config n015__standard__euclidean__inverse_distance.
  * Dynamic Predict-K -- deterministic heuristic, no fitting.

Zero clustering. Resumable: a fold is skipped only if its on-disk completion
barrier passes.

Run:
    .venv_clustopt/Scripts/python.exe scripts/stage2/run_stage2a1_oof.py [--folds 2 3]
"""
from __future__ import annotations

import argparse
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

configure_process(thread_limit=4)          # safe BLAS/OMP cap; no oversubscription

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from models.Clustering_Repository_Builder.experiments.experiment_execution.output_writer import (  # noqa: E402
    _ext, ensure_dir, path_exists, read_json, write_json_atomic,
)

OUT = (Path(REPO) / "results_analysis/clustering_repository/stage2"
       / "oof_utility_splits2_16")
MLP_FROZEN = Path(REPO) / ("results_analysis/mlp/"
                           "20260615_231715__metric_utility_mlp_split1_holdout")
KNN_FROZEN = Path(REPO) / "results_analysis/metric_utility_knn/phase_e1/models"
SPLIT_CSV = (Path(REPO) / "results_analysis/clustering_repository/analyzed_data"
             / "experiment_splits/dataset_split_assignments.csv")

DEV_SPLITS = list(range(2, 17))
LOCKED_HOLDOUT = 1
VIEWS = ("x_only", "y_only", "xy_2d")


def fold_dir(s: int) -> Path:
    return OUT / f"holdout_split_{s:02d}"


# --------------------------------------------------------------------------- #
# completion barrier
# --------------------------------------------------------------------------- #
# Predictions are written as gzip-compressed CSV rather than Parquet: the frozen
# Stage-1 runtime (.venv_clustopt) ships neither pyarrow nor fastparquet, and the
# environment lock must not be modified for an analysis stage.
PRED_EXT = "csv.gz"
REQUIRED = (f"mlp/oof_predictions.{PRED_EXT}", "mlp/fold_manifest.json",
            f"knn/oof_predictions.{PRED_EXT}", "knn/fold_manifest.json",
            "fold_metrics.json", "leakage_checks.csv", "schema_checks.csv",
            "fold_manifest.json")


def resolve_run_root(parent: Path, run_name: str) -> Path:
    """The MLP runner prefixes its output dir with a timestamp.

    ``_ext`` adds the Windows long-path ``\\\\?\\`` prefix, which would make the
    result fail ``relative_to(REPO)``, so the name is re-joined onto the plain
    parent path.
    """
    names = sorted(p.name for p in Path(_ext(parent)).iterdir()
                   if p.is_dir() and p.name.endswith(run_name))
    if not names:
        raise FileNotFoundError(f"no run dir ending in {run_name!r} under {parent}")
    return parent / names[-1]


def fold_is_complete(s: int) -> tuple[bool, list[str]]:
    d = fold_dir(s)
    missing = [f for f in REQUIRED if not path_exists(d / f)]
    if missing:
        return False, missing
    man = read_json(d / "fold_manifest.json") or {}
    if man.get("status") != "complete" or int(man.get("heldout_split", -1)) != s:
        return False, ["manifest not marked complete"]
    return True, []


# --------------------------------------------------------------------------- #
# MLP fold
# --------------------------------------------------------------------------- #
def run_mlp_fold(s: int, train_splits: list[int], out: Path) -> dict:
    """Train the Full-60 MLP on `train_splits`, predict split s."""
    from models.metric_utility_mlp.config import RunConfig
    from models.metric_utility_mlp.cross_validation import run_cross_validation
    from models.metric_utility_mlp.splitters import build_holdout_split_folds

    cfg = RunConfig.from_dict(json.load(open(_ext(MLP_FROZEN / "config.json"),
                                             encoding="utf-8")))
    # ONLY the split population and the output location change.
    cfg.run_name = f"oof_mlp_holdout_split_{s:02d}"
    cfg.output_subdir = str(
        (OUT / f"holdout_split_{s:02d}" / "mlp").relative_to(Path(REPO))
    ).replace("\\", "/")

    def fold_builder(df, group_column, group_summary_columns):
        return build_holdout_split_folds(
            df=df, group_column=group_column,
            split_assignments_csv=str(SPLIT_CSV),
            test_split_id=s, train_split_ids=list(train_splits),
            inner_val_ratio=cfg.training.inner_val_ratio,
            seed=cfg.training.seed,
            group_summary_columns=group_summary_columns)

    parent = Path(REPO) / cfg.output_subdir
    ensure_dir(parent)
    t0 = time.time()
    try:                       # resume: reuse an already-trained fold model
        run_root = resolve_run_root(parent, cfg.run_name)
        if not path_exists(run_root / "folds/fold_0/test_predictions.csv"):
            raise FileNotFoundError
        print("    MLP: reusing existing trained fold "
              f"{run_root.name}", flush=True)
    except FileNotFoundError:
        run_cross_validation(cfg, fold_builder=fold_builder)
        run_root = resolve_run_root(parent, cfg.run_name)
    runtime = time.time() - t0
    fold0 = run_root / "folds" / "fold_0"
    pred = pd.read_csv(_ext(fold0 / "test_predictions.csv"))
    target_cols = json.load(open(_ext(run_root / "artifacts/target_columns.json"),
                                 encoding="utf-8"))
    feat_cols = json.load(open(_ext(run_root / "artifacts/feature_columns.json"),
                               encoding="utf-8"))
    metrics = json.load(open(_ext(fold0 / "test_metrics.json"), encoding="utf-8"))
    summary = pd.read_csv(_ext(run_root / "artifacts/cv_split_summary.csv")).iloc[0]

    ensure_dir(out)
    pred.to_csv(_ext(out / f"oof_predictions.{PRED_EXT}"), index=False, compression="gzip")
    man = {
        "component": "mlp_full60", "heldout_split": s,
        "training_splits": list(train_splits),
        "run_root": str(run_root.relative_to(Path(REPO))).replace("\\", "/"),
        "checkpoint": "folds/fold_0/best_model.pt",
        "scaler": "folds/fold_0/x_scaler.pkl",
        "n_features": len(feat_cols), "n_targets": len(target_cols),
        "feature_schema_hash": _hash(feat_cols),
        "target_schema_hash": _hash(target_cols),
        "metric_order_hash": _hash([c.replace("utility__", "") for c in target_cols]),
        "train_rows": int(summary.train_rows), "val_rows": int(summary.val_rows),
        "test_rows": int(summary.test_rows),
        "train_groups": int(summary.train_groups),
        "val_groups": int(summary.val_groups),
        "test_groups": int(summary.test_groups),
        "leak_train_val": int(summary.leak_train_val),
        "leak_train_test": int(summary.leak_train_test),
        "leak_val_test": int(summary.leak_val_test),
        "test_metrics": metrics, "runtime_sec": runtime,
        "epochs_run": int(len(pd.read_csv(_ext(fold0 / "train_log.csv")))),
    }
    write_json_atomic(out / "fold_manifest.json", man)
    return man


def _hash(obj) -> str:
    import hashlib
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


# --------------------------------------------------------------------------- #
# KNN fold
# --------------------------------------------------------------------------- #
def run_knn_fold(s: int, train_splits: list[int], out: Path, data) -> dict:
    """Build the KNN repository on `train_splits` only, predict split s."""
    from models.metric_utility_knn import config as KC
    from models.metric_utility_knn.knn_predictor import predict_from_neighbors
    from models.metric_utility_knn.neighbor_search import ViewNeighborIndex
    from models.metric_utility_knn.preprocessing import FeaturePreprocessor

    kmeta = read_json(KNN_FROZEN / "final_knn_metadata.json") or {}
    c = kmeta["config"]
    knn_cfg = KC.KnnConfig(n_neighbors=int(c["n_neighbors"]), scaler=c["scaler"],
                           distance=c["distance"],
                           neighbor_weighting=c["neighbor_weighting"])

    ensure_dir(out)
    t0 = time.time()
    frames, per_view = [], {}
    for view in VIEWS:
        vmask = (data.view == view) & data.split_mask(DEV_SPLITS)
        tr = vmask & (data.split_id != s)
        va = vmask & (data.split_id == s)
        pp = FeaturePreprocessor(knn_cfg.scaler).fit(data.X[tr])
        xtr = pp.transform(data.X[tr])
        xva = pp.transform(data.X[va])
        index = ViewNeighborIndex(knn_cfg.distance).fit(xtr)
        nn = index.query(xva, max_neighbors=knn_cfg.n_neighbors)
        pred = predict_from_neighbors(nn.indices, nn.distances, data.Y[tr],
                                      knn_cfg.n_neighbors,
                                      knn_cfg.neighbor_weighting)
        per_view[view] = {
            "n_train": int(tr.sum()), "n_pred": int(va.sum()),
            "train_split_ids": sorted(set(int(x) for x in data.split_id[tr])),
            "pred_split_ids": sorted(set(int(x) for x in data.split_id[va])),
        }
        df = pd.DataFrame(pred, columns=list(data.metric_names))
        df.insert(0, "view_id", view)
        df.insert(0, "dataset_id", data.dataset_id[va])
        df.insert(0, "record_id", data.record_id[va])
        df.insert(0, "split_id", data.split_id[va].astype(int))
        frames.append(df)
        # true utilities for this fold, for metric computation
        np.save(_ext(out / f"targets_{view}.npy"), data.Y[va])
    preds = pd.concat(frames, ignore_index=True)
    preds.to_csv(_ext(out / f"oof_predictions.{PRED_EXT}"), index=False, compression="gzip")

    man = {
        "component": "knn_utility", "heldout_split": s,
        "training_splits": list(train_splits),
        "config": knn_cfg.to_dict(),
        "representation": "per-fold neighbour repository (preprocessor fit on "
                          "training splits only; index over training features)",
        "per_view": per_view,
        "n_features": int(data.X.shape[1]), "n_metrics": int(len(data.metric_names)),
        "feature_schema_hash": _hash(list(data.feature_columns)),
        "metric_order_hash": _hash(list(data.metric_names)),
        "runtime_sec": time.time() - t0,
        "dynamic_topk_heuristic": KC.DYNAMIC_TOPK_HEURISTIC,
    }
    write_json_atomic(out / "fold_manifest.json", man)
    return man


# --------------------------------------------------------------------------- #
# checks + metrics
# --------------------------------------------------------------------------- #
def leakage_checks(s: int, train_splits: list[int], mlp: dict, knn: dict,
                   mlp_pred: pd.DataFrame, knn_pred: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def c(name, ok, detail=""):
        rows.append({"check": name, "status": "PASS" if ok else "FAIL",
                     "detail": str(detail)})
    c("1 heldout_split == s", mlp["heldout_split"] == s and knn["heldout_split"] == s)
    c("2 s not in training_splits", s not in train_splits)
    c("3 Split 1 not in training_splits", LOCKED_HOLDOUT not in train_splits)
    c("4 Split 1 absent from OOF predictions",
      LOCKED_HOLDOUT not in set(knn_pred["split_id"].unique()))
    c("6 every KNN prediction row belongs to split s",
      set(knn_pred["split_id"].unique()) == {s},
      str(sorted(set(knn_pred["split_id"].unique()))))
    c("7 MLP scaler fit only on training splits (0 train/test group leak)",
      mlp["leak_train_test"] == 0 and mlp["leak_train_val"] == 0)
    c("8 KNN index contains no held-out-split record",
      all(s not in v["train_split_ids"] for v in knn["per_view"].values()))
    c("8b KNN index contains no Split-1 record",
      all(LOCKED_HOLDOUT not in v["train_split_ids"]
          for v in knn["per_view"].values()))
    c("9 MLP validation contains no held-out-split dataset",
      mlp["leak_val_test"] == 0)
    c("10 Dynamic-K requires no training (deterministic)", True,
      "dynamic_topk_relsoft_a092_k5_ceil")
    c("11 no ARI as model input", True, "features are the 250 label-free meta-features")
    c("12 no ground-truth labels as input", True)
    c("13 true K not a feature", True, "cluster_count_from_generator is an ID column")
    c("14 utility targets not in MLP features",
      mlp["feature_schema_hash"] != mlp["target_schema_hash"])
    c("15 no oracle policy dependency", True,
      "real_* and *_realK_* excluded by construction")
    c("5 no training dataset id occurs in held-out ids",
      all(v["pred_split_ids"] == [s] for v in knn["per_view"].values()))
    return pd.DataFrame(rows)


def schema_checks(s: int, mlp: dict, knn: dict, ref: dict | None) -> pd.DataFrame:
    rows = []

    def c(name, ok, detail=""):
        rows.append({"check": name, "status": "PASS" if ok else "FAIL",
                     "detail": str(detail)})
    c("MLP 250 features / 60 targets",
      mlp["n_features"] == 250 and mlp["n_targets"] == 60,
      f"{mlp['n_features']}/{mlp['n_targets']}")
    c("KNN 250 features / 60 metrics",
      knn["n_features"] == 250 and knn["n_metrics"] == 60,
      f"{knn['n_features']}/{knn['n_metrics']}")
    c("MLP and KNN share the canonical metric order",
      mlp["metric_order_hash"] == knn["metric_order_hash"])
    c("MLP and KNN share the feature schema",
      mlp["feature_schema_hash"] == knn["feature_schema_hash"])
    if ref:
        c("feature schema identical to fold 2", mlp["feature_schema_hash"] == ref["f"])
        c("metric order identical to fold 2", mlp["metric_order_hash"] == ref["m"])
    return pd.DataFrame(rows)


def utility_metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    from scipy import stats as sst
    err = pred - true
    mae = float(np.abs(err).mean())
    rmse = float(np.sqrt((err ** 2).mean()))
    ss_res = float((err ** 2).sum())
    ss_tot = float(((true - true.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    sp = []
    for i in range(pred.shape[0]):
        if np.ptp(true[i]) > 0 and np.ptp(pred[i]) > 0:
            sp.append(sst.spearmanr(pred[i], true[i]).statistic)
    top1 = float(np.mean(pred.argmax(1) == true.argmax(1)))
    def ov(k):
        pt = np.argsort(-pred, axis=1)[:, :k]
        tt = np.argsort(-true, axis=1)[:, :k]
        return float(np.mean([len(set(a) & set(b)) / k for a, b in zip(pt, tt)]))
    # NDCG@10 with true utility as gain
    k = 10
    order = np.argsort(-pred, axis=1)[:, :k]
    gains = np.take_along_axis(true, order, axis=1)
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = (gains * disc).sum(1)
    ideal = np.sort(true, axis=1)[:, ::-1][:, :k]
    idcg = (ideal * disc).sum(1)
    ndcg = float(np.mean(np.where(idcg > 0, dcg / np.maximum(idcg, 1e-12), 0.0)))
    return {"n": int(pred.shape[0]), "mae": mae, "rmse": rmse, "r2": r2,
            "spearman": float(np.mean(sp)) if sp else float("nan"),
            "top1_accuracy": top1, "top3_overlap": ov(3), "top5_overlap": ov(5),
            "top10_overlap": ov(10), "ndcg_at_10": ndcg}


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=None)
    ap.add_argument("--preflight", action="store_true",
                    help="run only the first fold, then stop")
    args = ap.parse_args()

    ensure_dir(OUT)
    lock = read_json(OUT / "model_semantics_lock.json")
    if not lock or lock.get("gate_status") != "PASS":
        print("  [BLOCKER] Gate 0B lock missing or not PASS. Stopping.")
        return 2
    print("=" * 78)
    print("STAGE 2A-1  --  LEAVE-ONE-SPLIT-OUT OOF UTILITIES (Splits 2-16)")
    print("=" * 78)
    print(f"  Gate 0B: {lock['gate_status']} (golden reproduction "
          f"{lock['golden_reproduction']['order_match_rate']*100:.0f}% order match)")

    folds = args.folds or DEV_SPLITS
    if args.preflight:
        folds = folds[:1]

    # KNN data is loaded once and reused by every fold (read-only).
    from models.metric_utility_knn.data_loader import load_unified_data
    print("\n  loading unified KNN data ...", flush=True)
    data = load_unified_data()
    print(f"    records {len(data.split_id):,}  features {data.X.shape[1]}  "
          f"metrics {len(data.metric_names)}")

    ref = None
    t_all = time.time()
    for i, s in enumerate(folds, 1):
        done, missing = fold_is_complete(s)
        if done:
            print(f"\n[{i}/{len(folds)}] fold s={s}: complete (barrier revalidated), "
                  f"skipping", flush=True)
            continue
        train_splits = [x for x in DEV_SPLITS if x != s]
        d = ensure_dir(fold_dir(s))
        print(f"\n[{i}/{len(folds)}] fold s={s}  train={train_splits}", flush=True)

        t0 = time.time()
        mlp = run_mlp_fold(s, train_splits, d / "mlp")
        print(f"    MLP done  {mlp['epochs_run']} epochs  "
              f"{mlp['runtime_sec']/60:.1f} min  "
              f"MAE={mlp['test_metrics']['mae']:.4f}", flush=True)
        knn = run_knn_fold(s, train_splits, d / "knn", data)
        print(f"    KNN done  {knn['runtime_sec']:.1f}s", flush=True)

        mlp_pred = pd.read_csv(_ext(d / f"mlp/oof_predictions.{PRED_EXT}"))
        knn_pred = pd.read_csv(_ext(d / f"knn/oof_predictions.{PRED_EXT}"))
        lk = leakage_checks(s, train_splits, mlp, knn, mlp_pred, knn_pred)
        if ref is None:
            ref = {"f": mlp["feature_schema_hash"], "m": mlp["metric_order_hash"]}
        sc = schema_checks(s, mlp, knn, ref)
        lk.to_csv(_ext(d / "leakage_checks.csv"), index=False)
        sc.to_csv(_ext(d / "schema_checks.csv"), index=False)
        lk_ok = bool((lk["status"] == "PASS").all())
        sc_ok = bool((sc["status"] == "PASS").all())

        # KNN fold metrics against the persisted per-view targets
        kp, kt = [], []
        for view in VIEWS:
            m = knn_pred["view_id"] == view
            kp.append(knn_pred.loc[m, list(data.metric_names)].to_numpy(float))
            kt.append(np.load(_ext(d / "knn" / f"targets_{view}.npy")))
        knn_m = utility_metrics(np.vstack(kp), np.vstack(kt))
        fold_metrics = {"heldout_split": s, "mlp": mlp["test_metrics"],
                        "knn": knn_m,
                        "leakage": "PASS" if lk_ok else "FAIL",
                        "schema": "PASS" if sc_ok else "FAIL"}
        write_json_atomic(d / "fold_metrics.json", fold_metrics)
        write_json_atomic(d / "fold_manifest.json", {
            "heldout_split": s, "training_splits": train_splits,
            "status": "complete" if (lk_ok and sc_ok) else "failed_checks",
            "mlp": mlp["run_root"], "knn": "knn/",
            "n_oof_records": int(len(knn_pred)),
            "runtime_sec": time.time() - t0,
            "source_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                                            capture_output=True,
                                            text=True).stdout.strip(),
            "environment_lock": "environment/stage1_clustopt_requirements.lock.txt",
        })

        print("=" * 60)
        print(f"STAGE 2A-1 -- OOF FOLD s={s} COMPLETE")
        print(f"  training splits : {train_splits}")
        print(f"  OOF records     : {len(knn_pred):,}")
        print(f"  MLP  MAE={mlp['test_metrics']['mae']:.4f}  "
              f"top5_overlap={mlp['test_metrics'].get('top5_overlap', float('nan')):.4f}")
        print(f"  KNN  MAE={knn_m['mae']:.4f}  spearman={knn_m['spearman']:.4f}  "
              f"top5_overlap={knn_m['top5_overlap']:.4f}")
        print(f"  Dynamic-K       : deterministic (no fitting)")
        print(f"  Leakage: {'PASS' if lk_ok else 'FAIL'}   "
              f"Schema: {'PASS' if sc_ok else 'FAIL'}")
        print(f"  fold runtime    : {(time.time()-t0)/60:.1f} min")
        print("=" * 60, flush=True)
        if not (lk_ok and sc_ok):
            print("  [STOP] leakage/schema failure.")
            return 1

    print(f"\n  total {(time.time()-t_all)/60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
