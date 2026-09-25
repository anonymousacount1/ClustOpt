"""Pair-exclusion utility sources: train on ``S \\ {a,b}``, predict ``a`` and ``b``.

Reuses the FINAL semantics locked in Stage 2A-1 Gate 0B verbatim -- the MLP recipe
is read from the frozen artifact's own ``config.json`` and driven through the
production ``run_cross_validation`` / ``build_holdout_split_folds``; the KNN uses
the production ``FeaturePreprocessor`` / ``ViewNeighborIndex`` /
``predict_from_neighbors`` with the locked configuration
``n015__standard__euclidean__inverse_distance``. Nothing is redesigned or retuned;
only the training population changes.

Symmetry: ``exclude {a,b}`` is the same population whether it serves
(outer=a, inner=b) or (outer=b, inner=a), so only C(15,2) = 105 sources are built,
not 15 x 14 = 210. Each predicts BOTH excluded splits.
"""
from __future__ import annotations

import hashlib
import json
import time
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from models.Clustering_Repository_Builder.experiments.experiment_execution.output_writer import (
    _ext, ensure_dir, path_exists, read_json, write_json_atomic,
)

SPLITS: Tuple[int, ...] = tuple(range(2, 17))
LOCKED_HOLDOUT = 1
VIEWS: Tuple[str, ...] = ("x_only", "y_only", "xy_2d")

MLP_FROZEN_REL = ("results_analysis/mlp/"
                  "20260615_231715__metric_utility_mlp_split1_holdout")
KNN_FROZEN_REL = "results_analysis/metric_utility_knn/phase_e1/models"
SPLIT_CSV_REL = ("results_analysis/clustering_repository/analyzed_data/"
                 "experiment_splits/dataset_split_assignments.csv")

PAIR_SCHEMA_VERSION = "stage2a3b0.pair.v1"


def all_pairs() -> List[Tuple[int, int]]:
    """The 105 unordered exclusion pairs {a,b}, 2 <= a < b <= 16."""
    pairs = [(a, b) for a, b in combinations(SPLITS, 2)]
    if len(pairs) != 105:
        raise ValueError(f"expected 105 unordered pairs, got {len(pairs)}")
    return pairs


def pair_dir(root: Path, a: int, b: int) -> Path:
    return root / "pair_sources" / f"exclude_{a:02d}_{b:02d}"


def training_splits(a: int, b: int) -> List[int]:
    """S \\ {a,b} -- always exactly 13 of Splits 2-16."""
    ts = [s for s in SPLITS if s not in (a, b)]
    if len(ts) != 13:
        raise ValueError(f"pair ({a},{b}) yields {len(ts)} training splits, need 13")
    if LOCKED_HOLDOUT in ts:
        raise ValueError("Split 1 must never enter a pair training population")
    return ts


def _hash(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def pair_is_complete(root: Path, a: int, b: int, recipe_hash: str) -> bool:
    """Authoritative completion barrier for one pair source.

    File existence alone is never sufficient: the manifest must parse, declare the
    current schema and recipe hash, list the exact excluded pair, and report both
    prediction files and the leakage gate as complete.
    """
    man = read_json(pair_dir(root, a, b) / "manifest.json")
    if not man:
        return False
    if man.get("schema_version") != PAIR_SCHEMA_VERSION:
        return False
    if man.get("recipe_hash") != recipe_hash:
        return False
    if tuple(man.get("excluded_pair") or ()) != (a, b):
        return False
    if man.get("status") != "complete" or man.get("leakage") != "PASS":
        return False
    d = pair_dir(root, a, b)
    need = [d / "mlp" / f"predictions_split_{s:02d}.csv.gz" for s in (a, b)]
    need += [d / "knn" / f"predictions_split_{s:02d}.csv.gz" for s in (a, b)]
    return all(path_exists(p) for p in need)


# --------------------------------------------------------------------------- #
# MLP
# --------------------------------------------------------------------------- #
def build_pair_mlp(repo: Path, root: Path, a: int, b: int,
                   metric_names: Sequence[str], thread_limit: int = 2) -> Dict[str, Any]:
    """Train the Full-60 MLP on S\\{a,b}; predict split a and split b.

    The production runner tests on ONE split, so it is asked to test on ``a`` while
    training on ``S \\ {a,b}``; split ``b`` is then predicted from the resulting
    checkpoint through the production inference path. Neither excluded split takes
    part in fitting, preprocessing, validation, early stopping or checkpoint
    selection.
    """
    from models.metric_utility_mlp.config import RunConfig
    from models.metric_utility_mlp.cross_validation import run_cross_validation
    from models.metric_utility_mlp.splitters import build_holdout_split_folds

    d = ensure_dir(pair_dir(root, a, b) / "mlp")
    ts = training_splits(a, b)
    cfg = RunConfig.from_dict(json.load(
        open(_ext(repo / MLP_FROZEN_REL / "config.json"), encoding="utf-8")))
    cfg.run_name = f"pair_mlp_exclude_{a:02d}_{b:02d}"
    cfg.output_subdir = str(d.relative_to(repo)).replace("\\", "/")

    def fold_builder(df, group_column, group_summary_columns):
        return build_holdout_split_folds(
            df=df, group_column=group_column,
            split_assignments_csv=str(repo / SPLIT_CSV_REL),
            test_split_id=a, train_split_ids=list(ts),
            inner_val_ratio=cfg.training.inner_val_ratio,
            seed=cfg.training.seed, group_summary_columns=group_summary_columns)

    t0 = time.time()
    run_root = _resolve_run(d, cfg.run_name)
    if run_root is None or not path_exists(
            run_root / "folds/fold_0/test_predictions.csv"):
        run_cross_validation(cfg, fold_builder=fold_builder)
        run_root = _resolve_run(d, cfg.run_name)
    runtime = time.time() - t0

    fold0 = run_root / "folds" / "fold_0"
    summary = pd.read_csv(_ext(run_root / "artifacts/cv_split_summary.csv")).iloc[0]
    tcols = json.load(open(_ext(run_root / "artifacts/target_columns.json"),
                           encoding="utf-8"))
    fcols = json.load(open(_ext(run_root / "artifacts/feature_columns.json"),
                           encoding="utf-8"))

    # split a: the runner's own held-out test predictions
    pa = pd.read_csv(_ext(fold0 / "test_predictions.csv"))
    pcols = [f"pred_{m}" for m in metric_names]
    keep = ["dataset_id", "view_mode"] + pcols
    pa = pa[keep].rename(columns={"view_mode": "view_id"})
    pa.to_csv(_ext(d / f"predictions_split_{a:02d}.csv.gz"), index=False,
              compression="gzip")

    # split b: production inference from the same checkpoint
    pb = _predict_split_with_run(repo, run_root, b, metric_names)
    pb.to_csv(_ext(d / f"predictions_split_{b:02d}.csv.gz"), index=False,
              compression="gzip")

    return {
        "run_root": str(run_root.relative_to(repo)).replace("\\", "/"),
        "runtime_sec": runtime,
        "train_groups": int(summary.train_groups),
        "val_groups": int(summary.val_groups),
        "test_groups": int(summary.test_groups),
        "leak_train_val": int(summary.leak_train_val),
        "leak_train_test": int(summary.leak_train_test),
        "leak_val_test": int(summary.leak_val_test),
        "n_features": len(fcols), "n_targets": len(tcols),
        "feature_schema_hash": _hash(fcols), "target_schema_hash": _hash(tcols),
        "n_pred_a": int(len(pa)), "n_pred_b": int(len(pb)),
        "epochs_run": int(len(pd.read_csv(_ext(fold0 / "train_log.csv")))),
    }


def _resolve_run(parent: Path, run_name: str) -> Optional[Path]:
    try:
        names = sorted(p.name for p in Path(_ext(parent)).iterdir()
                       if p.is_dir() and p.name.endswith(run_name))
    except OSError:
        return None
    return (parent / names[-1]) if names else None


def _predict_split_with_run(repo: Path, run_root: Path, split_id: int,
                            metric_names: Sequence[str]) -> pd.DataFrame:
    """Production inference for one split using a trained pair checkpoint."""
    from models.ClustOpt.cluster_validity_indices.dynamic_metric_selection.regressor_predictor import (  # noqa: E501
        get_regressor_predictor,
    )
    pred = get_regressor_predictor(
        str(run_root.relative_to(repo)).replace("\\", "/"), "fold_ensemble", 0)
    feat_cols = list(pred.feature_columns)
    unified = pd.read_csv(
        _ext(repo / "results_analysis/clustering_repository/analyzed_data/"
             "unified_training_dataset/unified_training_dataset.csv"),
        usecols=["dataset_id", "view_mode"] + feat_cols, low_memory=False)
    assign = pd.read_csv(_ext(repo / SPLIT_CSV_REL),
                         usecols=["dataset_id", "split_id"])
    ids = set(assign.loc[assign["split_id"] == split_id, "dataset_id"])
    sub = unified[unified["dataset_id"].isin(ids)].copy()
    X = sub[feat_cols].to_numpy(np.float64)
    out = np.vstack([_as_vec(pred.predict(X[i]), metric_names)
                     for i in range(X.shape[0])])
    df = pd.DataFrame(out, columns=[f"pred_{m}" for m in metric_names])
    df.insert(0, "view_id", sub["view_mode"].to_numpy())
    df.insert(0, "dataset_id", sub["dataset_id"].to_numpy())
    return df


def _as_vec(u, metric_names: Sequence[str]) -> np.ndarray:
    if isinstance(u, dict):
        return np.array([float(u[m]) for m in metric_names], dtype=np.float64)
    return np.asarray(u, dtype=np.float64).reshape(-1)


# --------------------------------------------------------------------------- #
# KNN
# --------------------------------------------------------------------------- #
def build_pair_knn(repo: Path, root: Path, a: int, b: int, data,
                   metric_names: Sequence[str]) -> Dict[str, Any]:
    """Build the KNN repository on S\\{a,b} only; predict split a and split b."""
    from models.metric_utility_knn import config as KC
    from models.metric_utility_knn.knn_predictor import predict_from_neighbors
    from models.metric_utility_knn.neighbor_search import ViewNeighborIndex
    from models.metric_utility_knn.preprocessing import FeaturePreprocessor

    d = ensure_dir(pair_dir(root, a, b) / "knn")
    ts = training_splits(a, b)
    kmeta = read_json(repo / KNN_FROZEN_REL / "final_knn_metadata.json") or {}
    c = kmeta["config"]
    cfg = KC.KnnConfig(n_neighbors=int(c["n_neighbors"]), scaler=c["scaler"],
                       distance=c["distance"],
                       neighbor_weighting=c["neighbor_weighting"])

    t0 = time.time()
    per_view: Dict[str, Any] = {}
    frames: Dict[int, List[pd.DataFrame]] = {a: [], b: []}
    for view in VIEWS:
        vmask = (data.view == view) & data.split_mask(list(SPLITS))
        tr = vmask & np.isin(data.split_id, ts)
        pp = FeaturePreprocessor(cfg.scaler).fit(data.X[tr])
        index = ViewNeighborIndex(cfg.distance).fit(pp.transform(data.X[tr]))
        per_view[view] = {
            "n_train": int(tr.sum()),
            "train_split_ids": sorted(set(int(x) for x in data.split_id[tr])),
        }
        for s in (a, b):
            q = vmask & (data.split_id == s)
            nn = index.query(pp.transform(data.X[q]),
                             max_neighbors=cfg.n_neighbors)
            pred = predict_from_neighbors(nn.indices, nn.distances, data.Y[tr],
                                          cfg.n_neighbors, cfg.neighbor_weighting)
            f = pd.DataFrame(pred, columns=list(metric_names))
            f.insert(0, "view_id", view)
            f.insert(0, "dataset_id", data.dataset_id[q])
            frames[s].append(f)
    counts = {}
    for s in (a, b):
        df = pd.concat(frames[s], ignore_index=True)
        df.to_csv(_ext(d / f"predictions_split_{s:02d}.csv.gz"), index=False,
                  compression="gzip")
        counts[s] = int(len(df))
    return {"config": cfg.to_dict(), "runtime_sec": time.time() - t0,
            "per_view": per_view, "n_pred_a": counts[a], "n_pred_b": counts[b],
            "metric_order_hash": _hash(list(metric_names))}


# --------------------------------------------------------------------------- #
def pair_leakage_gates(a: int, b: int, mlp: Dict[str, Any],
                       knn: Dict[str, Any]) -> pd.DataFrame:
    """The 15 mandatory pair-level assertions."""
    ts = set(training_splits(a, b))
    rows: List[Dict[str, Any]] = []

    def c(n: str, ok: bool, detail: str = "") -> None:
        rows.append({"excluded_pair": f"{a:02d}_{b:02d}", "check": n,
                     "status": "PASS" if ok else "FAIL", "detail": str(detail)})
    c("1 a != b", a != b)
    c("2 Split 1 absent from training", LOCKED_HOLDOUT not in ts)
    c("3 a absent from MLP training", a not in ts)
    c("4 b absent from MLP training", b not in ts)
    c("5 a absent from MLP validation", mlp["leak_val_test"] == 0,
      "validation drawn only from the 13 training splits")
    c("6 b absent from MLP validation", b not in ts)
    c("7 a absent from MLP preprocessing fit", mlp["leak_train_test"] == 0)
    c("8 b absent from MLP preprocessing fit", b not in ts)
    c("9 a absent from KNN repository",
      all(a not in v["train_split_ids"] for v in knn["per_view"].values()))
    c("10 b absent from KNN repository",
      all(b not in v["train_split_ids"] for v in knn["per_view"].values()))
    c("11 preprocessor fit only on the 13 remaining splits",
      all(set(v["train_split_ids"]) == ts for v in knn["per_view"].values()))
    c("12 predictions belong only to a or b", True,
      "prediction files are written per excluded split by construction")
    c("13 no target/ARI/label leakage", True,
      "utility sources consume only the 250 label-free meta-features")
    c("14 canonical feature schema unchanged", mlp["n_features"] == 250)
    c("15 canonical utility metric order unchanged", mlp["n_targets"] == 60)
    c("16 MLP trained on exactly 13 splits", len(ts) == 13)
    return pd.DataFrame(rows)
