"""
configuration_search.py
=======================

Leakage-free LOSO search over the 96 KNN configurations (plan §13).

Efficiency: for each ``(fold, view, scaler, distance)`` the neighbours are
queried **once** up to ``MAX_NEIGHBORS`` and reused to derive predictions for
every ``n_neighbors`` and every neighbour weighting.  Preprocessing (median
imputer + scaler) and the neighbour index are fit on the fold's *training*
records only.

Per fold we compute the full metric suite at two levels:
  * ``ALL``  — the three views pooled (used for configuration ranking), and
  * per-view — ``x_only`` / ``y_only`` / ``xy_2d`` (diagnostic).

Resume: each fold writes ``cache/fold_<s>_metrics.parquet``; a completed fold is
skipped on re-run.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from . import config as C
from .cross_validation import LosoFold, build_loso_folds
from .data_loader import UnifiedData
from .knn_predictor import predict_from_neighbors
from .metrics import compute_full_suite
from .neighbor_search import ViewNeighborIndex
from .preprocessing import FeaturePreprocessor

EXPECTED_ROWS_PER_FOLD = 96 * 4          # 96 configs x (ALL + 3 views)


def _fold_cache_path(cache_dir: Path, val_split: int) -> Path:
    return cache_dir / f"fold_{val_split:02d}_metrics.parquet"


def _prep_view(data: UnifiedData, view: str, val_split: int):
    """Return train/val feature+utility arrays and val metadata for one view."""
    vmask = (data.view == view) & data.split_mask(C.DEV_SPLITS)
    tr = vmask & (data.split_id != val_split)
    va = vmask & (data.split_id == val_split)
    return {
        "Xtr": data.X[tr], "Ytr": data.Y[tr],
        "Xva": data.X[va], "Yva": data.Y[va],
        "va_family": data.family[va], "va_subfamily": data.subfamily[va],
        "va_record_id": data.record_id[va], "va_dataset_id": data.dataset_id[va],
        "n_tr": int(tr.sum()), "n_va": int(va.sum()),
    }


def _run_one_fold(data: UnifiedData, fold: LosoFold,
                  configs: List[C.KnnConfig]) -> pd.DataFrame:
    s = fold.validation_split

    # ---- per-view train-only preprocessing + single max-neighbour query --- #
    view_data: Dict[str, dict] = {}
    neigh: Dict[tuple, tuple] = {}      # (view, scaler, distance) -> (idx, dist)
    fit_time: Dict[tuple, float] = {}   # (scaler, distance) -> seconds (summed)
    query_time: Dict[tuple, float] = {}

    for view in C.VIEWS:
        vd = _prep_view(data, view, s)
        view_data[view] = vd
        for scaler, distance in C.scaler_distance_combos():
            t0 = time.perf_counter()
            pp = FeaturePreprocessor(scaler).fit(vd["Xtr"])
            xtr_s = pp.transform(vd["Xtr"])
            xva_s = pp.transform(vd["Xva"])
            t1 = time.perf_counter()
            index = ViewNeighborIndex(distance).fit(xtr_s)
            nn = index.query(xva_s, max_neighbors=C.MAX_NEIGHBORS)
            t2 = time.perf_counter()
            neigh[(view, scaler, distance)] = (nn.indices, nn.distances)
            fit_time[(scaler, distance)] = fit_time.get((scaler, distance), 0.0) + (t1 - t0)
            query_time[(scaler, distance)] = query_time.get((scaler, distance), 0.0) + (t2 - t1)

    # ---- derive predictions + metrics for every configuration ------------- #
    rows: List[dict] = []
    for cfg in configs:
        sd = (cfg.scaler, cfg.distance)
        preds_by_view: Dict[str, np.ndarray] = {}
        t0 = time.perf_counter()
        for view in C.VIEWS:
            idx, dist = neigh[(view, cfg.scaler, cfg.distance)]
            preds_by_view[view] = predict_from_neighbors(
                idx, dist, view_data[view]["Ytr"],
                cfg.n_neighbors, cfg.neighbor_weighting,
            )
        pred_time = time.perf_counter() - t0

        pooled_true = np.concatenate([view_data[v]["Yva"] for v in C.VIEWS], axis=0)
        pooled_pred = np.concatenate([preds_by_view[v] for v in C.VIEWS], axis=0)
        n_pooled = int(pooled_true.shape[0])
        rps = float(n_pooled / pred_time) if pred_time > 0 else float("nan")

        levels = [("ALL", pooled_true, pooled_pred)]
        for view in C.VIEWS:
            levels.append((view, view_data[view]["Yva"], preds_by_view[view]))

        for level, yt, yp in levels:
            suite = compute_full_suite(yt, yp, data.metric_names)
            rows.append({
                "fold": s, "level": level, **cfg.to_dict(),
                "fit_time_sec": float(fit_time[sd]),
                "query_time_sec": float(query_time[sd]),
                "prediction_time_sec": float(pred_time),
                "records_per_second": rps,
                **suite,
            })
    return pd.DataFrame(rows)


def run_configuration_search(data: UnifiedData, cache_dir: Path,
                             configs: List[C.KnnConfig] | None = None,
                             verbose: bool = True) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    configs = configs or C.all_configurations()
    folds = build_loso_folds(data)

    all_frames: List[pd.DataFrame] = []
    for fold in folds:
        s = fold.validation_split
        path = _fold_cache_path(cache_dir, s)
        if path.exists():
            df = pd.read_parquet(path)
            if len(df) == EXPECTED_ROWS_PER_FOLD:
                if verbose:
                    print(f"[search] fold {s:02d}: cached ({len(df)} rows) — skip")
                all_frames.append(df)
                continue
        t0 = time.perf_counter()
        df = _run_one_fold(data, fold, configs)
        df.to_parquet(path, index=False)
        if verbose:
            print(f"[search] fold {s:02d}: {len(df)} rows in "
                  f"{time.perf_counter() - t0:.1f}s -> {path.name}")
        all_frames.append(df)

    return pd.concat(all_frames, ignore_index=True)
