"""
cross_validation.py
===================

Top-level orchestrator: builds the run directory, loads & splits the data,
trains and evaluates every fold, then aggregates results, writes reports, plots
and global artifacts.

Directory layout produced under ``<repo>/results_analysis/mlp/<ts>__<run>/``:

    config.json, run_summary.md, fold_results.csv, overall_results.csv
    folds/fold_<k>/...      (per-fold checkpoints, logs, predictions, metrics)
    plots/                  (cross-fold comparison plots)
    artifacts/              (global predictions, column lists, head mapping, ...)
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from . import artifact_writer as aw
from . import plots
from . import report_writer as rw
from .config import RunConfig
from .data_loader import load_unified_dataset
from .dataset import UtilityDataset
from .evaluator import evaluate_fold
from .head_groups import build_head_mapping
from .losses import build_loss
from .model import architecture_summary, build_model
from .preprocessing import FeaturePipeline, clean_and_clip_targets
from .splitters import build_folds, cv_split_summary_frame
from .trainer import train_one_fold
from .utils import resolve_device, set_seed


def _make_run_dir(cfg: RunConfig) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{ts}__{cfg.run_name}"
    run_dir = Path(cfg.repo_root) / cfg.output_subdir / name
    aw.ensure_dir(run_dir)
    return run_dir


def _build_loaders(
    x: np.ndarray, y: np.ndarray, idx: np.ndarray, pipeline: FeaturePipeline,
    batch_size: int, shuffle: bool, num_workers: int, fit: bool = False,
):
    if fit:
        pipeline.fit(x[idx])
    x_t, _ = pipeline.transform(x[idx])
    y_t, _ = clean_and_clip_targets(y[idx])
    ds = UtilityDataset(x_t, y_t, idx)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, drop_last=False,
    )
    return loader


def run_cross_validation(cfg: RunConfig, fold_builder=None) -> Dict:
    """Run the full train/eval/aggregate pipeline.

    ``fold_builder`` is an optional callable ``(df, group_column,
    group_summary_columns) -> List[FoldSplit]``.  When ``None`` (the default)
    the standard group-safe ``GroupKFold`` folds are built via
    :func:`splitters.build_folds`.  Passing a builder lets a caller inject a
    custom outer split (e.g. a predefined holdout split) while reusing the rest
    of the pipeline unchanged.
    """
    set_seed(cfg.training.seed)
    device = resolve_device(cfg.training.device)
    run_dir = _make_run_dir(cfg)
    print(f"[run] output directory: {run_dir}")
    print(f"[run] device: {device}")

    cfg.to_json(run_dir / "config.json")

    # ---- load & detect ----------------------------------------------------
    loaded = load_unified_dataset(cfg.data)
    df = loaded.df
    feat_cols = loaded.feature_columns
    target_cols = loaded.target_columns
    id_cols = loaded.id_columns
    print(f"[data] rows={len(df)} | id={len(id_cols)} feat={len(feat_cols)} targets={len(target_cols)}")

    target_names = [c[len(cfg.data.target_prefix):] if c.startswith(cfg.data.target_prefix) else c
                    for c in target_cols]

    head_info = build_head_mapping(target_cols, cfg.data.target_prefix)
    head_mapping = head_info["heads"]
    if head_info["unmatched"]:
        print(f"[head] {len(head_info['unmatched'])} metric(s) routed to 'other': {head_info['unmatched']}")

    # Materialise feature / target matrices once.
    x_all = df[feat_cols].to_numpy(dtype=np.float64)
    y_all = df[target_cols].to_numpy(dtype=np.float64)

    # ---- global artifacts (column lists, head mapping, architecture) ------
    artifacts_dir = aw.ensure_dir(run_dir / "artifacts")
    aw.write_json(artifacts_dir / "feature_columns.json", feat_cols)
    aw.write_json(artifacts_dir / "target_columns.json", target_cols)
    head_mapping_serialisable = {
        "head_order": head_info["head_order"],
        "heads": {h: idxs for h, idxs in head_mapping.items()},
        "metric_to_head": head_info["metric_to_head"],
        "unmatched": head_info["unmatched"],
        "target_columns": target_cols,
    }
    aw.write_json(artifacts_dir / "head_mapping.json", head_mapping_serialisable)
    aw.write_json(artifacts_dir / "target_head_mapping.json", head_mapping_serialisable)

    sample_model = build_model(
        cfg.model, head_mapping if cfg.model.type == "multi_head_mlp" else None
    )
    aw.write_text(artifacts_dir / "model_architecture.txt", architecture_summary(sample_model))
    del sample_model

    # ---- folds ------------------------------------------------------------
    group_summary_cols = [c for c in cfg.data.group_columns.values() if c in df.columns]
    if fold_builder is None:
        folds = build_folds(
            df=df,
            group_column=loaded.group_column,
            n_splits=cfg.training.n_splits,
            inner_val_ratio=cfg.training.inner_val_ratio,
            seed=cfg.training.seed,
            group_summary_columns=group_summary_cols,
        )
    else:
        folds = fold_builder(df, loaded.group_column, group_summary_cols)
    cv_summary = cv_split_summary_frame(folds)
    aw.write_csv(artifacts_dir / "cv_split_summary.csv", cv_summary)
    print("[split] leakage validated for all folds (see cv_split_summary.csv).")

    folds_to_run = cfg.training.folds_to_run
    if folds_to_run is None:
        folds_to_run = list(range(len(folds)))

    fold_rows: List[Dict] = []
    all_test_preds: List[pd.DataFrame] = []
    grouped_accumulator: Dict[str, List[pd.DataFrame]] = {}
    per_metric_accumulator: List[pd.DataFrame] = []

    folds_root = aw.ensure_dir(run_dir / "folds")

    for fs in folds:
        if fs.fold not in folds_to_run:
            continue
        print(f"\n[fold {fs.fold}] training ...")
        set_seed(cfg.training.seed + fs.fold)
        fold_dir = aw.ensure_dir(folds_root / f"fold_{fs.fold}")

        # Preprocessing + loaders.
        pipeline = FeaturePipeline()
        train_loader = _build_loaders(
            x_all, y_all, fs.train_idx, pipeline, cfg.training.batch_size,
            shuffle=True, num_workers=cfg.training.num_workers, fit=True,
        )
        val_loader = _build_loaders(
            x_all, y_all, fs.val_idx, pipeline, cfg.training.batch_size,
            shuffle=False, num_workers=cfg.training.num_workers,
        )
        test_loader = _build_loaders(
            x_all, y_all, fs.test_idx, pipeline, cfg.training.batch_size,
            shuffle=False, num_workers=cfg.training.num_workers,
        )
        pipeline.save(fold_dir / "x_scaler.pkl")

        # Per-fold reference artifacts.
        aw.write_json(fold_dir / "feature_columns.json", feat_cols)
        aw.write_json(fold_dir / "target_columns.json", target_cols)
        aw.write_json(fold_dir / "head_mapping.json", head_mapping_serialisable)
        aw.write_json(fold_dir / "split_summary.json", fs.summary)

        # Model + loss + train.
        model = build_model(
            cfg.model, head_mapping if cfg.model.type == "multi_head_mlp" else None
        )
        loss_fn = build_loss(cfg.loss).to(device)
        train_result = train_one_fold(
            model, train_loader, val_loader, loss_fn, cfg.training, device, fold_dir,
            verbose=True,
        )
        pd.DataFrame(train_result.train_log).to_csv(fold_dir / "train_log.csv", index=False)
        pd.DataFrame(train_result.val_log).to_csv(fold_dir / "val_log.csv", index=False)

        # Evaluate on the test fold (model already restored to best weights).
        summary, per_metric, pred_frame, grouped = evaluate_fold(
            model, test_loader, device, df, target_names, id_cols,
            cfg.data.group_columns, cfg.evaluation,
        )
        summary["fold"] = fs.fold
        summary["best_epoch"] = train_result.best_epoch
        summary["best_val_loss"] = train_result.best_val_loss

        aw.write_json(fold_dir / "test_metrics.json", summary)
        aw.write_csv(fold_dir / "per_metric_errors.csv", per_metric)
        aw.write_csv(fold_dir / "test_predictions.csv", pred_frame)

        grouped_dir = aw.ensure_dir(fold_dir / "grouped_metrics")
        for logical, gdf in grouped.items():
            aw.write_csv(grouped_dir / f"by_{logical}.csv", gdf)
            grouped_accumulator.setdefault(logical, []).append(gdf)

        # Per-fold plots.
        fold_plots = aw.ensure_dir(fold_dir / "plots")
        tr_df = pd.DataFrame(train_result.train_log)
        vl_df = pd.DataFrame(train_result.val_log)
        plots.plot_loss_curves(tr_df, vl_df, fold_plots / "loss_curves.png")
        plots.plot_mae_curve(vl_df, fold_plots / "mae_curve.png")
        plots.plot_per_metric_mae(per_metric, fold_plots / "per_metric_mae.png")
        for logical, gdf in grouped.items():
            plots.plot_grouped_performance(gdf, logical, "mae", fold_plots / f"by_{logical}_mae.png")

        # Fold report.
        report = rw.write_fold_report(
            fs.fold, train_result, summary, per_metric, grouped, fs.summary
        )
        aw.write_text(fold_dir / "test_metrics.md", report)

        # Accumulate.
        pf = pred_frame.copy()
        pf.insert(0, "fold", fs.fold)
        all_test_preds.append(pf)
        pm = per_metric.copy()
        pm["fold"] = fs.fold
        per_metric_accumulator.append(pm)
        fold_rows.append(summary)
        print(f"[fold {fs.fold}] test MAE={summary['mae']:.4f} top1={summary['top1_accuracy']:.3f} "
              f"top10_overlap={summary.get('top10_overlap', float('nan')):.3f} "
              f"ndcg@10={summary.get('ndcg@10', float('nan')):.3f}")

    # ---- aggregate --------------------------------------------------------
    result = _aggregate(
        cfg, run_dir, artifacts_dir, fold_rows, all_test_preds,
        per_metric_accumulator, grouped_accumulator,
    )
    print(f"\n[done] run summary written to {run_dir / 'run_summary.md'}")
    return result


def _aggregate(cfg, run_dir, artifacts_dir, fold_rows, all_test_preds,
               per_metric_accumulator, grouped_accumulator) -> Dict:
    if not fold_rows:
        print("[warn] no folds were run; nothing to aggregate.")
        return {"run_dir": str(run_dir), "fold_results": pd.DataFrame()}

    fold_results = pd.DataFrame(fold_rows)
    front = [c for c in ["fold", "best_epoch", "best_val_loss"] if c in fold_results.columns]
    fold_results = fold_results[front + [c for c in fold_results.columns if c not in front]]
    aw.write_csv(run_dir / "fold_results.csv", fold_results)

    # Mean ± std across folds for numeric metrics.
    numeric = fold_results.drop(columns=[c for c in ["fold", "best_epoch"] if c in fold_results.columns])
    overall = pd.DataFrame({
        "metric": numeric.columns,
        "mean": numeric.mean(numeric_only=True).values,
        "std": numeric.std(numeric_only=True).values,
    })
    aw.write_csv(run_dir / "overall_results.csv", overall)

    # Global predictions.
    if all_test_preds:
        all_preds = pd.concat(all_test_preds, ignore_index=True)
        aw.write_csv(artifacts_dir / "all_fold_test_predictions.csv", all_preds)
        oof = all_preds.drop(columns=["fold"]) if "fold" in all_preds.columns else all_preds
        aw.write_csv(artifacts_dir / "final_oof_predictions.csv", oof)

    # Worst metrics by mean MAE across folds.
    worst_metrics = pd.DataFrame()
    if per_metric_accumulator:
        pm_all = pd.concat(per_metric_accumulator, ignore_index=True)
        worst_metrics = (
            pm_all.groupby("metric", as_index=False)[["mae", "rmse", "r2"]].mean()
            .sort_values("mae", ascending=False, ignore_index=True)
        )
        aw.write_csv(artifacts_dir / "per_metric_errors_mean.csv", worst_metrics)

    # Worst groups by mean MAE across folds.
    worst_groups: Dict[str, pd.DataFrame] = {}
    for logical, gdfs in grouped_accumulator.items():
        g_all = pd.concat(gdfs, ignore_index=True)
        if "mae" in g_all.columns and logical in g_all.columns:
            agg = (
                g_all.groupby(logical, as_index=False)
                .agg({"mae": "mean", "top1_accuracy": "mean", "ndcg@10": "mean", "n_samples": "sum"})
                .sort_values("mae", ascending=False, ignore_index=True)
            )
            worst_groups[logical] = agg
            aw.write_csv(artifacts_dir / f"grouped_mean_by_{logical}.csv", agg)

    # Cross-fold plots.
    plots_dir = aw.ensure_dir(run_dir / "plots")
    plots.plot_metrics_by_fold(
        fold_results, ["top3_overlap", "top5_overlap", "top10_overlap"],
        plots_dir / "topk_overlap_by_fold.png", "Top-K overlap by fold")
    plots.plot_metrics_by_fold(
        fold_results, ["ndcg@3", "ndcg@5", "ndcg@10", "ndcg@all"],
        plots_dir / "ndcg_by_fold.png", "NDCG by fold")
    plots.plot_metrics_by_fold(
        fold_results, ["spearman_mean", "kendall_mean", "pairwise_ranking_accuracy"],
        plots_dir / "rank_corr_by_fold.png", "Ranking correlation by fold")
    plots.plot_metrics_by_fold(
        fold_results, ["mae", "rmse"], plots_dir / "regression_by_fold.png",
        "Regression error by fold")
    if not worst_metrics.empty:
        plots.plot_per_metric_mae(worst_metrics, plots_dir / "per_metric_mae_mean.png")

    # Run summary report.
    summary_md = rw.write_run_summary(
        cfg.run_name, fold_results, overall, worst_metrics, worst_groups, cfg.to_dict()
    )
    aw.write_text(run_dir / "run_summary.md", summary_md)

    return {
        "run_dir": str(run_dir),
        "fold_results": fold_results,
        "overall": overall,
        "worst_metrics": worst_metrics,
    }
