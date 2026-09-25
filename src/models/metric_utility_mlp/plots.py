"""
plots.py
========

Matplotlib (Agg backend) plotting helpers.  Every function saves a single,
labelled figure to ``out_path`` and closes it.  Failures are swallowed with a
warning so plotting never aborts a training run.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .artifact_writer import ensure_dir


def _save(fig, out_path: str | Path) -> None:
    p = Path(out_path)
    ensure_dir(p.parent)
    fig.tight_layout()
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _guard(fn):
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # pragma: no cover - plotting must never crash a run
            warnings.warn(f"plot {fn.__name__} failed: {exc}", stacklevel=2)
    return wrapper


@_guard
def plot_loss_curves(train_log: pd.DataFrame, val_log: pd.DataFrame, out_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_log["epoch"], train_log["train_loss"], label="train loss")
    ax.plot(val_log["epoch"], val_log["val_loss"], label="val loss")
    best = val_log.loc[val_log["val_loss"].idxmin()]
    ax.axvline(best["epoch"], color="grey", linestyle="--", alpha=0.6,
               label=f"best epoch {int(best['epoch'])}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("combined loss")
    ax.set_title("Training / validation loss")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, out_path)


@_guard
def plot_mae_curve(val_log: pd.DataFrame, out_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(val_log["epoch"], val_log["val_mae"], color="tab:green", label="val MAE")
    if "val_ndcg@10" in val_log.columns:
        ax2 = ax.twinx()
        ax2.plot(val_log["epoch"], val_log["val_ndcg@10"], color="tab:orange",
                 alpha=0.7, label="val NDCG@10")
        ax2.set_ylabel("NDCG@10")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MAE")
    ax.set_title("Validation MAE / NDCG@10 by epoch")
    ax.grid(alpha=0.3)
    _save(fig, out_path)


@_guard
def plot_metrics_by_fold(fold_results: pd.DataFrame, metric_cols: List[str],
                         out_path: str | Path, title: str) -> None:
    cols = [c for c in metric_cols if c in fold_results.columns]
    if not cols:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    folds = fold_results["fold"].to_numpy()
    width = 0.8 / max(1, len(cols))
    x = np.arange(len(folds))
    for i, c in enumerate(cols):
        vals = fold_results[c].to_numpy()
        bars = ax.bar(x + i * width, vals, width=width, label=c)
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, v),
                        ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x + width * (len(cols) - 1) / 2)
    ax.set_xticklabels([f"fold {int(f)}" for f in folds])
    ax.set_title(title)
    ax.set_ylabel("score")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    _save(fig, out_path)


@_guard
def plot_per_metric_mae(per_metric: pd.DataFrame, out_path: str | Path, top_n: int = 60) -> None:
    df = per_metric.sort_values("mae", ascending=True).tail(top_n)
    fig, ax = plt.subplots(figsize=(9, max(6, 0.22 * len(df))))
    ax.barh(df["metric"], df["mae"], color="tab:red", alpha=0.8)
    ax.set_xlabel("MAE")
    ax.set_title("Per-metric MAE (worst at top)")
    ax.grid(alpha=0.3, axis="x")
    _save(fig, out_path)


@_guard
def plot_grouped_performance(grouped: pd.DataFrame, group_key: str, metric: str,
                             out_path: str | Path) -> None:
    if metric not in grouped.columns or group_key not in grouped.columns:
        return
    df = grouped.sort_values(metric, ascending=True)
    fig, ax = plt.subplots(figsize=(9, max(4, 0.4 * len(df))))
    labels = [str(v)[:40] for v in df[group_key]]
    bars = ax.barh(labels, df[metric], color="tab:blue", alpha=0.8)
    for b, v in zip(bars, df[metric]):
        ax.annotate(f"{v:.3f}", (v, b.get_y() + b.get_height() / 2),
                    ha="left", va="center", fontsize=7)
    ax.set_xlabel(metric)
    ax.set_title(f"{metric} by {group_key}")
    ax.grid(alpha=0.3, axis="x")
    _save(fig, out_path)
