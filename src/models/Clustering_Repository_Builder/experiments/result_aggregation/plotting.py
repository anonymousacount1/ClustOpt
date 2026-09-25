"""Plot generation for the aggregation report.

All plots are best-effort: any individual plot failure is caught and logged so
a single bad figure never aborts aggregation. Uses a non-interactive matplotlib
backend.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _bar(values: pd.Series, title: str, ylabel: str, path: Path, *, rotate: int = 30) -> None:
    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(values) + 3), 5))
    values = values.dropna()
    ax.bar([str(i) for i in values.index], values.values, color="steelblue", edgecolor="black")
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=rotate, ha="right")
    _save(fig, path)


def plot_overall(overall_summary: pd.DataFrame, plots_dir: Path) -> List[str]:
    saved: List[str] = []
    if overall_summary.empty:
        return saved
    s = overall_summary.set_index("method_name")
    specs = [
        ("mean_ari", "Mean ARI by method", "mean ARI", "mean_ari_by_method.png"),
        ("median_ari", "Median ARI by method", "median ARI", "median_ari_by_method.png"),
        ("mean_nmi", "Mean NMI by method", "mean NMI", "mean_nmi_by_method.png"),
        ("mean_ami", "Mean AMI by method", "mean AMI", "mean_ami_by_method.png"),
        ("mean_runtime_sec", "Mean runtime by method", "seconds", "mean_runtime_by_method.png"),
        ("k_accuracy", "K accuracy by method", "accuracy", "k_accuracy_by_method.png"),
    ]
    for col, title, ylabel, fname in specs:
        if col in s.columns:
            try:
                _bar(s[col].sort_values(ascending=False), title, ylabel, plots_dir / fname)
                saved.append(str(plots_dir / fname))
            except Exception as exc:
                print(f"[plot] skip {fname}: {exc}", flush=True)
    return saved


def plot_winrates(winrate_df: pd.DataFrame, plots_dir: Path) -> List[str]:
    if winrate_df.empty:
        return []
    try:
        pivot = winrate_df.pivot(index="method", columns="baseline", values="win_rate")
        fig, ax = plt.subplots(figsize=(1.6 * len(pivot.columns) + 4, 0.8 * len(pivot) + 3))
        im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                v = pivot.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9)
        ax.set_title("Win rate: regressor method vs baseline", fontweight="bold")
        fig.colorbar(im, ax=ax, label="win rate")
        _save(fig, plots_dir / "winrate_heatmap.png")
        return [str(plots_dir / "winrate_heatmap.png")]
    except Exception as exc:
        print(f"[plot] skip winrate_heatmap: {exc}", flush=True)
        return []


def plot_family_method_heatmap(family_summary: pd.DataFrame, plots_dir: Path) -> List[str]:
    if family_summary.empty or "family" not in family_summary.columns:
        return []
    try:
        pivot = family_summary.pivot(index="family", columns="method_name", values="mean_ari")
        fig, ax = plt.subplots(figsize=(1.0 * len(pivot.columns) + 4, 0.5 * len(pivot) + 3))
        im = ax.imshow(pivot.values, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=40, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_title("Mean ARI: family × method", fontweight="bold")
        fig.colorbar(im, ax=ax, label="mean ARI")
        _save(fig, plots_dir / "family_method_ari_heatmap.png")
        return [str(plots_dir / "family_method_ari_heatmap.png")]
    except Exception as exc:
        print(f"[plot] skip family heatmap: {exc}", flush=True)
        return []


def plot_k_confusion(matrices: Dict[str, pd.DataFrame], plots_dir: Path) -> List[str]:
    saved: List[str] = []
    for method, cm in matrices.items():
        try:
            fig, ax = plt.subplots(figsize=(0.6 * len(cm.columns) + 3, 0.6 * len(cm.index) + 3))
            im = ax.imshow(cm.values, cmap="Blues", aspect="auto")
            ax.set_xticks(range(len(cm.columns)))
            ax.set_xticklabels(cm.columns)
            ax.set_yticks(range(len(cm.index)))
            ax.set_yticklabels(cm.index)
            ax.set_xlabel("selected_k")
            ax.set_ylabel("true_k")
            ax.set_title(f"K confusion: {method}", fontweight="bold")
            for i in range(len(cm.index)):
                for j in range(len(cm.columns)):
                    ax.text(j, i, str(int(cm.values[i, j])), ha="center", va="center", fontsize=8)
            fig.colorbar(im, ax=ax, label="count")
            path = plots_dir / "k_confusion" / f"{method}.png"
            _save(fig, path)
            saved.append(str(path))
        except Exception as exc:
            print(f"[plot] skip k_confusion {method}: {exc}", flush=True)
    return saved


def plot_metric_frequency(freq_df: pd.DataFrame, plots_dir: Path, *, top_n: int = 25) -> List[str]:
    if freq_df.empty or "metric" not in freq_df.columns:
        return []
    try:
        agg = freq_df.groupby("metric")["selection_count"].sum().sort_values(ascending=False).head(top_n)
        _bar(agg, "Selected-metric frequency (regressor methods)", "selection count",
             plots_dir / "selected_metric_frequency.png", rotate=60)
        return [str(plots_dir / "selected_metric_frequency.png")]
    except Exception as exc:
        print(f"[plot] skip metric frequency: {exc}", flush=True)
        return []


def _heatmap(pivot: pd.DataFrame, title: str, cbar: str, path: Path, *, cmap: str = "viridis") -> None:
    fig, ax = plt.subplots(figsize=(1.0 * len(pivot.columns) + 4, 0.5 * len(pivot) + 3))
    im = ax.imshow(pivot.values.astype(float), cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=40, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title(title, fontweight="bold")
    fig.colorbar(im, ax=ax, label=cbar)
    _save(fig, path)


def plot_trace_best_gap(gap_by_method: pd.DataFrame, plots_dir: Path) -> List[str]:
    """Mean gap and selected-is-trace-best rate per method."""
    saved: List[str] = []
    if gap_by_method.empty or "method_name" not in gap_by_method.columns:
        return saved
    s = gap_by_method.set_index("method_name")
    specs = [
        ("mean_ari_gap_to_trace_best", "Mean ARI gap to trace-best by method",
         "mean gap (lower=better)", "mean_gap_by_method.png", True),
        ("selected_is_trace_best_rate", "Selected-is-trace-best rate by method",
         "rate (higher=better)", "selected_is_trace_best_by_method.png", False),
    ]
    for col, title, ylabel, fname, asc in specs:
        if col in s.columns:
            try:
                _bar(s[col].sort_values(ascending=asc), title, ylabel,
                     plots_dir / "trace_best_gap" / fname)
                saved.append(str(plots_dir / "trace_best_gap" / fname))
            except Exception as exc:
                print(f"[plot] skip {fname}: {exc}", flush=True)
    return saved


def plot_gap_family_method_heatmap(gap_by_family_method: pd.DataFrame, plots_dir: Path) -> List[str]:
    if gap_by_family_method.empty or "family" not in gap_by_family_method.columns:
        return []
    try:
        pivot = gap_by_family_method.pivot(
            index="family", columns="method_name", values="mean_ari_gap_to_trace_best")
        path = plots_dir / "trace_best_gap" / "gap_by_family_method_heatmap.png"
        _heatmap(pivot, "ARI gap to trace-best: family × method", "mean gap", path, cmap="magma")
        return [str(path)]
    except Exception as exc:
        print(f"[plot] skip gap family heatmap: {exc}", flush=True)
        return []


def plot_cross_heatmaps(family_cluster_ari: pd.DataFrame, family_cluster_gap: pd.DataFrame,
                        plots_dir: Path) -> List[str]:
    saved: List[str] = []
    cd = plots_dir / "cross_analysis"
    if not family_cluster_ari.empty and {"family", "cluster_count"} <= set(family_cluster_ari.columns):
        try:
            pivot = family_cluster_ari.pivot_table(index="family", columns="cluster_count",
                                                   values="mean_ari", aggfunc="mean")
            _heatmap(pivot, "Mean ARI: family × cluster_count", "mean ARI",
                     cd / "family_cluster_count_ari_heatmap.png", cmap="viridis")
            saved.append(str(cd / "family_cluster_count_ari_heatmap.png"))
        except Exception as exc:
            print(f"[plot] skip family×cc ari heatmap: {exc}", flush=True)
    if not family_cluster_gap.empty and {"family", "cluster_count"} <= set(family_cluster_gap.columns):
        try:
            pivot = family_cluster_gap.pivot_table(index="family", columns="cluster_count",
                                                   values="mean_ari_gap_to_trace_best", aggfunc="mean")
            _heatmap(pivot, "ARI gap to trace-best: family × cluster_count", "mean gap",
                     cd / "family_cluster_count_gap_heatmap.png", cmap="magma")
            saved.append(str(cd / "family_cluster_count_gap_heatmap.png"))
        except Exception as exc:
            print(f"[plot] skip family×cc gap heatmap: {exc}", flush=True)
    return saved


def plot_topk(topk_df: pd.DataFrame, plots_dir: Path) -> List[str]:
    saved: List[str] = []
    if topk_df.empty or "top_k" not in topk_df.columns:
        return saved
    s = topk_df.sort_values("top_k").set_index("top_k")
    specs = [
        ("mean_ari", "Top-K regressor: mean ARI", "mean ARI", "topk_mean_ari.png"),
        ("mean_ari_gap_to_trace_best", "Top-K regressor: gap to trace-best",
         "mean gap", "topk_gap_to_trace_best.png"),
        ("k_accuracy", "Top-K regressor: K accuracy", "accuracy", "topk_k_accuracy.png"),
    ]
    for col, title, ylabel, fname in specs:
        if col in s.columns:
            try:
                _bar(s[col], title, ylabel, plots_dir / "topk" / fname, rotate=0)
                saved.append(str(plots_dir / "topk" / fname))
            except Exception as exc:
                print(f"[plot] skip {fname}: {exc}", flush=True)
    return saved


def plot_runtime_boxplot(best_view_raw: pd.DataFrame, plots_dir: Path) -> List[str]:
    """Per-method runtime distribution box plot (ML2DAC Fig. 6 style).

    Reproduces the paper's runtime comparison figure: one box per method over the
    per-dataset (best-view) runtimes, on a **log** y-axis, with the median
    annotated under each box. Works for both ClustOpt methods and the external
    baselines (every method records ``runtime_sec``).
    """
    if best_view_raw is None or best_view_raw.empty \
            or "runtime_sec" not in best_view_raw.columns:
        return []
    try:
        succ = best_view_raw[best_view_raw["status"] == "success"].copy()
        succ["_rt"] = pd.to_numeric(succ["runtime_sec"], errors="coerce")
        succ = succ.dropna(subset=["_rt"])
        succ = succ[succ["_rt"] > 0]
        if succ.empty:
            return []
        # Order methods by median runtime (cheapest first).
        order = (succ.groupby("method_name")["_rt"].median()
                 .sort_values().index.tolist())
        data = [succ.loc[succ["method_name"] == m, "_rt"].to_numpy() for m in order]
        medians = [float(np.median(d)) for d in data]
        fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(order) + 3), 5.5))
        ax.boxplot(data, labels=[str(m) for m in order], showfliers=True,
                   flierprops=dict(markerfacecolor="grey", markersize=4, alpha=0.4))
        ax.set_yscale("log")
        ax.set_ylabel("Runtime (seconds, log scale)")
        ax.set_title("Runtime distribution by method", fontweight="bold")
        ax.grid(axis="y", which="both", linestyle="--", alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
        for i, med in enumerate(medians, start=1):
            ax.text(i, med, f"{med:.1f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color="black")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        _save(fig, plots_dir / "runtime_boxplot.png")
        return [str(plots_dir / "runtime_boxplot.png")]
    except Exception as exc:
        print(f"[plot] skip runtime_boxplot: {exc}", flush=True)
        return []


def generate_all_plots(
    *,
    plots_dir: Path,
    overall_summary: pd.DataFrame,
    family_summary: pd.DataFrame,
    winrate_df: pd.DataFrame,
    k_confusion: Dict[str, pd.DataFrame],
    metric_freq: pd.DataFrame,
    best_view_raw: Optional[pd.DataFrame] = None,
) -> List[str]:
    plots_dir.mkdir(parents=True, exist_ok=True)
    saved: List[str] = []
    saved += plot_overall(overall_summary, plots_dir)
    saved += plot_winrates(winrate_df, plots_dir)
    saved += plot_family_method_heatmap(family_summary, plots_dir)
    saved += plot_k_confusion(k_confusion, plots_dir)
    saved += plot_metric_frequency(metric_freq, plots_dir)
    if best_view_raw is not None:
        saved += plot_runtime_boxplot(best_view_raw, plots_dir)
    return saved


def generate_extended_plots(
    *,
    plots_dir: Path,
    gap_by_method: pd.DataFrame,
    gap_by_family_method: pd.DataFrame,
    family_cluster_ari: pd.DataFrame,
    family_cluster_gap: pd.DataFrame,
    topk_df: pd.DataFrame,
) -> List[str]:
    saved: List[str] = []
    saved += plot_trace_best_gap(gap_by_method, plots_dir)
    saved += plot_gap_family_method_heatmap(gap_by_family_method, plots_dir)
    saved += plot_cross_heatmaps(family_cluster_ari, family_cluster_gap, plots_dir)
    saved += plot_topk(topk_df, plots_dir)
    return saved
