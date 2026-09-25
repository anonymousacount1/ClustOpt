"""
plotting.py
===========

All Phase-E1 figures.  Every plot is wrapped so a single failure never aborts
the run; the master :func:`generate_all_plots` returns the list of files it
produced.  Uses a non-interactive Agg backend.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C

plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight",
                     "axes.grid": True, "grid.alpha": 0.3})


def _save(fig, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def _combo_label(row) -> str:
    return f"{row['scaler'][:4]}/{row['distance'][:3]}/{row['neighbor_weighting'][:3]}"


# --------------------------------------------------------------------------- #
# Ranking
# --------------------------------------------------------------------------- #
def plot_ranking(ranking: pd.DataFrame, plots_dir: Path, selected_id: str,
                 out: List[str]) -> None:
    for name, n in [("composite_configuration_ranking.png", len(ranking)),
                    ("top20_configuration_ranking.png", 20)]:
        sub = ranking.head(n)
        fig, ax = plt.subplots(figsize=(9, max(4, 0.28 * len(sub))))
        colors = ["#d62728" if cid == selected_id else "#1f77b4"
                  for cid in sub["config_id"]]
        ax.barh(range(len(sub)), sub["mean_composite_score"], color=colors,
                xerr=sub["standard_error"], error_kw={"alpha": 0.4})
        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels(sub["config_id"], fontsize=6)
        ax.invert_yaxis()
        ax.set_xlabel("Mean composite score (+/- 1 SE)")
        ax.set_title(f"Configuration ranking (red = one-SE selected: {selected_id})")
        out.append(_save(fig, plots_dir / name))


# --------------------------------------------------------------------------- #
# K-neighbours sensitivity
# --------------------------------------------------------------------------- #
def _per_config_means(fold_all: pd.DataFrame, metric: str) -> pd.DataFrame:
    g = (fold_all.groupby(["n_neighbors", "scaler", "distance",
                           "neighbor_weighting"])[metric].mean().reset_index())
    return g


def plot_metric_vs_k(fold_all: pd.DataFrame, metric: str, plots_dir: Path,
                     fname: str, ylabel: str, out: List[str]) -> None:
    g = _per_config_means(fold_all, metric)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for (sc, di, we), sub in g.groupby(["scaler", "distance",
                                        "neighbor_weighting"]):
        sub = sub.sort_values("n_neighbors")
        ax.plot(sub["n_neighbors"], sub[metric], marker="o", markersize=3,
                label=f"{sc[:4]}/{di[:3]}/{we[:3]}", alpha=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("n_neighbors (K_neighbors)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs n_neighbors (LOSO mean)")
    ax.legend(fontsize=6, ncol=2)
    out.append(_save(fig, plots_dir / fname))


def plot_k_sensitivity_overview(fold_all: pd.DataFrame, plots_dir: Path,
                                out: List[str]) -> None:
    metrics = [("mae", "MAE"), ("spearman_mean", "Spearman"),
               ("ndcg@all", "NDCG@all"),
               ("dynamic_k_exact_accuracy", "Dyn-K exact acc")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (metric, lab) in zip(axes.ravel(), metrics):
        g = fold_all.groupby("n_neighbors")[metric].agg(["mean", "std"])
        ax.errorbar(g.index, g["mean"], yerr=g["std"], marker="o", capsize=3)
        ax.set_xscale("log")
        ax.set_xlabel("n_neighbors")
        ax.set_ylabel(lab)
        ax.set_title(lab)
    fig.suptitle("K_neighbors sensitivity (mean +/- std over configs & folds)")
    out.append(_save(fig, plots_dir / "k_neighbors_sensitivity.png"))


def plot_k_by_metric(per_metric_by_k: pd.DataFrame, plots_dir: Path,
                     out: List[str]) -> None:
    if per_metric_by_k is None or per_metric_by_k.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    pivot = per_metric_by_k.pivot(index="n_neighbors", columns="metric_name",
                                  values="mae")
    for col in pivot.columns:
        ax.plot(pivot.index, pivot[col], alpha=0.5, linewidth=0.8)
    ax.plot(pivot.index, pivot.mean(axis=1), color="k", linewidth=2.5,
            label="mean over metrics")
    ax.set_xscale("log")
    ax.set_xlabel("n_neighbors")
    ax.set_ylabel("Per-metric MAE")
    ax.set_title("Per-metric MAE vs n_neighbors")
    ax.legend(fontsize=7)
    out.append(_save(fig, plots_dir / "k_neighbors_by_metric.png"))


# --------------------------------------------------------------------------- #
# Component comparisons
# --------------------------------------------------------------------------- #
def plot_component_comparisons(ranking: pd.DataFrame, plots_dir: Path,
                               out: List[str]) -> None:
    for comp, fname in [("scaler", "scaler_comparison.png"),
                        ("distance", "distance_metric_comparison.png"),
                        ("neighbor_weighting", "neighbor_weighting_comparison.png")]:
        fig, ax = plt.subplots(figsize=(5, 4))
        data = [ranking[ranking[comp] == v]["mean_composite_score"].values
                for v in sorted(ranking[comp].unique())]
        ax.boxplot(data, labels=sorted(ranking[comp].unique()))
        ax.set_ylabel("Mean composite score")
        ax.set_title(f"Composite score by {comp}")
        out.append(_save(fig, plots_dir / fname))

    # combined scaler x distance x weighting
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ranking = ranking.copy()
    ranking["_combo"] = ranking.apply(_combo_label, axis=1)
    order = (ranking.groupby("_combo")["mean_composite_score"].mean()
             .sort_values(ascending=False).index)
    data = [ranking[ranking["_combo"] == c]["mean_composite_score"].values
            for c in order]
    ax.boxplot(data, labels=order)
    ax.set_ylabel("Mean composite score")
    ax.set_title("Composite score by scaler / distance / weighting")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    out.append(_save(fig, plots_dir / "scaler_distance_weighting_comparison.png"))


# --------------------------------------------------------------------------- #
# Fold stability
# --------------------------------------------------------------------------- #
def plot_fold_stability(fold_composite: pd.DataFrame, plots_dir: Path,
                        selected_id: str, out: List[str]) -> None:
    top = (fold_composite.groupby("config_id")["composite_score"].mean()
           .sort_values(ascending=False).head(15).index.tolist())
    if selected_id not in top:
        top = [selected_id] + top[:14]
    fig, ax = plt.subplots(figsize=(10, 5))
    data = [fold_composite[fold_composite["config_id"] == c]["composite_score"].values
            for c in top]
    bp = ax.boxplot(data, labels=top, showmeans=True)
    ax.set_ylabel("Composite score across folds")
    ax.set_title("Fold stability of top configurations")
    plt.setp(ax.get_xticklabels(), rotation=90, fontsize=6)
    for name in ("fold_composite_score_boxplot.png", "fold_stability.png"):
        out.append(_save(fig, plots_dir / name))


def plot_fold_metric_heatmap(fold_all: pd.DataFrame, selected_id: str,
                             plots_dir: Path, out: List[str]) -> None:
    sub = fold_all[(fold_all["config_id"] == selected_id) &
                   (fold_all["level"] == "ALL")]
    metrics = ["mae", "rmse", "r2", "spearman_mean", "ndcg@all",
               "top5_overlap", "dynamic_k_exact_accuracy"]
    metrics = [m for m in metrics if m in sub.columns]
    mat = sub.sort_values("fold")[metrics].to_numpy().T
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(mat, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels(metrics, fontsize=7)
    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels(sorted(sub["fold"].unique()), fontsize=7)
    ax.set_xlabel("Validation fold (split)")
    ax.set_title(f"Selected config per-fold metrics: {selected_id}")
    fig.colorbar(im, ax=ax, fraction=0.03)
    out.append(_save(fig, plots_dir / "fold_metric_heatmap.png"))


# --------------------------------------------------------------------------- #
# Domain diagnostics (from OOF selected-config predictions)
# --------------------------------------------------------------------------- #
def plot_view_performance(view_metrics: pd.DataFrame, plots_dir: Path,
                          out: List[str]) -> None:
    metrics = ["mae", "spearman_mean", "ndcg@all", "dynamic_k_exact_accuracy"]
    metrics = [m for m in metrics if m in view_metrics.columns]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(view_metrics))
    w = 0.8 / len(metrics)
    for i, m in enumerate(metrics):
        ax.bar(x + i * w, view_metrics[m], width=w, label=m)
    ax.set_xticks(x + 0.4 - w / 2)
    ax.set_xticklabels(view_metrics["view"])
    ax.set_title("Per-view performance (OOF, selected config)")
    ax.legend(fontsize=7)
    out.append(_save(fig, plots_dir / "view_performance.png"))


def plot_group_heatmap(group_metrics: pd.DataFrame, group_col: str,
                       plots_dir: Path, fname: str, title: str,
                       out: List[str], top_n: int = 30) -> None:
    if group_metrics is None or group_metrics.empty:
        return
    metrics = ["mae", "rmse", "spearman_mean", "ndcg@all",
               "top5_overlap", "dynamic_k_exact_accuracy"]
    metrics = [m for m in metrics if m in group_metrics.columns]
    gm = group_metrics.sort_values("n_records", ascending=False).head(top_n)
    mat = gm[metrics].to_numpy()
    # per-column normalise for visual contrast
    mn = mat.min(axis=0, keepdims=True)
    rng = np.ptp(mat, axis=0, keepdims=True)
    norm = (mat - mn) / np.where(rng > 0, rng, 1.0)
    fig, ax = plt.subplots(figsize=(8, max(4, 0.28 * len(gm))))
    im = ax.imshow(norm, aspect="auto", cmap="magma")
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(gm)))
    ax.set_yticklabels(gm[group_col].astype(str), fontsize=6)
    ax.set_title(title + " (column-normalised)")
    fig.colorbar(im, ax=ax, fraction=0.03)
    out.append(_save(fig, plots_dir / fname))


# --------------------------------------------------------------------------- #
# Prediction quality
# --------------------------------------------------------------------------- #
def plot_prediction_quality(y_true: np.ndarray, y_pred: np.ndarray,
                            plots_dir: Path, out: List[str],
                            tag: str = "") -> None:
    rng = np.random.default_rng(0)
    flat_t = y_true.ravel()
    flat_p = y_pred.ravel()
    if flat_t.size > 200000:
        sel = rng.choice(flat_t.size, 200000, replace=False)
        flat_t, flat_p = flat_t[sel], flat_p[sel]
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.hexbin(flat_t, flat_p, gridsize=60, cmap="Blues", mincnt=1)
    ax.plot([0, 1], [0, 1], "r--", linewidth=1)
    ax.set_xlabel("True utility")
    ax.set_ylabel("Predicted utility")
    ax.set_title(f"Predicted vs true utility {tag}")
    out.append(_save(fig, plots_dir / "predicted_vs_true_utility.png"))

    err = (y_pred - y_true).ravel()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(err, bins=100, color="#555")
    ax.axvline(0, color="r", linestyle="--")
    ax.set_xlabel("Predicted - true utility")
    ax.set_ylabel("Count")
    ax.set_title(f"Utility error distribution {tag}")
    out.append(_save(fig, plots_dir / "utility_error_distribution.png"))


def plot_per_metric(per_metric: pd.DataFrame, plots_dir: Path,
                    out: List[str]) -> None:
    pm = per_metric.sort_values("mae", ascending=False)
    fig, ax = plt.subplots(figsize=(8, max(5, 0.2 * len(pm))))
    ax.barh(range(len(pm)), pm["mae"], color="#c44")
    ax.set_yticks(range(len(pm)))
    ax.set_yticklabels(pm["metric_name"], fontsize=5)
    ax.invert_yaxis()
    ax.set_xlabel("MAE")
    ax.set_title("Per-metric MAE (OOF, selected config)")
    out.append(_save(fig, plots_dir / "per_metric_error.png"))

    pm2 = per_metric.sort_values("r2")
    fig, ax = plt.subplots(figsize=(8, max(5, 0.2 * len(pm2))))
    ax.barh(range(len(pm2)), pm2["r2"], color="#3a7")
    ax.set_yticks(range(len(pm2)))
    ax.set_yticklabels(pm2["metric_name"], fontsize=5)
    ax.invert_yaxis()
    ax.set_xlabel("R^2")
    ax.set_title("Per-metric R^2 (OOF, selected config)")
    out.append(_save(fig, plots_dir / "per_metric_r2.png"))


# --------------------------------------------------------------------------- #
# Dynamic-K
# --------------------------------------------------------------------------- #
def plot_dynamic_k(confusion: pd.DataFrame, k_true: np.ndarray,
                   k_pred: np.ndarray, plots_dir: Path, out: List[str]) -> None:
    mat = confusion.to_numpy()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mat, cmap="Blues")
    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels(range(1, mat.shape[1] + 1))
    ax.set_yticks(range(mat.shape[0]))
    ax.set_yticklabels(range(1, mat.shape[0] + 1))
    ax.set_xlabel("Predicted K_metrics")
    ax.set_ylabel("True K_metrics")
    ax.set_title("Dynamic Top-K confusion matrix")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if mat[i, j]:
                ax.text(j, i, int(mat[i, j]), ha="center", va="center",
                        fontsize=6,
                        color="white" if mat[i, j] > mat.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.04)
    out.append(_save(fig, plots_dir / "dynamic_k_confusion_matrix.png"))

    diff = k_pred - k_true
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.arange(diff.min() - 0.5, diff.max() + 1.5)
    ax.hist(diff, bins=bins, color="#764")
    ax.axvline(0, color="r", linestyle="--")
    ax.set_xlabel("Predicted - true K_metrics")
    ax.set_ylabel("Count")
    ax.set_title("Dynamic Top-K signed error distribution")
    out.append(_save(fig, plots_dir / "dynamic_k_error_distribution.png"))


# --------------------------------------------------------------------------- #
# Neighbour geometry
# --------------------------------------------------------------------------- #
def plot_neighbor_geometry(neighbor_distances: np.ndarray,
                           effective_counts: np.ndarray, plots_dir: Path,
                           out: List[str]) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(neighbor_distances.ravel(), bins=80, color="#38a")
    ax.set_xlabel("Neighbour distance")
    ax.set_ylabel("Count")
    ax.set_title("Neighbour distance distribution (selected config)")
    out.append(_save(fig, plots_dir / "neighbor_distance_distribution.png"))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(effective_counts, bins=40, color="#a63")
    ax.set_xlabel("Effective neighbour count")
    ax.set_ylabel("Count")
    ax.set_title("Effective neighbour count distribution")
    out.append(_save(fig, plots_dir / "effective_neighbor_count_distribution.png"))


# --------------------------------------------------------------------------- #
# Holdout KNN vs MLP
# --------------------------------------------------------------------------- #
def plot_split1_vs_mlp(comparison: pd.DataFrame, plots_dir: Path,
                       out: List[str]) -> None:
    key = ["mae", "rmse", "r2", "spearman_mean", "ndcg@all", "top1_accuracy",
           "top3_overlap", "top5_overlap", "top10_overlap",
           "pairwise_ranking_accuracy"]
    sub = comparison[comparison["metric"].isin(key)].copy()
    sub = sub.set_index("metric").reindex([k for k in key if k in
                                           set(comparison["metric"])]).reset_index()
    x = np.arange(len(sub))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - 0.2, sub["knn"], width=0.4, label="KNN", color="#1f77b4")
    ax.bar(x + 0.2, sub["mlp"], width=0.4, label="MLP", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(sub["metric"], rotation=45, ha="right", fontsize=8)
    ax.set_title("Split-1 locked: KNN vs MLP")
    ax.legend()
    out.append(_save(fig, plots_dir / "split1_knn_vs_mlp.png"))

    comp = comparison.copy()
    fig, ax = plt.subplots(figsize=(9, max(4, 0.25 * len(comp))))
    colors = ["#2a2" if w == "KNN" else "#c22" if w == "MLP" else "#888"
              for w in comp["winner"]]
    ax.barh(range(len(comp)), comp["knn_minus_mlp"], color=colors)
    ax.set_yticks(range(len(comp)))
    ax.set_yticklabels(comp["metric"], fontsize=6)
    ax.invert_yaxis()
    ax.axvline(0, color="k", linewidth=0.8)
    ax.set_xlabel("KNN - MLP")
    ax.set_title("Split-1 metric delta vs MLP (green=KNN better, red=MLP better)")
    out.append(_save(fig, plots_dir / "split1_metric_delta_vs_mlp.png"))


def generate_all_plots(ctx: Dict, plots_dir: Path) -> List[str]:
    plots_dir.mkdir(parents=True, exist_ok=True)
    out: List[str] = []

    def _try(fn, *a, **k):
        try:
            fn(*a, **k)
        except Exception as e:  # noqa: BLE001
            print(f"[plot] {fn.__name__} failed: {e}")

    ranking = ctx["ranking"]
    fold_all = ctx["fold_metrics_all"]        # level == ALL
    sel = ctx["selected_id"]

    _try(plot_ranking, ranking, plots_dir, sel, out)
    _try(plot_metric_vs_k, fold_all, "spearman_mean", plots_dir,
         "spearman_vs_k.png", "Spearman", out)
    _try(plot_metric_vs_k, fold_all, "ndcg@3", plots_dir, "ndcg3_vs_k.png",
         "NDCG@3", out)
    _try(plot_metric_vs_k, fold_all, "ndcg@5", plots_dir, "ndcg5_vs_k.png",
         "NDCG@5", out)
    _try(plot_metric_vs_k, fold_all, "top5_overlap", plots_dir,
         "top5_overlap_vs_k.png", "Top-5 overlap", out)
    _try(plot_metric_vs_k, fold_all, "mae", plots_dir, "mae_vs_k.png", "MAE", out)
    _try(plot_metric_vs_k, fold_all, "dynamic_k_exact_accuracy", plots_dir,
         "dynamic_k_agreement_vs_k.png", "Dyn-K exact accuracy", out)
    _try(plot_metric_vs_k, fold_all, "query_time_sec", plots_dir,
         "runtime_vs_k.png", "Query time (s)", out)
    _try(plot_k_sensitivity_overview, fold_all, plots_dir, out)
    _try(plot_k_by_metric, ctx.get("per_metric_by_k"), plots_dir, out)

    _try(plot_component_comparisons, ranking, plots_dir, out)
    _try(plot_fold_stability, ctx["fold_composite"], plots_dir, sel, out)
    _try(plot_fold_metric_heatmap, ctx["fold_metrics_all_levels"], sel,
         plots_dir, out)

    _try(plot_view_performance, ctx["view_metrics"], plots_dir, out)
    _try(plot_group_heatmap, ctx["family_metrics"], "family", plots_dir,
         "family_performance_heatmap.png", "Per-family performance", out)
    _try(plot_group_heatmap, ctx["subfamily_metrics"], "subfamily", plots_dir,
         "subfamily_performance_heatmap.png", "Per-subfamily performance", out)

    _try(plot_prediction_quality, ctx["oof_true"], ctx["oof_pred"], plots_dir,
         out, "(OOF)")
    _try(plot_per_metric, ctx["per_metric"], plots_dir, out)
    _try(plot_dynamic_k, ctx["oof_dynamic_k_confusion"], ctx["oof_k_true"],
         ctx["oof_k_pred"], plots_dir, out)
    _try(plot_neighbor_geometry, ctx["neighbor_distances"],
         ctx["effective_counts"], plots_dir, out)
    _try(plot_split1_vs_mlp, ctx["mlp_comparison"], plots_dir, out)

    return out
