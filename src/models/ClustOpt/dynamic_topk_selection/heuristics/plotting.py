"""Plots for the selected-heuristic post-analysis (sections 3 & 8)."""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

MAX_K = 10
DPI = 130


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def k_distribution(df, path):
    k = df["selected_k"].astype(int)
    ks = np.arange(1, MAX_K + 1)
    pct = [float((k == v).mean() * 100) for v in ks]
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    bars = ax.bar(ks, pct, color="#3b7dd8", edgecolor="white")
    ax.set_title("Selected K distribution — dynamic_topk_relsoft_a092_k5_ceil")
    ax.set_xlabel("K (metrics selected)")
    ax.set_ylabel("% of records")
    ax.set_xticks(ks)
    for b, p in zip(bars, pct):
        ax.text(b.get_x() + b.get_width() / 2, p + 0.4, f"{p:.1f}",
                ha="center", fontsize=7)
    _save(fig, path)


def _group_heatmap(df, by, path, title, fontsize=7):
    groups = sorted(df[by].unique())
    mat = np.zeros((len(groups), MAX_K))
    for i, g in enumerate(groups):
        kk = df.loc[df[by] == g, "selected_k"].astype(int)
        for v in range(1, MAX_K + 1):
            mat[i, v - 1] = (kk == v).mean() * 100
    fig, ax = plt.subplots(figsize=(8, max(3.5, len(groups) * 0.22)))
    im = ax.imshow(mat, aspect="auto", cmap="viridis")
    ax.set_xticks(range(MAX_K)); ax.set_xticklabels(range(1, MAX_K + 1))
    ax.set_yticks(range(len(groups))); ax.set_yticklabels(groups, fontsize=fontsize)
    ax.set_xlabel("K"); ax.set_title(title)
    fig.colorbar(im, ax=ax, label="% of records")
    _save(fig, path)


def k_by_view(df, path):
    views = sorted(df["view_type"].unique())
    ks = np.arange(1, MAX_K + 1)
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    width = 0.8 / max(len(views), 1)
    for i, v in enumerate(views):
        kk = df.loc[df["view_type"] == v, "selected_k"].astype(int)
        pct = [float((kk == x).mean() * 100) for x in ks]
        ax.bar(ks + i * width - 0.4, pct, width=width, label=v)
    ax.set_title("Selected K by view"); ax.set_xlabel("K")
    ax.set_ylabel("% of records"); ax.set_xticks(ks); ax.legend()
    _save(fig, path)


def _mean_k_bar(df, by, path, title, fontsize=7):
    g = df.groupby(by)["selected_k"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(8, max(3.5, len(g) * 0.22)))
    ax.barh(g.index.astype(str), g.values, color="#5cb85c")
    ax.axvline(df["selected_k"].mean(), color="#d9534f", ls="--", lw=1.2,
               label=f"global mean={df['selected_k'].mean():.2f}")
    ax.set_xlabel("mean K"); ax.set_title(title); ax.legend(fontsize=8)
    ax.tick_params(axis="y", labelsize=fontsize)
    _save(fig, path)


def make_all(plot_dir, df):
    os.makedirs(plot_dir, exist_ok=True)
    k_distribution(df, os.path.join(plot_dir,
                                    "selected_heuristic_k_distribution.png"))
    fam_heat = "Selected K by family (% of records)"
    _group_heatmap(df, "family",
                   os.path.join(plot_dir,
                                "selected_heuristic_k_by_family_heatmap.png"),
                   fam_heat)
    _group_heatmap(df, "family",
                   os.path.join(plot_dir, "selected_heuristic_k_by_family.png"),
                   fam_heat)
    _group_heatmap(df, "subfamily",
                   os.path.join(plot_dir,
                                "selected_heuristic_k_by_subfamily.png"),
                   "Selected K by subfamily (% of records)", fontsize=5)
    k_by_view(df, os.path.join(plot_dir, "selected_heuristic_k_by_view.png"))
    _mean_k_bar(df, "family",
                os.path.join(plot_dir,
                             "selected_heuristic_mean_k_by_family.png"),
                "Mean selected K by family")
    _mean_k_bar(df, "subfamily",
                os.path.join(plot_dir,
                             "selected_heuristic_mean_k_by_subfamily.png"),
                "Mean selected K by subfamily", fontsize=4)
