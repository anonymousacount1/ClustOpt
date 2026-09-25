"""All figures for the Dynamic Top-K pre-analysis.

Uses the non-interactive Agg backend so it runs headless. Each public function
writes one (or a small family of) PNG(s) into ``plots/`` and never raises on
empty / degenerate input -- a missing figure is logged, not fatal.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from heuristic_candidates import MAX_K  # noqa: E402
from reporting import PRIMARY_HEURISTIC, SELECTED_CANDIDATES  # noqa: E402

DPI = 130


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def _hist(df, col, path, title, xlabel, bins=60, clip=None):
    s = df[col].replace([np.inf, -np.inf], np.nan).dropna()
    if clip is not None:
        s = s.clip(*clip)
    if not len(s):
        return
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.hist(s, bins=bins, color="#3b7dd8", edgecolor="white", linewidth=0.3)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("records")
    ax.axvline(s.median(), color="#d9534f", ls="--", lw=1.2,
               label=f"median={s.median():.3f}")
    ax.legend()
    _save(fig, path)


def _sorted_curve_matrix(wide_df, summary_df):
    """rows x ranks matrix of each record's utilities sorted descending."""
    ok = set(summary_df.loc[summary_df["status"] == "ok", "record_id"])
    w = wide_df[wide_df["record_id"].isin(ok)]
    cols = [c for c in w.columns if c.startswith("metric_")]
    vals = w[cols].to_numpy(dtype=float)
    vals = np.where(np.isfinite(vals), vals, np.nan)
    # sort each row descending, NaNs last
    srt = -np.sort(-np.nan_to_num(vals, nan=-np.inf), axis=1)
    srt = np.where(np.isfinite(srt), srt, np.nan)
    return srt


def make_all(out_dir, summary_df, wide_df, dist):
    plot_dir = os.path.join(out_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    df = summary_df[summary_df["status"] == "ok"].copy()
    made = []

    def p(name):
        made.append(name)
        return os.path.join(plot_dir, name)

    # ---- value distributions ----
    _hist(df, "max", p("utility_max_distribution.png"),
          "Utility max per record", "max utility")
    _hist(df, "mean", p("utility_mean_distribution.png"),
          "Utility mean per record", "mean utility")
    _hist(df, "std", p("utility_std_distribution.png"),
          "Utility std per record", "std utility")

    # ---- entropy / effective K ----
    _hist(df, "normalized_entropy", p("utility_entropy_distribution.png"),
          "Normalized entropy per record", "normalized entropy")
    _hist(df, "participation_ratio", p("utility_effective_k_distribution.png"),
          "Effective number of metrics (participation ratio)",
          "participation ratio", clip=(0, 60))

    # ---- sorted curves ----
    srt = _sorted_curve_matrix(wide_df, summary_df)
    if srt.size:
        ranks = np.arange(1, srt.shape[1] + 1)
        meanc = np.nanmean(srt, axis=0)
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        ax.plot(ranks, meanc, marker="o", ms=3, color="#3b7dd8")
        ax.set_title("Mean sorted utility curve")
        ax.set_xlabel("rank position (1 = best metric)")
        ax.set_ylabel("mean utility")
        ax.grid(alpha=.3)
        _save(fig, p("sorted_utility_mean_curve.png"))

        pcts = {"p10": 10, "p25": 25, "median": 50, "p75": 75, "p90": 90}
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        for lbl, q in pcts.items():
            ax.plot(ranks, np.nanpercentile(srt, q, axis=0), marker=".",
                    ms=3, label=lbl)
        ax.set_title("Sorted utility curve percentiles")
        ax.set_xlabel("rank position (1 = best metric)")
        ax.set_ylabel("utility")
        ax.legend()
        ax.grid(alpha=.3)
        _save(fig, p("sorted_utility_percentile_curves.png"))

    # ---- gap analysis ----
    _hist(df, "largest_gap_position_1_to_10",
          p("gap_position_distribution.png"),
          "Largest-gap position (ranks 1-10)", "elbow position", bins=10)
    _hist(df, "largest_gap_value_1_to_10",
          p("largest_gap_value_distribution.png"),
          "Largest-gap value (ranks 1-10)", "gap magnitude")
    # alias required by the spec output list
    _hist(df, "largest_gap_value_1_to_10", p("gap_distribution.png"),
          "Largest-gap value (ranks 1-10)", "gap magnitude")

    # ---- heuristic K distributions ----
    def _k_bar(cols, path, title):
        cols = [c for c in cols if c in df.columns]
        if not cols:
            return
        ks = np.arange(1, MAX_K + 1)
        fig, ax = plt.subplots(figsize=(max(7, len(cols) * 0.9), 4.5))
        width = 0.8 / len(cols)
        for i, c in enumerate(cols):
            counts = df[c].value_counts(normalize=True) * 100
            ax.bar(ks + i * width - 0.4, [counts.get(k, 0) for k in ks],
                   width=width, label=c)
        ax.set_title(title)
        ax.set_xlabel("K")
        ax.set_ylabel("% of records")
        ax.set_xticks(ks)
        ax.legend(fontsize=7, ncol=2)
        _save(fig, path)

    from heuristic_candidates import heuristic_columns
    _k_bar(heuristic_columns(), p("heuristic_k_distribution_all.png"),
           "K distribution -- all heuristics")
    _k_bar(SELECTED_CANDIDATES,
           p("heuristic_k_distribution_selected_candidates.png"),
           "K distribution -- selected candidates")
    # spec alias
    _k_bar(SELECTED_CANDIDATES, p("heuristic_k_distribution.png"),
           "K distribution -- selected candidates")

    # ---- heuristic K by family heatmap (primary candidate) ----
    prim = PRIMARY_HEURISTIC
    if prim in df.columns:
        fams = sorted(df["family_id"].unique())
        mat = np.zeros((len(fams), MAX_K))
        for i, fam in enumerate(fams):
            vc = df.loc[df["family_id"] == fam, prim].value_counts(normalize=True) * 100
            for k in range(1, MAX_K + 1):
                mat[i, k - 1] = vc.get(k, 0)
        fig, ax = plt.subplots(figsize=(8, max(4, len(fams) * 0.4)))
        im = ax.imshow(mat, aspect="auto", cmap="viridis")
        ax.set_xticks(range(MAX_K))
        ax.set_xticklabels(range(1, MAX_K + 1))
        ax.set_yticks(range(len(fams)))
        ax.set_yticklabels(fams, fontsize=7)
        ax.set_xlabel("K")
        ax.set_title(f"K distribution by family ({prim}, % of records)")
        fig.colorbar(im, ax=ax, label="% records")
        _save(fig, p("heuristic_k_by_family_heatmap.png"))

        # ---- by view ----
        views = sorted(df["view_type"].unique())
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        ks = np.arange(1, MAX_K + 1)
        width = 0.8 / max(len(views), 1)
        for i, v in enumerate(views):
            vc = df.loc[df["view_type"] == v, prim].value_counts(normalize=True) * 100
            ax.bar(ks + i * width - 0.4, [vc.get(k, 0) for k in ks],
                   width=width, label=v)
        ax.set_title(f"K distribution by view ({prim})")
        ax.set_xlabel("K")
        ax.set_ylabel("% of records")
        ax.set_xticks(ks)
        ax.legend()
        _save(fig, p("heuristic_k_by_view.png"))

    return made
