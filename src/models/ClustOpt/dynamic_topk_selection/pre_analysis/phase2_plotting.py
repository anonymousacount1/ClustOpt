"""Phase-2 figures (section 12). Headless Agg backend; never fatal on bad input."""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from heuristic_refinement import CAPS, FINE_ALPHAS  # noqa: E402

DPI = 130


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def relative_threshold_sweep(metrics, plot_dir):
    rows = []
    for a in FINE_ALPHAS:
        c = f"k_rel_a{int(round(a*100)):03d}"
        m = metrics[metrics["heuristic"] == c]
        if len(m):
            rows.append((a, m.iloc[0]))
    if not rows:
        return
    al = [r[0] for r in rows]
    meanK = [r[1]["mean_K"] for r in rows]
    p25 = [r[1]["P_2_5"] for r in rows]
    p10 = [r[1]["P_eq10"] for r in rows]
    bell = [r[1]["bell_fit"] for r in rows]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot(al, meanK, "o-", color="#3b7dd8", label="mean K")
    ax.set_xlabel("alpha (relative threshold)")
    ax.set_ylabel("mean K", color="#3b7dd8")
    ax.invert_xaxis()
    ax2 = ax.twinx()
    ax2.plot(al, p25, "s--", color="#5cb85c", label="P(2<=K<=5)")
    ax2.plot(al, p10, "^--", color="#d9534f", label="P(K==10)")
    ax2.plot(al, bell, "d:", color="#9467bd", label="bell_fit")
    ax2.set_ylabel("probability / bell_fit")
    ax.set_title("Relative-threshold fine sweep")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], fontsize=8, loc="center right")
    _save(fig, os.path.join(plot_dir, "relative_threshold_sweep.png"))


def cap_sensitivity(metrics, plot_dir, alpha=0.90):
    tag = f"a{int(round(alpha*100)):03d}"
    rows = []
    for cap in CAPS:
        c = f"k_relcap_{tag}_c{cap}"
        m = metrics[metrics["heuristic"] == c]
        if len(m):
            rows.append((cap, m.iloc[0]))
    if not rows:
        return
    caps = [r[0] for r in rows]
    meanK = [r[1]["mean_K"] for r in rows]
    bell = [r[1]["bell_fit"] for r in rows]
    p25 = [r[1]["P_2_5"] for r in rows]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot(caps, meanK, "o-", color="#3b7dd8", label="mean K")
    ax.set_xlabel(f"hard cap (alpha={alpha})")
    ax.set_ylabel("mean K", color="#3b7dd8")
    ax2 = ax.twinx()
    ax2.plot(caps, bell, "d--", color="#9467bd", label="bell_fit")
    ax2.plot(caps, p25, "s--", color="#5cb85c", label="P(2<=K<=5)")
    ax2.set_ylabel("bell_fit / P(2<=K<=5)")
    ax.set_title("Hard-cap sensitivity")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], fontsize=8)
    _save(fig, os.path.join(plot_dir, "cap_sensitivity.png"))


def heuristic_ranking(metrics, plot_dir, top=20):
    d = metrics.sort_values("composite_score", ascending=False).head(top)[::-1]
    fig, ax = plt.subplots(figsize=(8, max(4, top * 0.32)))
    ax.barh(d["heuristic"], d["composite_score"], color="#3b7dd8")
    ax.set_xlabel("composite score")
    ax.set_title(f"Top {top} heuristics by composite score")
    ax.tick_params(axis="y", labelsize=7)
    _save(fig, os.path.join(plot_dir, "heuristic_ranking.png"))


def bell_score_comparison(metrics, plot_dir, shortlist):
    d = metrics[metrics["heuristic"].isin(shortlist)].copy()
    d = d.sort_values("bell_fit")
    fig, ax = plt.subplots(figsize=(8, max(4, len(d) * 0.32)))
    colors = ["#5cb85c" if "phase1" not in f and "elbow(phase1)" not in f
              else "#999999" for f in d["family"]]
    ax.barh(d["heuristic"], d["bell_fit"], color=colors)
    ax.set_xlabel("bell_fit (1 = ideal discrete bell)")
    ax.set_title("Bell-score comparison (shortlist)")
    ax.tick_params(axis="y", labelsize=7)
    _save(fig, os.path.join(plot_dir, "bell_score_comparison.png"))


def agreement_heatmap(exact_df, plot_dir):
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(exact_df.to_numpy(), cmap="magma", vmin=0, vmax=100)
    ax.set_xticks(range(len(exact_df.columns)))
    ax.set_xticklabels(exact_df.columns, rotation=90, fontsize=6)
    ax.set_yticks(range(len(exact_df.index)))
    ax.set_yticklabels(exact_df.index, fontsize=6)
    ax.set_title("Exact K-agreement % between heuristics (shortlist)")
    fig.colorbar(im, ax=ax, label="% records identical K")
    _save(fig, os.path.join(plot_dir, "agreement_heatmap.png"))


def family_variance_heatmap(full, plot_dir, shortlist):
    cols = [c for c in shortlist if c in full.columns]
    fams = sorted(full["family_id"].unique())
    mat = np.array([[full.loc[full["family_id"] == fam, c].mean()
                     for fam in fams] for c in cols])
    fig, ax = plt.subplots(figsize=(max(8, len(fams) * 0.55), max(4, len(cols) * 0.34)))
    im = ax.imshow(mat, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(fams)))
    ax.set_xticklabels(fams, rotation=90, fontsize=6)
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(cols, fontsize=7)
    ax.set_title("Mean K by family (shortlist) -- visualises cross-family variance")
    fig.colorbar(im, ax=ax, label="mean K")
    _save(fig, os.path.join(plot_dir, "family_variance_heatmap.png"))


def make_all(plot_dir, metrics, full, exact_df, shortlist):
    os.makedirs(plot_dir, exist_ok=True)
    relative_threshold_sweep(metrics, plot_dir)
    cap_sensitivity(metrics, plot_dir)
    heuristic_ranking(metrics, plot_dir)
    bell_score_comparison(metrics, plot_dir, shortlist)
    agreement_heatmap(exact_df, plot_dir)
    family_variance_heatmap(full, plot_dir, shortlist)
