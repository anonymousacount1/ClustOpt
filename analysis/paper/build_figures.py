"""Build the paper's figure files from the evidence tables. Presentation only: no recomputation.

Output names follow an earlier numbering of the figures. ``docs/paper_results_index.md``
("Figure map") gives, for every figure of the paper, the file built here or the released data
it is drawn from; several figures of the paper are drawn directly in the manuscript.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_paths import (AC, C_ACCENT, C_AUTOCLUST, C_CLUSTOPT, C_CLUSTOPT_L,
                       C_GRID, C_ML2DAC, C_NEUTRAL, C_ORACLE, DATA, EVI,
                       FAM_PLAIN, FIG, MAN, NAME, S1, S2, S2B5, S3, SR,
                       SRC_COLOR, SRC_LABEL, UNI, apply_style, arm_color,
                       record, write_prov)

plt = apply_style()
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

W = 5.5  # ICLR text width, inches


def save(fig, name):
    fig.savefig(FIG / (name + ".pdf"))
    fig.savefig(FIG / (name + ".png"), dpi=210)
    plt.close(fig)
    print("  figure:", name)


def nogrid(ax):
    ax.grid(False)


def square_window(ax, x, y, pad=0.06):
    """Equal-aspect square window centred on the cloud: uniform panel sizes."""
    cx = (float(np.nanmin(x)) + float(np.nanmax(x))) / 2.0
    cy = (float(np.nanmin(y)) + float(np.nanmax(y))) / 2.0
    r = max(float(np.nanmax(x)) - float(np.nanmin(x)),
            float(np.nanmax(y)) - float(np.nanmin(y))) / 2.0
    r = r * (1.0 + pad) if r > 0 else 1.0
    ax.set_xlim(cx - r, cx + r); ax.set_ylim(cy - r, cy + r)
    ax.set_aspect("equal", adjustable="box")


def panel_tag(ax, s, dx=-0.085, dy=1.045):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=8.4, fontweight="bold",
            va="top", ha="left")


# --------------------------------------------------------------- SS19 naming
_METRIC_NAME = {
    "silhouette": "Silhouette",
    "noise_aware_silhouette": "Noise-aware silhouette",
    "gridness_fft_acf": "Gridness (FFT/ACF)",
    "gridness_2d": "Gridness (occupancy)",
    "axial_symmetry": "Axial symmetry",
    "connected_components_count": "Connected-component count",
    "connectivity_components": "Connectivity components",
    "contour_graph_connectivity": "Contour-graph connectivity",
    "convexity_ratio": "Convexity ratio",
    "convexity_ratio_image_based": "Convexity ratio (image)",
    "corner_junction_density": "Corner/junction density",
    "parallel_bands": "Parallel bands",
    "ribbon_thickness": "Ribbon thickness",
    "alpha_shape_compactness": "Alpha-shape compactness",
    "hough_arc_circle_strength": "Hough arc/circle strength",
    "hough_line_strength": "Hough line strength",
    "mst_smoothness": "MST smoothness",
    "glcm_haralick": "GLCM Haralick",
    "lbp_stationarity": "LBP stationarity",
}


def pretty_metric(mid):
    """Human-readable metric label; canonical ids stay in the tables."""
    m = str(mid)
    if m in _METRIC_NAME:
        return _METRIC_NAME[m]
    s = m.replace("_", " ")
    return s[:1].upper() + s[1:]


# an opaque halo for point annotations, so the Pareto guide and the grid never
# strike through a label (v15 figure audit)
LBOX = dict(facecolor="white", edgecolor="none", pad=0.6, alpha=1.0)

# ==================================================================== FIG 1
def fig1_problem_and_architecture():
    """Panel A: three structural motifs. Panel B: the ClustOpt pipeline."""
    gal = json.loads((EVI / "gallery_points.json").read_text())
    picks = ["arcs_circles_rings", "ladder_grid_lattice",
             "standard_clustering_blobs"]
    fig = plt.figure(figsize=(W, 2.10))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.52, 1.95], hspace=0.10,
                          wspace=0.06)

    # Panels are labelled by FAMILY, not by an asserted visual property: the
    # representative is fixed by the declared meta-feature-centroid rule and is
    # not always the most visually typical member of its family.
    titles = {"arcs_circles_rings": "Arcs \\& rings",
              "ladder_grid_lattice": "Ladder \\& grid",
              "standard_clustering_blobs": "Isotropic blobs"}
    for j, fam in enumerate(picks):
        ax = fig.add_subplot(gs[0, j])
        g = gal[fam]
        x = np.array(g["x"]); y = np.array(g["y"]); lab = np.array(g["label"])
        for i, u in enumerate(sorted(set(lab.tolist()))):
            m = lab == u
            ax.scatter(x[m], y[m], s=0.5, linewidths=0,
                       color=plt.get_cmap("tab10")(i % 10), rasterized=True)
        ax.set_xticks([]); ax.set_yticks([]); nogrid(ax)
        square_window(ax, x, y)
        ax.set_title(titles[fam].replace("\&","&"), fontsize=6.6, pad=2.0)
        for sp in ax.spines.values():
            sp.set_color("#BBBBBB"); sp.set_linewidth(0.5)
        if j == 0:
            panel_tag(ax, "A", dx=-0.30, dy=1.66)

    ax = fig.add_subplot(gs[1, :])
    ax.set_xlim(0, 106); ax.set_ylim(0, 60); ax.axis("off")
    panel_tag(ax, "B", dx=-0.045, dy=1.00)

    def box(x, y, w, h, txt, fc, ec, fs=5.6, tc="black", ls="solid"):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.30,rounding_size=0.9",
            linewidth=0.6, edgecolor=ec, facecolor=fc, zorder=3,
            linestyle=ls))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center",
                fontsize=fs, color=tc, zorder=4, linespacing=1.18)

    def arrow(x0, y0, x1, y1, c="#5A5A5A", ls="-", lw=0.7):
        ax.add_patch(FancyArrowPatch(
            (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=5.5,
            linewidth=lw, color=c, linestyle=ls, zorder=2,
            shrinkA=0, shrinkB=0))

    BL = "#E8F1F8"; OR = "#FDF0E6"; GY = "#F3F3F3"; CT = "#DCEAF6"
    CTRL = "#2E75B6"
    h = 7.6

    # ---------------- control row: the policy selector and what it chooses
    yS = 49.0
    box(1.0, yS, 24.0, h, "policy selector\n(PP v2)", CT, CTRL, fs=5.8)
    choices = [(28.0, 18.0, "predictor\nMLP / KNN"),
               (48.0, 24.0, "subset rule\ntop-1/3/5/10 / dynamic"),
               (73.0, 17.0, "weighting\nRAW / SOFTMAX")]
    for x, w, t in choices:
        box(x, yS, w, h, t, "#FFFFFF", CTRL, fs=5.3, ls=(0, (2.2, 1.4)))
    for x, w, _ in choices:
        arrow(25.0, yS + h / 2, x, yS + h / 2, c=CTRL, ls=(0, (2.2, 1.4)))

    # ---------------- data row 1
    yA = 34.0
    r1 = [(1.0, 18.0, "dataset view $(D,v)$", GY, "#9A9A9A"),
          (23.0, 19.0, "label-free\nmeta-features", BL, C_CLUSTOPT),
          (46.0, 21.0, "predicted utility\nprofile $\\hat{u}$", BL, C_CLUSTOPT),
          (71.0, 24.0, "objective $J_{S}$\n(weighted top-$k$)", BL, C_CLUSTOPT)]
    for x, w, t, fc, ec in r1:
        box(x, yA, w, h, t, fc, ec)
    for (x0, w0, *_), (x1, *_) in zip(r1, r1[1:]):
        arrow(x0 + w0, yA + h / 2, x1, yA + h / 2)
    # control edges land on the components they parameterise
    arrow(37.0, yS, 56.5, yA + h, c=CTRL, ls=(0, (2.2, 1.4)))   # predictor
    arrow(60.0, yS, 83.0, yA + h, c=CTRL, ls=(0, (2.2, 1.4)))   # subset rule
    arrow(81.5, yS, 89.0, yA + h, c=CTRL, ls=(0, (2.2, 1.4)))   # weighting
    # meta-features also feed the selector
    arrow(32.5, yA + h, 13.0, yS, c=CTRL, ls=(0, (2.2, 1.4)))

    # ---------------- data row 2
    yB = 18.5
    r2 = [(1.0, 16.0, "search", BL, C_CLUSTOPT),
          (21.0, 19.0, "candidate slate", GY, "#9A9A9A"),
          (44.0, 22.0, "reranker $r_\\psi$\n(regime-specific)", OR, C_AUTOCLUST),
          (70.0, 20.0, "returned partition", GY, "#9A9A9A")]
    for x, w, t, fc, ec in r2:
        box(x, yB, w, h, t, fc, ec)
    for (x0, w0, *_), (x1, *_) in zip(r2, r2[1:]):
        arrow(x0 + w0, yB + h / 2, x1, yB + h / 2)
    ax.plot([95.0, 99.6, 99.6, 9.0], [yA + h / 2, yA + h / 2, yA - 3.6,
                                      yA - 3.6], color="#5A5A5A", lw=0.7,
            zorder=2, solid_capstyle="round")
    arrow(9.0, yA - 3.6, 9.0, yB + h, lw=0.7)
    # the selector also picks the reranker regime
    arrow(13.0, yS, 55.0, yB + h, c=CTRL, ls=(0, (2.2, 1.4)))

    # ---------------- offline band
    yC = 2.5
    ax.add_patch(FancyBboxPatch(
        (1.0, yC), 89.0, 7.0, boxstyle="round,pad=0.30,rounding_size=0.9",
        linewidth=0.6, edgecolor="#9A9A9A", facecolor="#FBFBFB", zorder=3,
        linestyle=(0, (2.4, 1.5))))
    ax.text(45.5, yC + 3.5,
            "offline, labelled development data only: utility targets, "
            "utility predictors, policy-selector targets, reranker targets",
            ha="center", va="center", fontsize=5.3, color="#4A4A4A", zorder=4)
    arrow(18.0, yC + 7.0, 18.0, yB, c="#9A9A9A", ls=(0, (2.2, 1.4)))
    arrow(52.0, yC + 7.0, 52.0, yB, c="#9A9A9A", ls=(0, (2.2, 1.4)))
    ax.text(92.0, yC + 3.5, "offline", fontsize=5.4, color="#6A6A6A",
            va="center", ha="left", style="italic")
    ax.text(92.0, yB + h / 2, "online\n(no labels)", fontsize=5.4,
            color="#6A6A6A", va="center", ha="left", style="italic",
            linespacing=1.2)
    ax.text(92.0, yS + h / 2, "control", fontsize=5.4, color=CTRL,
            va="center", ha="left", style="italic")

    save(fig, "fig1_problem_and_architecture")
    record("fig1_problem_and_architecture",
           ["analyzed_data/<family>/.../data_data.csv"],
           "schematic; point clouds are stored repository datasets chosen by "
           "the deterministic rule in manifests/gallery_selection.json")


# ==================================================================== FIG 2
def fig2_controlled_mechanism():
    off = pd.read_csv(S1 / "offline_fixed32" / "offline_2x3_overall_summary.csv")
    onl = pd.read_csv(S1 / "online_fixed50" / "online_2x3_overall_summary.csv")
    co = pd.read_csv(S1 / "offline_fixed32" / "offline_2x3_paired_contrasts.csv")
    cn = pd.read_csv(S1 / "online_fixed50" / "online_2x3_paired_contrasts.csv")
    prog = pd.read_csv(EVI / "controlled_progression.csv")
    acm = pd.read_csv(S2 / "autoclust_matched_comparison.csv").iloc[0]

    fig = plt.figure(figsize=(W, 1.52))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.45], wspace=0.32)

    # ---- A: absolute means, both regimes
    ax = fig.add_subplot(gs[0])
    strat = ["uniform", "predicted", "oracle"]
    xs = np.arange(3)
    wd = 0.34
    for k, (df, mk, lbl, vc) in enumerate(
            [(off, "o", "fixed", "mean_best_view_ari"),
             (onl, "s", "online", "mean_ari")]):
        d = df.set_index("arm")
        for inv, code, col in [("Head14", "H", "#9ECAE1"),
                               ("Full60", "F", C_CLUSTOPT)]:
            ys = [d.loc[code + s[0].upper(), vc] for s in strat]
            lo = [d.loc[code + s[0].upper(), "mean_ci_lo"] for s in strat]
            hi = [d.loc[code + s[0].upper(), "mean_ci_hi"] for s in strat]
            off_x = (-wd / 2 if inv == "Head14" else wd / 2)
            ax.errorbar(xs + off_x + (k - 0.5) * 0.09, ys,
                        yerr=[np.array(ys) - lo, np.array(hi) - ys],
                        fmt=mk, ms=3.0, lw=0, elinewidth=0.6, capsize=1.5,
                        color=col, mfc=col if k == 0 else "white",
                        mec=col, mew=0.8,
                        label=("%s, %s" % (inv, lbl)))
    ax.set_xticks(xs)
    ax.set_xticklabels(["uniform", "predicted", "oracle"])
    ax.set_ylabel("Mean Best-View ARI")
    ax.set_title("Inventory $\\times$ weighting", fontsize=7.4)
    # the four combined labels overran panel A's right spine at full length,
    # so the regime is abbreviated here and spelled out in the caption
    ax.legend(ncol=2, fontsize=5.0, loc="upper left", columnspacing=0.6,
              handletextpad=0.3, borderaxespad=0.12, handlelength=1.0)
    ax.set_ylim(0.30, 0.79)
    panel_tag(ax, "A", dx=-0.30, dy=1.13)

    # ---- C: controlled progression including the pre-reranker point
    ax = fig.add_subplot(gs[1])
    lbls = ["Fixed\nno rerank", "Fixed\n+ rerank", "PPv1\n+ rerank",
            "PPv2\n+ rerank", "Oracle\n+ rerank"]
    xs = np.arange(len(prog))
    cols = [C_CLUSTOPT_L, C_CLUSTOPT, C_CLUSTOPT, C_CLUSTOPT, C_ORACLE]
    for x, (_, r), c in zip(xs, prog.iterrows(), cols):
        if np.isfinite(r["lo"]):
            ax.plot([x, x], [r["lo"], r["hi"]], color=c, lw=0.9, zorder=2)
            ax.plot([x - 0.055, x + 0.055], [r["lo"]] * 2, color=c, lw=0.7)
            ax.plot([x - 0.055, x + 0.055], [r["hi"]] * 2, color=c, lw=0.7)
        ax.plot([x], [r["mean"]], "o", ms=4.0, color=c, zorder=4,
                mec="white", mew=0.6)
        # PPv1's label lands on the AutoClust rule; halo it like the scatters
        ax.annotate("%.3f" % r["mean"], (x, r["mean"]),
                    textcoords="offset points", xytext=(7.0, 5.5),
                    fontsize=6.0, va="center", color="#2A2A2A",
                    bbox=LBOX, zorder=6)
    ax.axhline(float(acm.autoclust_reference_value), color=C_AUTOCLUST,
               lw=0.85, ls=(0, (3.2, 1.8)), zorder=1)
    # sitting on the line, this label's halo erased the point values beside
    # it, so it moves to the empty top-left corner; the dashed line is the
    # only orange rule in the panel and the label names its value
    ax.text(0.015, 0.975,
            "AutoClust Extended \u2014 matched search domain (%.3f)"
            % acm.autoclust_reference_value, fontsize=5.7,
            transform=ax.transAxes,
            color=C_AUTOCLUST, ha="left", va="top",
            bbox=dict(facecolor="white", edgecolor="none", pad=0.5,
                      alpha=1.0))
    ax.set_xticks(xs); ax.set_xticklabels(lbls, fontsize=5.9)
    ax.set_xlim(-0.45, len(prog) - 0.25)
    lo = float(prog["mean"].min()); hi = float(prog["hi"].max())
    ax.set_ylim(lo - 0.012, hi + 0.010)
    ax.set_ylabel("Mean Best-View ARI")
    ax.set_title("Controlled Split-1 replay: where the gain comes from",
                 fontsize=7.4)
    panel_tag(ax, "B", dx=-0.115, dy=1.13)

    save(fig, "fig2_controlled_mechanism")
    record("fig2_controlled_mechanism",
           ["stage1_2x3_aggregation/*/overall_summary.csv",
            "stage1_2x3_aggregation/*/paired_contrasts.csv",
            "candidate_reranker_split1_online_final/per_policy_results.csv",
            "stage2c_final_v2_split1_online_replay/four_level_online_table.csv"],
           "stored means, stored bootstrap intervals, stored Holm decisions")


# ==================================================================== FIG 3
def fig3_family_delta():
    """Main text: the 12-family difference from matched AutoClust."""
    g = pd.read_csv(EVI / "controlled_family_deltas.csv")
    fig, ax = plt.subplots(figsize=(W, 2.00))
    xs = np.arange(len(g))
    ax.bar(xs - 0.19, g.d_v2_ac, width=0.38, color=C_CLUSTOPT,
           label="PPv2 + rerank $-$ AutoClust Extended", zorder=3)
    ax.bar(xs + 0.19, g.d_fixed_ac, width=0.38, color=C_CLUSTOPT_L,
           label="Fixed policy + rerank $-$ AutoClust Extended", zorder=3)
    ax.axhline(0, color="#5A5A5A", lw=0.7, zorder=4)
    ax.axhspan(-0.01, 0.01, color="#EDEDED", zorder=0)
    ax.set_xticks(xs)
    ax.set_xticklabels(g.family, rotation=30, ha="right", fontsize=5.9)
    ax.set_ylabel("$\Delta$ mean ARI")
    ax.legend(fontsize=5.8, loc="upper right", ncol=1)
    ax.text(0.005, 0.055, "shaded: $|\Delta|\leq 0.01$", fontsize=5.5,
            color="#6A6A6A", transform=ax.transAxes)
    ax.set_xlim(-0.65, len(g) - 0.35)
    save(fig, "fig3_family_delta")
    record("fig3_family_delta", ["controlled_family_deltas.csv (derived)"],
           "descriptive family means over the frozen per-dataset Split-1 rows")


def figE_selection_heatmap():
    """Appendix: family x index selection structure."""
    sel = pd.read_csv(EVI / "family_metric_selection_matrix.csv", index_col=0)
    core = json.loads((EVI / "evidence_summary.json").read_text())["tailoring"]
    fig, ax = plt.subplots(figsize=(W, 2.85))
    order = [c for c in FAM_PLAIN if c in sel.columns]
    M = sel[order].to_numpy(dtype=float)
    Z = M - M.mean(axis=1, keepdims=True)
    v = np.nanmax(np.abs(Z))
    im = ax.imshow(Z, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels([FAM_PLAIN[c] for c in order], rotation=32, ha="right",
                       fontsize=5.8)
    ax.set_yticks(np.arange(len(sel.index)))
    ax.set_yticklabels([pretty_metric(m) for m in sel.index], fontsize=5.8)
    nogrid(ax)
    ncore = len(core["core"]); n = len(sel.index)
    ax.axhline(ncore - 0.5, color="#2A2A2A", lw=0.9)
    ax.text(-0.235, 1.0 - (ncore / n) / 2, "core", transform=ax.transAxes,
            fontsize=5.8, rotation=90, va="center", ha="center", color="#3A3A3A")
    ax.text(-0.235, (1.0 - ncore / n) / 2, "specialists",
            transform=ax.transAxes, fontsize=5.8, rotation=90, va="center",
            ha="center", color="#3A3A3A")
    cb = fig.colorbar(im, ax=ax, fraction=0.019, pad=0.012)
    cb.ax.tick_params(labelsize=5.4)
    cb.set_label("deviation from the index's own mean", fontsize=5.6)
    save(fig, "figE_selection_heatmap")
    record("figE_selection_heatmap",
           ["family_subfamily_specialization/family_metric_selection.csv.gz"],
           "stored in-family selection rates, row-centred for display only")


def figF_core_external():
    """Appendix: external per-source utility of the three core indices."""
    gc = pd.read_csv(SR / "generalist_core_review.csv")
    idcol = "metric_id"
    srcs = ["utility_fcps", "utility_sklearn_shapes", "utility_real_pca"]
    lbl = {"utility_fcps": "FCPS", "utility_sklearn_shapes": "scikit-learn",
           "utility_real_pca": "real (PCA)"}
    fig, ax = plt.subplots(figsize=(W * 0.74, 1.55))
    ys = np.arange(len(gc))
    for j, sname in enumerate(srcs):
        key = sname.replace("utility_", "")
        ax.scatter(gc[sname], ys + (j - 1) * 0.20, s=14,
                   color=SRC_COLOR.get(key, C_NEUTRAL), label=lbl[sname],
                   zorder=3, edgecolor="white", linewidth=0.4)
    ax.set_yticks(ys)
    ax.set_yticklabels([str(m).replace("_", " ") for m in gc[idcol]],
                       fontsize=6.0)
    ax.set_xlabel("Utility on the independent corpus")
    ax.legend(fontsize=5.6, ncol=1, loc="upper left",
              bbox_to_anchor=(1.004, 1.02), borderaxespad=0.0)
    ax.set_ylim(-0.6, len(gc) - 0.4)
    lo = float(np.nanmin(gc[srcs].to_numpy()))
    hi = float(np.nanmax(gc[srcs].to_numpy()))
    pad = 0.09 * (hi - lo)
    ax.set_xlim(lo - pad, hi + pad)
    save(fig, "figF_core_external")
    record("figF_core_external", ["scientific_review/generalist_core_review.csv"],
           "stored per-source available-case utilities")


# ==================================================================== FIG 4
def fig4_external():
    """Main: accuracy against measured cost, all 14 arms (Gate 17)."""
    tv = pd.read_csv(SR / "runtime_performance_tradeoff.csv").set_index("short")

    fig, ax = plt.subplots(figsize=(W, 1.72))
    xc = [c for c in tv.columns if "runtime" in c and "median" in c][0]
    yc = [c for c in tv.columns if "ari" in c.lower() and "mean" in c][0]
    HEAD = {"C5": "ClustOpt PPv2+R", "C4": "ClustOpt KNN+R",
            "A0": "AutoClust Original", "M1": "ML2DAC +Established"}

    # Pareto guide first, so it sits under the markers
    if "pareto_efficient" in tv.columns:
        fr = tv[tv.pareto_efficient].sort_values(xc)
        ax.plot(fr[xc], fr[yc], color="#BFBFBF", lw=0.7, ls=(0, (3, 2.5)),
                zorder=1)

    for k in tv.index:
        big = k in HEAD
        ax.scatter(tv.loc[k, xc], tv.loc[k, yc], s=52 if big else 22,
                   color=arm_color(k), alpha=1.0 if big else 0.75,
                   zorder=4 if big else 3, edgecolor="white",
                   linewidth=0.7 if big else 0.4)

    # headline arms: words. everything else: its two-character code.
    OFF_H = {"C5": (8, 1.0), "C4": (-8, 3.0), "A0": (8, 1.0), "M1": (8, 1.0)}
    for k in tv.index:
        x, y = tv.loc[k, xc], tv.loc[k, yc]
        if k in HEAD:
            dx, dy = OFF_H[k]
            ax.annotate(HEAD[k], (x, y), textcoords="offset points",
                        xytext=(dx, dy), fontsize=6.1,
                        ha="left" if dx >= 0 else "right", va="center",
                        color=arm_color(k), fontweight="bold",
                        bbox=LBOX, zorder=6)
        else:
            # A2/A3 and M0/M2 sit close together; nudge those four apart
            OFF_C = {"A2": (-9, 2.0), "A3": (9, -1.0), "M0": (9, -1.0),
                     "M2": (0, -9.5)}
            dx, dy = OFF_C.get(k, (0, -9.5))
            ax.annotate(k, (x, y), textcoords="offset points",
                        xytext=(dx, dy), fontsize=5.6,
                        ha="center" if dx == 0 else
                        ("left" if dx > 0 else "right"),
                        va="center", color=arm_color(k),
                        bbox=LBOX, zorder=6)

    ax.set_xscale("log")
    ticks = [20, 40, 80, 160]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_xlim(16.0, 200.0)
    ymin, ymax = float(tv[yc].min()), float(tv[yc].max())
    ax.set_ylim(ymin - 0.16 * (ymax - ymin), ymax + 0.13 * (ymax - ymin))
    ax.set_xlabel("Median three-view runtime (s, log scale)")
    ax.set_ylabel("Mean Best-View ARI")

    # a compact family key, since most points carry only a code
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ls="", ms=4.2, color=c, label=t)
               for c, t in ((C_CLUSTOPT, "ClustOpt (C0–C5)"),
                            (C_AUTOCLUST, "AutoClust (A0–A3)"),
                            (C_ML2DAC, "ML2DAC (M0–M3)"))]
    ax.legend(handles=handles, fontsize=5.6, loc="lower right", frameon=False,
              handletextpad=0.3, borderaxespad=0.3, labelspacing=0.25)

    save(fig, "fig4_external")
    record("fig4_external",
           ["scientific_review/runtime_performance_tradeoff.csv"],
           "stored runtime medians and mean ARIs; the Pareto flag is the "
           "stored one")
def figG_paired_delta():
    """Per-dataset paired deltas against each baseline (an earlier figure; not used by the paper)."""
    per = pd.read_csv(EVI / "external_per_dataset.csv")
    meta = pd.read_csv(EVI / "external_corpus_metadata.csv")
    per = per.drop(columns=["source"], errors="ignore").merge(
        meta[["dataset_id", "source"]], on="dataset_id")

    fig, ax = plt.subplots(figsize=(W * 0.72, 2.15))
    rng = np.random.default_rng(7)
    groups = [("d_C5_A0", "vs. AutoClust\n(best variant)"),
              ("d_C5_M1", "vs. ML2DAC\n(best variant)")]
    for gi, (col, lab) in enumerate(groups):
        for si, src in enumerate(["fcps", "sklearn_shapes", "real_pca"]):
            sub = per[per.source == src]
            xj = gi + (si - 1) * 0.22 + rng.uniform(-0.045, 0.045, len(sub))
            ax.scatter(xj, sub[col], s=8.5, color=SRC_COLOR[src],
                       alpha=0.85, linewidths=0, zorder=3,
                       label=SRC_LABEL[src] if gi == 0 else None)
        mu = per[col].mean()
        ax.plot([gi - 0.36, gi + 0.36], [mu, mu], color="#2A2A2A", lw=1.0,
                zorder=2)          # behind the points: data first
        ax.text(gi, float(per[col].max()) + 0.055, "mean %+.3f" % mu,
                ha="center", va="bottom", fontsize=6.0, color="#2A2A2A")
    ax.axhline(0, color="#8A8A8A", lw=0.6, zorder=1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([g[1] for g in groups], fontsize=6.2)
    ax.set_xlim(-0.5, 1.5)
    _all = pd.concat([per["d_C5_A0"], per["d_C5_M1"]])
    ax.set_ylim(float(_all.min()) - 0.05, float(_all.max()) + 0.17)
    ax.set_ylabel("Per-dataset paired $\\Delta$ Best-View ARI")
    ax.legend(fontsize=5.6, loc="upper left", ncol=3, columnspacing=0.8,
              handletextpad=0.3, bbox_to_anchor=(-0.02, 1.15),
              borderaxespad=0.0)
    save(fig, "figG_paired_delta")
    record("figG_paired_delta",
           ["scientific_review/per_dataset_method_matrix.csv"],
           "stored per-dataset ARIs")


def fig3_metric():
    """Main metric figure: core -> specialists -> selection tracks utility."""
    gc = pd.read_csv(SR / "generalist_core_review.csv")
    fa = pd.read_csv(EVI / "stage3_family_tierA.csv")
    al = pd.read_csv(EVI / "stage3_alignment.csv")

    fig = plt.figure(figsize=(W, 1.80))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.74, 1.40, 0.86], wspace=1.28)

    # ---------------- A: the generalist core, externally
    ax = fig.add_subplot(gs[0])
    srcs = ["utility_fcps", "utility_sklearn_shapes", "utility_real_pca"]
    lbl = {"utility_fcps": "FCPS", "utility_sklearn_shapes": "scikit-learn",
           "utility_real_pca": "real (PCA)"}
    ys = np.arange(len(gc))
    for j, sname in enumerate(srcs):
        key = sname.replace("utility_", "")
        ax.scatter(gc[sname], ys + (j - 1) * 0.20, s=17,
                   color=SRC_COLOR.get(key, C_NEUTRAL), label=lbl[sname],
                   zorder=3, edgecolor="white", linewidth=0.4)
    ax.set_yticks(ys)
    ax.set_yticklabels([pretty_metric(m) for m in gc["metric_id"]],
                       fontsize=5.9)
    ax.set_xlabel("Index utility", fontsize=6.6)
    ax.set_title("A  Generalist core transfers", fontsize=7.0, loc="left",
                 pad=11.0)
    ax.legend(fontsize=5.2, loc="lower left", bbox_to_anchor=(0.0, 1.005),
              ncol=3, frameon=False, handletextpad=0.2, columnspacing=0.7,
              borderaxespad=0.0)
    ax.set_ylim(-0.7, len(gc) - 0.3)
    lo = float(np.nanmin(gc[srcs].to_numpy()))
    hi = float(np.nanmax(gc[srcs].to_numpy()))
    pad = 0.22 * (hi - lo)
    ax.set_xlim(lo - pad, hi + pad)
    ax.tick_params(labelsize=5.9)

    # ---------------- B: sparse family x specialist matrix
    # v16: this panel used marker area AND shade for the same quantity, which
    # made it slower to read than it needed to be. It is now a sparse matrix
    # with ONE encoding -- cell colour is the utility uplift, and a colour bar
    # states that. Membership, values and ordering rule are unchanged, and all
    # 14 frozen Tier-A relations are still drawn.
    ax = fig.add_subplot(gs[1])
    # deterministic rule: every metric appearing in a Tier-A family relation,
    # deduplicated; rows ordered by the metric's maximum Tier-A uplift.
    order = (fa.groupby("metric_id")["DeltaU"].max()
             .sort_values(ascending=False).index.tolist())
    fams = list(FAM_PLAIN.keys())   # all 12, so empty families are visible
    xi = {f: i for i, f in enumerate(fams)}
    yi = {m: i for i, m in enumerate(order)}
    dmax = float(fa["DeltaU"].max())
    M = np.full((len(order), len(fams)), np.nan)
    for _, r in fa.iterrows():
        M[yi[r["metric_id"]], xi[r["group"]]] = float(r["DeltaU"])
    assert int(np.isfinite(M).sum()) == len(fa), "lost a Tier-A relation"

    # Blues truncated away from white: the weakest relation (0.022) must still
    # read as a filled cell against the empty ones, which are left white
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "tierA", plt.get_cmap("Blues")(np.linspace(0.28, 0.95, 256)))
    # a faint rule grid so the empty cells read as part of a matrix
    for i in range(len(order) + 1):
        ax.plot([0, len(fams)], [i, i], color=C_GRID, lw=0.3, zorder=1)
    for j in range(len(fams) + 1):
        ax.plot([j, j], [0, len(order)], color=C_GRID, lw=0.3, zorder=1)
    pm = ax.pcolormesh(np.ma.masked_invalid(M), cmap=cmap, vmin=0.0,
                       vmax=dmax, edgecolors="#2A2A2A", linewidth=0.35,
                       zorder=2)
    ax.set_xlim(0, len(fams)); ax.set_ylim(len(order), 0)
    ax.set_xticks(np.arange(len(fams)) + 0.5)
    ax.set_xticklabels([FAM_PLAIN.get(f, f) for f in fams], rotation=90,
                       fontsize=5.1)
    ax.set_yticks(np.arange(len(order)) + 0.5)
    ax.set_yticklabels([pretty_metric(m) for m in order], fontsize=5.2)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("B  Specialists differ by family", fontsize=7.0, loc="left",
                 pad=11.0)
    cb = fig.colorbar(pm, ax=ax, fraction=0.030, pad=0.015, aspect=15)
    cb.set_label("Utility uplift $\\Delta U$", fontsize=5.0, labelpad=1.2)
    cb.set_ticks([0.0, 0.05, 0.10])
    cb.ax.tick_params(labelsize=4.7, length=1.4, width=0.35, pad=1.0)
    cb.outline.set_linewidth(0.35)

    # ---------------- C: selection tracks utility, per family
    ax = fig.add_subplot(gs[2])
    a = al.sort_values("spearman_util_sel_all").reset_index(drop=True)
    ys = np.arange(len(a))
    ax.scatter(a["spearman_util_sel_all"], ys, s=19, color=C_CLUSTOPT,
               zorder=3, edgecolor="white", linewidth=0.4)
    mu = float(a["spearman_util_sel_all"].mean())
    ax.axvline(mu, color=C_ACCENT, lw=0.9, ls=(0, (3, 2)), zorder=2)
    # panel C is too narrow to hold this label anywhere without covering
    # either a family name, the right spine or the top marker; the caption
    # already gives the mean and says the rule is dashed, so it is dropped
    ax.set_yticks(ys)
    ax.set_yticklabels([FAM_PLAIN.get(g, g) for g in a["group"]], fontsize=5.4)
    ax.set_xlabel("Spearman($U$, selection freq.)", fontsize=6.6)
    ax.set_title("C  Selection tracks utility", fontsize=7.0, loc="left",
                 pad=11.0)
    ax.set_ylim(-0.7, len(a) - 0.3)
    lo = float(a["spearman_util_sel_all"].min())
    hi = float(a["spearman_util_sel_all"].max())
    ax.set_xlim(lo - 0.05, hi + 0.05)
    ax.tick_params(labelsize=5.9)

    save(fig, "fig3_metric")
    record("fig3_metric",
           ["scientific_review/generalist_core_review.csv",
            "family_subfamily_specialization/family_specialists_tierA.csv",
            "family_subfamily_specialization/family_utility_selection_alignment.csv"],
           "stored per-source utilities, stored Tier-A uplifts and stored "
           "per-family Spearman values; no statistic recomputed")
def figA_gallery():
    gal = json.loads((EVI / "gallery_points.json").read_text())
    fams = [f for f in FAM_PLAIN if f in gal]
    fig, axes = plt.subplots(3, 4, figsize=(W, 4.35))
    for ax, fam in zip(axes.ravel(), fams):
        g = gal[fam]
        x = np.array(g["x"]); y = np.array(g["y"]); lab = np.array(g["label"])
        for i, u in enumerate(sorted(set(lab.tolist()))):
            m = lab == u
            ax.scatter(x[m], y[m], s=0.85, linewidths=0,
                       color=plt.get_cmap("tab10")(i % 10), rasterized=True)
        ax.set_title(FAM_PLAIN[fam], fontsize=6.4, pad=2.0)
        ax.set_xticks([]); ax.set_yticks([]); nogrid(ax)
        square_window(ax, x, y)
        for sp in ax.spines.values():
            sp.set_color("#BBBBBB"); sp.set_linewidth(0.5)
    for ax in axes.ravel()[len(fams):]:
        ax.axis("off")
    fig.subplots_adjust(hspace=0.30, wspace=0.06)
    save(fig, "figA_family_gallery")
    record("figA_family_gallery", ["analyzed_data/.../data_data.csv"],
           "stored xy_2d point clouds, coloured by generator ground truth; "
           "representative chosen by the deterministic meta-feature-centroid "
           "rule recorded in manifests/gallery_selection.json")


def figB_family_absolute():
    g = pd.read_csv(EVI / "controlled_family_deltas.csv")
    cols = ["fixed", "v1", "v2", "vbs", "autoclust"]
    lab = ["Fixed+R", "PPv1+R", "PPv2+R", "Oracle+R", "AutoClust Ext."]
    M = g.set_index("family")[cols].to_numpy(dtype=float).T
    fig, ax = plt.subplots(figsize=(W, 2.35))
    im = ax.imshow(M, cmap="viridis", aspect="auto", vmin=0.15, vmax=0.90)
    ax.set_xticks(np.arange(len(g)))
    ax.set_xticklabels(g.family, rotation=34, ha="right", fontsize=6.0)
    ax.set_yticks(np.arange(len(cols))); ax.set_yticklabels(lab, fontsize=6.2)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, "%.2f" % M[i, j], ha="center", va="center",
                    fontsize=4.9,
                    color="white" if M[i, j] < 0.60 else "#101010")
    nogrid(ax)
    cb = fig.colorbar(im, ax=ax, fraction=0.020, pad=0.010)
    cb.ax.tick_params(labelsize=5.2); cb.set_label("mean ARI", fontsize=5.6)
    ax.set_title("Absolute controlled mean Best-View ARI by structural family",
                 fontsize=7.4)
    save(fig, "figB_family_absolute")
    record("figB_family_absolute", ["controlled_family_deltas.csv (derived)"],
           "descriptive family means over the frozen per-dataset Split-1 rows")


def figC_single_view_pareto():
    xy = pd.read_csv(SR / "runtime_xy2d_tradeoff.csv").set_index("short")
    fig, ax = plt.subplots(figsize=(W * 0.72, 2.35))
    xc = [c for c in xy.columns if "runtime" in c and "median" in c][0]
    yc = [c for c in xy.columns if "ari" in c.lower() and "mean" in c][0]
    HEAD = {"C5": "ClustOpt PPv2+R", "C4": "ClustOpt KNN+R",
            "A0": "AutoClust Original", "M1": "ML2DAC +Established"}

    if "pareto_efficient" in xy.columns:
        fr = xy[xy.pareto_efficient].sort_values(xc)
        ax.plot(fr[xc], fr[yc], color="#BFBFBF", lw=0.7, ls=(0, (3, 2.5)),
                zorder=1)
    for k in xy.index:
        big = k in HEAD
        ax.scatter(xy.loc[k, xc], xy.loc[k, yc], s=42 if big else 18,
                   color=arm_color(k), alpha=1.0 if big else 0.75,
                   zorder=4 if big else 3, edgecolor="white",
                   linewidth=0.6 if big else 0.35)

    OFF_H = {"C5": (-8, 3.0), "C4": (7, -2.0), "A0": (-8, 2.0), "M1": (7, 1.0)}
    OFF_C = {"C3": (6.5, 0.0), "C2": (7, -1.0), "C1": (0, -9.0),
             "C0": (-7, -1.0), "A1": (0, -8.5), "A2": (-8, 3.0),
             "A3": (7, -2.0), "M0": (6.5, -1.0), "M2": (-7, -1.0),
             "M3": (0, 8.0)}
    for k in xy.index:
        x, y = xy.loc[k, xc], xy.loc[k, yc]
        if k in HEAD:
            dx, dy = OFF_H[k]
            ax.annotate(HEAD[k], (x, y), textcoords="offset points",
                        xytext=(dx, dy), fontsize=5.9, va="center",
                        ha="left" if dx >= 0 else "right",
                        color=arm_color(k), fontweight="bold",
                        bbox=LBOX, zorder=6)
        else:
            dx, dy = OFF_C.get(k, (0, -9.0))
            ax.annotate(k, (x, y), textcoords="offset points",
                        xytext=(dx, dy), fontsize=5.5, va="center",
                        ha="center" if dx == 0 else
                        ("left" if dx > 0 else "right"),
                        color=arm_color(k), bbox=LBOX, zorder=6)

    xmin, xmax = float(xy[xc].min()), float(xy[xc].max())
    ymin, ymax = float(xy[yc].min()), float(xy[yc].max())
    ax.set_xlim(xmin - 0.10 * (xmax - xmin), xmax + 0.12 * (xmax - xmin))
    ax.set_ylim(ymin - 0.10 * (ymax - ymin), ymax + 0.16 * (ymax - ymin))
    ax.set_xlabel("Median single-view runtime (s)")
    ax.set_ylabel("Mean single-view ARI")

    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ls="", ms=4.0, color=c, label=t)
               for c, t in ((C_CLUSTOPT, "ClustOpt (C0–C5)"),
                            (C_AUTOCLUST, "AutoClust (A0–A3)"),
                            (C_ML2DAC, "ML2DAC (M0–M3)"))]
    ax.legend(handles=handles, fontsize=5.4, loc="lower right", frameon=False,
              handletextpad=0.3, borderaxespad=0.3, labelspacing=0.25)

    save(fig, "figC_single_view_pareto")
    record("figC_single_view_pareto",
           ["scientific_review/runtime_xy2d_tradeoff.csv"],
           "stored single-view medians and means on the matching scope")


def figD_arm_lineage():
    fig, ax = plt.subplots(figsize=(W, 3.05))
    ax.set_xlim(0, 100); ax.set_ylim(-1.5, 49.5); ax.axis("off")

    def box(x, y, w, h, t, fc, ec, fs=5.5):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.30,rounding_size=0.9",
            linewidth=0.55, edgecolor=ec, facecolor=fc, zorder=3))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs,
                zorder=4, linespacing=1.2)

    def arr(x0, y0, x1, y1, c="#7A7A7A"):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                     mutation_scale=5.0, linewidth=0.6,
                                     color=c, zorder=2, shrinkA=0, shrinkB=0))

    BL = "#EAF2F9"; GY = "#F4F4F4"; OR = "#FDF0E6"
    rows = [
        (38.0, "Stage 1", [("Head14 / Full60\nuniform", GY),
                           ("MLP Top-5\npredicted", BL),
                           ("subset\noracle", GY)]),
        (25.5, "Stage 2", [("20 policies\nMLP/KNN $\\times$ $k$ $\\times$ weight", BL),
                           ("10 rerankers\none per regime", OR),
                           ("PPv1 / PPv2\npolicy selector", BL)]),
        (13.0, "Split-1 replay", [("Fixed+R", BL), ("PPv1+R / PPv2+R", BL),
                                  ("policy oracle+R", GY)]),
        (0.5, "Stage 4", [("C0 / C1 / C2\nno rerank", BL),
                          ("C3 / C4 / C5\n+ rerank", BL),
                          ("A0\u2013A3, M0\u2013M3\nbaselines", OR)]),
    ]
    for y, tag, items in rows:
        ax.text(-1.0, y + 4.6, tag, fontsize=6.0, fontweight="bold",
                ha="left", va="center", color="#3A3A3A")
        x = 20.0
        for t, fc in items:
            box(x, y, 24.0, 9.2, t, fc, "#8A8A8A")
            x += 26.5
    for y0, y1 in ((38.0, 25.5 + 9.2), (25.5, 13.0 + 9.2), (13.0, 0.5 + 9.2)):
        for cx in (32.0, 58.5, 85.0):
            arr(cx, y0, cx, y1)
    save(fig, "figD_arm_lineage")
    record("figD_arm_lineage", ["method_registry.json", "policy_to_reranker_mapping.json"],
           "schematic of the frozen arm lineage; no numeric content")


# ============================================================ gallery source
def build_gallery_points():
    """Deterministic, non-performance representative selection (brief s10).

    Rule, declared before any plot was inspected: for each family, take the
    dataset whose 250-dimensional frozen label-free meta-feature vector (xy_2d
    view, z-scored within the family) is closest in Euclidean distance to the
    family centroid. Ties break on dataset_id ascending. Nothing about ARI,
    utility, difficulty or any model output enters the rule.
    """
    rule = ("per family, the xy_2d record whose frozen 250-dimensional "
            "label-free meta-feature vector, z-scored within the family, is "
            "closest to the family centroid; ties broken on dataset_id. No "
            "performance quantity enters the rule.")
    head = pd.read_csv(UNI, nrows=1)
    feat = [c for c in head.columns[19:] if not c.startswith("utility__")]
    use = ["dataset_id", "view_mode", "source_dataset_dir",
           "feature_names_original"] + feat
    d = pd.read_csv(UNI, usecols=use)
    d = d[d.view_mode == "xy_2d"].copy()
    d["fam"] = d.source_dataset_dir.str.replace("\\", "/", regex=False) \
        .str.split("/analyzed_data/").str[-1].str.split("/").str[0]

    out, chosen = {}, {}
    for fam, sub in d.groupby("fam"):
        if fam not in FAM_PLAIN:
            continue
        X = sub[feat].to_numpy(dtype=float)
        X = np.where(np.isfinite(X), X, np.nan)
        mu = np.nanmean(X, axis=0); sd = np.nanstd(X, axis=0)
        keep = np.isfinite(mu) & (sd > 0)
        Z = (X[:, keep] - mu[keep]) / sd[keep]
        Z = np.where(np.isfinite(Z), Z, 0.0)
        dist = np.linalg.norm(Z - Z.mean(axis=0), axis=1)
        ordr = np.lexsort((sub.dataset_id.to_numpy(), dist))
        pick = sub.iloc[ordr[0]]
        p = Path(str(pick.source_dataset_dir)) / "data_data.csv"
        t = pd.read_csv(p)
        import ast as _ast
        names = _ast.literal_eval(str(pick.feature_names_original))
        cx, cy = names[0], names[1]
        out[fam] = {"x": t[cx].tolist(), "y": t[cy].tolist(),
                    "label": t["cluster_id"].tolist(),
                    "axes": [cx, cy], "equal_aspect": bool(
                        {cx, cy} == {"x", "y"})}
        chosen[fam] = {"dataset_id": pick.dataset_id, "axes": [cx, cy],
                       "path": str(p).replace("\\", "/").split("/analyzed_data/")[-1],
                       "n_points": int(len(t)),
                       "n_clusters": int(t["cluster_id"].nunique()),
                       "distance_to_centroid": float(dist[ordr[0]])}
        print("  gallery pick:", fam, pick.dataset_id)
    (EVI / "gallery_points.json").write_text(json.dumps(out), encoding="utf-8")
    (MAN / "gallery_selection.json").write_text(
        json.dumps({"selection_rule": rule,
                    "declared_before_inspection": True,
                    "selected": chosen}, indent=2), encoding="utf-8")


def figH_inventory_effect():
    """Appendix F: the Full60-Head14 paired effect, both regimes."""
    co = pd.read_csv(S1 / "offline_fixed32" / "offline_2x3_paired_contrasts.csv")
    cn = pd.read_csv(S1 / "online_fixed50" / "online_2x3_paired_contrasts.csv")
    fig, ax = plt.subplots(figsize=(W * 0.62, 1.85))
    pairs = [("FU-HU", "uniform"), ("FP-HP", "predicted"), ("FO-HO", "oracle")]
    for k, (df, col, lbl, mk) in enumerate(
            [(co, "#7F7F7F", "fixed budget", "o"),
             (cn, C_CLUSTOPT, "online search", "s")]):
        d = df.set_index("hypothesis_id")
        ys, lo, hi, rej = [], [], [], []
        for hid, _ in pairs:
            r = d.loc[hid]
            ys.append(r.mean_delta); lo.append(r.mean_delta_ci_low)
            hi.append(r.mean_delta_ci_high); rej.append(bool(r.holm_reject))
        xp = np.arange(3) + (k - 0.5) * 0.20
        ax.errorbar(xp, ys, yerr=[np.array(ys) - np.array(lo),
                                  np.array(hi) - np.array(ys)],
                    fmt=mk, ms=3.6, lw=0, elinewidth=0.7, capsize=1.8,
                    color=col, label=lbl)
        # anchored on the mean, the star landed inside the interval whenever
        # the effect was negative; anchor it on the upper bound instead
        for x, h, r in zip(xp, hi, rej):
            if r:
                ax.text(x, h + 0.006, "$\\ast$", ha="center", fontsize=7.0,
                        va="bottom", color=col)
    ax.axhline(0, color="#8A8A8A", lw=0.6)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels([p[1] for p in pairs])
    ax.set_ylabel("Full60 $-$ Head14 (ARI)")
    ax.margins(y=0.16)            # the $\\ast$ markers sit above the bars
    ax.legend(fontsize=6.0, loc="lower right", frameon=False)
    save(fig, "figH_inventory_effect")
    record("figH_inventory_effect",
           ["stage1_2x3_aggregation/*/paired_contrasts.csv"],
           "stored paired means, stored bootstrap intervals, stored Holm "
           "decisions")


if __name__ == "__main__":
    if not (EVI / "gallery_points.json").exists() or "--gallery" in sys.argv:
        build_gallery_points()
    fig1_problem_and_architecture()
    fig2_controlled_mechanism()
    fig3_family_delta()
    fig4_external(); figG_paired_delta(); fig3_metric()
    figE_selection_heatmap()
    figH_inventory_effect()
    figA_gallery()
    figB_family_absolute()
    figC_single_view_pareto()
    figD_arm_lineage()
    write_prov()
    print("figure files done")
