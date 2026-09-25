"""Stage 3C figures. Rendering only -- reads the frozen Stage-3C tables."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metric_atlas import common as C  # noqa: E402

SUB_REL = "family_subfamily_specialization"
GROUP_ORDER = ("Original", "Established", "New46-Pattern", "New46-Image")
GROUP_COLOR = {"Original": "#1f4e79", "Established": "#2e8b57",
               "New46-Pattern": "#c1121f", "New46-Image": "#e07a00"}
SHORT = {"arcs_circles_rings": "arcs", "deinterleaving_pri_toa_amp_like": "deint",
         "hollow_topological_components": "hollow", "hybrid_mixed_geometry": "hybrid",
         "ladder_grid_lattice": "ladder", "linear_bands_parallel_stripes": "bands",
         "parabolic_curved_chains": "parab", "piecewise_polylines_corners": "piece",
         "polygons_rectangles_frames": "polyg", "radial_spiral_meander": "radial",
         "standard_clustering_blobs": "blobs", "texture_density_fields": "texture"}
plt.rcParams.update({"font.size": 7, "figure.dpi": 160,
                     "savefig.bbox": "tight", "axes.grid": False})


def _sub(A: pd.DataFrame, m: str) -> str:
    r = A[A.metric_id == m]
    if not len(r):
        return "Other"
    g, s = r["group"].iloc[0], r["subgroup"].iloc[0]
    return s if isinstance(s, str) and s.startswith("New46") else g


def _ordered(A: pd.DataFrame, metrics) -> list:
    tag = {m: _sub(A, m) for m in metrics}
    return sorted(metrics, key=lambda m: (GROUP_ORDER.index(tag[m])
                                          if tag[m] in GROUP_ORDER else 9, m))


def _band(ax, order, tag, x=-0.6):
    start = 0
    for i in range(1, len(order) + 1):
        if i == len(order) or tag[order[i]] != tag[order[start]]:
            ax.add_patch(plt.Rectangle((x, start - 0.5), 0.45, i - start,
                                       color=GROUP_COLOR.get(tag[order[start]], "#888"),
                                       clip_on=False))
            ax.text(x - 0.25, (start + i) / 2 - 0.5, tag[order[start]].replace("New46-", ""),
                    rotation=90, va="center", ha="center", fontsize=6)
            start = i


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=None)
    args = ap.parse_args(argv)
    repo = Path(args.repo_root).resolve() if args.repo_root else C.REPO
    out = repo / C.OUT_REL / SUB_REL
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    FAM = pd.read_csv(C.ext(out / "family_metric_full_matrix.csv.gz"))
    A = pd.read_csv(C.ext(repo / C.OUT_REL / "canonical_metric_atlas.csv"))
    B = pd.read_csv(C.ext(out / "family_core_breadth.csv"))
    fa = pd.read_csv(C.ext(out / "family_specialists_tierA.csv"))
    SIG = pd.read_csv(C.ext(out / "family_signature.csv"))
    sa = pd.read_csv(C.ext(out / "subfamily_specialists_tierA.csv"))
    SUB = pd.read_csv(C.ext(out / "subfamily_metric_full_matrix.csv.gz"))

    fams = sorted(FAM["group"].unique())
    metrics = _ordered(A, sorted(FAM["metric_id"].unique()))
    tag = {m: _sub(A, m) for m in metrics}
    xlab = [SHORT.get(f, f[:7]) for f in fams]

    def pivot(col):
        return FAM.pivot(index="metric_id", columns="group",
                         values=col).reindex(metrics)[fams]

    # ---- FIG 1: family true-utility rank heatmap ----------------------------
    R = pivot("utility_rank_group")
    fig, ax = plt.subplots(figsize=(5.2, 9.5))
    im = ax.imshow(R.to_numpy(float), aspect="auto", cmap="viridis_r", vmin=1, vmax=60)
    ax.set_xticks(range(len(fams))); ax.set_xticklabels(xlab, rotation=90)
    ax.set_yticks(range(len(metrics))); ax.set_yticklabels(metrics, fontsize=5)
    _band(ax, metrics, tag)
    ax.set_title("Fig 1 — family true-utility rank (1 = best of 60)")
    fig.colorbar(im, ax=ax, shrink=0.4, label="rank within family")
    fig.savefig(fig_dir / "fig1_family_utility_rank_heatmap.png"); plt.close(fig)

    # ---- FIG 2: selection-enrichment heatmap -------------------------------
    E = pivot("enrichment").replace([np.inf, -np.inf], np.nan)
    fig, ax = plt.subplots(figsize=(5.2, 9.5))
    im = ax.imshow(np.log2(E.to_numpy(float)), aspect="auto", cmap="RdBu_r",
                   norm=TwoSlopeNorm(vcenter=0, vmin=-2, vmax=2))
    ax.set_xticks(range(len(fams))); ax.set_xticklabels(xlab, rotation=90)
    ax.set_yticks(range(len(metrics))); ax.set_yticklabels(metrics, fontsize=5)
    _band(ax, metrics, tag)
    ax.set_title("Fig 2 — ClustOpt selection enrichment (log2, family vs outside)")
    fig.colorbar(im, ax=ax, shrink=0.4, label="log2 enrichment")
    fig.savefig(fig_dir / "fig2_family_selection_enrichment.png"); plt.close(fig)

    # ---- FIG 3: New46 DeltaU heatmap with Tier-A marks ---------------------
    n46 = [m for m in metrics if tag[m].startswith("New46")]
    D = FAM.pivot(index="metric_id", columns="group",
                  values="DeltaU").reindex(n46)[fams]
    fig, ax = plt.subplots(figsize=(5.2, 8.0))
    v = np.nanmax(np.abs(D.to_numpy(float)))
    im = ax.imshow(D.to_numpy(float), aspect="auto", cmap="RdBu_r",
                   norm=TwoSlopeNorm(vcenter=0, vmin=-v, vmax=v))
    for _, r in fa.iterrows():
        if r["metric_id"] in n46 and r["group"] in fams:
            ax.plot(fams.index(r["group"]), n46.index(r["metric_id"]), "k*",
                    ms=6, mew=0.4)
    ax.set_xticks(range(len(fams))); ax.set_xticklabels(xlab, rotation=90)
    ax.set_yticks(range(len(n46))); ax.set_yticklabels(n46, fontsize=5)
    _band(ax, n46, tag)
    ax.set_title("Fig 3 — New46 utility uplift ΔU (family − outside)\n★ = Tier-A specialist")
    fig.colorbar(im, ax=ax, shrink=0.4, label="ΔU")
    fig.savefig(fig_dir / "fig3_new46_specialization_heatmap.png"); plt.close(fig)

    # ---- FIG 4: core vs specialist map -------------------------------------
    B = B.copy()
    B["tag"] = [_sub(A, m) for m in B["metric_id"]]
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    for g in GROUP_ORDER:
        s = B[B.tag == g]
        ax.scatter(s["B_U_top5"], s["selection_specificity"],
                   s=8 + 900 * s["global_selection_rate"], alpha=0.75,
                   c=GROUP_COLOR[g], label=g, edgecolors="k", linewidths=0.3)
    for _, r in B[B.is_core_candidate].iterrows():
        ax.annotate(r["metric_id"], (r["B_U_top5"], r["selection_specificity"]),
                    fontsize=5.5, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("utility breadth: families where Top-5 (of 12)")
    ax.set_ylabel("selection specificity (1 − normalised entropy)")
    ax.set_title("Fig 4 — core vs specialist map (size = global selection rate)")
    ax.legend(fontsize=6, frameon=False)
    fig.savefig(fig_dir / "fig4_core_vs_specialist_map.png"); plt.close(fig)

    # ---- FIG 5: utility vs selection alignment, faceted --------------------
    fig, axes = plt.subplots(3, 4, figsize=(9, 6.4), sharex=True, sharey=True)
    for axi, f in zip(axes.ravel(), fams):
        s = FAM[FAM.group == f]
        for g in GROUP_ORDER:
            ss = s[s["metric_id"].isin([m for m in metrics if tag[m] == g])]
            axi.scatter(ss["U_group"], ss["S_group"], s=5, c=GROUP_COLOR[g],
                        alpha=0.8, edgecolors="none")
        rho = SIG.loc[SIG.family == f, "spearman_util_sel_all"]
        axi.set_title("%s  ρ=%.2f" % (SHORT.get(f, f[:7]),
                                      float(rho.iloc[0]) if len(rho) else np.nan),
                      fontsize=6.5)
    fig.supxlabel("true utility in family"); fig.supylabel("selection rate in family")
    fig.suptitle("Fig 5 — utility vs selection alignment, by family", fontsize=9)
    fig.savefig(fig_dir / "fig5_utility_selection_alignment.png"); plt.close(fig)

    # ---- FIG 6: family signature matrix ------------------------------------
    core = list(B[B.is_core_candidate]["metric_id"])
    spec = sorted(fa["metric_id"].unique())
    missed = pd.read_csv(C.ext(out / "exploited_vs_missed_specialists.csv"))
    miss = missed[missed.classification == "B_missed_specialist"]
    rowset = core + [m for m in spec if m not in core] + \
        sorted(set(miss["metric_id"]) - set(core) - set(spec))
    Mx = np.zeros((len(rowset), len(fams)))
    for i, m in enumerate(rowset):
        for j, f in enumerate(fams):
            if m in core and len(FAM[(FAM.metric_id == m) & (FAM.group == f)
                                     & (FAM.utility_rank_group <= 10)]):
                Mx[i, j] = 1
            if len(fa[(fa.metric_id == m) & (fa.group == f)]):
                Mx[i, j] = 2
            if len(miss[(miss.metric_id == m) & (miss.group == f)]):
                Mx[i, j] = 3
    fig, ax = plt.subplots(figsize=(5.4, 0.22 * len(rowset) + 1.6))
    ax.imshow(Mx, aspect="auto", cmap=matplotlib.colors.ListedColormap(
        ["#f2f2f2", "#1f4e79", "#c1121f", "#e0a800"]), vmin=0, vmax=3)
    ax.set_xticks(range(len(fams))); ax.set_xticklabels(xlab, rotation=90)
    ax.set_yticks(range(len(rowset))); ax.set_yticklabels(rowset, fontsize=5.5)
    ax.set_title("Fig 6 — family signature\nblue = core present, red = Tier-A specialist, "
                 "amber = missed specialist", fontsize=7)
    fig.savefig(fig_dir / "fig6_family_signature.png"); plt.close(fig)

    # ---- FIG 7: specialization vs downstream gain --------------------------
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.1))
    for axi, (col, lab) in zip(axes, [
            ("new46_selection_share", "New46 selection share"),
            ("n_tier_a", "# Tier-A specialists"),
            ("max_new46_utility_uplift", "max New46 ΔU")]):
        axi.scatter(SIG[col], SIG["full60_minus_head14"], s=18, c="#1f4e79")
        for _, r in SIG.iterrows():
            axi.annotate(SHORT.get(r["family"], ""), (r[col], r["full60_minus_head14"]),
                         fontsize=5, xytext=(2, 2), textcoords="offset points")
        rho = SIG[col].corr(SIG["full60_minus_head14"], method="spearman")
        axi.axhline(0, color="grey", lw=0.5)
        axi.set_xlabel(lab); axi.set_title("ρ = %.3f" % rho, fontsize=7)
    axes[0].set_ylabel("Full60 − Head14 ARI (Split 1)")
    fig.suptitle("Fig 7 — specialisation vs downstream gain — ASSOCIATIVE, n=12 families",
                 fontsize=9)
    fig.savefig(fig_dir / "fig7_specialization_vs_downstream.png"); plt.close(fig)

    # ---- FIG 8: subfamily specialisation (Tier-A metrics only) -------------
    sm = sorted(sa["metric_id"].unique())
    subs = (SUB[["group", "parent_family"]].drop_duplicates()
            .sort_values(["parent_family", "group"]))
    order_sub = list(subs["group"])
    P = SUB[SUB.metric_id.isin(sm)].pivot(index="metric_id", columns="group",
                                          values="DeltaU").reindex(sm)[order_sub]
    fig, ax = plt.subplots(figsize=(11, 0.3 * len(sm) + 2))
    v = np.nanmax(np.abs(P.to_numpy(float)))
    im = ax.imshow(P.to_numpy(float), aspect="auto", cmap="RdBu_r",
                   norm=TwoSlopeNorm(vcenter=0, vmin=-v, vmax=v))
    for _, r in sa.iterrows():
        if r["metric_id"] in sm and r["group"] in order_sub:
            ax.plot(order_sub.index(r["group"]), sm.index(r["metric_id"]), "k*",
                    ms=5, mew=0.3)
    bounds, prev = [], None
    for i, s in enumerate(order_sub):
        f = subs[subs.group == s]["parent_family"].iloc[0]
        if f != prev:
            bounds.append((i, f)); prev = f
    for i, _ in bounds[1:]:
        ax.axvline(i - 0.5, color="k", lw=0.6)
    ax.set_xticks([i for i, _ in bounds])
    ax.set_xticklabels([SHORT.get(f, f[:6]) for _, f in bounds], rotation=90, fontsize=6)
    ax.set_yticks(range(len(sm))); ax.set_yticklabels(sm, fontsize=6)
    ax.set_title("Fig 8 — subfamily ΔU (vs sibling subfamilies) for Tier-A specialist "
                 "metrics; ★ = Tier-A subfamily specialist", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.5, label="ΔU vs siblings")
    fig.savefig(fig_dir / "fig8_subfamily_specialization.png"); plt.close(fig)

    print("[3c-fig] wrote 8 figures -> %s" % fig_dir, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
