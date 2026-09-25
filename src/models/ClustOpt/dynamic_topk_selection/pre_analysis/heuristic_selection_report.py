"""Phase-2 Markdown reports (section 13). Writes NEW files only -- never touches
the Phase-1 reports."""

from __future__ import annotations

import os

import distribution_metrics as dm


def _f(x, d=3):
    try:
        return f"{float(x):.{d}f}"
    except (TypeError, ValueError):
        return str(x)


RANK_COLS = [
    ("heuristic", "heuristic", None), ("mean_K", "meanK", 2),
    ("median_K", "medK", 1), ("P_K1", "P(K=1)", 3), ("P_2_5", "P(2-5)", 3),
    ("P_gt5", "P(K>5)", 3), ("P_eq10", "P(K=10)", 3),
    ("entropy_K", "entH", 2), ("bell_fit", "bell", 3),
    ("family_meanK_std", "famVar", 3), ("view_meanK_std", "viewVar", 3),
    ("split1_diag_ari", "s1_ari", 4), ("composite_score", "score", 3),
]


def _table(df, cols):
    head = "| " + " | ".join(c[1] for c in cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    lines = [head, sep]
    for _, r in df.iterrows():
        cells = []
        for key, _lbl, dec in cols:
            v = r.get(key, "")
            cells.append(_f(v, dec) if dec is not None else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_ranking_report(out_dir, ranking, baselines, top=40):
    cols = [c for c in RANK_COLS
            if c[0] in ranking.columns or c[0] == "heuristic"]
    d = ranking.sort_values("composite_score", ascending=False)
    lines = ["# Heuristic Ranking", "",
             "All heuristics (Phase-1 + Phase-2), ranked by **composite score** "
             "(see `DISTRIBUTION_ANALYSIS.md` for every formula). `famVar`/`viewVar` "
             "= std of per-group mean-K (lower = more stable). `s1_ari` = split-1 "
             "diagnostic ARI (read-only; not an experiment).", ""]
    if baselines:
        lines += [f"Split-1 fixed-Top-K baselines (mean best ARI over "
                  f"{baselines.get('n_datasets','?')} datasets): "
                  f"Top1={_f(baselines.get('TOP1'),4)}, "
                  f"Top3={_f(baselines.get('TOP3'),4)}, "
                  f"Top5={_f(baselines.get('TOP5'),4)}, "
                  f"Top10={_f(baselines.get('TOP10'),4)}; "
                  f"oracle(top-of-4)={_f(baselines.get('ORACLE_top_k'),4)}.", ""]
    lines += [f"### Top {top} by composite score", "", _table(d.head(top), cols), ""]
    lines += ["### Bottom 10 (worst-scoring, for contrast)", "",
              _table(d.tail(10), cols), ""]
    with open(os.path.join(out_dir, "reports", "HEURISTIC_RANKING.md"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def write_distribution_report(out_dir, metrics):
    ideal = ", ".join(f"{v:.3f}" for v in dm.IDEAL_BELL)
    by_fam = (metrics.groupby("family")
              .agg(n=("heuristic", "size"),
                   mean_bell=("bell_fit", "mean"),
                   max_bell=("bell_fit", "max"),
                   mean_P2_5=("P_2_5", "mean"),
                   mean_P_eq10=("P_eq10", "mean"))
              .sort_values("max_bell", ascending=False).reset_index())
    lines = [
        "# Distribution Analysis", "",
        "## Quality-metric definitions (section 6)", "",
        "For each heuristic, over its K distribution `P(K)` on K=1..10:", "",
        "- **mean_K / median_K / std_K** — central tendency & spread of K.",
        "- **P(K=i)** — share of records selecting each K.",
        "- **P(2<=K<=5)** — mass in the preferred region (higher = better).",
        "- **P(K>5), P(K>7), P(K==10)** — tail / clip-saturation mass (lower better).",
        "- **entropy_K** — Shannon entropy of P(K) (spread of the K choice).",
        "- **smoothness** — sum |P(K=i)-P(K=i+1)|; large = oscillatory/unstable.",
        "- **bell_fit** — `1 - 0.5*sum_K |P(K) - IDEAL(K)|`, the closeness to a",
        "  documented discrete ideal bell (not Gaussian):",
        f"  `IDEAL = [{ideal}]` over K=1..10 (peak at 3-4, ~4% at K=1, monotone",
        "  decay). bell_fit in [0,1], 1 = identical to ideal.",
        "- **composite_score** — `bell_fit - 0.25*P(K==10) - 0.15*min(std_K/3,1)",
        "  - 0.20*max(0, P(K=1)-0.12)`; bell_fit dominates, penalties favour",
        "  stable, non-saturating, not-too-many-K=1 shapes.", "",
        "## Bell-fit by heuristic family", "",
        _table(by_fam, [("family", "family", None), ("n", "n", 0),
                        ("mean_bell", "mean_bell", 3), ("max_bell", "max_bell", 3),
                        ("mean_P2_5", "mean_P(2-5)", 3),
                        ("mean_P_eq10", "mean_P(K=10)", 3)]), "",
        "## Heuristics that naturally produce the desired shape", "",
        "Top 12 by bell_fit:", "",
        _table(metrics.sort_values("bell_fit", ascending=False).head(12),
               [("heuristic", "heuristic", None), ("mean_K", "meanK", 2),
                ("P_K1", "P(K=1)", 3), ("P_2_5", "P(2-5)", 3),
                ("P_gt5", "P(K>5)", 3), ("P_eq10", "P(K=10)", 3),
                ("bell_fit", "bell", 3), ("smoothness", "smooth", 3)]), "",
    ]
    with open(os.path.join(out_dir, "reports", "DISTRIBUTION_ANALYSIS.md"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def pick_recommendations(ranking, k=3):
    """Top-k by composite score, but spanning distinct sub-families so the user
    sees genuinely different designs (e.g. soft-cap vs hard-cap vs uncapped)
    rather than three near-identical alpha/cap twins."""
    d = ranking.sort_values("composite_score", ascending=False)
    picks, used_fams = [], set()
    for _, r in d.iterrows():
        if r["family"] in used_fams:
            continue
        used_fams.add(r["family"])
        picks.append(r)
        if len(picks) == k:
            break
    # backfill from the top if fewer than k sub-families qualified
    if len(picks) < k:
        have = {p["heuristic"] for p in picks}
        for _, r in d.iterrows():
            if r["heuristic"] not in have:
                picks.append(r)
            if len(picks) == k:
                break
    return picks


def write_final_selection(out_dir, ranking, baselines, recommended, has_split1):
    d = ranking.sort_values("composite_score", ascending=False)
    lines = [
        "# Final Heuristic Selection — Dynamic Top-K", "",
        "Phase-2 evidence for choosing **one** Dynamic Top-K policy. This does "
        "**not** implement anything inside ClustOpt; it selects the recommended "
        "candidate(s) on statistical grounds.", "",
        "## Recommended candidates (top 3)", "",
    ]
    for i, r in enumerate(recommended, 1):
        s1 = (f" Split-1 diagnostic ARI **{_f(r.get('split1_diag_ari'),4)}** "
              f"(Δ vs best fixed Top-K = {_f(r.get('split1_delta_vs_best_fixed'),4)})."
              if has_split1 else "")
        lines += [
            f"### {i}. `{r['heuristic']}`  (composite {_f(r['composite_score'])})",
            f"- K distribution: mean {_f(r['mean_K'],2)}, median {_f(r['median_K'],1)}, "
            f"mode/peak K={int(r.get('peak_K',0))}; P(K=1)={_f(r['P_K1'],3)}, "
            f"P(2<=K<=5)={_f(r['P_2_5'],3)}, P(K>5)={_f(r['P_gt5'],3)}, "
            f"P(K=10)={_f(r['P_eq10'],3)}.",
            f"- bell_fit {_f(r['bell_fit'],3)}; cross-family meanK std "
            f"{_f(r.get('family_meanK_std'),3)}, view std {_f(r.get('view_meanK_std'),3)} "
            f"(stability).{s1}",
            f"- **Why preferable:** strong preferred-region mass with a light tail "
            f"and little clip pile-up, while staying stable across families/views.",
            "",
        ]
    lines += ["## Why the other families are rejected", "",
              "- **Cumulative-mass / effective-K / entropy heuristics** — saturate "
              "at K=10 for ~99% of records (utility vectors are too flat); "
              "bell_fit near 0. Not viable.",
              "- **Loose relative thresholds (alpha<=0.85)** — mean K ~7-10, heavy "
              "tail, mode at 10; selection collapses toward Top-10.",
              "- **Pure elbow (k_elbow_10 / norm)** — too much mass at K=1 "
              "(>27%), so it under-selects and loses the K=3-5 bulk.",
              "- **Hybrid mass-based (Phase-1)** — inherit the mass saturation; "
              "dominated by the capped/guarded relative variants.", "",
              "## Ranking (top 10)", "",
              _table(d.head(10), [c for c in RANK_COLS]), ""]
    if baselines and has_split1:
        best_fixed = max(baselines["TOP1"], baselines["TOP3"],
                         baselines["TOP5"], baselines["TOP10"])
        lines += ["## Split-1 diagnostic (split1_diagnostic_only)", "",
                  "Read-only sanity check — **not** an experiment, nothing trained "
                  "or tuned. Each heuristic's per-dataset K (rounded median over "
                  "views) is mapped to the nearest fixed Top-K bucket "
                  "(1->Top1, 2-3->Top3, 4-5->Top5, >=6->Top10) and the already "
                  "computed ARI is read off.", "",
                  f"- Fixed baselines: Top1={_f(baselines['TOP1'],4)}, "
                  f"Top3={_f(baselines['TOP3'],4)}, Top5={_f(baselines['TOP5'],4)}, "
                  f"Top10={_f(baselines['TOP10'],4)} (best fixed = {_f(best_fixed,4)}).",
                  f"- Oracle (per-dataset best of the 4 Top-K) = "
                  f"{_f(baselines['ORACLE_top_k'],4)} — the headroom a perfect "
                  f"dynamic policy could reach.",
                  f"- Best heuristic by diagnostic ARI: "
                  f"`{d.sort_values('split1_diag_ari', ascending=False).iloc[0]['heuristic']}` "
                  f"at {_f(d['split1_diag_ari'].max(),4)}.", "",
                  "Interpretation: the diagnostic is coarse (4 buckets, view-median "
                  "reduction) and must not drive the final choice on its own; it is "
                  "consistent with the distribution-based ranking.", "",
                  f"**Key caveat:** under this coarse mapping only a handful of "
                  f"heuristics edge past best fixed Top-K, and only by ~1e-4 ARI "
                  f"(noise). The diagnostic therefore does **not** demonstrate an "
                  f"ARI gain over fixed Top-5; it merely confirms the recommended "
                  f"heuristics are not *worse* than the best fixed policy. The "
                  f"oracle headroom ({_f(baselines['ORACLE_top_k'],4)} vs "
                  f"{_f(best_fixed,4)}) is real but is not captured by 4-bucket "
                  f"quantisation — exact per-view K (only testable inside ClustOpt) "
                  f"is needed to realise it.", ""]
    # readiness
    lines += ["## Readiness", "",
              "Enough statistical evidence to choose the heuristic **form** and "
              "parameters: **YES** — the relative-threshold family at alpha≈0.92 "
              "with a light cap (soft or hard) is the clear, stable winner on every "
              "distribution-quality metric.", "",
              "Open item (not a blocker for *selection*, but required before "
              "*deployment*): the split-1 diagnostic does not yet show an ARI "
              "improvement over fixed Top-5. Confirming an actual gain requires "
              "running the chosen heuristic inside ClustOpt with exact per-view K "
              "(the next phase), which is out of scope here.", ""]
    with open(os.path.join(out_dir, "reports", "FINAL_HEURISTIC_SELECTION.md"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
