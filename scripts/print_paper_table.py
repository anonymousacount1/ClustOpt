"""Print the numbers of a paper table directly from the released result files.

    python scripts/print_paper_table.py table1
    python scripts/print_paper_table.py table16
    python scripts/print_paper_table.py table17
    python scripts/print_paper_table.py all [--csv OUTDIR]

Values are formatted exactly as printed in the paper (rounding half up).
Nothing is recomputed from raw data; every value is read from ``results/``.
See ``docs/paper_results_index.md`` for the file behind every table and figure.
"""
from __future__ import annotations

import argparse
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SWEEP = ROOT / "results/controlled/criterion_baselines/seeded_sweep"
FRAME = ROOT / "results/controlled/criterion_baselines/canonical_frame"
RERANK = ROOT / "results/controlled/reranking"

# seeded-sweep arm id -> paper row label (Tables 1, 16, 17)
SWEEP_ROWS = [
    ("e1_random", "Random search (no criterion)"),
    ("stage1_full60_uniform_fixed50", "Uniform over FULL60"),
    ("e1_best1", "Best single index (development, per view)"),
    ("e1_core3", "Static three-index core"),
    ("e1_meanprof", "Dataset-independent mean profile, top-10"),
    ("e1_c4_seed", "Predicted profile, top-10 (CLUSTOPT)"),
]
REFERENCE_ROWS = [  # Table 16 conditioned reference arms (same seeded protocol)
    ("stage1_full60_mlp_top5_raw_fixed50", "Neural predicted profile, top-5 (conditioned reference)"),
    ("stage1_full60_oracle_top5_raw_fixed50", "True (oracle) profile, top-5 (ceiling reference)"),
]
CLUSTOPT = "e1_c4_seed"


def r(x, nd, sign=False):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "-"
    q = Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP)
    s = format(q, "f")
    if sign and q >= 0:
        s = "+" + s
    return s


def p_fmt(p):
    if p >= 0.01:
        return r(p, 3)
    m, e = ("%.1e" % p).split("e")
    return "%se%d" % (m, int(e))


def _sweep():
    a = pd.read_csv(SWEEP / "arm_summary.csv").set_index("arm")
    c = pd.read_csv(SWEEP / "paired_contrasts.csv")
    return a, c


def _contrast(c, arm, unit):
    row = c[(c.baseline_arm == arm) & (c.unit == unit)]
    return None if row.empty else row.iloc[0]


def table1():
    """Search criteria, controlled held-out split (1,055 datasets)."""
    a, _ = _sweep()
    rows = []
    for arm, label in SWEEP_ROWS:
        s = a.loc[arm]
        rows.append({"row": label, "protocol": "seeded sweep", "endpoint": r(s.mean_endpoint_bestview, 3),
                     "ci": "[%s, %s]" % (r(s.ci_lo, 3), r(s.ci_hi, 3)),
                     "single": r(s.mean_endpoint_xy, 3), "slate_oracle": r(s.mean_slate_oracle, 3)})
    h = pd.read_csv(FRAME / "controlled_headline_stats.csv")
    lvl = h[(h.contrast == "LEVEL MEAN") & (h.unit == "bestview_ari")].set_index("arm_a")
    pp = pd.read_csv(RERANK / "per_policy_results.csv").set_index("policy_id")
    rr = lvl.loc["fixed_knn_top10_raw_reranked"]
    rows.append({"row": "CLUSTOPT, reranked", "protocol": "replay", "endpoint": r(rr.mean_a, 3),
                 "ci": "[%s, %s]" % (r(rr.ci_lo, 3), r(rr.ci_hi, 3)), "single": "-",
                 "slate_oracle": r(pp.loc["knn_top10_raw", "visited_oracle_mean_bestview"], 3)})
    for arm, label in (("AutoClust_Extended_InDomain_same_search_space", "AutoClust, extended inventory"),
                       ("ML2DAC_OriginalPlusEstablished_InDomain_same_search_space", "ML2DAC, corrected, +established")):
        rows.append({"row": label, "protocol": "replay", "endpoint": r(lvl.loc[arm, "mean_a"], 3),
                     "ci": "-", "single": "-", "slate_oracle": "-"})
    return pd.DataFrame(rows)


def table16():
    """Criterion baselines, seeded sweep: CLUSTOPT minus arm on best-view ARI."""
    a, c = _sweep()
    rows = []
    for arm, label in SWEEP_ROWS + REFERENCE_ROWS:
        s = a.loc[arm]
        row = {"row": label, "best_view": r(s.mean_endpoint_bestview, 4), "single": r(s.mean_endpoint_xy, 4),
               "oracle": r(s.mean_slate_oracle, 4), "delta": "-", "ci": "-", "wtl": "-", "p": "-"}
        if arm != CLUSTOPT:
            k = _contrast(c, arm, "bestview_endpoint")
            row.update({"delta": r(k.mean_delta, 4, sign=True),
                        "ci": "[%s, %s]" % (r(k.ci_lo, 3, sign=True), r(k.ci_hi, 3, sign=True)),
                        "wtl": "%d/%d/%d" % (k.wins_eps, k.ties_eps, k.losses_eps), "p": p_fmt(k.p_value)})
        rows.append(row)
    return pd.DataFrame(rows)


def table17():
    """Slate ceilings vs returned endpoints (steering = slate-oracle difference; selection = remainder)."""
    a, c = _sweep()
    rows = []
    for arm, label in SWEEP_ROWS:
        s = a.loc[arm]
        row = {"row": label, "slate_oracle": r(s.mean_slate_oracle, 4), "endpoint": r(s.mean_endpoint_bestview, 4),
               "gap_to_oracle": r(s.mean_endpoint_to_oracle_gap, 4), "steering": "-", "selection": "-"}
        if arm != CLUSTOPT:
            e = _contrast(c, arm, "bestview_endpoint")
            o = _contrast(c, arm, "slate_oracle")
            row.update({"steering": r(o.mean_delta, 4, sign=True),
                        "selection": r(e.mean_delta - o.mean_delta, 4, sign=True)})
        rows.append(row)
    return pd.DataFrame(rows)


def table2():
    """Primary external corpus (50 datasets): single view, best view by source, median three-view runtime."""
    m = pd.read_csv(ROOT / "results/external/summary/method_master_table.csv").set_index("short")
    rt = pd.read_csv(ROOT / "results/external/runtime/runtime_three_view_bestview_cost.csv").set_index("short")
    rows = []
    for code, label in (("C4", "CLUSTOPT"), ("C1", "without reranking"), ("C5", "with policy selector"),
                        ("A0", "AutoClust, original"), ("A3", "AutoClust, extended"), ("M1", "ML2DAC, +established")):
        s = m.loc[code]
        rows.append({"code": code, "system": label, "single": r(s.xy_2d_ari_mean, 3), "all": r(s.bestview_ari_mean, 3),
                     "fcps": r(s.bestview_mean_fcps, 3), "sklearn": r(s.bestview_mean_sklearn_shapes, 3),
                     "real": r(s.bestview_mean_real_pca, 3),
                     "runtime_s": r(rt.loc[code, "three_view_median"], 0)})
    return pd.DataFrame(rows)


def table15():
    """Four-level progression on the replay (rows 1-2: exact reranking isolation; rows 3-5: selectors, ceiling)."""
    h = pd.read_csv(FRAME / "controlled_headline_stats.csv")
    lvl = h[(h.contrast == "LEVEL MEAN") & (h.unit == "bestview_ari")].set_index("arm_a")
    four = pd.read_csv(ROOT / "results/controlled/policy_selection/four_level_online_table.csv").set_index("level")
    rows = []
    for arm, label in (("fixed_knn_top10_raw_unreranked", "Fixed policy (knn_top10_raw), no rerank"),
                       ("fixed_knn_top10_raw_reranked", "Fixed policy + rerank (= CLUSTOPT)")):
        s = lvl.loc[arm]
        rows.append({"row": label, "mean": r(s.mean_a, 4), "ci": "[%s, %s]" % (r(s.ci_lo, 3), r(s.ci_hi, 3)),
                     "source": "canonical_frame/controlled_headline_stats.csv"})
    for lev, label in (("B", "Policy selector v1 + rerank"), ("C", "Policy selector v2 + rerank"),
                       ("D", "Policy oracle + rerank (ceiling)")):
        s = four.loc[lev]
        rows.append({"row": label, "mean": r(s.mean_bestview, 4), "ci": "[%s, %s]" % (r(s.ci_lo, 3), r(s.ci_hi, 3)),
                     "source": "policy_selection/four_level_online_table.csv"})
    return pd.DataFrame(rows)


def section5_3():
    """Section 5.3: reranked CLUSTOPT against matched AutoClust (extended inventory), replay."""
    h = pd.read_csv(FRAME / "controlled_headline_stats.csv")
    lvl = h[(h.contrast == "LEVEL MEAN") & (h.unit == "bestview_ari")].set_index("arm_a")
    a = r(lvl.loc["fixed_knn_top10_raw_reranked", "mean_a"], 3)
    b = r(lvl.loc["AutoClust_Extended_InDomain_same_search_space", "mean_a"], 3)
    return pd.DataFrame([{"row": "CLUSTOPT (reranked) vs AutoClust extended (matched)", "text": "%s against %s" % (a, b)}])


TABLES = {"table1": table1, "table2": table2, "table15": table15, "table16": table16, "table17": table17,
          "section5_3": section5_3}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("table", choices=sorted(TABLES) + ["all"])
    ap.add_argument("--csv", default=None, help="also write <table>.csv into this folder")
    a = ap.parse_args(argv)
    names = sorted(TABLES) if a.table == "all" else [a.table]
    for n in names:
        df = TABLES[n]()
        print("\n== %s: %s" % (n, TABLES[n].__doc__.strip().splitlines()[0]))
        print(df.to_string(index=False))
        if a.csv:
            Path(a.csv).mkdir(parents=True, exist_ok=True)
            df.to_csv(Path(a.csv) / ("%s.csv" % n), index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
