"""Criterion baselines on the controlled held-out split (1,055 datasets), replay protocol.

Reports, at full precision and paired against CLUSTOPT's search-objective maximiser (policy
``knn_top10_raw`` before reranking), the criterion arms present in the replay frame: searches
under a single index (silhouette, DBCV, Davies-Bouldin, Calinski-Harabasz), uniform weighting
over the four classical indices and over all 60, and AutoML4Clust. The seeded-sweep criterion
baselines of Tables 1, 16 and 17 are a separate protocol
(``results/controlled/criterion_baselines/seeded_sweep/``).

Analysis only: nothing is re-run or refitted.
Inputs : ``results/controlled/per_dataset/*.parquet`` and released controlled summaries.
Outputs: ``analysis/output/statistics/controlled_criterion_baselines.csv`` and
         ``criterion_baseline_notes.json`` (released copies in
         ``results/controlled/criterion_baselines/canonical_frame/``).

Some keys of ``criterion_baseline_notes.json`` (for example ``table2_cited_value`` and
``answer_for_TBD_*``) keep the names they had when the analysis was written against an earlier
draft of the paper; they are kept so that the output matches the released file. The current
table numbering is in ``docs/paper_results_index.md``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from stats_common import paired_block, row_bootstrap_ci  # noqa: E402

sys.path.insert(0, str(HERE.parent))
from release_paths import LEGACY_ROOT, output_dir  # noqa: E402
REPO = LEGACY_ROOT
CR = REPO / "results_analysis" / "clustering_repository"
CANON = CR / "paper_analysis_v2" / "13_raw_and_reproducibility"
MAE = CR / "metric_analysis_existing_domain"
S1 = CR / "stage1_2x3_aggregation"
OUT = output_dir("statistics")

POLICY = "knn_top10_raw"
ARMS = {
    "silhouette_single": "Search under silhouette alone",
    "dbcv_single": "Search under DBCV alone",
    "calinski_harabasz_single": "Search under Calinski-Harabasz alone",
    "davies_bouldin_single": "Search under Davies-Bouldin alone",
    "uniform_classic_cvi": "Uniform over the 4 classical CVIs",
    "all_metrics_uniform": "Uniform over FULL60",
    "AutoML4Clust_phase_B": "AutoML4Clust (silhouette), authors' code",
}


def main() -> int:
    bv = pd.read_parquet(CANON / "canonical_best_view_frame.parquet",
                         columns=["dataset_id", "split_id", "method_name",
                                  "best_view_ari", "ami"])
    run = pd.read_parquet(CANON / "canonical_run_frame.parquet",
                          columns=["dataset_id", "view_id", "method_name", "ari"])
    run = run[run.view_id == "xy_2d"]

    base = bv[bv.method_name == POLICY].set_index("dataset_id")
    ids = sorted(base.index)
    ref = base.loc[ids, "best_view_ari"].to_numpy()
    ref_xy = (run[run.method_name == POLICY].set_index("dataset_id")
              .loc[ids, "ari"].to_numpy())

    rows = []
    for arm, desc in ARMS.items():
        sub = bv[bv.method_name == arm]
        if sub.empty:
            rows.append({"arm": arm, "description": desc,
                         "status": "NOT PRESENT IN THE CANONICAL FRAME"})
            continue
        s = sub.set_index("dataset_id").loc[ids]
        v = s["best_view_ari"].to_numpy()
        xy = (run[run.method_name == arm].set_index("dataset_id")
              .loc[ids, "ari"].to_numpy())
        ci = row_bootstrap_ci(v)
        blk = paired_block(ref, v)          # ClustOpt J-max minus baseline
        rows.append({
            "arm": arm, "description": desc, "status": "PRESENT",
            "n": len(ids),
            "bestview_mean": float(v.mean()),
            "bestview_ci_lo": ci["lo"], "bestview_ci_hi": ci["hi"],
            "xy_only_mean": float(np.nanmean(xy)),
            "bestview_ami_mean": float(s["ami"].mean()),
            "clustopt_Jmax_minus_arm_mean_delta": blk["mean_delta"],
            "delta_ci_lo": blk["ci_lo"], "delta_ci_hi": blk["ci_hi"],
            "wins": blk["wins"], "ties": blk["ties"], "losses": blk["losses"],
            "wins_eps": blk["wins_eps"], "ties_eps": blk["ties_eps"],
            "losses_eps": blk["losses_eps"],
            "p_value": blk["p_value"], "rank_biserial": blk["rank_biserial"],
        })

    ci = row_bootstrap_ci(ref)
    rows.append({"arm": POLICY,
                 "description": "ClustOpt J-maximiser (deployed policy, no rerank)",
                 "status": "PRESENT", "n": len(ids),
                 "bestview_mean": float(ref.mean()),
                 "bestview_ci_lo": ci["lo"], "bestview_ci_hi": ci["hi"],
                 "xy_only_mean": float(np.nanmean(ref_xy)),
                 "bestview_ami_mean": float(base.loc[ids, "ami"].mean())})

    df = pd.DataFrame(rows).sort_values("bestview_mean", ascending=False,
                                        na_position="last")
    df.to_csv(OUT / "controlled_criterion_baselines.csv", index=False)

    # ------------------------- the 0.339 comparability defect, quantified ----
    fac = pd.read_csv(S1 / "online_fixed50" / "online_2x3_overall_summary.csv")
    fu = float(fac.loc[fac.arm == "FU", "mean_ari"].iloc[0])
    e2e = float(bv.loc[bv.method_name == "all_metrics_uniform",
                       "best_view_ari"].mean())
    # best single index on the development splits, by mean TRUE utility
    usage = pd.read_csv(MAE / "clustopt_metric_usage.csv")
    top = usage.sort_values("mean_true_utility", ascending=False).head(5)

    note = {
        "uniform_full60_comparability": {
            "table2_cited_value": 0.339,
            "stage1_factorial_FU_online_mean": fu,
            "stage1_factorial_arm_id": "stage1_full60_uniform_fixed50",
            "stage1_factorial_note": ("the factorial arms use the MLP top-5 "
                                      "seeding path; the ClustOpt rows of "
                                      "Table 2 (0.558 / 0.616) come from the "
                                      "end-to-end KNN top-10 replay"),
            "end_to_end_uniform_full60_arm_id": "all_metrics_uniform",
            "end_to_end_uniform_full60_mean": e2e,
            "difference": e2e - fu,
            "verdict": ("Table 2 currently mixes protocols. An end-to-end "
                        "uniform-over-FULL60 arm ALREADY EXISTS on Split 1 "
                        "under the same search space and 50-evaluation budget "
                        "and scores %.6f, not %.3f. Using it, the conditioning "
                        "margin at the J-max level is %+.6f, not %+.6f."
                        % (e2e, fu, float(ref.mean()) - e2e,
                           float(ref.mean()) - fu)),
        },
        "best_single_index_on_development": {
            "definition": "highest mean TRUE utility over splits 2-16",
            "source": "metric_analysis_existing_domain/clustopt_metric_usage.csv",
            "ranking_top5": top[["metric_id", "mean_true_utility"]]
            .to_dict(orient="records"),
            "answer_for_TBD_ctrl_best_single_name": str(top.iloc[0]["metric_id"]),
            "caveat": ("the NAME is determinable now; its search ARI is not, "
                       "because no single-index arm for it was ever run"),
        },
    }
    (OUT / "criterion_baseline_notes.json").write_text(
        json.dumps(note, indent=2, default=str), encoding="utf-8")

    pd.set_option("display.width", 300)
    print(df.to_string())
    print()
    print(json.dumps(note, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
