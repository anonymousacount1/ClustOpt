"""Controlled headline statistics for CLUSTOPT (C4) on the held-out split, replay protocol.

Recomputes from released per-dataset records the paired comparisons behind Table 1 (last three
rows), Section 5.3, Table 15 (rows 1-2), Table 19 and Fig. 8: reranked CLUSTOPT (fixed policy
``knn_top10_raw`` with the ``KNN_TOP10`` reranker) against the same policy without reranking,
matched AutoClust (extended inventory) and ML2DAC. Level means with bootstrap intervals,
Wilcoxon tests, rank-biserial effects, wins/ties/losses, and per-family differences.

The AutoClust comparator is ``AutoClust_Extended_InDomain_same_search_space`` (matched search
domain). The output also contains an auxiliary row for the unconstrained arm
``AutoClust_Extended_InDomain`` ("NATIVE domain"); it is not a comparator in the paper.

Analysis only: no clustering, search or model fitting.
Inputs : ``results/controlled/per_dataset/*.parquet``,
         ``results/controlled/reranking/per_dataset_policy_results.csv.gz``.
Outputs: ``analysis/output/statistics/controlled_headline_stats.csv``,
         ``controlled_family_stats.csv``, ``controlled_provenance.json`` (released copies in
         ``results/controlled/criterion_baselines/canonical_frame/``).
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
RR = CR / "candidate_reranker_split1_online_final"
OUT = output_dir("statistics")

POLICY = "knn_top10_raw"
AC_MATCHED = "AutoClust_Extended_InDomain_same_search_space"
AC_NATIVE = "AutoClust_Extended_InDomain"
ML_MATCHED = "ML2DAC_Extended_InDomain_same_search_space"
ML1_MATCHED = "ML2DAC_OriginalPlusEstablished_InDomain_same_search_space"

PROV: dict = {}
BLOCKERS: list = []


def rel(p: Path) -> str:
    return str(p.relative_to(REPO)).replace("\\", "/")


def main() -> int:
    # ------------------------------------------------------------------ load
    bvcols = ["dataset_id", "split_id", "family", "family_id", "method_name",
              "best_view_ari", "best_view_id", "ami"]
    bv = pd.read_parquet(CANON / "canonical_best_view_frame.parquet", columns=bvcols)
    assert set(bv["split_id"].unique()) == {1}, "canonical frame is not Split-1 only"

    fixed_un = bv[bv.method_name == POLICY].set_index("dataset_id")
    ac_m = bv[bv.method_name == AC_MATCHED].set_index("dataset_id")
    ac_n = bv[bv.method_name == AC_NATIVE].set_index("dataset_id")
    ml_m = bv[bv.method_name == ML_MATCHED].set_index("dataset_id")
    ml1_m = bv[bv.method_name == ML1_MATCHED].set_index("dataset_id")

    pol = pd.read_csv(RR / "per_dataset_policy_results.csv.gz")
    pol = pol[pol.policy_id == POLICY].set_index("dataset_id")

    # ------------------------------------------------ SAFETY: dataset alignment
    ids = sorted(set(fixed_un.index))
    for name, fr in (("AC matched", ac_m), ("AC native", ac_n),
                     ("ML extended matched", ml_m), ("reranker policy table", pol)):
        missing = set(ids) - set(fr.index)
        extra = set(fr.index) - set(ids)
        if missing or extra:
            raise SystemExit("STOP: dataset id misalignment vs %s: "
                             "%d missing, %d extra" % (name, len(missing), len(extra)))
    # the reranker table's `original` column must be the SAME quantity as the
    # canonical frame's unreranked Best-View ARI -- if it is not, the two
    # artifacts disagree and the run must stop rather than pick one.
    resid = np.abs(pol.loc[ids, "original"].to_numpy()
                   - fixed_un.loc[ids, "best_view_ari"].to_numpy())
    PROV["alignment_check"] = {
        "n_datasets": len(ids),
        "max_abs_residual_reranker_original_vs_canonical_unreranked": float(resid.max()),
        "n_disagreeing_gt_1e-9": int((resid > 1e-9).sum()),
        "verdict": ("IDENTICAL" if resid.max() <= 1e-9
                    else "DISAGREE -- STOP CONDITION"),
    }
    if resid.max() > 1e-9:
        raise SystemExit("STOP: the reranker package's unreranked column and the "
                         "canonical Split-1 frame disagree (max |d| = %.3e). "
                         "Two candidate artifacts disagree about a headline "
                         "number; not silently choosing one." % resid.max())

    fx_un = fixed_un.loc[ids, "best_view_ari"].to_numpy()
    fx_rr = pol.loc[ids, "reranked"].to_numpy()
    fx_or = pol.loc[ids, "oracle"].to_numpy()
    a_ext = ac_m.loc[ids, "best_view_ari"].to_numpy()
    a_ext_nat = ac_n.loc[ids, "best_view_ari"].to_numpy()
    m_ext = ml_m.loc[ids, "best_view_ari"].to_numpy()
    m_est = ml1_m.loc[ids, "best_view_ari"].to_numpy()

    # --------------------------------------------------------------- B1 headline
    rows = []

    def add(label, a, b, a_name, b_name, unit="bestview_ari", note=""):
        blk = paired_block(a, b)
        blk.update({"contrast": label, "arm_a": a_name, "arm_b": b_name,
                    "unit": unit, "population": "controlled Split-1 (%d datasets)"
                    % len(ids), "note": note})
        rows.append(blk)
        return blk

    add("Fixed knn_top10_raw +rerank  vs  AutoClust Extended (MATCHED domain)",
        fx_rr, a_ext, "fixed_knn_top10_raw_reranked", AC_MATCHED,
        note="PRIMARY: this is the C4-equivalent controlled contrast")
    add("Fixed knn_top10_raw +rerank  vs  AutoClust Extended (NATIVE domain)",
        fx_rr, a_ext_nat, "fixed_knn_top10_raw_reranked", AC_NATIVE,
        note="SECONDARY: AutoClust searched its own, unconstrained space")
    add("Fixed knn_top10_raw +rerank  vs  Fixed knn_top10_raw (no rerank)",
        fx_rr, fx_un, "fixed_knn_top10_raw_reranked",
        "fixed_knn_top10_raw_unreranked",
        note="exact reranker ablation: same policy, same slate")
    add("Fixed knn_top10_raw (no rerank)  vs  AutoClust Extended (MATCHED)",
        fx_un, a_ext, "fixed_knn_top10_raw_unreranked", AC_MATCHED)
    add("Fixed knn_top10_raw +rerank  vs  ML2DAC Extended (MATCHED)",
        fx_rr, m_ext, "fixed_knn_top10_raw_reranked", ML_MATCHED)
    add("Fixed knn_top10_raw +rerank  vs  ML2DAC +Established (MATCHED)",
        fx_rr, m_est, "fixed_knn_top10_raw_reranked", ML1_MATCHED,
        note="the controlled counterpart of external arm M1")

    # level means + slate oracle
    for nm, v in (("fixed_knn_top10_raw_unreranked", fx_un),
                  ("fixed_knn_top10_raw_reranked", fx_rr),
                  ("fixed_knn_top10_raw_slate_oracle", fx_or),
                  (AC_MATCHED, a_ext), (AC_NATIVE, a_ext_nat),
                  (ML_MATCHED, m_ext), (ML1_MATCHED, m_est)):
        ci = row_bootstrap_ci(v)
        rows.append({"contrast": "LEVEL MEAN", "arm_a": nm, "arm_b": "",
                     "unit": "bestview_ari",
                     "population": "controlled Split-1 (%d datasets)" % len(ids),
                     "n": len(v), "mean_a": float(np.mean(v)),
                     "mean_b": np.nan, "mean_delta": np.nan,
                     "ci_lo": ci["lo"], "ci_hi": ci["hi"],
                     "bootstrap_unit": "dataset_row", "note": ""})

    # --------------------------------------------------------- B3 single-view xy
    run = pd.read_parquet(CANON / "canonical_run_frame.parquet",
                          columns=["dataset_id", "view_id", "method_name", "ari",
                                   "ami", "status"])
    run = run[run.view_id == "xy_2d"]
    xy_ac = (run[run.method_name == AC_MATCHED]
             .set_index("dataset_id").loc[ids, "ari"].to_numpy())
    xy_ac_ami = (run[run.method_name == AC_MATCHED]
                 .set_index("dataset_id").loc[ids, "ami"].to_numpy())
    xy_un_canon = (run[run.method_name == POLICY]
                   .set_index("dataset_id").loc[ids, "ari"].to_numpy())
    xy_un_ami = (run[run.method_name == POLICY]
                 .set_index("dataset_id").loc[ids, "ami"].to_numpy())

    per_run = pd.read_csv(RR / "per_run_reranker_results.csv.gz")
    pr = per_run[(per_run.policy_id == POLICY) & (per_run.view_id == "xy_2d")] \
        .set_index("dataset_id")
    xy_un_rr = pr.loc[ids, "original_selected_ari"].to_numpy()
    xy_rr = pr.loc[ids, "reranked_selected_ari"].to_numpy()

    xy_resid = float(np.nanmax(np.abs(xy_un_rr - xy_un_canon)))
    PROV["xy_alignment_check"] = {
        "max_abs_residual_reranker_xy_original_vs_canonical_xy": xy_resid,
        "verdict": "IDENTICAL" if xy_resid <= 1e-9 else "DISAGREE"}

    add("xy-only: Fixed +rerank vs AutoClust Extended (MATCHED)",
        xy_rr, xy_ac, "fixed_knn_top10_raw_reranked", AC_MATCHED,
        unit="xy_2d_ari")
    add("xy-only: Fixed +rerank vs Fixed (no rerank)",
        xy_rr, xy_un_rr, "fixed_knn_top10_raw_reranked",
        "fixed_knn_top10_raw_unreranked", unit="xy_2d_ari")
    add("xy-only: Fixed (no rerank) vs AutoClust Extended (MATCHED)",
        xy_un_rr, xy_ac, "fixed_knn_top10_raw_unreranked", AC_MATCHED,
        unit="xy_2d_ari")
    for nm, v in (("fixed_knn_top10_raw_unreranked", xy_un_rr),
                  ("fixed_knn_top10_raw_reranked", xy_rr),
                  (AC_MATCHED, xy_ac)):
        ci = row_bootstrap_ci(v)
        rows.append({"contrast": "LEVEL MEAN", "arm_a": nm, "arm_b": "",
                     "unit": "xy_2d_ari",
                     "population": "controlled Split-1 (%d datasets)" % len(ids),
                     "n": len(v), "mean_a": float(np.mean(v)),
                     "mean_b": np.nan, "mean_delta": np.nan,
                     "ci_lo": ci["lo"], "ci_hi": ci["hi"],
                     "bootstrap_unit": "dataset_row", "note": ""})

    # ----------------------------------------------------------------- B4 AMI
    ami_un = fixed_un.loc[ids, "ami"].to_numpy()
    ami_ac = ac_m.loc[ids, "ami"].to_numpy()
    add("AMI: Fixed (no rerank) vs AutoClust Extended (MATCHED), Best-View",
        ami_un, ami_ac, "fixed_knn_top10_raw_unreranked", AC_MATCHED,
        unit="bestview_ami",
        note="AVAILABLE: both sides persist AMI at the ARI-selected view")
    add("AMI (xy-only): Fixed (no rerank) vs AutoClust Extended (MATCHED)",
        xy_un_ami, xy_ac_ami, "fixed_knn_top10_raw_unreranked", AC_MATCHED,
        unit="xy_2d_ami")
    for lbl in ("AMI: Fixed +rerank vs Fixed (no rerank)",
                "AMI: Fixed +rerank vs AutoClust Extended (MATCHED)"):
        rows.append({"contrast": lbl, "arm_a": "fixed_knn_top10_raw_reranked",
                     "arm_b": "", "unit": "bestview_ami",
                     "population": "controlled Split-1 (%d datasets)" % len(ids),
                     "n": len(ids), "mean_a": np.nan, "mean_b": np.nan,
                     "mean_delta": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
                     "bootstrap_unit": "", "note": "BLOCKED -- see e8_inventory"})
    BLOCKERS.append({
        "item": "B4 controlled AMI for any RERANKED arm",
        "status": "BLOCKED",
        "reason": ("The reranker returns a different candidate from the search "
                   "slate. The per-trial trace "
                   "analyzed_data/<...>/experiments/split_01/knn_top10_raw/<view>/"
                   "clustopt_results.csv stores per-trial ARI and metric values "
                   "but NOT per-trial labels and NOT per-trial AMI; "
                   "external_metrics.json holds AMI only for the run's own "
                   "J-maximiser. The Stage-2B reranker packages store slate "
                   "selections and ARI only. Computing AMI for the reranked "
                   "candidate therefore requires re-instantiating and re-running "
                   "that clustering configuration, which this task forbids."),
        "what_is_available": ("AMI for the UNRERANKED fixed policy and for "
                             "AutoClust/ML2DAC, at Best-View and xy_2d."),
        "unblocks_with": "E-series re-execution that persists per-trial labels or AMI",
    })

    df = pd.DataFrame(rows)
    front = ["contrast", "arm_a", "arm_b", "unit", "population", "n", "mean_a",
             "mean_b", "mean_delta", "median_delta", "ci_lo", "ci_hi",
             "bootstrap_unit", "wins", "ties", "losses", "wins_eps", "ties_eps",
             "losses_eps", "statistic", "p_value", "rank_biserial", "valid", "note"]
    df = df[[c for c in front if c in df.columns]
            + [c for c in df.columns if c not in front]]
    df.to_csv(OUT / "controlled_headline_stats.csv", index=False)

    # --------------------------------------------------------- B2 family table
    fam = fixed_un.loc[ids, "family_id"].to_numpy()
    fr = pd.DataFrame({"dataset_id": ids, "family_id": fam,
                       "fixed_rerank": fx_rr, "fixed_norerank": fx_un,
                       "autoclust_ext_matched": a_ext,
                       "autoclust_ext_native": a_ext_nat})
    fr["delta_fixed_rerank_minus_autoclust"] = (fr.fixed_rerank
                                                - fr.autoclust_ext_matched)
    g = fr.groupby("family_id")
    famtab = g.agg(n=("dataset_id", "size"),
                   fixed_rerank_mean=("fixed_rerank", "mean"),
                   fixed_norerank_mean=("fixed_norerank", "mean"),
                   autoclust_ext_matched_mean=("autoclust_ext_matched", "mean"),
                   autoclust_ext_native_mean=("autoclust_ext_native", "mean"),
                   ).reset_index()
    famtab["delta_fixed_minus_autoclust"] = (famtab.fixed_rerank_mean
                                             - famtab.autoclust_ext_matched_mean)
    # per-family paired stats (dataset is the unit within the family)
    extra = []
    for f_id, sub in g:
        blk = paired_block(sub.fixed_rerank.to_numpy(),
                           sub.autoclust_ext_matched.to_numpy())
        extra.append({"family_id": f_id, "ci_lo": blk["ci_lo"],
                      "ci_hi": blk["ci_hi"], "wins": blk["wins"],
                      "ties": blk["ties"], "losses": blk["losses"],
                      "wins_eps": blk["wins_eps"], "ties_eps": blk["ties_eps"],
                      "losses_eps": blk["losses_eps"],
                      "p_value": blk["p_value"],
                      "rank_biserial": blk["rank_biserial"]})
    famtab = famtab.merge(pd.DataFrame(extra), on="family_id")
    famtab = famtab.sort_values("delta_fixed_minus_autoclust", ascending=False)
    famtab.to_csv(OUT / "controlled_family_stats.csv", index=False)

    d = famtab["delta_fixed_minus_autoclust"]
    PROV["family_summary"] = {
        "n_families": int(len(famtab)),
        "n_positive": int((d > 0).sum()),
        "n_negative": int((d < 0).sum()),
        "n_within_pm_0.01": int((d.abs() <= 0.01).sum()),
        "max_abs_delta": float(d.abs().max()),
        "max_abs_delta_family": str(famtab.loc[d.abs().idxmax(), "family_id"]),
        "deinterleaving_delta": float(
            famtab.loc[famtab.family_id == "deinterleaving_pri_toa_amp_like",
                       "delta_fixed_minus_autoclust"].iloc[0]),
        "bands_stripes_delta": float(
            famtab.loc[famtab.family_id == "linear_bands_parallel_stripes",
                       "delta_fixed_minus_autoclust"].iloc[0]),
        "arcs_rings_delta": float(
            famtab.loc[famtab.family_id == "arcs_circles_rings",
                       "delta_fixed_minus_autoclust"].iloc[0]),
        "blobs_delta": float(
            famtab.loc[famtab.family_id == "standard_clustering_blobs",
                       "delta_fixed_minus_autoclust"].iloc[0]),
    }

    PROV["sources"] = {
        "canonical_best_view_frame": rel(CANON / "canonical_best_view_frame.parquet"),
        "canonical_run_frame": rel(CANON / "canonical_run_frame.parquet"),
        "per_dataset_policy_results": rel(RR / "per_dataset_policy_results.csv.gz"),
        "per_run_reranker_results": rel(RR / "per_run_reranker_results.csv.gz"),
        "method_label_map": rel(CR / "paper_analysis_v2" / "00_protocol_and_definitions"
                                / "method_label_map.csv"),
    }
    PROV["blockers"] = BLOCKERS
    (OUT / "controlled_provenance.json").write_text(
        json.dumps(PROV, indent=2, default=str), encoding="utf-8")

    pd.set_option("display.width", 260)
    print(json.dumps(PROV["alignment_check"], indent=1))
    print(json.dumps(PROV["xy_alignment_check"], indent=1))
    print(df[["contrast", "n", "mean_a", "mean_b", "mean_delta", "ci_lo", "ci_hi",
              "wins", "ties", "losses", "wins_eps", "ties_eps", "losses_eps",
              "p_value", "rank_biserial"]].to_string())
    print()
    print(famtab.to_string())
    print()
    print(json.dumps(PROV["family_summary"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
