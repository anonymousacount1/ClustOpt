"""Stage 3B runner, part 3: cross-method summary, publication tables, forward plan."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metric_atlas import common as C          # noqa: E402
from metric_atlas import summarise as SM      # noqa: E402


def log(m: str) -> None:
    print("[3b] %s" % m, flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=None)
    args = ap.parse_args(argv)
    repo = Path(args.repo_root).resolve() if args.repo_root else C.REPO
    out = C.out_dir(repo)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                            capture_output=True, text=True).stdout.strip()
    print("=" * 78)
    print("STAGE 3B  METRIC ATTRIBUTION  (part 3: summaries and forward plan)")
    print("=" * 78, flush=True)

    rd = lambda n, gz=False: pd.read_csv(C.ext(out / n))  # noqa: E731
    A = rd("canonical_metric_atlas.csv")
    G = rd("metric_global_utility.csv")
    S = rd("metric_group_utility_summary.csv")
    Ct = rd("metric_group_contrasts.csv")
    Cp = rd("new46_complementarity.csv")
    Cov = rd("new46_unique_win_coverage.csv")
    Go = rd("group_oracle_coverage.csv")
    U = rd("clustopt_metric_usage.csv")
    Q = rd("clustopt_metric_prediction_quality.csv")
    CF = rd("clustopt_family_metric_usage.csv")
    MS = rd("ml2dac_metric_selection.csv")
    Arm = rd("ml2dac_cvi_star_diagnostics.csv")
    MR = rd("ml2dac_per_cvi_recall.csv")
    MF = rd("ml2dac_family_metric_selection.csv")
    Gp = rd("autoclust_group_reliance.csv")
    AR = rd("autoclust_metric_reliance.csv")
    AF = rd("autoclust_family_group_reliance.csv")
    FAM = pd.read_csv(C.ext(out / "metric_family_utility.csv.gz"))
    SUB = pd.read_csv(C.ext(out / "metric_subfamily_utility.csv.gz"))

    # ------------------------------------------------- cross-method summaries
    MG = SM.metric_group_summary(S, Q, Arm, Gp)
    C.write_csv(MG, out / "metric_group_summary.csv")

    MT = SM.metric_publication_table(A, G, Cp, U, MS, FAM, AR)
    C.write_csv(MT, out / "metric_publication_table.csv")

    fam_map = (pd.read_csv(C.ext(repo / C.OOF_REL),
                          usecols=["family_id", "family_name"])
               .drop_duplicates().set_index("family_id")["family_name"].to_dict())
    FT = SM.family_publication_table(A, FAM, Cov, CF, MF, AF, fam_map)
    C.write_csv(FT, out / "family_publication_table.csv")

    subcov = Cov[Cov.level == "subfamily"].rename(columns={"key": "subfamily_id"})
    SP = SUB.merge(subcov[["subfamily_id", "new46_best_frac", "new46_best_n"]],
                   on="subfamily_id", how="left")
    C.write_csv(SP, out / "subfamily_publication_table.csv.gz", gz=True)

    C.write_json(SM.comparator_coverage(A, repo), out / "comparator_coverage.json")

    # ------------------------------------------------------- multiple testing
    fams = {
        "new46_vs_best_comparator": {
            "n_tests": int(len(Cp)), "method": "Benjamini-Hochberg",
            "n_q_below_0p05": int((Cp["wilcoxon_q_bh"] < 0.05).sum()),
            "n_ci_excludes_zero_positive": int((Cp["ci_low"] > 0).sum()),
            "n_ci_excludes_zero_negative": int((Cp["ci_high"] < 0).sum()),
            "note": ("the comparator is the MAX over 14 established metrics, so a "
                     "negative mean is close to structural; the discriminating "
                     "quantities are frac_beats_best_comparator and oracle_win_freq"),
        },
        "group_utility_contrasts": {
            "n_tests": int(len(Ct)), "method": "Benjamini-Hochberg",
            "n_q_below_0p05": int((Ct["wilcoxon_q_bh"] < 0.05).sum()),
        },
    }
    C.write_json({
        "stage": "3B", "created_at": C.now(),
        "policy": ("tests were declared per family BEFORE values were inspected; "
                   "no threshold was chosen after seeing outcomes and no metric "
                   "was dropped from a family"),
        "inferential_unit": "dataset", "bootstrap_B": C.BOOT_B,
        "bootstrap_seed": C.BOOT_SEED, "families": fams,
    }, out / "multiple_testing_summary.json")

    # ------------------------------------------------------- publication path
    n_top10_new = int((G[(G.group == "New46")]["utility_rank_global"] <= 10).sum())
    n_oracle_2pct = int((Cp["oracle_win_freq"] > 0.02).sum())
    new46_best_global = float(Cov[Cov.level == "global"]["new46_best_frac"].iloc[0])
    a3 = Arm[Arm.arm == "A3"].iloc[0]
    a0 = Arm[Arm.arm == "A0"].iloc[0]
    assess = {
        "stage": "3B", "created_at": C.now(), "source_commit": commit,
        "default_recommendation": "ONE INTEGRATED PAPER",
        "separate_metrics_paper_authorised_now": False,
        "evidence_for_integration": {
            "method_dependence_is_the_finding": (
                "the identical inventory helps AutoClust (+0.033474, p=3.6e-40) and "
                "does not help ML2DAC (-0.005757, ns); the between-method difference "
                "is significant on every factorial term"),
            "mechanism_is_measured": (
                "ML2DAC CVI-selection accuracy falls from %.4f at 7 candidates to "
                "%.4f at 63" % (a0["selection_accuracy_vs_cvi_star"],
                                a3["selection_accuracy_vs_cvi_star"])),
            "inventory_raises_the_ceiling": (
                "group oracle utility rises A0 %.4f -> A3 %.4f, improving %.1f%% of "
                "datasets" % (Go[Go.inventory == "A0_Original"]["mean_best_utility"].iloc[0],
                              Go[Go.inventory == "A3_Extended"]["mean_best_utility"].iloc[0],
                              100 * Go[Go.inventory == "A3_Extended"]["frac_datasets_improved_vs_A0"].iloc[0])),
        },
        "evidence_against_a_standalone_metrics_paper_today": {
            "no_new46_metric_beats_the_comparator_pool_on_average": int((Cp["ci_low"] > 0).sum()) == 0,
            "average_new46_metric_is_weaker": (
                "group mean utility New46 %.4f vs Established %.4f vs Original %.4f"
                % (S[S.group == "New46"]["mean_of_metric_means"].iloc[0],
                   S[S.group == "Established"]["mean_of_metric_means"].iloc[0],
                   S[S.group == "Original"]["mean_of_metric_means"].iloc[0])),
            "benefit_is_partly_concentrated": (
                "%d New46 metrics reach the global top-10 by utility and %d exceed a "
                "2%% oracle-win rate, out of 44 live" % (n_top10_new, n_oracle_2pct)),
            "existing_domain_only": (
                "these metrics were designed against this synthetic repository, so "
                "existing-domain strength cannot establish external validity"),
            "modern_comparators_incomplete": "CDbw / CVNN / CVDD / DCSI are absent",
        },
        "evidence_that_would_justify_separation": [
            "several New46 metrics individually strong on data not used to design them",
            "interpretable geometry/family alignment reproduced out of domain",
            "complementary coverage beyond strong established AND modern CVIs",
            "robustness on an independent domain",
            "acceptable computational cost",
        ],
        "key_existing_domain_numbers": {
            "new46_is_best_available_metric_frac": new46_best_global,
            "new46_share_of_clustopt_top5_slots": float(
                Q[Q.group == "New46"]["share_of_selection_slots"].iloc[0]),
            "ml2dac_a3_new46_selected_share": float(a3["selected_share_New46"]),
            "ml2dac_a3_cvi_star_new46_share": float(a3["cvi_star_share_New46"]),
            "autoclust_new46_group_reliance_pct": float(
                Gp[Gp.group == "New46"]["rmse_increase_pct"].iloc[0]),
        },
    }
    C.write_json(assess, out / "publication_path_assessment.json")

    # ---------------------------------------------------- external plan (frozen)
    live = A[A.is_live]
    focal = list(Cp.sort_values("oracle_win_freq", ascending=False)
                 .head(10)["metric_id"])
    plan = {
        "stage": "3B (prepared, NOT executed)", "created_at": C.now(),
        "source_commit": commit, "external_experiment_executed": False,
        "external_datasets_downloaded": False,
        "metrics_to_evaluate": {
            "policy": "ALL 46 designed metrics, including the 2 dead here",
            "designed": 46,
            "live_on_existing_domain": int((live.group == "New46").sum()),
            "dead_here_but_retained_for_external_evaluation": list(
                A[~A.is_live]["metric_id"]),
            "all_new46_ids": sorted(A[A.group == "New46"]["metric_id"]),
            "comparators_already_available": sorted(
                A[A.group.isin(["Original", "Established"]) & A.is_live]["metric_id"]),
        },
        "focal_shortlist": {
            "metrics": focal,
            "selection_rule": ("PREDECLARED: top 10 live New46 metrics by "
                               "existing-domain oracle-win frequency"),
            "status": "HYPOTHESIS-GENERATING ONLY; not externally validated",
            "does_not_replace_full_set": True,
        },
        "comparator_plan": {
            "already_present": ["dbcv", "s_dbw", "silhouette",
                                "noise_aware_silhouette", "calinski_harabasz",
                                "davies_bouldin", "dunn_index", "cop",
                                "coggins_jain_index"],
            "to_consider_adding": ["CDbw", "CVNN", "CVDD", "DCSI"],
            "rule": "no comparison may be claimed against a metric not evaluated",
        },
        "planned_per_index_statistics": [
            "mean/median canonical utility with dataset-level bootstrap CI",
            "rank and Top-k coverage among all evaluated metrics",
            "difference vs the best established+modern comparator",
            "oracle-win frequency and unique-win coverage",
            "per-geometry/per-family breakdown",
            "runtime per metric per record",
        ],
        "multiplicity_plan": {
            "method": "Benjamini-Hochberg",
            "families": ["per-index vs comparator pool", "per-geometry alignment",
                         "group-level contrasts"],
            "report": "raw p and adjusted q for every test",
        },
        "predeclared_hypotheses": {
            "H1": ("a minority of New46 metrics, not the group mean, carries the "
                   "existing-domain benefit; this should reproduce out of domain"),
            "H2": ("Pattern metrics align with curve/line/lattice geometries and "
                   "Image metrics with texture/morphology geometries"),
            "H3": ("New46 raises the achievable oracle ceiling even where the "
                   "average New46 metric is weaker than the average classic CVI"),
            "H4": ("benefit depends on the consuming mechanism: full-vector "
                   "consumers gain, single-index selectors do not"),
        },
        "runtime_fields_to_capture": ["per-metric wall time", "per-record total",
                                      "peak memory", "failure/NaN rate"],
        "inferential_unit": "dataset",
        "forbidden_until_executed": [
            "any claim of external validation",
            "any claim that a focal metric is validated",
            "any single-metric causal ARI claim",
        ],
    }
    C.write_json(plan, out / "external_metric_validation_plan.json")

    # ------------------------------------------------------------- provenance
    C.write_json({
        "stage": "3B", "created_at": C.now(), "source_commit": commit,
        "analysis_only": True, "clustering_runs": 0, "models_trained": 0,
        "mkr_rebuilds": 0, "external_datasets": 0,
        "metric_inventory_modified": False,
        "sources": {
            "utility": C.OOF_REL,
            "master_mkr": C.MKR_REL + "/master",
            "derived_baselines": C.MKR_REL + "/derived",
            "stage1_bundle": C.STAGE1_REL,
            "stage3a_decompositions": C.EXT_REL,
            "split1_results": C.ANALYZED_REL + "/<family>/subfamilies/<subfamily>/"
                              "<dataset>/experiments/split_01/<METHOD_ID>/",
        },
        "populations": {
            "intrinsic_utility": "Splits 2-16, 16,013 datasets, 48,039 records",
            "ml2dac_selection": "Split 1, 3,165 view records per arm",
            "autoclust_reliance": "Split 1, 101,274 held-out evaluation rows",
        },
        "existing_domain_only": True,
        "evidence_levels": {
            "1_causal_bundle": "Stage-1 arms and Stage-3A A0/A1/A2/A3 ARI ablations",
            "2_association": "utility, ranks, selection frequency, permutation reliance",
            "3_causal_single_metric": "NOT ESTABLISHED; no leave-one-metric-out exists",
        },
    }, out / "provenance.json")

    C.write_json({
        "stage": "3B", "experiment": "existing-domain metric attribution atlas",
        "created_at": C.now(), "source_commit": commit,
        "parts": ["atlas+utility+structure+clustopt", "ml2dac+autoclust",
                  "summaries+publication+external plan"],
        "n_live_metrics": int(len(live)), "n_dead_metrics": int((~A.is_live).sum()),
        "n_designed_new46": int((A.group == "New46").sum()),
        "n_live_new46": int(((A.group == "New46") & A.is_live).sum()),
        "recommendation": "ONE INTEGRATED PAPER",
        "next_step": "independent-domain metric validation (NOT started)",
    }, out / "experiment_manifest.json")

    log("part 3 complete: %d live metrics, %d New46 in global top-10, "
        "%d New46 above 2%% oracle-win" % (len(live), n_top10_new, n_oracle_2pct))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
