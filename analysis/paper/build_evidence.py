"""Extract the evidence tables used by the paper's tables and figures from the released results.

Read-only with respect to all scientific outputs: it writes descriptive aggregations to
``analysis/output/paper/evidence`` for ``build_tables.py`` and ``build_figures.py``.
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_paths import (AC, CORPUS, CR, DATA, EVI, FAM_PLAIN, IDBCFG, KNNRUN,
                       MLPRUN, RRPREP, S1, S2, S2B5, S3, S3P, SR, STAGE1REP,
                       UNI, CATALOGUE)

OUT = {}


def dump(name, obj):
    (EVI / name).write_text(json.dumps(obj, indent=1, default=str),
                            encoding="utf-8")
    print("  evidence:", name)


def dumpcsv(name, df):
    df.to_csv(EVI / name, index=False)
    print("  evidence:", name, df.shape)


# ===================================================== 1. repository shape
def repository():
    cols = ["dataset_id", "view_mode", "family_name", "subfamily_id",
            "difficulty", "cluster_count_from_generator", "n_points",
            "source_dataset_dir"]
    # released dataset catalogue (one row per generated dataset) instead of the
    # 269 MB training table: identical dataset population and descriptors
    s = pd.read_csv(CATALOGUE)
    d = pd.DataFrame({"dataset_id": s.dataset_id, "view_mode": "xy_2d",
                      "family_name": s.family_name, "subfamily_id": s.subfamily_id,
                      "difficulty": s.difficulty,
                      "cluster_count_from_generator": s.cluster_count,
                      "n_points": s.n_points, "family_dir": s.family_id})

    fam = d.groupby("family_dir").agg(
        n_datasets=("dataset_id", "size"),
        n_subfamilies=("subfamily_id", "nunique"),
        n_points_min=("n_points", "min"), n_points_max=("n_points", "max"),
        n_points_median=("n_points", "median"),
        k_min=("cluster_count_from_generator", "min"),
        k_max=("cluster_count_from_generator", "max")).reset_index()
    fam["family"] = fam.family_dir.map(FAM_PLAIN)
    dumpcsv("repo_family_summary.csv", fam)

    sub = d.groupby(["family_dir", "subfamily_id"]).agg(
        n_datasets=("dataset_id", "size"),
        n_points_min=("n_points", "min"), n_points_max=("n_points", "max"),
        k_min=("cluster_count_from_generator", "min"),
        k_max=("cluster_count_from_generator", "max"),
        difficulties=("difficulty", lambda s: "/".join(
            sorted(set(s), key=lambda x: ["easy", "medium", "hard",
                                          "very_hard"].index(x))))).reset_index()
    sub["family"] = sub.family_dir.map(FAM_PLAIN)
    dumpcsv("repo_subfamily_summary.csv", sub)

    diff = d.groupby(["family_dir", "difficulty"]).size().unstack(fill_value=0)
    dumpcsv("repo_family_difficulty.csv", diff.reset_index())

    OUT["repository"] = {
        "n_datasets": int(len(d)), "n_families": int(d.family_dir.nunique()),
        "n_subfamilies": int(d.subfamily_id.nunique()),
        "n_records": int(len(d) * 3),
        "n_points_min": int(d.n_points.min()),
        "n_points_max": int(d.n_points.max()),
        "n_points_median": float(d.n_points.median()),
        "k_values": sorted(int(x) for x in
                           d.cluster_count_from_generator.unique()),
        "difficulty_levels": ["easy", "medium", "hard", "very_hard"],
        "generator": "HYBRID_SCM_V2",
        "bounds": "unit square [0,1]^2",
    }
    return d


# ============================================== 2. controlled family deltas
def controlled_families():
    fam = pd.read_csv(S1 / "offline_fixed32" / "offline_2x3_dataset_results.csv",
                      usecols=["dataset_id", "family"]).drop_duplicates(
                          "dataset_id")
    per = pd.read_csv(S2 / "per_dataset_online_results.csv.gz")
    acp = pd.read_csv(AC / ("per_dataset_bestview__AutoClust_Extended_"
                            "InDomain_same_search_space.csv")).rename(
        columns={"bestview_ari": "AC_matched"})
    m = per.merge(acp, on="dataset_id").merge(fam, on="dataset_id", how="left")
    assert len(m) == 1055, len(m)

    g = m.groupby("family").agg(
        n=("dataset_id", "size"), fixed=("A_fixed", "mean"),
        v1=("B_v1", "mean"), v2=("C_v2", "mean"), vbs=("D_vbs", "mean"),
        autoclust=("AC_matched", "mean")).reset_index()
    g["d_v2_ac"] = g.v2 - g.autoclust
    g["d_fixed_ac"] = g.fixed - g.autoclust
    g["d_vbs_v2"] = g.vbs - g.v2
    g = g.rename(columns={"family": "family_id"});g["family"] = g.family_id.map(FAM_PLAIN)
    g = g.sort_values("d_v2_ac", ascending=False).reset_index(drop=True)
    dumpcsv("controlled_family_deltas.csv", g)

    within = int((g.d_v2_ac.abs() <= 0.01).sum())
    OUT["controlled_families"] = {
        "n_families": int(len(g)),
        "n_within_0.01_v2_vs_autoclust": within,
        "n_outside_0.01": int(len(g) - within),
        "largest_positive": {"family": g.iloc[0].family,
                             "delta": float(g.iloc[0].d_v2_ac)},
        "largest_negative": {"family": g.iloc[-1].family,
                             "delta": float(g.iloc[-1].d_v2_ac)},
        "n_clustopt_ahead": int((g.d_v2_ac > 0).sum()),
        "n_autoclust_ahead": int((g.d_v2_ac < 0).sum()),
        "deinterleaving_delta": float(
            g.loc[g.family_id == "deinterleaving_pri_toa_amp_like",
                  "d_v2_ac"].iloc[0]),
        "max_abs_delta": float(g.d_v2_ac.abs().max()),
    }
    return g


# ================================================ 3. controlled progression
def controlled_progression():
    four = pd.read_csv(S2 / "four_level_online_table.csv")
    pol = pd.read_csv(S2B5 / "per_policy_results.csv")
    fixed_id = json.loads((S2 / "development_selected_fixed_policy.json")
                          .read_text())["policy_id"]
    row = pol[pol.policy_id == fixed_id].iloc[0]
    ac = pd.read_csv(S2 / "autoclust_matched_comparison.csv").iloc[0]

    # the pre-reranker point is the SAME policy on the SAME split-1 replay;
    # its reranked value reproduces four-level level A exactly.
    assert abs(float(row.reranked_mean_bestview)
               - float(four[four.level == "A"].mean_bestview.iloc[0])) < 1e-12

    prog = pd.DataFrame([
        {"key": "pre", "label": "Fixed policy, no rerank",
         "mean": float(row.original_mean_bestview), "lo": np.nan, "hi": np.nan,
         "kind": "deployable"},
        {"key": "A", "label": "Fixed policy + rerank",
         "mean": float(four[four.level == "A"].mean_bestview.iloc[0]),
         "lo": float(four[four.level == "A"].ci_lo.iloc[0]),
         "hi": float(four[four.level == "A"].ci_hi.iloc[0]),
         "kind": "deployable"},
        {"key": "B", "label": "PPv1 + rerank",
         "mean": float(four[four.level == "B"].mean_bestview.iloc[0]),
         "lo": float(four[four.level == "B"].ci_lo.iloc[0]),
         "hi": float(four[four.level == "B"].ci_hi.iloc[0]),
         "kind": "deployable"},
        {"key": "C", "label": "PPv2 + rerank",
         "mean": float(four[four.level == "C"].mean_bestview.iloc[0]),
         "lo": float(four[four.level == "C"].ci_lo.iloc[0]),
         "hi": float(four[four.level == "C"].ci_hi.iloc[0]),
         "kind": "deployable"},
        {"key": "D", "label": "Policy oracle + rerank",
         "mean": float(four[four.level == "D"].mean_bestview.iloc[0]),
         "lo": float(four[four.level == "D"].ci_lo.iloc[0]),
         "hi": float(four[four.level == "D"].ci_hi.iloc[0]),
         "kind": "oracle"},
    ])
    dumpcsv("controlled_progression.csv", prog)
    OUT["controlled_progression"] = {
        "pre_reranker_available": True,
        "pre_reranker_policy": fixed_id,
        "pre_reranker_mean": float(row.original_mean_bestview),
        "pre_reranker_gain": float(row.absolute_gain),
        "pre_reranker_gain_ci": [float(row.gain_ci_lo), float(row.gain_ci_hi)],
        "pre_reranker_improved": int(row.datasets_improved),
        "pre_reranker_unchanged": int(row.datasets_unchanged),
        "pre_reranker_worsened": int(row.datasets_worsened),
        "slate_oracle": float(row.visited_oracle_mean_bestview),
        "recovery_aggregate": float(row.recovery_aggregate),
        "autoclust_matched": float(ac.autoclust_reference_value),
        "comparability_note": ("same Split-1 replay, same policy, same frozen "
                               "protocol: the reranked column reproduces "
                               "four-level level A to machine precision"),
    }
    dumpcsv("controlled_policy_prepost.csv",
            pol[["policy_id", "utility_source", "k_mode", "weighting_mode",
                 "original_mean_bestview", "reranked_mean_bestview",
                 "visited_oracle_mean_bestview", "absolute_gain",
                 "gain_ci_lo", "gain_ci_hi", "recovery_aggregate",
                 "datasets_improved", "datasets_unchanged",
                 "datasets_worsened"]])
    return prog


# ================================================== 4. stage-3 internal
def stage3():
    hv = json.loads((S3 / "hypothesis_verdict.json").read_text())
    rec = pd.read_csv(S3 / "family_true_topk_recall.csv")
    aln = pd.read_csv(S3 / "family_utility_selection_alignment.csv")
    pvi = pd.read_csv(S3 / "pattern_vs_image_family.csv")
    tierA_f = pd.read_csv(S3 / "family_specialists_tierA.csv")
    tierA_s = pd.read_csv(S3 / "subfamily_specialists_tierA.csv")
    core = pd.read_csv(S3 / "core_candidates.csv")

    OUT["stage3"] = {
        "verdict": hv["verdict"],
        "core_metrics": hv["core"]["metrics"],
        "core_definition": hv["core"]["definition"],
        "family": hv["family"], "subfamily": hv["subfamily"],
        "alignment": hv["alignment"],
        "selection_composition": hv["selection_composition"],
        "jaccard": hv["jaccard"],
        "new46_true_top5_recall_mean": float(rec.new46_true_top5_recall.mean()),
        "true_top5_recall_mean": float(rec.true_top5_recall.mean()),
        "true_top1_hit_mean": float(rec.true_top1_hit.mean()),
        "new46_selection_precision_mean": float(
            rec.new46_selection_precision.mean()),
        "alignment_min": float(aln.spearman_util_sel_all.min()),
        "alignment_max": float(aln.spearman_util_sel_all.max()),
        "n_tierA_family_rows": int(len(tierA_f)),
        "n_tierA_subfamily_rows": int(len(tierA_s)),
        "tierA_unique_metrics": sorted(tierA_f.metric_id.unique().tolist()),
        "tierA_subfamily_unique_metrics": sorted(
            tierA_s.metric_id.unique().tolist()),
    }
    pat = pvi.groupby("subgroup").agg(
        mean_utility=("mean_utility", "mean"),
        selection_share=("selection_share", "mean"),
        n_tier_a=("n_tier_a", "sum"),
        top5_count=("top5_count", "sum")).reset_index()
    OUT["stage3"]["pattern_vs_image"] = pat.to_dict("records")
    dumpcsv("stage3_pattern_vs_image.csv", pvi)
    dumpcsv("stage3_alignment.csv", aln[[
        "group", "n_datasets", "spearman_util_sel_all",
        "spearman_util_sel_Original", "spearman_util_sel_Established",
        "spearman_util_sel_New46"]])
    dumpcsv("stage3_family_tierA.csv", tierA_f[[
        "group", "metric_id", "metric_group", "subgroup", "n_datasets",
        "U_group", "U_comparator", "DeltaU", "DeltaU_ci_low", "DeltaU_ci_high",
        "DeltaU_q_bh", "S_group", "S_comparator", "enrichment",
        "unique_specialist"]])
    dumpcsv("stage3_subfamily_tierA.csv", tierA_s[[
        "group", "parent_family", "metric_id", "metric_group", "n_datasets",
        "U_group", "U_comparator", "DeltaU", "DeltaU_ci_low", "DeltaU_ci_high",
        "DeltaU_q_bh", "enrichment", "subfamily_only"]])
    dumpcsv("stage3_core.csv", core)
    dumpcsv("stage3_family_recall.csv", rec)
    return tierA_f, tierA_s


# ============================== 5. family x metric selection / utility
def tailoring():
    sel = pd.read_csv(S3 / "family_metric_selection.csv.gz")
    uti = pd.read_csv(S3 / "family_metric_utility.csv.gz")
    core = json.loads((S3 / "core_definition.json").read_text())["core_candidates"]
    tierA = pd.read_csv(S3 / "family_specialists_tierA.csv")
    tierB = pd.read_csv(S3 / "family_specialists_tierB.csv")

    # transparent, pre-declared metric selection rule: the stable core plus
    # every metric carrying Tier-A or Tier-B family specialisation evidence.
    spec = sorted(set(tierA.metric_id) | set(tierB.metric_id))
    keep = list(dict.fromkeys(list(core) + [m for m in spec if m not in core]))
    rule = ("stable generalist core (Top-5 by true utility AND by selection "
            "frequency in >=6 of 12 families) followed by every metric with "
            "Tier-A or Tier-B family-level specialisation evidence; no "
            "manual choice")

    s = sel[sel.metric_id.isin(keep)].pivot(index="metric_id", columns="group",
                                            values="S_group")
    u = uti[uti.metric_id.isin(keep)].pivot(index="metric_id", columns="group",
                                            values="U_group")
    s = s.reindex(keep)
    u = u.reindex(keep)
    s.to_csv(EVI / "family_metric_selection_matrix.csv")
    u.to_csv(EVI / "family_metric_utility_matrix.csv")
    OUT["tailoring"] = {
        "selection_rule": rule, "metrics": keep,
        "core": list(core), "n_specialists": len(keep) - len(core),
        "n_families": int(s.shape[1]),
    }
    print("  evidence: family_metric_{selection,utility}_matrix.csv",
          s.shape)
    return s, u


# ================================================= 6. external corpus meta
def external_corpus():
    d = json.loads(CORPUS.read_text())
    rows = []
    for e in d["entries"]:
        si = e["source_identity"]
        rows.append({
            "dataset_id": e["dataset_id"], "source": e["source_group"],
            "n_samples": e["n_samples"],
            "dim_after_cleaning": e["preparation"].get("dim_after_cleaning"),
            "prepared_dim": e["prepared_dimensionality"],
            "pca_applied": e["preparation"]["pca_applied"],
            "explained_variance_total": e["preparation"].get(
                "explained_variance_total"),
            "generator": si.get("generator"), "difficulty": si.get("difficulty"),
            "openml_id": si.get("openml_data_id"),
            "redistribution": e["redistribution"]["raw_redistribution"],
            "representation_checksum": e["representation_checksum"],
        })
    df = pd.DataFrame(rows)
    dumpcsv("external_corpus_metadata.csv", df)
    OUT["external_corpus"] = {
        "n_datasets": int(len(df)),
        "by_source": df.source.value_counts().to_dict(),
        "manifest_hash": d["frozen_corpus_manifest_hash"],
        "checksums_match": d["checksums_match_frozen_manifest"],
        "pca_reduced": d["pca_reduced_datasets"],
        "redistribution": df.redistribution.value_counts().to_dict(),
        "n_samples_min": int(df.n_samples.min()),
        "n_samples_max": int(df.n_samples.max()),
        "view_definitions": d["view_definitions"],
    }
    return df


# ================================================ 7. external per-dataset
def external_results(meta):
    mat = pd.read_csv(SR / "per_dataset_method_matrix.csv")
    keep = ["dataset_id", "source"] + [c for c in ("C5", "C4", "A0", "M1")
                                       if c in mat.columns]
    df = mat[keep].copy()
    df["d_C5_A0"] = df.C5 - df.A0
    df["d_C5_M1"] = df.C5 - df.M1
    df["d_C4_A0"] = df.C4 - df.A0
    df["d_C4_M1"] = df.C4 - df.M1
    dumpcsv("external_per_dataset.csv", df)
    by = df.groupby("source")[["C5", "C4", "A0", "M1", "d_C5_A0",
                               "d_C5_M1"]].mean().reset_index()
    dumpcsv("external_by_source.csv", by)
    OUT["external_by_source"] = by.to_dict("records")
    return df


# ================================================== 8. arm catalogue
def arms():
    reg = json.loads((IDBCFG / "method_registry.json").read_text())
    rows = []
    for m in reg["methods"]:
        short = m["method_id"].split("_")[2]
        rows.append({
            "stage": "4 (external)", "code": short,
            "method_id": m["method_id"], "framework": m["framework"],
            "label": m.get("label", m["method_id"]),
            "utility_source": m.get("utility_source", "--"),
            "top_k": m.get("top_k") if m.get("top_k") is not None else "--",
            "weighting": m.get("weighting", "--"),
            "policy": m.get("policy_predictor") or "--",
            "reranker": m.get("reranker_id") or "no",
            "status": "deployable",
        })
    df = pd.DataFrame(rows)
    dumpcsv("arm_catalogue_stage4.csv", df)

    pol = pd.read_csv(S2B5 / "per_policy_results.csv")[
        ["policy_id", "utility_source", "k_mode", "weighting_mode"]]
    dumpcsv("policy_set.csv", pol)
    mp = json.loads((RRPREP / "policy_to_reranker_mapping.json").read_text())
    OUT["policy_set"] = {"n_policies": mp["n_policies"],
                         "n_rerankers": mp["n_rerankers"]}
    rr = json.loads((RRPREP / "final_reranker_registry.json").read_text())
    OUT["reranker"] = {
        "n_rerankers": rr["n_rerankers"],
        "n_features": rr["rerankers"][0]["n_features"],
        "n_training_rows": rr["rerankers"][0]["n_training_rows"],
        "regimes": [r["reranker_id"] for r in rr["rerankers"]],
    }
    return df


# =============================================== 9. predictor quality
def predictor():
    ov = pd.read_csv(MLPRUN / "overall_results.csv").set_index("metric")["mean"]
    cmp_ = pd.read_csv(STAGE1REP / "head14_vs_full60_eval_table.csv")
    dumpcsv("predictor_head14_vs_full60.csv", cmp_)
    view = pd.read_csv(MLPRUN / "artifacts" / "grouped_mean_by_view_id.csv")
    dumpcsv("predictor_by_view.csv", view)
    cfg = json.loads((MLPRUN / "config.json").read_text())
    knn = json.loads((KNNRUN / "selected_configuration.json").read_text())
    knnm = json.loads((KNNRUN / "models" / "final_knn_metadata.json").read_text())
    OUT["predictor"] = {
        "mlp": {"input_dim": cfg["model"]["input_dim"],
                "output_dim": cfg["model"]["output_dim"],
                "hidden": cfg["model"]["hidden_dims"],
                "dropout": cfg["model"]["dropout"],
                "activation": cfg["model"]["activation"],
                "batch_norm": cfg["model"]["batch_norm"],
                "head_hidden": cfg["model"]["head_hidden_dim"],
                "head_dropout": cfg["model"]["head_dropout"],
                "loss": cfg["loss"], "training": cfg["training"],
                "n_eval_records": int(ov["n_samples"])},
        "knn": {"config": knn["selection"]["one_se_selected_configuration"],
                "rule": knn["selection"]["selection_rule"],
                "one_se_threshold": knn["selection"]["one_se_threshold"],
                "n_candidates_within_one_se":
                    knn["selection"]["n_candidates_within_one_se"],
                "candidates": knn["selection"]["candidate_config_ids"],
                "dynamic_topk": knn["dynamic_topk_heuristic"],
                "n_features": knnm["n_features"], "n_metrics": knnm["n_metrics"],
                "per_view_train": knnm["per_view_train_counts"]},
    }
    pp = json.loads((S2 / "final_v2_model_manifest.json").read_text())
    pp1 = json.loads((S2 / "final_v1_model_manifest.json").read_text())
    OUT["policy_predictor"] = {
        "v2": {k: pp[k] for k in ("config_id", "model_config", "seed",
                                  "target_semantics", "n_train_rows",
                                  "input_dim", "output_dim", "split_1_used")},
        "v1_target_semantics": pp1.get("target_semantics"),
        "v1_config_id": pp1.get("config_id"),
    }


# ================================================ 10. search space
def search_space():
    sys.path.insert(0, str(CR.parents[1]))
    src = (CR.parents[1] / "models" / "ClustOpt" / "configs" / "experiments"
           / "generate_experiment_configs.py").read_text(encoding="utf-8")
    ns = {}
    body = src.split("# ----------------------------------------------------------------- space helpers")[1]
    body = body.split("# ------------------------------------------------------------- shared blocks")[0]
    exec("from typing import Any, Dict, List\n" + body, ns)
    spaces = {v: ns["build_search_space"](v)
              for v in ("x_only", "y_only", "xy_2d")}
    dump("search_space.json", spaces)
    OUT["search"] = {
        "budget_evaluations": 50, "sampler": "Optuna TPE",
        "early_stopping": False, "k_range": [2, 5],
        "base_seed": 1234,
        "algorithms": sorted({a.split("-")[0] for s in spaces.values()
                              for a in s}),
        "n_algorithm_blocks": {v: len(s) for v, s in spaces.items()},
    }
    return spaces


# ================================================ 11. metric groups
def metrics():
    mm = pd.read_csv(SR / "metric_master_table.csv")
    dumpcsv("metric_master.csv", mm)
    grp = pd.read_csv(CR / "metric_analysis_existing_domain"
                      / "metric_group_summary.csv")
    dumpcsv("metric_group_summary_internal.csv", grp)
    OUT["metrics"] = {"n_primary": int(len(mm)),
                      "columns": list(mm.columns)}
    return mm


if __name__ == "__main__":
    print("building evidence tables ...")
    repository()
    controlled_families()
    controlled_progression()
    stage3()
    tailoring()
    meta = external_corpus()
    external_results(meta)
    arms()
    predictor()
    search_space()
    metrics()
    dump("evidence_summary.json", OUT)
    print("done")
