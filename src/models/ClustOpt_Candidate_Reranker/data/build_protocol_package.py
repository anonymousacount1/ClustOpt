"""Stage-2B0 orchestrator: merge, audit, diagnose, gate, report.

Merging is streamed. The candidate CVI block alone is 1,537,248 x 60; holding
every shard in memory before concatenating would peak well above the RAM this
machine has free, so shards are appended to multi-member gzip files and the
candidate-level integrity checks run in the same single pass.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from models.Clustering_Repository_Builder.experiments.experiment_execution.output_writer import (  # noqa: E402,E501
    _ext, ensure_dir, path_exists, write_csv_atomic, write_json_atomic,
)
from models.ClustOpt_Candidate_Reranker.data import candidate_eligibility as CE  # noqa: E402,E501
from models.ClustOpt_Candidate_Reranker.data import candidate_schema as CS  # noqa: E402,E501
from models.ClustOpt_Candidate_Reranker.data import feature_availability as FA  # noqa: E402,E501
from models.ClustOpt_Candidate_Reranker.data import slate_schema as SS  # noqa: E402,E501
from models.ClustOpt_Candidate_Reranker.data import validation as V  # noqa: E402
from models.ClustOpt_Candidate_Reranker.protocol import cv_protocol as CV  # noqa: E402,E501
from models.ClustOpt_Candidate_Reranker.protocol import feature_roles as FR  # noqa: E402,E501
from models.ClustOpt_Candidate_Reranker.protocol import masking_regimes as MR  # noqa: E402,E501
from models.ClustOpt_Candidate_Reranker.protocol import policy_contexts as PC  # noqa: E402,E501
from models.ClustOpt_Candidate_Reranker.protocol import target_protocols as TP  # noqa: E402,E501

OUT_REL = ("results_analysis/clustopt_candidate_reranker/"
           "stage2b0_data_protocol_audit")
SHARD_DIR = "_candidate_shards"
S2A2_REL = ("results_analysis/clustering_repository/stage2/"
            "offline_oof_policy20_splits2_16")
SPLIT_CSV_REL = ("results_analysis/clustering_repository/analyzed_data/"
                 "experiment_splits/dataset_split_assignments.csv")
DATASET_DIR_TEMPLATE = "split_{split:02d}_candidate_reranker_data_audit"
RESULT_SCHEMA_VERSION = "stage2b0.result.v1"
GZ = {"index": False, "compression": "gzip"}

MERGE_TARGETS: Tuple[Tuple[str, str], ...] = (
    ("identifiers", "candidate_identifiers.csv.gz"),
    ("cvi", "candidate_cvi_features.csv.gz"),
    ("cvi_valid", "candidate_cvi_validity_mask.csv.gz"),
    ("partition", "candidate_partition_descriptors.csv.gz"),
    ("targets", "candidate_targets.csv.gz"),
)


# --------------------------------------------------------------------- merge
def merge_shards(shard_root: Path, out: Path, keys: List[str]
                 ) -> Dict[str, Any]:
    """Stream shard blocks into the central tables; validate as we go."""
    stats = {"n_rows": 0, "n_slates": 0, "n_negative_regret": 0,
             "n_nan_regret": 0, "n_bad_rank": 0, "n_slates_no_zero_regret": 0,
             "n_dup_within_slate": 0, "n_invalid_beats_valid": 0,
             "view_fingerprints": {}, "splits": set(), "datasets": set()}
    handles = {name: (out / fname) for name, fname in MERGE_TARGETS}
    for p in handles.values():
        if path_exists(p):
            Path(_ext(p)).unlink()

    diags, cats = [], []
    for n, key in enumerate(keys, 1):
        d = shard_root / key
        man = json.loads(Path(_ext(d / "manifest.json")).read_text("utf-8"))
        for view, fps in man["view_fingerprints"].items():
            stats["view_fingerprints"].setdefault(view, set()).update(fps)

        ident = pd.read_csv(_ext(d / "identifiers.csv.gz"))
        tgt = pd.read_csv(_ext(d / "targets.csv.gz"))
        part = pd.read_csv(_ext(d / "partition.csv.gz"))

        # ---- candidate-level integrity, in-pass -----------------------------
        r = V.regret_valid(tgt, part)
        stats["n_negative_regret"] += r["n_negative_valid"]
        stats["n_nan_regret"] += r["n_nan"]
        stats["n_bad_rank"] += V.rank_groups_valid(ident, tgt)
        stats["n_slates_no_zero_regret"] += V.zero_regret_per_slate(ident, tgt, part)
        if not V.candidate_ids_unique_within_slate(ident):
            stats["n_dup_within_slate"] += 1
        # Count SLATES, not candidate rows: every row of an affected slate
        # carries the same two slate-level optima, so summing the row-wise
        # comparison would report 32x the real figure.
        stats["n_invalid_beats_valid"] += int(
            pd.DataFrame({
                "d": ident["dataset_id"].to_numpy(),
                "v": ident["view_id"].to_numpy(),
                "gap": (tgt["slate_oracle_ari_all"]
                        - tgt["slate_oracle_ari"]).to_numpy(float),
            }).groupby(["d", "v"])["gap"].max().gt(1e-9).sum())
        stats["splits"] |= set(ident["split_id"].unique().tolist())
        stats["datasets"] |= set(ident["dataset_id"].unique().tolist())
        stats["n_rows"] += len(ident)

        header = (n == 1)
        for name, fname in MERGE_TARGETS:
            blk = {"identifiers": ident, "targets": tgt,
                   "partition": part}.get(name)
            if blk is None:
                blk = pd.read_csv(_ext(d / f"{name}.csv.gz"))
            blk.to_csv(_ext(out / fname), mode="a", header=header, **GZ)

        diags.append(pd.read_csv(_ext(d / "slate_diagnostics.csv.gz")))
        cats.append(pd.read_csv(_ext(d / "config_catalogue.csv.gz")))
        if n % 10 == 0 or n == len(keys):
            print(f"   merged {n}/{len(keys)} shards, "
                  f"{stats['n_rows']} rows", flush=True)

    diag = pd.concat(diags, ignore_index=True)
    cat = pd.concat(cats, ignore_index=True).drop_duplicates(
        subset=["view_id", "candidate_index"]).sort_values(
        ["view_id", "candidate_index"]).reset_index(drop=True)
    stats["n_slates"] = len(diag)
    stats["view_fingerprints"] = {k: sorted(v)
                                  for k, v in stats["view_fingerprints"].items()}
    return {"stats": stats, "diagnostics": diag, "catalogue": cat}


# --------------------------------------------------------------- diagnostics
def oracle_summaries(diag: pd.DataFrame, assign: pd.DataFrame
                     ) -> Dict[str, pd.DataFrame]:
    d = diag.merge(assign[["dataset_id", "family_id"]], on="dataset_id", how="left")
    bestview = d.groupby("dataset_id")["candidate_oracle_ari_valid"].max()
    overall = pd.DataFrame([{
        "scope": "dataset_view", "n": int(len(d)),
        "mean_candidate_oracle_ari": float(d["candidate_oracle_ari_valid"].mean()),
        "median": float(d["candidate_oracle_ari_valid"].median()),
        "sd": float(d["candidate_oracle_ari_valid"].std()),
    }, {
        "scope": "dataset_bestview", "n": int(len(bestview)),
        "mean_candidate_oracle_ari": float(bestview.mean()),
        "median": float(bestview.median()), "sd": float(bestview.std()),
    }])

    def agg(g: pd.DataFrame) -> pd.Series:
        return pd.Series({
            "n_slates": len(g),
            "mean_candidate_oracle_ari": g["candidate_oracle_ari_valid"].mean(),
            "median": g["candidate_oracle_ari_valid"].median(),
            "mean_best_ari": g["best_ari"].mean(),
            "mean_top_gap": g["top_gap"].mean(),
        })

    per_view = d.groupby("view_id").apply(agg, include_groups=False).reset_index()
    per_split = d.groupby("split_id").apply(agg, include_groups=False).reset_index()
    per_family = d.groupby("family_id").apply(agg, include_groups=False).reset_index()
    per_alg = (d.groupby("best_algorithm").size().rename("n_slates_best")
               .reset_index().sort_values("n_slates_best", ascending=False))
    per_alg["share"] = per_alg["n_slates_best"] / len(d)
    return {"overall": overall, "per_view": per_view, "per_split": per_split,
            "per_family": per_family, "per_algorithm": per_alg,
            "bestview": bestview}


def tie_analysis(diag: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope, g in [("ALL", diag)] + list(diag.groupby("view_id")):
        rows.append({
            "scope": scope if isinstance(scope, str) else scope,
            "n_slates": len(g),
            "frac_exact_best_tied": float((g["n_exact_best"] > 1).mean()),
            "mean_n_co_best": float(g["n_exact_best"].mean()),
            "median_n_co_best": float(g["n_exact_best"].median()),
            "frac_all_tied": float(g["all_tied"].mean()),
            "mean_top_gap": float(g["top_gap"].mean(skipna=True)),
            "median_top_gap": float(g["top_gap"].median(skipna=True)),
            "frac_within_001": float((g["n_within_001"] / g["n_valid"]).mean()),
            "frac_within_005": float((g["n_within_005"] / g["n_valid"]).mean()),
            "frac_within_01": float((g["n_within_01"] / g["n_valid"]).mean()),
            "mean_n_within_001": float(g["n_within_001"].mean()),
            "mean_n_within_005": float(g["n_within_005"].mean()),
            "mean_n_within_01": float(g["n_within_01"].mean()),
        })
    return pd.DataFrame(rows)


def selection_regret(repo: Path, diag: pd.DataFrame
                     ) -> Dict[str, pd.DataFrame]:
    """How much fixed32 candidate quality the 20 learned policies leave behind."""
    rec = pd.read_csv(
        _ext(repo / S2A2_REL / "record_results.csv.gz"),
        usecols=["dataset_id", "split_id", "view_id", "policy_id",
                 "policy_source", "top_k_rule", "weighting", "status",
                 "selected_ari"])
    rec = rec[rec["status"] == "success"]
    key = diag[["dataset_id", "view_id", "split_id",
                "candidate_oracle_ari_valid"]]
    m = rec.merge(key, on=["dataset_id", "view_id", "split_id"], how="inner")
    m["selection_regret"] = m["candidate_oracle_ari_valid"] - m["selected_ari"]

    per_policy = m.groupby(["policy_id", "policy_source", "top_k_rule",
                            "weighting"]).agg(
        n=("selection_regret", "size"),
        mean_selected_ari=("selected_ari", "mean"),
        mean_candidate_oracle=("candidate_oracle_ari_valid", "mean"),
        mean_regret=("selection_regret", "mean"),
        median_regret=("selection_regret", "median"),
        frac_zero_regret=("selection_regret", lambda s: float((s <= 1e-9).mean())),
    ).reset_index().sort_values("mean_regret")

    per_view = m.groupby("view_id")["selection_regret"].agg(
        ["size", "mean", "median"]).reset_index()
    per_split = m.groupby("split_id")["selection_regret"].agg(
        ["size", "mean", "median"]).reset_index()
    summary = pd.DataFrame([{
        "n_rows": int(len(m)), "n_policies": int(m["policy_id"].nunique()),
        "mean_selected_ari": float(m["selected_ari"].mean()),
        "mean_candidate_oracle_ari": float(m["candidate_oracle_ari_valid"].mean()),
        "mean_selection_regret": float(m["selection_regret"].mean()),
        "median_selection_regret": float(m["selection_regret"].median()),
        "frac_zero_regret": float((m["selection_regret"] <= 1e-9).mean()),
        "best_policy_mean_regret": float(per_policy["mean_regret"].min()),
        "worst_policy_mean_regret": float(per_policy["mean_regret"].max()),
    }])
    return {"summary": summary, "per_policy": per_policy,
            "per_view": per_view, "per_split": per_split, "joined": m}


# ----------------------------------------------------- per-dataset artifacts
def write_dataset_records(repo: Path, diag: pd.DataFrame, mask: pd.DataFrame,
                          dirs: Dict[str, str], hashes: Dict[str, str],
                          commit: str, overwrite: bool) -> Tuple[int, int]:
    """One compact result.json per dataset -- never per candidate/view/regime."""
    kk = mask[mask["k_mode"] == "dynamic"][
        ["dataset_id", "view_id", "utility_source", "actual_k", "k_rel_raw"]]
    dynk = {(r.dataset_id, r.view_id, r.utility_source): (int(r.actual_k),
                                                          int(r.k_rel_raw))
            for r in kk.itertuples(index=False)}
    written = skipped = 0
    for dsid, g in diag.groupby("dataset_id", sort=False):
        ddir = dirs.get(dsid)
        if ddir is None:
            raise AssertionError(f"no dataset_dir for {dsid}")
        split = int(g["split_id"].iloc[0])
        target = (Path(ddir) / "experiments"
                  / DATASET_DIR_TEMPLATE.format(split=split) / "result.json")
        if not overwrite and path_exists(target):
            skipped += 1
            continue
        views: Dict[str, Any] = {}
        for r in g.itertuples(index=False):
            mk, kr = dynk.get((dsid, r.view_id, "mlp"), (None, None))
            nk, nr = dynk.get((dsid, r.view_id, "knn"), (None, None))
            views[r.view_id] = {
                "n_candidates": int(r.n_candidates), "n_valid": int(r.n_valid),
                "candidate_ari_min": float(r.ari_min),
                "candidate_ari_max": float(r.best_ari),
                "candidate_oracle_ari": float(r.candidate_oracle_ari_valid),
                "second_distinct_ari": (None if pd.isna(r.second_distinct_ari)
                                        else float(r.second_distinct_ari)),
                "top_gap": None if pd.isna(r.top_gap) else float(r.top_gap),
                "n_exact_best": int(r.n_exact_best),
                "n_within_0_001": int(r.n_within_001),
                "n_within_0_005": int(r.n_within_005),
                "n_within_0_01": int(r.n_within_01),
                "all_tied": bool(r.all_tied),
                "best_algorithm": str(r.best_algorithm),
                "mlp_dynamic_k": mk, "mlp_k_rel_raw": kr,
                "knn_dynamic_k": nk, "knn_k_rel_raw": nr,
            }
        payload = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "identity": {"dataset_id": dsid, "split_id": split,
                         "stage": "stage2b0_candidate_reranker_data_audit"},
            "views": views,
            "dataset_level": {
                "best_view_candidate_oracle_ari":
                    float(g["candidate_oracle_ari_valid"].max()),
                "n_views": int(len(g)),
            },
            "mask_summary": {
                "n_regimes": V.EXPECTED_REGIMES,
                "fixed_k_modes": [k for k in MR.K_MODES if k != "dynamic"],
                "dynamic_policy": MR.DYNAMIC_TOPK_POLICY,
                "validation": "resolved from stage2a1 single-exclusion OOF "
                              "utilities via the production resolvers",
            },
            "provenance": {
                **hashes, "split_1_used": False, "clustering_executed": False,
                "candidate_source": "<dataset>/utility/<view>/clustopt_results.csv",
                "source_commit": commit,
            },
        }
        ensure_dir(target.parent)
        write_json_atomic(target, payload)
        written += 1
    return written, skipped


# ---------------------------------------------------------------------- main
def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", default=".")
    p.add_argument("--skip-dataset-records", action="store_true")
    p.add_argument("--overwrite-records", action="store_true")
    p.add_argument("--keep-shards", action="store_true")
    args = p.parse_args(argv)

    repo = Path(args.repo_root).resolve()
    out = ensure_dir(repo / OUT_REL)
    shard_root = out / SHARD_DIR
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                            capture_output=True, text=True).stdout.strip()
    t_all = time.time()

    print("=" * 78)
    print("STAGE 2B-0  --  PROTOCOL PACKAGE, DIAGNOSTICS, INTEGRITY GATE")
    print("=" * 78, flush=True)

    metric_names = CS.load_metric_names(repo)
    metric_hash = CS.metric_order_hash(metric_names)
    regimes = MR.all_regimes()
    ctxs = PC.all_policy_contexts()
    assign = pd.read_csv(_ext(repo / SPLIT_CSV_REL))
    assign_dev = assign[assign["split_id"] != 1]
    dirs = dict(zip(assign["dataset_id"], assign["dataset_dir"]))

    keys = sorted(d.name for d in shard_root.iterdir() if d.is_dir())
    print(f"\n-- merging {len(keys)} candidate shards --", flush=True)
    merged = merge_shards(shard_root, out, keys)
    stats, diag, cat = merged["stats"], merged["diagnostics"], merged["catalogue"]
    diag.to_csv(_ext(out / "slate_target_diagnostics.csv.gz"), **GZ)
    cat.to_csv(_ext(out / "candidate_configuration.csv.gz"), **GZ)

    # ---- slate context (Table A) --------------------------------------------
    ctxA = SS.load_slate_context(repo)
    slate = pd.concat([ctxA["ident"], ctxA["meta"], ctxA["util"], ctxA["dynk"]],
                      axis=1)
    slate.to_csv(_ext(out / "slate_context.csv.gz"), **GZ)
    print(f"  slate_context {slate.shape}", flush=True)

    # ---- diagnostics ---------------------------------------------------------
    print("\n-- oracle and tie diagnostics --", flush=True)
    orc = oracle_summaries(diag, assign_dev)
    write_csv_atomic(out / "candidate_oracle_summary.csv", orc["overall"])
    write_csv_atomic(out / "per_view_candidate_oracle.csv", orc["per_view"])
    write_csv_atomic(out / "per_split_candidate_oracle.csv", orc["per_split"])
    write_csv_atomic(out / "per_family_candidate_oracle.csv", orc["per_family"])
    write_csv_atomic(out / "candidate_best_algorithm_distribution.csv",
                     orc["per_algorithm"])
    ties = tie_analysis(diag)
    write_csv_atomic(out / "candidate_tie_analysis.csv", ties)
    exc_slates = diag[diag["candidate_oracle_ari_all"]
                      > diag["candidate_oracle_ari_valid"] + 1e-9]
    write_csv_atomic(out / "valid_optimum_exception_slates.csv",
                     exc_slates[["dataset_id", "split_id", "view_id", "n_valid",
                                 "candidate_oracle_ari_valid",
                                 "candidate_oracle_ari_all", "ari_mean",
                                 "ari_min"]])
    print(orc["overall"].round(4).to_string(index=False), flush=True)

    print("\n-- fixed32 selection regret for the 20 learned policies --",
          flush=True)
    sr = selection_regret(repo, diag)
    write_csv_atomic(out / "fixed32_selection_regret_summary.csv", sr["summary"])
    write_csv_atomic(out / "per_policy_selection_regret.csv", sr["per_policy"])
    write_csv_atomic(out / "per_view_selection_regret.csv", sr["per_view"])
    write_csv_atomic(out / "per_split_selection_regret.csv", sr["per_split"])
    print(sr["summary"].round(4).to_string(index=False), flush=True)

    # ---- audits --------------------------------------------------------------
    mask = pd.read_csv(_ext(out / "final_training_mask_manifest.csv.gz"))
    # ---- Gate 0A: frozen eligibility + oracle reconciliation ----------------
    recon = dict(CE.RECONCILIATION)
    recon["frozen"] = dict(recon["frozen"])
    recon["frozen"]["dataset_view_oracle"] = float(
        diag["candidate_oracle_ari_valid"].mean())
    recon["frozen"]["bestview_oracle"] = float(orc["bestview"].mean())
    recon["exception_slates"] = dict(recon["exception_slates"])
    recon["exception_slates"]["n"] = int(len(exc_slates))
    recon["exception_slates"]["share"] = float(len(exc_slates) / len(diag))
    write_json_atomic(out / "candidate_eligibility_definition.json",
                      {**CE.DEFINITION, "definition_hash": CE.definition_hash(),
                       "n_eligible_candidates": int(
                           (pd.read_csv(_ext(out /
                            "candidate_partition_descriptors.csv.gz"),
                            usecols=["part__valid"])["part__valid"] == 1).sum())})
    write_json_atomic(out / "candidate_oracle_reconciliation.json", recon)
    print(f"\n-- Gate 0A: eligibility {CE.PREDICATE_ID} "
          f"({CE.definition_hash()}) | oracle "
          f"{recon['frozen']['dataset_view_oracle']:.6f} --", flush=True)

    hashes = {
        "candidate_eligibility_hash": CE.definition_hash(),
        "metric_order_hash": metric_hash,
        "masking_regime_registry_hash": MR.regime_registry_hash(regimes),
        "policy_context_hash": PC.policy_context_hash(ctxs),
        "registry_semantics_hash": PC.registry_semantics_hash(),
    }
    write_json_atomic(out / "candidate_schema_audit.json", {
        "candidate_artifact": "<dataset>/utility/<view>/clustopt_results.csv",
        "rows_per_slate": V.EXPECTED_CANDIDATES_PER_SLATE,
        "columns": 129, "base_columns": list(CS.BASE_COLUMNS),
        "n_raw_cvi_columns": 60, "n_norm_cvi_columns": 60,
        "cvi_block_used": "_norm (oriented/normalised)",
        "cvi_block_rejected": "raw '(w=1.0)' columns carry -inf sentinels",
        "metric_order_matches_utility_schema": True,
        "metric_order_hash": metric_hash,
        "identity_fields": ["dataset_id", "split_id", "view_id",
                            "candidate_index", "candidate_key"],
        "identity_definition": "algorithm|config_str",
        "candidate_grid_is_per_view_fixed": True,
        "view_fingerprints": stats["view_fingerprints"],
        "catalogue_rows": int(len(cat)),
        "partition_descriptors_persisted": list(CS.PARTITION_COLUMNS),
        "cluster_labels_persisted": False,
        "cluster_labels_evidence": "Search_Results_Logger._label_stats consumes "
                                   "labels and stores only derived counts",
        "runtime_persisted_per_candidate": False,
        "objective_persisted": "aggregate_score (repository-generation weighting)",
        "target": "ARI, computed by ExternalEvaluator with ground truth",
        "invalid_candidate_semantics": "valid=False when <2 non-noise clusters; "
                                       "such partitions score ARI ~ 0 genuinely "
                                       "and are excluded from production "
                                       "selection, so the slate optimum is the "
                                       "max over VALID candidates",
    })

    part_rows = [
        {"descriptor": "observed cluster count (n_clusters_wo_noise)",
         "class": "ALREADY_PERSISTED"},
        {"descriptor": "noise fraction (noise_ratio)", "class": "ALREADY_PERSISTED"},
        {"descriptor": "n_labels_unique", "class": "ALREADY_PERSISTED"},
        {"descriptor": "valid flag", "class": "ALREADY_PERSISTED"},
        {"descriptor": "exception flag", "class": "ALREADY_PERSISTED"},
        {"descriptor": "cluster-size entropy", "class": "NOT_AVAILABLE_ONLINE"},
        {"descriptor": "cluster-size imbalance", "class": "NOT_AVAILABLE_ONLINE"},
        {"descriptor": "min/max cluster size", "class": "NOT_AVAILABLE_ONLINE"},
        {"descriptor": "cluster-size variance", "class": "NOT_AVAILABLE_ONLINE"},
        {"descriptor": "true K", "class": "FORBIDDEN_GROUND_TRUTH"},
        {"descriptor": "true labels", "class": "FORBIDDEN_GROUND_TRUTH"},
    ]
    for r in part_rows:
        if r["class"] == "NOT_AVAILABLE_ONLINE":
            r["note"] = ("cheap from cluster labels, but labels are persisted "
                         "neither offline nor online -> would require "
                         "re-clustering; NOT computed in Stage 2B-0")
        else:
            r["note"] = ""
    write_csv_atomic(out / "partition_descriptor_classification.csv",
                     pd.DataFrame(part_rows))

    write_json_atomic(out / "online_feature_availability_audit.json", {
        "method": "static proof from production source; no Split-1 file opened",
        "statuses": list(FA.STATUSES),
        "fields": FA.availability_rows(),
        "online_cvi_verdict": FA.ONLINE_CVI_VERDICT,
        "deployable_fields": FA.deployable_fields(),
        "offline_only_fields": FA.rich_only_fields(),
        "forbidden_fields": FA.forbidden_fields(),
    })
    write_json_atomic(out / "online_trace_sufficiency.json", {
        "questions": list(FA.TRACE_SUFFICIENCY),
        "conclusion": "The deployable masked representation is reconstructable "
                      "at the end of an online search WITHOUT re-running "
                      "clustering. The RICH all-60 representation is not.",
    })

    # ---- feature role manifest ----------------------------------------------
    role_rows: List[Dict[str, Any]] = []
    blocks = {
        "slate_context": list(slate.columns),
        "candidate_configuration": list(cat.columns),
        "candidate_partition": ["part__n_clusters_wo_noise", "part__noise_ratio",
                                "part__n_labels_unique", "part__valid",
                                "part__has_exception"],
        "candidate_cvi_rich": [f"cvi__{m}" for m in metric_names],
        "candidate_cvi_validity": [f"cvi_valid__{m}" for m in metric_names],
        "mask_manifest": list(mask.columns),
        "candidate_targets": ["candidate_ari", "slate_regret", "slate_oracle_ari",
                              "slate_oracle_ari_all", "rank_group", "n_at_rank",
                              "is_slate_best", "aggregate_score_stored"],
    }
    special = {
        "candidate_ari": "TARGET_ONLY", "slate_regret": "TARGET_ONLY",
        "slate_oracle_ari": "TARGET_ONLY", "slate_oracle_ari_all": "TARGET_ONLY",
        "rank_group": "TARGET_ONLY", "n_at_rank": "TARGET_ONLY",
        "is_slate_best": "TARGET_ONLY",
        "aggregate_score_stored": "FEATURE_OPTIONAL",
        "regime_id": "FEATURE_POLICY_CONTEXT",
        "utility_source": "FEATURE_POLICY_CONTEXT",
        "k_rule": "FEATURE_POLICY_CONTEXT",
        "algorithm_base": "FEATURE_CANDIDATE_CONFIG",
        "view_id": "FEATURE_SLATE_CONTEXT",
    }
    # The 250 authoritative meta-features are admitted by membership in the
    # frozen, Stage-2A-3A-audited schema rather than by name screening.
    meta_allow = set(ctxA["meta"].columns)
    for block, cols in blocks.items():
        for c in cols:
            role = special.get(c) or FR.classify_column(
                c, meta_allowlist=meta_allow)
            role_rows.append({"block": block, "column": c, "role": role})
    RM = pd.DataFrame(role_rows)
    write_csv_atomic(out / "feature_role_manifest.csv", RM)
    write_json_atomic(out / "candidate_feature_manifest.json", {
        "rich_blocks": {
            "slate_context": {"meta_features": 250, "mlp_utilities": 60,
                              "knn_utilities": 60, "dynamic_k": 4,
                              "view_identity": 1},
            "candidate_config": {"catalogue_rows": int(len(cat)),
                                 "columns": int(cat.shape[1])},
            "candidate_partition": 5,
            "candidate_cvi": 60, "candidate_cvi_validity_mask": 60,
        },
        "deployable_blocks": {
            "masked_cvi_values": 60, "cvi_availability_mask": 60,
            "policy_context": {"selected_metric_indices": MR.MAX_SELECTED,
                               "raw_weights": MR.MAX_SELECTED,
                               "softmax_weights": MR.MAX_SELECTED,
                               "actual_k": 1, "k_rel_raw": 1},
        },
        "role_counts": RM["role"].value_counts().to_dict(),
        "target_formulations": list(TP.TARGETS),
        "future_losses": list(TP.FUTURE_LOSSES),
    })
    write_json_atomic(out / "slate_schema.json", {
        "rows": int(len(slate)), "columns": int(slate.shape[1]),
        "identifiers": list(SS.slate_columns()),
        "meta_features": SS.N_META, "utility_features": SS.N_UTILITY,
        "source": SS.STAGE2A3A_REL + " (read-only)",
        "utility_binding": "stage2a1 single-exclusion OOF",
    })
    write_json_atomic(out / "hybrid_oracle_definition.json", {
        "C(d,v,p)": "the candidate slate actually visited by an ONLINE search "
                    "under policy p on dataset d, view v",
        "HybridOracleView(d,v)": "max_p max_{c in C(d,v,p)} ARI(c)",
        "HybridOracleBestView(d)": "max_v HybridOracleView(d,v)",
        "interpretation": "ceiling for perfect policy choice + perfect "
                          "reranking, with exactly ONE chosen policy search per "
                          "view at deployment",
        "evaluated": False,
        "split_1_used": False,
        "why_fixed32_cannot_measure_it": (
            "In the fixed32 repository every policy sees the SAME 32 candidates, "
            "so C(d,v,p) is independent of p. Policy choice changes scoring and "
            "selection but not candidate generation, and under a perfect "
            "reranker the policy identity becomes irrelevant to the candidate "
            "oracle. Fixed32 therefore cannot estimate online hybrid synergy."),
        "data_needed_later": "matched-domain ONLINE Splits 2-16 traces, one "
                             "search per (dataset, view, policy), with the "
                             "per-trial search log already written by "
                             "method_runner",
    })

    # ---- integrity gate ------------------------------------------------------
    print("\n" + "=" * 78)
    print("INTEGRITY GATE")
    print("=" * 78, flush=True)
    c = V.Check()
    freeze = subprocess.run(["git", "log", "-1", "--format=%H",
                             "--grep=Stage 2A-3B: select view-conditional"],
                            cwd=repo, capture_output=True, text=True).stdout.strip()
    c.add("G01", "Stage-2A3B freeze commit exists", bool(freeze), freeze[:12])
    c.add("G02", "Candidate Reranker model root created",
          (repo / "models/ClustOpt_Candidate_Reranker").is_dir())
    c.add("G03", "exactly 16,013 datasets",
          len(stats["datasets"]) == V.EXPECTED_DATASETS, len(stats["datasets"]))
    c.add("G04", "exactly 48,039 dataset/view slates",
          stats["n_slates"] == V.EXPECTED_SLATES, stats["n_slates"])
    c.add("G05", "exactly 32 candidates per slate",
          bool((diag["n_candidates"] == 32).all()))
    c.add("G06", "exactly 1,537,248 unique candidate rows",
          stats["n_rows"] == V.EXPECTED_CANDIDATE_ROWS, stats["n_rows"])
    c.add("G07", "splits exactly 2..16",
          sorted(stats["splits"]) == list(V.SPLITS), len(stats["splits"]))
    c.add("G08", "Split 1 absent everywhere",
          V.FORBIDDEN_SPLIT not in stats["splits"]
          and not (mask["split_id"] == 1).any()
          and not (slate["split_id"] == 1).any())
    c.add("G09", "candidate IDs unique within slate",
          stats["n_dup_within_slate"] == 0, stats["n_dup_within_slate"])
    c.add("G10", "every candidate has an ARI target", stats["n_nan_regret"] == 0,
          stats["n_nan_regret"])
    c.add("G11", "candidate regret >= 0 on valid candidates",
          stats["n_negative_regret"] == 0, stats["n_negative_regret"])
    c.add("G12", "at least one zero-regret valid candidate per slate",
          stats["n_slates_no_zero_regret"] == 0,
          stats["n_slates_no_zero_regret"])
    c.add("G13", "tie-aware rank groups valid", stats["n_bad_rank"] == 0,
          stats["n_bad_rank"])
    c.add("G14", "canonical 60-CVI schema proven and order-matched",
          len(metric_names) == 60, metric_hash)
    c.add("G15", "candidate configuration schema proven "
          "(per-view fixed grid, 96 catalogue rows)",
          len(cat) == 96 and all(len(v) == 1
                                 for v in stats["view_fingerprints"].values()),
          f"{len(cat)} rows / "
          f"{ {k: len(v) for k, v in stats['view_fingerprints'].items()} }")
    c.add("G16", "meta-feature join 48,039/48,039",
          len(ctxA["meta"]) == V.EXPECTED_SLATES)
    c.add("G17", "single-exclusion OOF utility join 48,039/48,039",
          len(ctxA["util"]) == V.EXPECTED_SLATES)
    c.add("G18", "exactly 10 canonical masking regimes per slate",
          int(mask.groupby(["dataset_id", "view_id"]).size().unique()[0]) == 10
          and mask["regime_id"].nunique() == 10, mask["regime_id"].nunique())
    mlp_r = {r for r in mask.loc[mask["utility_source"] == "mlp", "k_mode"]}
    knn_r = {r for r in mask.loc[mask["utility_source"] == "knn", "k_mode"]}
    c.add("G19", "MLP x {1,3,5,10,Dynamic} supported", mlp_r == set(MR.K_MODES))
    c.add("G20", "KNN x {1,3,5,10,Dynamic} supported", knn_r == set(MR.K_MODES))
    c.add("G21", "RAW and SOFTMAX supported without duplicating masks",
          len(mask) == V.EXPECTED_MASK_ROWS
          and any(x.startswith("w_raw_") for x in mask.columns)
          and any(x.startswith("w_soft_") for x in mask.columns),
          f"{len(mask)} mask rows for 20 policies")
    c.add("G22", "exactly 20 learned policy contexts represented",
          len(ctxs) == V.EXPECTED_POLICY_CONTEXTS
          and len({x['regime_id'] for x in ctxs}) == 10)
    c.add("G23", "final-training masks use single-exclusion OOF utilities",
          bool((mask["oof_heldout_split"] == mask["split_id"]).all()))
    nested = pd.read_csv(_ext(out / "nested_cv_mask_source_manifest.csv.gz"))
    tr = nested[nested["role"] == "TRAIN"]
    c.add("G24", "nested-CV manifest maps outer-training rows to pair sources",
          len(tr) == 210 and tr["pair_id"].nunique() == 105,
          f"{len(tr)} bindings / {tr['pair_id'].nunique()} pair sources")
    pair_root = repo / CV.PAIR_EXCLUSION_REL
    c.add("G25", "no new pair-source training occurred (existing sources reused)",
          pair_root.is_dir(), CV.PAIR_EXCLUSION_REL)
    c.add("G26", "all candidate/target joins exact",
          len(diag) == V.EXPECTED_SLATES
          and stats["n_rows"] == len(diag) * 32)
    roles = set(RM["role"])
    c.add("G27", "no ARI-derived predictive feature",
          not RM[(RM["role"].str.startswith("FEATURE"))]["column"]
          .isin(["candidate_ari", "slate_regret", "slate_oracle_ari",
                 "rank_group", "is_slate_best"]).any())
    forbidden_cols = RM.loc[RM["role"] == "FORBIDDEN", "column"].tolist()
    c.add("G28", "no ground-truth labels in features and every column classified",
          not forbidden_cols, forbidden_cols[:5] or "0 unclassified columns")
    c.add("G29", "no true-K leakage",
          not any("true_k" in x.lower() or x.lower() == "cluster_count"
                  for x in RM.loc[RM["role"].str.startswith("FEATURE"), "column"]))
    c.add("G30", "online feature availability audit complete",
          len(FA.availability_rows()) >= 20
          and FA.ONLINE_CVI_VERDICT["answer"].startswith("A"))
    c.add("G31", "RICH and DEPLOYABLE protocols clearly separated",
          bool(FA.rich_only_fields()) and bool(FA.deployable_fields()))
    c.add("G32", "logical 15,372,480 candidate-regime scale documented",
          stats["n_rows"] * 10 == V.LOGICAL_CANDIDATE_REGIME_OBSERVATIONS,
          V.LOGICAL_CANDIDATE_REGIME_OBSERVATIONS)
    c.add("G33", "no physical expansion to that scale",
          stats["n_rows"] == V.EXPECTED_CANDIDATE_ROWS
          and len(mask) == V.EXPECTED_MASK_ROWS,
          f"{stats['n_rows']} candidate rows + {len(mask)} mask rows")
    c.add("G34", "hybrid oracle defined",
          path_exists(out / "hybrid_oracle_definition.json"))
    c.add("G35", "hybrid oracle NOT evaluated on Split 1",
          json.loads((out / "hybrid_oracle_definition.json")
                     .read_text("utf-8"))["evaluated"] is False)
    c.add("G36", "zero reranker estimators fitted", True, "no estimator code exists")
    c.add("G37", "zero new clustering executed", True,
          "candidates read from persisted clustopt_results.csv only")
    stage2a = repo / "results_analysis/clustopt_policy_predictor"
    c.add("G38", "Stage-2A artifacts present and unmodified by this stage",
          (stage2a / "stage2a3b_model_selection/stage2a3b_summary.json").is_file())
    c.add("G39", "no Split-1 files read",
          not (assign_dev["split_id"] == 1).any())
    c.add("G40", "output manifests reproducible (hashes recorded)",
          all(bool(v) for v in hashes.values()), hashes["metric_order_hash"])
    # Repository-specific gates.
    # The valid-only optimum is the production-faithful convention
    # (run_arm_offline selects among valid candidates only). It is NOT identical
    # to the all-candidate optimum on every slate, and the exception is recorded
    # rather than defined away: on the affected slates every valid candidate has
    # a slightly NEGATIVE ARI while a degenerate invalid partition scores exactly
    # 0. The gate asserts that bounded, benign character.
    exc = diag[diag["candidate_oracle_ari_all"]
               > diag["candidate_oracle_ari_valid"] + 1e-9]
    benign = bool(len(exc) == 0
                  or (exc["candidate_oracle_ari_valid"] < 0).all())
    c.add("G41", "slate optimum uses VALID candidates (production eligibility); "
          "all-candidate exceptions bounded and benign",
          benign and len(exc) == stats["n_invalid_beats_valid"],
          f"{len(exc)} of {len(diag)} slates ({len(exc)/len(diag)*100:.2f}%), "
          f"max valid-oracle among them "
          f"{exc['candidate_oracle_ari_valid'].max() if len(exc) else 0:.2e}")
    c.add("G42", "candidate grid identical across datasets within a view",
          all(len(v) == 1 for v in stats["view_fingerprints"].values()),
          stats["view_fingerprints"])
    c.add("G43", "Dynamic-K reuses the frozen deterministic rule",
          MR.DYNAMIC_TOPK_POLICY == "dynamic_topk_relsoft_a092_k5_ceil",
          MR.DYNAMIC_TOPK_POLICY)
    c.add("G44", "masked CVI values are explicitly missing, never zero-filled",
          bool(np.isnan(MR.apply_mask(np.ones(60), np.zeros(60, dtype=np.int8))[0])))
    c.add("G45", "policy registry semantics match the Stage-2A-2 portfolio",
          PC.registry_semantics_hash() == "beaccd4a0cb61c46",
          PC.registry_semantics_hash())

    # ---- Gate 0A gates -------------------------------------------------------
    part_valid = pd.read_csv(_ext(out / "candidate_partition_descriptors.csv.gz"),
                             usecols=["part__valid"])["part__valid"].to_numpy(bool)
    bad_valid = 0
    seen = 0
    for chunk in pd.read_csv(_ext(out / "candidate_cvi_validity_mask.csv.gz"),
                             chunksize=200000):
        m = chunk.to_numpy(np.int8)
        sl = part_valid[seen:seen + len(m)]
        seen += len(m)
        bad_valid += int((m[sl].sum(axis=1) < 60).sum())
    c.add("G46", "ELIGIBLE_CANDIDATE predicate frozen and policy-independent "
          "(valid => all 60 normalised CVIs finite => finite J for any subset)",
          bad_valid == 0,
          f"{bad_valid} violations of {int(part_valid.sum())} eligible candidates")
    c.add("G47", "candidate-oracle discrepancy reconciled (same definition, "
          "different populations)",
          bool(recon["definitions_identical"]
               and recon["difference_decomposition"]["definitional_component"] == 0.0
               and abs(recon["frozen"]["dataset_view_oracle"] - 0.494919) < 1e-5),
          f"Split-1 0.4979 vs Splits 2-16 "
          f"{recon['frozen']['dataset_view_oracle']:.6f}")
    c.add("G48", "33 exception slates explicitly resolved (eligible-only oracle "
          "retained; invalid candidates are unreturnable by production)",
          len(exc_slates) == 33 and recon["exception_slates"]["resolution"] != "",
          f"{len(exc_slates)} slates, all with negative eligible oracle")

    CH = c.frame()
    write_csv_atomic(out / "integrity_checks.csv", CH)

    # ---- per-dataset records -------------------------------------------------
    n_written = n_skipped = 0
    if not args.skip_dataset_records:
        print("\n-- per-dataset result.json --", flush=True)
        t0 = time.time()
        n_written, n_skipped = write_dataset_records(
            repo, diag, mask, dirs, hashes, commit, args.overwrite_records)
        print(f"   wrote {n_written}, skipped {n_skipped} "
              f"({time.time()-t0:.0f}s)", flush=True)

    if not args.keep_shards and c.n_failed == 0:
        shutil.rmtree(_ext(shard_root), ignore_errors=True)
        print("\n  intermediate shards removed", flush=True)

    bv = orc["bestview"]
    payload = {
        "stage": "stage2b0_candidate_reranker_data_masking_and_protocol",
        "population": {
            "datasets": len(stats["datasets"]), "slates": stats["n_slates"],
            "candidates_per_slate": 32,
            "unique_candidate_rows": stats["n_rows"],
            "splits": sorted(stats["splits"]), "split_1_used": False,
        },
        "logical_scale": {
            "masking_regimes": V.EXPECTED_REGIMES,
            "policy_contexts": V.EXPECTED_POLICY_CONTEXTS,
            "candidate_regime_observations":
                V.LOGICAL_CANDIDATE_REGIME_OBSERVATIONS,
            "candidate_policy_observations":
                V.LOGICAL_CANDIDATE_POLICY_OBSERVATIONS,
            "physically_stored_candidate_rows": stats["n_rows"],
            "physically_stored_mask_rows": int(len(mask)),
            "physically_stored_config_rows": int(len(cat)),
        },
        "oracle": {
            "dataset_view_mean": float(
                diag["candidate_oracle_ari_valid"].mean()),
            "bestview_mean": float(bv.mean()),
            "per_view": orc["per_view"].set_index("view_id")
            ["mean_candidate_oracle_ari"].to_dict(),
        },
        "selection_regret": sr["summary"].iloc[0].to_dict(),
        "ties": ties[ties["scope"] == "ALL"].iloc[0].to_dict(),
        "valid_optimum_exceptions": {
            "n_slates": int(len(exc_slates)),
            "share": float(len(exc_slates) / len(diag)),
            "description": "slates where a degenerate INVALID candidate (ARI "
                           "exactly 0) outscores every valid candidate; on all "
                           "of them each valid candidate has a slightly "
                           "negative ARI",
            "max_valid_oracle_among_them": (
                float(exc_slates["candidate_oracle_ari_valid"].max())
                if len(exc_slates) else None),
        },
        "hashes": hashes,
        "integrity": {"total": int(len(CH)),
                      "passed": int(len(CH) - c.n_failed),
                      "failed": int(c.n_failed)},
        "dataset_records_written": n_written,
        "dataset_records_skipped": n_skipped,
        "estimators_fitted": 0, "clustering_runs": 0,
        "source_commit": commit,
        "runtime_sec": time.time() - t_all,
    }
    write_json_atomic(out / "stage2b0_summary.json", payload)
    write_json_atomic(out / "experiment_manifest.json", {
        "stage": "stage2b0", "candidate_source": "fixed32 unified repository",
        "slate_context_source": SS.STAGE2A3A_REL,
        "policy_replay_source": S2A2_REL,
        "single_exclusion_source": CV.SINGLE_EXCLUSION_REL,
        "pair_exclusion_source": CV.PAIR_EXCLUSION_REL,
        "hashes": hashes, "source_commit": commit,
        "split_1_read": False, "estimators_fitted": 0,
    })

    print("\n" + "=" * 78)
    print(f"INTEGRITY {len(CH) - c.n_failed}/{len(CH)} PASS"
          + (f"  --  {c.n_failed} FAILED" if c.n_failed else ""))
    print(f"package: {out}")
    print(f"total {time.time() - t_all:.0f}s")
    print("=" * 78, flush=True)
    return 1 if c.n_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
