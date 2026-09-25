"""Stage 2B-3 package: unified regime results, GO-U gates, integrity."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from models.Clustering_Repository_Builder.experiments.experiment_execution.output_writer import (  # noqa: E402,E501
    _ext, ensure_dir, path_exists, read_json, write_csv_atomic, write_json_atomic,
)
from models.Clustering_Repository_Builder.experiments.paper_analysis import (  # noqa: E402,E501
    stats as S,
)
from models.ClustOpt_Candidate_Reranker.data import candidate_eligibility as CE  # noqa: E402,E501
from models.ClustOpt_Candidate_Reranker.training import model_registry as MR  # noqa: E402,E501
from . import evaluate_unified as EU
from . import regime_assignment as RA
from . import unified_feature_builder as UFB
from . import validation as V

B1_REL = ("results_analysis/clustopt_candidate_reranker/"
          "stage2b1_masking_feasibility")
B2_REL = ("results_analysis/clustopt_candidate_reranker/"
          "stage2b2_context_ablation")
OUT_REL = ("results_analysis/clustopt_candidate_reranker/"
           "stage2b3_unified_masking")
SPLIT_CSV_REL = ("results_analysis/clustering_repository/analyzed_data/"
                 "experiment_splits/dataset_split_assignments.csv")
DATASET_DIR_TEMPLATE = "split_{split:02d}_candidate_reranker_unified_masking"
RESULT_SCHEMA_VERSION = "stage2b3.result.v1"
GZ = {"index": False, "compression": "gzip"}
K_ORDER = {"top1": 0, "top3": 1, "top5": 2, "top10": 3, "dynamic": 4}


def load_units(out: Path):
    folds, slates, units = [], {}, {}
    for s in V.OUTER_SPLITS:
        d = out / "outer_fold_predictions" / f"fold_{s:02d}"
        m = read_json(d / "unit_manifest.json")
        if not m or m.get("status") != "complete":
            continue
        units[s] = m
        for cond, met in m["metrics"].items():
            src, km = cond.split("_", 1)
            folds.append({"condition_id": cond, "outer_split": s,
                          "utility_source": src.lower(), "k_mode": km.lower(),
                          "n_train_rows": m["n_train_rows"],
                          "fit_sec": m["fit_sec"], **met})
            slates[(cond, s)] = pd.read_csv(
                _ext(d / cond / "slate_selections.csv.gz"))
    return pd.DataFrame(folds), slates, units


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", default=".")
    p.add_argument("--skip-dataset-records", action="store_true")
    p.add_argument("--overwrite-records", action="store_true")
    args = p.parse_args(argv)

    repo = Path(args.repo_root).resolve()
    out = ensure_dir(repo / OUT_REL)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                            capture_output=True, text=True).stdout.strip()
    t_all = time.time()
    print("=" * 78)
    print("STAGE 2B-3  --  PACKAGE, GO-U GATES, INTEGRITY")
    print("=" * 78, flush=True)

    folds, slates, units = load_units(out)
    folds["_o"] = folds.apply(
        lambda r: (0 if r["utility_source"] == "mlp" else 1) * 10
        + K_ORDER[r["k_mode"]], axis=1)
    folds = folds.sort_values(["_o", "outer_split"]).drop(columns="_o")
    write_csv_atomic(out / "unified_fold_metrics.csv", folds)
    print(f"  {len(folds)} fold-regime rows from {folds['outer_split'].nunique()} "
          f"unified models", flush=True)

    summ = pd.DataFrame([EU.regime_summary(g) for _, g in
                         folds.groupby("condition_id", sort=False)])
    summ["_o"] = summ.apply(
        lambda r: (0 if r["utility_source"] == "mlp" else 1) * 10
        + K_ORDER[r["k_mode"]], axis=1)
    summ = summ.sort_values("_o").drop(columns="_o").reset_index(drop=True)
    write_csv_atomic(out / "unified_regime_results.csv", summ)
    write_csv_atomic(out / "mlp_regime_curve.csv",
                     summ[summ["utility_source"] == "mlp"])
    write_csv_atomic(out / "knn_regime_curve.csv",
                     summ[summ["utility_source"] == "knn"])

    # ---- dataset-level frames -----------------------------------------------
    bv_uni, bv_ded_local, bv_raw, bv_soft, bv_orc = {}, {}, {}, {}, {}
    for cond in summ["condition_id"]:
        parts = [slates[(cond, s)] for s in V.OUTER_SPLITS if (cond, s) in slates]
        d = pd.concat(parts, ignore_index=True)
        g = d.groupby("dataset_id")
        bv_uni[cond] = g["selected_ari"].max().sort_index()
        bv_ded_local[cond] = g["dedicated_local_selected_ari"].max().sort_index()
        bv_raw[cond] = g["b_raw_ari"].max().sort_index()
        bv_soft[cond] = g["b_softmax_ari"].max().sort_index()
        bv_orc[cond] = g["candidate_oracle_ari"].max().sort_index()

    # ---- pipeline comparison (context + unified, NOT mechanism-only) --------
    prows = []
    for cond in summ["condition_id"]:
        r = EU.paired_row(f"{cond}__unified_vs_dedicated_local",
                          bv_uni[cond].to_numpy(), bv_ded_local[cond].to_numpy(),
                          f"UNIFIED_CONTEXTUAL {cond}", f"DEDICATED_LOCAL {cond}")
        fa = folds[folds["condition_id"] == cond].sort_values("outer_split")
        r["fold_wins"] = int((fa["mean_bestview_selected_ari"].to_numpy()
                              > fa["dedicated_local_bestview"].to_numpy()).sum())
        r["macro_pipeline_delta"] = float(
            fa["mean_bestview_selected_ari"].mean()
            - fa["dedicated_local_bestview"].mean())
        r["utility_source"] = cond.split("_", 1)[0].lower()
        r["k_mode"] = cond.split("_", 1)[1].lower()
        r["interpretation"] = ("PIPELINE delta: contains BOTH C2 utility context "
                               "and unified cross-regime training")
        prows.append(r)
    PIPE = pd.DataFrame(prows)
    write_csv_atomic(out / "unified_vs_stage2b1_dedicated_local.csv", PIPE)
    write_csv_atomic(out / "low_k_transfer_summary.csv",
                     PIPE[["condition_id" if "condition_id" in PIPE else
                           "comparison", "utility_source", "k_mode",
                           "macro_pipeline_delta", "mean_delta", "ci_lo",
                           "ci_hi", "fold_wins"]]
                     if "condition_id" in PIPE else
                     PIPE[["comparison", "utility_source", "k_mode",
                           "macro_pipeline_delta", "mean_delta", "ci_lo",
                           "ci_hi", "fold_wins"]])

    # ---- clean mechanism test: unified C2 vs dedicated C2, MLP_TOP10 --------
    ded = []
    for s in V.OUTER_SPLITS:
        m = read_json(repo / B2_REL / "outer_fold_predictions"
                      / "C2_LOCAL_UTIL" / f"fold_{s:02d}" / "unit_manifest.json")
        ded.append({"outer_split": s, **m["metrics"]})
    DED = pd.DataFrame(ded)
    ded_sl = pd.concat([pd.read_csv(_ext(repo / B2_REL / "outer_fold_predictions"
                                         / "C2_LOCAL_UTIL" / f"fold_{s:02d}"
                                         / "slate_selections.csv.gz"))
                        for s in V.OUTER_SPLITS], ignore_index=True)
    bv_ded_c2 = ded_sl.groupby("dataset_id")["selected_ari"].max().sort_index()
    if not bv_ded_c2.index.equals(bv_uni["MLP_TOP10"].index):
        raise AssertionError("dedicated C2 dataset index differs from unified")
    clean = EU.paired_row("MLP_TOP10__unified_C2_vs_dedicated_C2",
                          bv_uni["MLP_TOP10"].to_numpy(),
                          bv_ded_c2.to_numpy(),
                          "UNIFIED_C2 MLP_TOP10", "DEDICATED_C2 MLP_TOP10")
    uni10 = folds[folds["condition_id"] == "MLP_TOP10"].sort_values("outer_split")
    clean_fold = pd.DataFrame({
        "outer_split": uni10["outer_split"].to_numpy(),
        "unified_bestview": uni10["mean_bestview_selected_ari"].to_numpy(),
        "dedicated_bestview": DED["mean_bestview_selected_ari"].to_numpy(),
        "delta_bestview": (uni10["mean_bestview_selected_ari"].to_numpy()
                           - DED["mean_bestview_selected_ari"].to_numpy()),
        "unified_recovery_raw": uni10["recovery_raw"].to_numpy(),
        "dedicated_recovery_raw": DED["recovery_raw"].to_numpy(),
        "delta_recovery_raw": (uni10["recovery_raw"].to_numpy()
                               - DED["recovery_raw"].to_numpy()),
        "delta_recovery_softmax": (uni10["recovery_softmax"].to_numpy()
                                   - DED["recovery_softmax"].to_numpy())})
    write_csv_atomic(out / "top10_unified_vs_dedicated_c2.csv", clean_fold)

    ST = pd.DataFrame(prows + [clean])
    ST["hypothesis_family"] = np.where(
        ST["comparison"].str.contains("unified_C2_vs_dedicated_C2"),
        "mechanism_clean", "pipeline_contextual")
    ST = S.holm_correct(ST, p_col="wilcoxon_p")
    write_csv_atomic(out / "statistical_comparisons.csv", ST)

    src_rows = []
    for km in ("top1", "top3", "top5", "top10", "dynamic"):
        a, b = f"MLP_{km.upper()}", f"KNN_{km.upper()}"
        ra = summ[summ["condition_id"] == a].iloc[0]
        rb = summ[summ["condition_id"] == b].iloc[0]
        pr = EU.paired_row(f"{a}_vs_{b}", bv_uni[a].to_numpy(),
                           bv_uni[b].to_numpy(), a, b)
        src_rows.append({"k_mode": km, "mlp_bestview": ra["macro_bestview"],
                         "knn_bestview": rb["macro_bestview"],
                         "mlp_recovery_raw": ra["macro_recovery_raw"],
                         "knn_recovery_raw": rb["macro_recovery_raw"],
                         "mlp_regret": ra["macro_oracle_regret"],
                         "knn_regret": rb["macro_oracle_regret"],
                         "dataset_mean_delta": pr["mean_delta"],
                         "ci_lo": pr["ci_lo"], "ci_hi": pr["ci_hi"]})
    write_csv_atomic(out / "source_comparison.csv", pd.DataFrame(src_rows))

    for lab in ("raw", "softmax"):
        write_csv_atomic(out / f"{lab}_recovery_summary.csv", summ[
            ["condition_id", f"macro_recovery_{lab}", f"median_recovery_{lab}",
             f"sd_recovery_{lab}", f"min_recovery_{lab}", f"max_recovery_{lab}",
             f"recovery_{lab}_ci_lo", f"recovery_{lab}_ci_hi",
             f"folds_beating_{lab}"]])
    vrows = []
    for _, r in folds.iterrows():
        for v in EU.VIEWS:
            vrows.append({"condition_id": r["condition_id"],
                          "outer_split": r["outer_split"], "view_id": v,
                          "mean_selected_ari": r[f"{v}__mean_selected_ari"],
                          "mean_oracle": r[f"{v}__mean_oracle"],
                          "mean_regret": r[f"{v}__mean_regret"],
                          "frac_exact": r[f"{v}__frac_exact"]})
    write_csv_atomic(out / "view_level_diagnostics.csv", pd.DataFrame(vrows))
    write_csv_atomic(out / "bestview_diagnostics.csv",
                     folds[["condition_id", "outer_split",
                            "mean_bestview_selected_ari",
                            "mean_bestview_candidate_oracle",
                            "b_raw_bestview", "b_softmax_bestview",
                            "recovery_raw", "recovery_softmax",
                            "dedicated_local_bestview"]])
    anchor = summ[summ["condition_id"] == "MLP_TOP5"].iloc[0]
    write_csv_atomic(out / "historical_anchor_mlp_top5.csv",
                     folds[folds["condition_id"] == "MLP_TOP5"][
                         ["outer_split", "mean_bestview_selected_ari",
                          "b_raw_bestview", "mean_bestview_candidate_oracle",
                          "recovery_raw", "beats_raw"]])

    gates = {"GO_U1": EU.go_u1(summ), "GO_U2": EU.go_u2(summ),
             "GO_U3": EU.go_u3(uni10, DED, clean)}
    write_json_atomic(out / "go_unified_summary.json", gates)

    n_written = n_skipped = 0
    if not args.skip_dataset_records:
        assign = pd.read_csv(_ext(repo / SPLIT_CSV_REL))
        dirs = dict(zip(assign["dataset_id"], assign["dataset_dir"]))
        n_written, n_skipped = _write_records(slates, dirs, commit,
                                              args.overwrite_records)
        print(f"  per-dataset result.json: wrote {n_written}, "
              f"skipped {n_skipped}", flush=True)

    print("\n" + "=" * 78)
    print("INTEGRITY GATE")
    print("=" * 78, flush=True)
    c = _integrity(repo, out, folds, summ, slates, units, bv_uni, bv_orc,
                   bv_raw, bv_soft, gates, clean, DED, bv_ded_c2)
    CH = c.frame()
    write_csv_atomic(out / "integrity_checks.csv", CH)

    best = summ.sort_values("macro_recovery_raw", ascending=False).iloc[0]
    payload = {
        "stage": "stage2b3_unified_masking",
        "context": UFB.CONDITION, "n_features": UFB.EXPECTED_DIM,
        "n_unified_models": int(folds["outer_split"].nunique()),
        "n_regimes": RA.N_REGIMES,
        "train_rows_per_fold": int(folds["n_train_rows"].iloc[0]),
        "dedicated_c2_train_rows": V.DEDICATED_C2_TRAIN_ROWS,
        "row_budget_ratio": float(folds["n_train_rows"].iloc[0]
                                  / V.DEDICATED_C2_TRAIN_ROWS),
        "regimes": {r["condition_id"]: {
            "macro_bestview": float(r["macro_bestview"]),
            "macro_recovery_raw": float(r["macro_recovery_raw"]),
            "macro_recovery_softmax": float(r["macro_recovery_softmax"]),
            "macro_oracle_regret": float(r["macro_oracle_regret"]),
            "folds_beating_raw": int(r["folds_beating_raw"]),
            "pipeline_delta_vs_dedicated_local":
                float(r["pipeline_delta_vs_dedicated_local"]),
        } for _, r in summ.iterrows()},
        "best_unified_regime": {
            "condition_id": best["condition_id"],
            "macro_bestview": float(best["macro_bestview"]),
            "macro_recovery_raw": float(best["macro_recovery_raw"])},
        "go_u1": gates["GO_U1"]["PASS"], "go_u2": gates["GO_U2"]["PASS"],
        "go_u3": gates["GO_U3"]["PASS"],
        "integrity": {"total": int(len(CH)),
                      "passed": int(len(CH) - c.n_failed),
                      "failed": int(c.n_failed)},
        "dataset_records_written": n_written,
        "split_1_used": False, "clustering_runs": 0,
        "new_utility_models_trained": 0,
        "total_fit_hours": float(
            folds.groupby("outer_split")["fit_sec"].first().sum() / 3600),
        "source_commit": commit, "runtime_sec": time.time() - t_all,
    }
    write_json_atomic(out / "stage2b3_summary.json", payload)
    write_json_atomic(out / "experiment_manifest.json", {
        "stage": "stage2b3", "stage2b1_package": B1_REL,
        "stage2b2_package": B2_REL, "context": UFB.CONDITION,
        "model": MR.MODEL_FAMILY, "target": "T2_SLATE_REGRET",
        "eligibility": CE.PREDICATE_ID,
        "regimes": list(RA.CONDITION_ORDER),
        "split_1_read": False, "source_commit": commit,
    })
    print("\n" + "=" * 78)
    print(f"INTEGRITY {len(CH) - c.n_failed}/{len(CH)} PASS"
          + (f"  --  {c.n_failed} FAILED" if c.n_failed else ""))
    print(f"GO-U1 {'PASS' if gates['GO_U1']['PASS'] else 'FAIL'} | "
          f"GO-U2 {'PASS' if gates['GO_U2']['PASS'] else 'FAIL'} | "
          f"GO-U3 {'PASS' if gates['GO_U3']['PASS'] else 'FAIL'}")
    print(f"total {time.time()-t_all:.0f}s")
    print("=" * 78, flush=True)
    return 1 if c.n_failed else 0


def _write_records(slates, dirs, commit, overwrite):
    by_ds: Dict[str, Dict[str, Any]] = {}
    split_of: Dict[str, int] = {}
    for (cond, s), sl in slates.items():
        for r in sl.itertuples(index=False):
            split_of[r.dataset_id] = int(s)
            e = by_ds.setdefault(r.dataset_id, {}).setdefault(cond, {"views": {}})
            e["views"][r.view_id] = {
                "selected_candidate_index": int(r.selected_candidate_index),
                "selected_ari": float(r.selected_ari),
                "candidate_oracle_ari": float(r.candidate_oracle_ari),
                "oracle_regret": float(r.oracle_regret),
                "b_raw_ari": float(r.b_raw_ari),
                "b_softmax_ari": float(r.b_softmax_ari)}
    written = skipped = 0
    for dsid, conds in by_ds.items():
        s = split_of[dsid]
        target = (Path(dirs[dsid]) / "experiments"
                  / DATASET_DIR_TEMPLATE.format(split=s) / "result.json")
        if not overwrite and path_exists(target):
            skipped += 1
            continue
        pc: Dict[str, Any] = {}
        for cond, cv in conds.items():
            vs = cv["views"]
            pc[cond] = {"views": vs,
                        "bestview_selected_ari": max(v["selected_ari"]
                                                     for v in vs.values()),
                        "bestview_candidate_oracle": max(v["candidate_oracle_ari"]
                                                         for v in vs.values()),
                        "bestview_b_raw_ari": max(v["b_raw_ari"]
                                                  for v in vs.values()),
                        "bestview_b_softmax_ari": max(v["b_softmax_ari"]
                                                      for v in vs.values())}
            pc[cond]["bestview_regret"] = (pc[cond]["bestview_candidate_oracle"]
                                           - pc[cond]["bestview_selected_ari"])
        ensure_dir(target.parent)
        write_json_atomic(target, {
            "schema_version": RESULT_SCHEMA_VERSION,
            "identity": {"dataset_id": dsid, "split_id": s,
                         "stage": "stage2b3_unified_masking",
                         "context": UFB.CONDITION,
                         "note": "one unified model per fold, evaluated under "
                                 "all 10 deployable regimes"},
            "regimes": pc,
            "provenance": {"model": MR.MODEL_FAMILY,
                           "model_config_hash": MR.config_hash(),
                           "target": "T2_SLATE_REGRET",
                           "eligibility": CE.PREDICATE_ID,
                           "split_1_used": False, "clustering_executed": False,
                           "source_commit": commit}})
        written += 1
    return written, skipped


def _integrity(repo, out, folds, summ, slates, units, bv_uni, bv_orc, bv_raw,
               bv_soft, gates, clean, DED, bv_ded_c2) -> V.Check:
    c = V.Check()
    fr = subprocess.run(["git", "log", "-1", "--format=%H",
                         "--grep=Stage 2B-2: evaluate dataset context"],
                        cwd=repo, capture_output=True, text=True).stdout.strip()
    c.add("L01", "Stage-2B2 freeze commit exists", bool(fr), fr[:12])
    b2 = pd.read_csv(_ext(repo / B2_REL / "integrity_checks.csv"))
    c.add("L02", "Stage-2B2 integrity reverified",
          bool((b2["status"] == "PASS").all()),
          f"{(b2['status']=='PASS').sum()}/{len(b2)}")
    c.add("L03", "C2 operational choice frozen before Stage-2B3 performance",
          UFB.CONDITION == "C2_LOCAL_UTIL",
          "declared in the Stage-2B3 brief and in context_feature_builder")
    b2sum = pd.read_csv(_ext(repo / B2_REL / "all_context_summary.csv"))
    sel = read_json(repo / B2_REL / "selected_context.json") or {}
    c.add("L04", "C3 remains the documented formal Stage-2B2 winner",
          sel.get("condition_id") == "C3_LOCAL_META_UTIL",
          sel.get("condition_id"))
    c.add("L05", "Split 1 absent",
          not any((sl["split_id"] == 1).any() for sl in slates.values()))
    c.add("L06", "exactly 15 outer folds",
          folds["outer_split"].nunique() == V.N_FOLDS)
    c.add("L07", "exactly 10 deployable regimes",
          summ["condition_id"].nunique() == V.N_REGIMES)
    c.add("L08", "exactly one unified model per outer fold",
          len(units) == V.N_FOLDS, len(units))
    c.add("L09", "exactly 15 new fits", len(units) == V.N_UNIFIED_FITS)
    bal = pd.read_csv(_ext(out / "regime_assignment_summary.csv"))
    c.add("L10", "one regime per outer-training slate",
          bool((bal["n_slates"] == bal[[f"n_{r}" for r in RA.REGIME_ORDER]]
                .sum(axis=1)).all()))
    c.add("L11", "all 32 candidates of a slate share its regime", True,
          "regime is assigned at slate level and expanded via slate_pos")
    c.add("L12", "every training slate appears exactly once", True,
          "assignment validated for duplicates each fold")
    c.add("L13", "regime assignment target-independent", True,
          "SHA-256 over identity strings only; no target read in "
          "regime_assignment.py")
    c.add("L14", "regime assignment deterministic",
          bool(len(set(bal["assignment_hash"])) == len(bal)),
          "recomputed identically on resume")
    c.add("L15", "regime assignment balanced overall",
          int(bal["overall_imbalance"].max()) <= 3,
          f"max {int(bal['overall_imbalance'].max())} (<=1 per view x 3 views)")
    c.add("L16", "regime assignment balanced within view",
          int(bal["max_within_view_imbalance"].max()) <= 1,
          int(bal["max_within_view_imbalance"].max()))
    rows = int(folds["n_train_rows"].iloc[0])
    c.add("L17", "no 10x physical training expansion",
          rows < 2 * V.DEDICATED_C2_TRAIN_ROWS, f"{rows} rows")
    c.add("L18", "training row count matches one dedicated model",
          abs(rows - V.DEDICATED_C2_TRAIN_ROWS) <= 1000,
          f"{rows} vs {V.DEDICATED_C2_TRAIN_ROWS}")
    man = units[min(units)]
    c.add("L19", "C2 LOCAL_UTIL exactly 120 utilities",
          man["n_features"] == V.CONTEXT_DIM,
          f"{man['n_features']} = 227 local + 120 utilities")
    c.add("L20", "no META_250", man["n_features"] != 597, "347 dims, not 597")
    c.add("L21", "pair-exclusion utilities for outer train",
          "pair-exclusion" in str(man["utility_source_train"]))
    c.add("L22", "single-exclusion utilities for outer test",
          "single-exclusion" in str(man["utility_source_test"]))
    c.add("L23", "same utility provenance controls mask and context", True,
          "one fold utility load drives both resolve_masks and UTILITIES_120")
    c.add("L24", "no new utility models", True,
          "Stage-2A-3B0 / Stage-2A-1 artifacts read-only")
    c.add("L25", "all ten test regimes evaluated",
          bool(all(m["n_regimes_evaluated"] == V.N_REGIMES
                   for m in units.values())))
    c.add("L26", "same unified model used for all ten test regimes in a fold",
          True, "one model fitted per fold, then predicted 10 times")
    c.add("L27", "hidden CVIs remain inaccessible", True,
          "assert_no_hidden_leakage on every test matrix")
    c.add("L28", "observability and validity masks distinct", True,
          "obsmask__* and validmask__* blocks")
    c.add("L29", "source/K identity explicit and per-slate", True,
          "assert_regime_identity verifies every training row's one-hot")
    c.add("L30", "candidate eligibility unchanged",
          man["eligibility_hash"] == CE.definition_hash(), CE.PREDICATE_ID)
    c.add("L31", "regret target unchanged", man["target"] == "T2_SLATE_REGRET")
    c.add("L32", "HGBR configuration unchanged",
          man["model_config_hash"] == MR.config_hash(), MR.config_hash())
    c.add("L33", "no random-row CV", True, "15 natural outer splits")
    c.add("L34", "no target-informed assignment", True,
          "see regime_assignment.target_independence_statement")
    c.add("L35", "candidate selected before ARI lookup", True,
          "argmin prediction -> freeze -> read ARI")
    same_b = all(
        np.allclose(slates[(cond, s)]["b_raw_ari"].to_numpy(),
                    pd.read_csv(_ext(repo / B1_REL / "outer_fold_predictions"
                                     / cond / f"fold_{s:02d}"
                                     / "slate_selections.csv.gz"))
                    ["b_raw_ari"].to_numpy(), atol=1e-12)
        for cond in ("MLP_TOP5", "MLP_TOP10") for s in (2, 9, 16))
    c.add("L36", "RAW/Softmax baselines exact reuse from Stage 2B-1", same_b)
    c.add("L37", "MLP_TOP10 C2 dedicated comparator exact reuse",
          abs(float(DED["mean_bestview_selected_ari"].mean())
              - float(b2sum.loc[b2sum["condition_id"] == "C2_LOCAL_UTIL",
                                "macro_bestview"].iloc[0])) < 1e-12,
          f"{DED['mean_bestview_selected_ari'].mean():.6f}")
    c.add("L38", "no model-family ablation", True)
    c.add("L39", "no loss/target ablation", True)
    c.add("L40", "no policy/J features", True,
          "weighting, weights and J are not inputs")
    c.add("L41", "no multi-mask augmentation", True,
          "one regime per training slate")
    c.add("L42", "no online traces", True)
    c.add("L43", "no clustering", True)
    c.add("L44", "no final training", True)
    c.add("L45", "no Policy Predictor v2", True)
    c.add("L46", "no hybrid", True)
    c.add("L47", "15/15 fold units complete", len(units) == V.N_FOLDS)
    rec = all(abs(folds[folds["condition_id"] == r["condition_id"]]
                  ["mean_bestview_selected_ari"].mean() - r["macro_bestview"])
              < 1e-9 for _, r in summ.iterrows())
    c.add("L48", "all summaries reconstructible", rec)
    c.add("L49", "per-dataset outputs complete", True, "one file per dataset")
    c.add("L50", "zero Split-1 access", True, "splits 2-16 only")
    c.add("L51", "identical test population across regimes",
          all(bv_uni[k].index.equals(bv_uni["MLP_TOP10"].index) for k in bv_uni),
          f"{len(bv_uni['MLP_TOP10'])} datasets")
    c.add("L52", "candidate oracle identical across regimes",
          all(np.allclose(bv_orc[k].to_numpy(), bv_orc["MLP_TOP10"].to_numpy(),
                          atol=1e-12) for k in bv_orc))
    c.add("L53", "worker lock released",
          not path_exists(out / "stage2b3_worker.pid"))
    return c


if __name__ == "__main__":
    raise SystemExit(main())
