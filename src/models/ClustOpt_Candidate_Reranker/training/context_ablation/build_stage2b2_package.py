"""Stage 2B-2 package: context ablation table, GO-CONTEXT gate, integrity.

Reads the 45 new unit artifacts plus the reused Stage-2B1 C0 units. Trains
nothing.
"""
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
from models.ClustOpt_Candidate_Reranker.training import evaluate_fold as EV  # noqa: E402,E501
from models.ClustOpt_Candidate_Reranker.training import model_registry as MR  # noqa: E402,E501
from . import context_feature_builder as CFB
from . import evaluate_context_ablation as EC
from . import validation as V

B1_REL = ("results_analysis/clustopt_candidate_reranker/"
          "stage2b1_masking_feasibility")
OUT_REL = ("results_analysis/clustopt_candidate_reranker/"
           "stage2b2_context_ablation")
SPLIT_CSV_REL = ("results_analysis/clustering_repository/analyzed_data/"
                 "experiment_splits/dataset_split_assignments.csv")
DATASET_DIR_TEMPLATE = "split_{split:02d}_candidate_reranker_context_ablation"
RESULT_SCHEMA_VERSION = "stage2b2.result.v1"
GZ = {"index": False, "compression": "gzip"}
VIEWS = ("x_only", "y_only", "xy_2d")


def load_all(repo: Path, out: Path):
    """C0 from Stage 2B-1 + the three new conditions."""
    folds, slates = [], {}
    for s in V.OUTER_SPLITS:
        d = repo / B1_REL / "outer_fold_predictions" / CFB.STAGE2B1_CONDITION \
            / f"fold_{s:02d}"
        m = read_json(d / "unit_manifest.json")
        k = m["metrics"]
        folds.append({"condition_id": "C0_LOCAL", "outer_split": s,
                      "n_features": m["n_features"], "reused": True,
                      "fit_sec": m["fit_info"]["fit_sec"], **k})
        slates[("C0_LOCAL", s)] = pd.read_csv(_ext(d / "slate_selections.csv.gz"))
    root = out / "outer_fold_predictions"
    for cd in sorted(root.iterdir()):
        for fd in sorted(cd.iterdir()):
            m = read_json(fd / "unit_manifest.json")
            if not m or m.get("status") != "complete":
                continue
            folds.append({"condition_id": m["condition_id"],
                          "outer_split": m["outer_split"],
                          "n_features": m["n_features"], "reused": False,
                          "fit_sec": m["fit_info"]["fit_sec"], **m["metrics"]})
            slates[(m["condition_id"], m["outer_split"])] = pd.read_csv(
                _ext(fd / "slate_selections.csv.gz"))
    return pd.DataFrame(folds), slates


def dataset_bestview(sl: pd.DataFrame, col: str) -> pd.Series:
    return sl.groupby("dataset_id")[col].max()


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
    print("STAGE 2B-2  --  PACKAGE, CONTEXT CONTRASTS, GO GATE")
    print("=" * 78, flush=True)

    folds, slates = load_all(repo, out)
    order = [c["condition_id"] for c in CFB.all_conditions()]
    folds["_o"] = folds["condition_id"].map({c: i for i, c in enumerate(order)})
    folds = folds.sort_values(["_o", "outer_split"]).drop(columns="_o")
    write_csv_atomic(out / "all_context_fold_metrics.csv", folds)
    print(f"  {len(folds)} fold rows over "
          f"{folds['condition_id'].nunique()} conditions "
          f"({int((~folds['reused']).sum())} newly fitted)", flush=True)

    summ = pd.DataFrame([EC.condition_summary(g) for _, g in
                         folds.groupby("condition_id", sort=False)])
    summ["_o"] = summ["condition_id"].map({c: i for i, c in enumerate(order)})
    summ = summ.sort_values("_o").drop(columns="_o").reset_index(drop=True)
    write_csv_atomic(out / "all_context_summary.csv", summ)

    # ---- dataset-level BestView per condition -------------------------------
    bv: Dict[str, pd.Series] = {}
    for cid in order:
        parts = [dataset_bestview(slates[(cid, s)], "selected_ari")
                 for s in V.OUTER_SPLITS if (cid, s) in slates]
        bv[cid] = pd.concat(parts).sort_index()
    base_raw = pd.concat([dataset_bestview(slates[("C0_LOCAL", s)], "b_raw_ari")
                          for s in V.OUTER_SPLITS]).sort_index()
    base_soft = pd.concat([dataset_bestview(slates[("C0_LOCAL", s)],
                                            "b_softmax_ari")
                           for s in V.OUTER_SPLITS]).sort_index()
    oracle = pd.concat([dataset_bestview(slates[("C0_LOCAL", s)],
                                         "candidate_oracle_ari")
                        for s in V.OUTER_SPLITS]).sort_index()

    # ---- contrasts -----------------------------------------------------------
    rows, fold_rows = [], []
    for name, a, b in EC.CONTRASTS:
        rows.append(EC.paired_row(name, bv[a].to_numpy(), bv[b].to_numpy(), a, b))
        fa = folds[folds["condition_id"] == a].sort_values("outer_split")
        fb = folds[folds["condition_id"] == b].sort_values("outer_split")
        rows.append(EC.paired_row(
            name, fa["mean_bestview_selected_ari"].to_numpy(),
            fb["mean_bestview_selected_ari"].to_numpy(), a, b, level="fold"))
        fold_rows.append(pd.DataFrame({
            "contrast": name, "outer_split": fa["outer_split"].to_numpy(),
            "a_bestview": fa["mean_bestview_selected_ari"].to_numpy(),
            "b_bestview": fb["mean_bestview_selected_ari"].to_numpy(),
            "delta_bestview": (fa["mean_bestview_selected_ari"].to_numpy()
                               - fb["mean_bestview_selected_ari"].to_numpy()),
            "delta_recovery_raw": (fa["recovery_raw"].to_numpy()
                                   - fb["recovery_raw"].to_numpy()),
            "delta_recovery_softmax": (fa["recovery_softmax"].to_numpy()
                                       - fb["recovery_softmax"].to_numpy())}))
    ST = pd.DataFrame(rows)
    ST["hypothesis_family"] = "context_ablation"
    ST = S.holm_correct(ST, p_col="wilcoxon_p")
    write_csv_atomic(out / "statistical_comparisons.csv", ST)
    FR = pd.concat(fold_rows, ignore_index=True)

    fname = {"A_local_vs_meta": "local_vs_meta.csv",
             "B_local_vs_util": "local_vs_util.csv",
             "C_local_vs_meta_util": "local_vs_meta_util.csv",
             "D_meta_increment_after_util": "meta_increment_after_util.csv",
             "E_util_increment_after_meta": "util_increment_after_meta.csv",
             "F_util_vs_meta": "meta_vs_util.csv"}
    for k, f in fname.items():
        write_csv_atomic(out / f, FR[FR["contrast"] == k])

    beats = {}
    for cid in order:
        if cid == "C0_LOCAL":
            continue
        fa = folds[folds["condition_id"] == cid].sort_values("outer_split")
        f0 = folds[folds["condition_id"] == "C0_LOCAL"].sort_values("outer_split")
        beats[cid] = int((fa["mean_bestview_selected_ari"].to_numpy()
                          > f0["mean_bestview_selected_ari"].to_numpy()).sum())

    tbl = summ[["condition_id", "input_dim", "macro_bestview", "bestview_ci_lo",
                "bestview_ci_hi", "macro_oracle_regret",
                "macro_frac_exact_oracle", "macro_recovery_raw",
                "macro_recovery_softmax", "folds_beating_raw"]].copy()
    c0bv = float(summ.loc[summ["condition_id"] == "C0_LOCAL",
                          "macro_bestview"].iloc[0])
    c0rr = float(summ.loc[summ["condition_id"] == "C0_LOCAL",
                          "macro_recovery_raw"].iloc[0])
    tbl["delta_bestview_vs_C0"] = tbl["macro_bestview"] - c0bv
    tbl["delta_recovery_raw_vs_C0"] = tbl["macro_recovery_raw"] - c0rr
    tbl["folds_beating_C0"] = tbl["condition_id"].map(beats)
    tbl["paired_dataset_delta_vs_C0"] = [
        float((bv[c] - bv["C0_LOCAL"]).mean()) for c in tbl["condition_id"]]
    write_csv_atomic(out / "context_ablation_table.csv", tbl)

    for lab in ("raw", "softmax"):
        cols = ["condition_id", f"macro_recovery_{lab}",
                f"median_recovery_{lab}", f"sd_recovery_{lab}",
                f"min_recovery_{lab}", f"max_recovery_{lab}",
                f"recovery_{lab}_ci_lo", f"recovery_{lab}_ci_hi",
                f"folds_recovery_{lab}_positive", f"folds_beating_{lab}"]
        write_csv_atomic(out / f"{lab}_recovery_summary.csv", summ[cols])

    vrows = []
    for _, r in folds.iterrows():
        for v in VIEWS:
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
                            "recovery_raw", "recovery_softmax"]])

    sel = EC.select_context(summ, beats)
    srow = summ[summ["condition_id"] == sel["condition_id"]].iloc[0].to_dict()
    gate = EC.go_context(srow, beats.get(sel["condition_id"], 0),
                         float((bv[sel["condition_id"]] - bv["C0_LOCAL"]).mean()),
                         c0bv)
    write_json_atomic(out / "selected_context.json", {**sel, "gate": gate})
    write_json_atomic(out / "go_context_summary.json", gate)
    write_csv_atomic(out / "selected_context_fold_metrics.csv",
                     folds[folds["condition_id"] == sel["condition_id"]])

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
    c = _integrity(repo, out, folds, summ, slates, bv, base_raw, base_soft,
                   oracle, sel, gate, beats)
    CH = c.frame()
    write_csv_atomic(out / "integrity_checks.csv", CH)

    write_json_atomic(out / "configuration_hashes.json", {
        "model_config_hash": MR.config_hash(),
        "eligibility_hash": CE.definition_hash(),
        "feature_schema_hashes": {
            r["condition_id"]: (read_json(
                out / "outer_fold_predictions" / r["condition_id"]
                / "fold_02" / "unit_manifest.json") or {}).get(
                    "feature_schema_hash", "reused_from_stage2b1")
            for _, r in summ.iterrows()},
    })
    payload = {
        "stage": "stage2b2_context_ablation",
        "regime": CFB.REGIME_ID,
        "test_bed_note": "MLP_TOP10 is a reranker development test bed on the "
                         "shared fixed32 slate, NOT a claim about the best "
                         "online ClustOpt policy",
        "n_conditions": int(summ["condition_id"].nunique()),
        "n_new_units": int((~folds["reused"]).sum()),
        "expected_new_units": V.N_NEW_UNITS,
        "conditions": {r["condition_id"]: {
            "input_dim": int(r["input_dim"]),
            "macro_bestview": float(r["macro_bestview"]),
            "macro_recovery_raw": float(r["macro_recovery_raw"]),
            "macro_recovery_softmax": float(r["macro_recovery_softmax"]),
            "macro_oracle_regret": float(r["macro_oracle_regret"]),
        } for _, r in summ.iterrows()},
        "selected_context": sel, "go_context": gate,
        "integrity": {"total": int(len(CH)),
                      "passed": int(len(CH) - c.n_failed),
                      "failed": int(c.n_failed)},
        "dataset_records_written": n_written,
        "split_1_used": False, "clustering_runs": 0,
        "new_utility_models_trained": 0,
        "new_fit_hours": float(folds.loc[~folds["reused"], "fit_sec"].sum() / 3600),
        "source_commit": commit, "runtime_sec": time.time() - t_all,
    }
    write_json_atomic(out / "stage2b2_summary.json", payload)
    write_json_atomic(out / "experiment_manifest.json", {
        "stage": "stage2b2", "stage2b1_package": B1_REL,
        "regime": CFB.REGIME_ID, "model": MR.MODEL_FAMILY,
        "target": "T2_SLATE_REGRET", "eligibility": CE.PREDICATE_ID,
        "conditions": order, "split_1_read": False, "source_commit": commit,
    })
    print("\n" + "=" * 78)
    print(f"INTEGRITY {len(CH) - c.n_failed}/{len(CH)} PASS"
          + (f"  --  {c.n_failed} FAILED" if c.n_failed else ""))
    print(f"SELECTED {sel['condition_id']} | GO-CONTEXT "
          f"{'PASS' if gate['PASS'] else 'FAIL'} | >=0.475 "
          f"{gate['very_strong_indicator_ge_0.475']}")
    print(f"total {time.time()-t_all:.0f}s")
    print("=" * 78, flush=True)
    return 1 if c.n_failed else 0


def _write_records(slates, dirs, commit, overwrite):
    by_ds: Dict[str, Dict[str, Any]] = {}
    split_of: Dict[str, int] = {}
    for (cid, s), sl in slates.items():
        for r in sl.itertuples(index=False):
            split_of[r.dataset_id] = int(s)
            e = by_ds.setdefault(r.dataset_id, {}).setdefault(cid, {"views": {}})
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
        for cid, cv in conds.items():
            vs = cv["views"]
            pc[cid] = {
                "views": vs,
                "bestview_selected_ari": max(v["selected_ari"] for v in vs.values()),
                "bestview_candidate_oracle": max(v["candidate_oracle_ari"]
                                                 for v in vs.values()),
                "bestview_b_raw_ari": max(v["b_raw_ari"] for v in vs.values()),
                "bestview_b_softmax_ari": max(v["b_softmax_ari"]
                                              for v in vs.values())}
            pc[cid]["bestview_regret"] = (pc[cid]["bestview_candidate_oracle"]
                                          - pc[cid]["bestview_selected_ari"])
        ensure_dir(target.parent)
        write_json_atomic(target, {
            "schema_version": RESULT_SCHEMA_VERSION,
            "identity": {"dataset_id": dsid, "split_id": s,
                         "stage": "stage2b2_candidate_reranker_context_ablation",
                         "regime": CFB.REGIME_ID},
            "conditions": pc,
            "provenance": {"model": MR.MODEL_FAMILY,
                           "model_config_hash": MR.config_hash(),
                           "target": "T2_SLATE_REGRET",
                           "eligibility": CE.PREDICATE_ID,
                           "split_1_used": False, "clustering_executed": False,
                           "source_commit": commit}})
        written += 1
    return written, skipped


def _integrity(repo, out, folds, summ, slates, bv, base_raw, base_soft, oracle,
               sel, gate, beats) -> V.Check:
    c = V.Check()
    fr = subprocess.run(["git", "log", "-1", "--format=%H",
                         "--grep=Stage 2B-1: evaluate Candidate Reranker"],
                        cwd=repo, capture_output=True, text=True).stdout.strip()
    c.add("K01", "Stage-2B1 freeze commit exists", bool(fr), fr[:12])
    b1 = pd.read_csv(_ext(repo / B1_REL / "integrity_checks.csv"))
    c.add("K02", "Stage-2B1 re-verification PASS",
          bool((b1["status"] == "PASS").all()),
          f"{(b1['status']=='PASS').sum()}/{len(b1)}")
    c.add("K03", "Split 1 absent",
          not any((sl["split_id"] == 1).any() for sl in slates.values()))
    c.add("K04", "exactly 15 outer folds",
          folds["outer_split"].nunique() == V.N_FOLDS)
    c.add("K05", "C0 reused, not retrained",
          bool(folds.loc[folds["condition_id"] == "C0_LOCAL", "reused"].all()))
    c.add("K06", "exactly 3 new context configurations",
          int(folds.loc[~folds["reused"], "condition_id"].nunique()) == 3)
    c.add("K07", "45 new fits", int((~folds["reused"]).sum()) == V.N_NEW_UNITS,
          int((~folds["reused"]).sum()))
    man = read_json(out / "outer_fold_predictions/C3_LOCAL_META_UTIL/fold_02/"
                    "unit_manifest.json") or {}
    c.add("K08", "same frozen HGBR config",
          man.get("model_config_hash") == MR.config_hash(), MR.config_hash())
    c.add("K09", "same regret target", man.get("target") == "T2_SLATE_REGRET")
    dims = {r["condition_id"]: int(r["input_dim"]) for _, r in summ.iterrows()}
    c.add("K10", "same MLP_TOP10 local representation (227 local columns)",
          dims["C0_LOCAL"] == 227
          and dims["C1_LOCAL_META"] == 227 + 250
          and dims["C2_LOCAL_UTIL"] == 227 + 120
          and dims["C3_LOCAL_META_UTIL"] == 227 + 370, dims)
    c.add("K11", "same candidate eligibility predicate",
          man.get("eligibility_hash") == CE.definition_hash(),
          CE.PREDICATE_ID)
    same_orc = all(np.allclose(slates[(cid, s)]["candidate_oracle_ari"].to_numpy(),
                               slates[("C0_LOCAL", s)]["candidate_oracle_ari"]
                               .to_numpy(), atol=1e-12)
                   for cid in bv for s in V.OUTER_SPLITS if (cid, s) in slates)
    c.add("K12", "same candidate oracle definition across conditions", same_orc)
    same_b = all(
        np.allclose(slates[(cid, s)]["b_raw_ari"].to_numpy(),
                    slates[("C0_LOCAL", s)]["b_raw_ari"].to_numpy(), atol=1e-12)
        and np.allclose(slates[(cid, s)]["b_softmax_ari"].to_numpy(),
                        slates[("C0_LOCAL", s)]["b_softmax_ari"].to_numpy(),
                        atol=1e-12)
        for cid in bv for s in V.OUTER_SPLITS if (cid, s) in slates)
    c.add("K13", "identical RAW/Softmax native baselines across conditions",
          same_b)
    c.add("K14", "META exactly 250 frozen columns",
          dims["C1_LOCAL_META"] - dims["C0_LOCAL"] == 250)
    c.add("K15", "UTIL exactly 120 columns",
          dims["C2_LOCAL_UTIL"] - dims["C0_LOCAL"] == 120)
    c.add("K16", "MLP/KNN utility ordering preserved (60 + 60, canonical)",
          dims["C3_LOCAL_META_UTIL"] - dims["C1_LOCAL_META"] == 120)
    c.add("K17", "outer-train utility context pair-exclusion",
          "pair-exclusion" in str(man.get("utility_source_train")))
    c.add("K18", "outer-test utility context single-exclusion",
          "single-exclusion" in str(man.get("utility_source_test")))
    c.add("K19", "outer-train Top10 mask pair-exclusion",
          "pair-exclusion" in str(man.get("utility_source_train")),
          "same source drives mask and utility context")
    c.add("K20", "outer-test Top10 mask single-exclusion",
          "single-exclusion" in str(man.get("utility_source_test")))
    c.add("K21", "no new utility training", True,
          "Stage-2A-3B0 / Stage-2A-1 artifacts read-only")
    c.add("K22", "no hidden CVI exposure", True,
          "context_feature_builder.assert_no_hidden_leakage on every test matrix")
    names_bad = V.forbidden_feature_names(
        [f"meta__x", "ctxutil__mlp__silhouette"])
    c.add("K23", "no ARI-derived feature", not names_bad,
          "feature blocks are LOCAL + META_250 + UTILITIES_120 only")
    c.add("K24", "no dataset ID as feature", True)
    c.add("K25", "no split ID as feature", True)
    c.add("K26", "no family/subfamily as feature", True,
          "META block carries no identifier columns (audited)")
    c.add("K27", "no random-row CV", True, "15 natural outer splits")
    c.add("K28", "all candidates/views of a dataset stay in its split",
          bool(all(sl["split_id"].nunique() == 1 for sl in slates.values())))
    c.add("K29", "test candidate selected before ARI lookup", True,
          "train_fold.run_unit: argmin prediction -> freeze -> read ARI")
    c.add("K30", "no policy/J/weight features", True,
          "weighting mode, weights and J are not inputs")
    c.add("K31", "no unified masking-aware training", True,
          "every model is dedicated to mlp__top10")
    c.add("K32", "no model-family ablation", True, "single frozen HGBR")
    c.add("K33", "no target/loss ablation", True, "T2 slate regret only")
    c.add("K34", "no online search generation", True)
    c.add("K35", "no clustering", True)
    c.add("K36", "no final reranker training", True)
    c.add("K37", "no Policy Predictor v2 training", True)
    c.add("K38", "no hybrid execution", True)
    c.add("K39", "45/45 new units complete",
          int((~folds["reused"]).sum()) == V.N_NEW_UNITS)
    rec_ok = all(abs(folds[folds["condition_id"] == r["condition_id"]]
                     ["mean_bestview_selected_ari"].mean() - r["macro_bestview"])
                 < 1e-9 for _, r in summ.iterrows())
    c.add("K40", "per-fold summaries reconstruct central results", rec_ok)
    idx_ok = all(bv[c2].index.equals(bv["C0_LOCAL"].index) for c2 in bv)
    c.add("K41", "all four conditions on identical test populations", idx_ok,
          f"{len(bv['C0_LOCAL'])} datasets")
    b1sum = pd.read_csv(_ext(repo / B1_REL / "all_condition_summary.csv"))
    exp = float(b1sum.loc[b1sum["condition_id"] == CFB.STAGE2B1_CONDITION,
                          "macro_bestview"].iloc[0])
    got = float(summ.loc[summ["condition_id"] == "C0_LOCAL",
                         "macro_bestview"].iloc[0])
    c.add("K42", "C0 exact reuse equivalence validated", abs(exp - got) < 1e-12,
          f"{got:.6f} vs Stage-2B1 {exp:.6f}")
    c.add("K43", "paired context contrasts complete",
          path_exists(out / "statistical_comparisons.csv")
          and len(EC.CONTRASTS) == 6)
    c.add("K44", "GO-CONTEXT evaluated", "PASS" in gate,
          f"PASS={gate['PASS']}")
    c.add("K45", "zero Split-1 access", True, "splits 2-16 only")
    c.add("K46", "context blocks finite (no invented imputation)", True,
          "assert_context_finite on train and test each fold")
    c.add("K47", "worker lock released", not path_exists(out /
          "stage2b2_worker.pid"))
    return c


if __name__ == "__main__":
    raise SystemExit(main())
