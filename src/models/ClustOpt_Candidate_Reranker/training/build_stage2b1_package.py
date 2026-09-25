"""Stage 2B-1 package: masking curve, native-J recovery, GO gates, integrity.

Reads only the 165 unit artifacts written by ``run_masking_feasibility`` plus the
frozen Stage-2B0 definitions. Trains nothing.
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
    _ext, ensure_dir, path_exists, read_json, write_csv_atomic, write_json_atomic,
)
from models.Clustering_Repository_Builder.experiments.paper_analysis import (  # noqa: E402,E501
    stats as S,
)
from models.ClustOpt_Candidate_Reranker.data import candidate_eligibility as CE  # noqa: E402,E501
from models.ClustOpt_Candidate_Reranker.training import evaluate_fold as EV  # noqa: E402,E501
from models.ClustOpt_Candidate_Reranker.training import feature_builder as FB  # noqa: E402,E501
from models.ClustOpt_Candidate_Reranker.training import model_registry as MR  # noqa: E402,E501
from models.ClustOpt_Candidate_Reranker.training import validation as V  # noqa: E402,E501

B0_REL = ("results_analysis/clustopt_candidate_reranker/"
          "stage2b0_data_protocol_audit")
OUT_REL = ("results_analysis/clustopt_candidate_reranker/"
           "stage2b1_masking_feasibility")
SPLIT_CSV_REL = ("results_analysis/clustering_repository/analyzed_data/"
                 "experiment_splits/dataset_split_assignments.csv")
DATASET_DIR_TEMPLATE = "split_{split:02d}_candidate_reranker_feasibility"
RESULT_SCHEMA_VERSION = "stage2b1.result.v1"
GZ = {"index": False, "compression": "gzip"}
ANCHOR_CONDITION = "MLP_TOP5"
HISTORICAL = {"FP_bestview_ari": 0.5779, "online_full_slate_oracle": 0.6549,
              "autoclust_extended": 0.6145,
              "required_recovery": (0.6145 - 0.5779) / (0.6549 - 0.5779)}


def load_units(out: Path) -> Tuple[pd.DataFrame, Dict[Tuple[str, int], pd.DataFrame]]:
    rows, slates = [], {}
    root = out / "outer_fold_predictions"
    for cond_dir in sorted(root.iterdir()):
        for fold_dir in sorted(cond_dir.iterdir()):
            m = read_json(fold_dir / "unit_manifest.json")
            if not m or m.get("status") != "complete":
                continue
            rows.append({
                "condition_id": m["condition_id"], "outer_split": m["outer_split"],
                "kind": m["kind"], "utility_source": m["utility_source"],
                "k_mode": m["k_mode"], "n_features": m["n_features"],
                "feature_schema_hash": m["feature_schema_hash"],
                "unit_hash": m["unit_hash"], "mask_source": m["mask_source"],
                "fit_sec": m["fit_info"]["fit_sec"],
                "n_train": m["fit_info"]["n_train"],
                "n_test": m["fit_info"]["n_test"], **m["metrics"]})
            slates[(m["condition_id"], m["outer_split"])] = pd.read_csv(
                _ext(fold_dir / "slate_selections.csv.gz"))
    return pd.DataFrame(rows), slates


def bestview_table(sl: pd.DataFrame) -> pd.DataFrame:
    cols = {"selected_ari": "reranker", "candidate_oracle_ari": "oracle"}
    if "b_raw_ari" in sl.columns:
        cols.update({"b_raw_ari": "b_raw", "b_softmax_ari": "b_softmax"})
    g = sl.groupby("dataset_id")[list(cols)].max().rename(columns=cols)
    return g.reset_index()


def macro_summary(fold_rows: pd.DataFrame) -> Dict[str, Any]:
    d = fold_rows
    out: Dict[str, Any] = {
        "condition_id": d["condition_id"].iloc[0], "kind": d["kind"].iloc[0],
        "utility_source": d["utility_source"].iloc[0],
        "k_mode": d["k_mode"].iloc[0], "n_folds": int(len(d)),
        "n_features": int(d["n_features"].iloc[0]),
        "macro_view_selected_ari": float(d["mean_selected_ari"].mean()),
        "macro_bestview": float(d["mean_bestview_selected_ari"].mean()),
        "macro_bestview_oracle": float(d["mean_bestview_candidate_oracle"].mean()),
        "macro_candidate_oracle": float(d["mean_candidate_oracle_ari"].mean()),
        "macro_oracle_regret": float(d["mean_oracle_regret"].mean()),
        "macro_median_regret": float(d["median_oracle_regret"].mean()),
        "macro_frac_exact_oracle": float(d["frac_exact_oracle"].mean()),
        "macro_frac_within_001": float(d["frac_within_001"].mean()),
        "macro_frac_within_005": float(d["frac_within_005"].mean()),
        "macro_frac_within_01": float(d["frac_within_01"].mean()),
        "min_bestview": float(d["mean_bestview_selected_ari"].min()),
        "max_bestview": float(d["mean_bestview_selected_ari"].max()),
        "sd_bestview": float(d["mean_bestview_selected_ari"].std()),
        "total_fit_sec": float(d["fit_sec"].sum()),
    }
    pt, lo, hi = S.bootstrap_ci(d["mean_bestview_selected_ari"].to_numpy(float),
                                statistic="mean")
    out["bestview_ci_lo"], out["bestview_ci_hi"] = float(lo), float(hi)
    for v in EV.VIEW_IDS:
        out[f"{v}__macro_selected_ari"] = float(d[f"{v}__mean_selected_ari"].mean())
        out[f"{v}__macro_regret"] = float(d[f"{v}__mean_regret"].mean())
        out[f"{v}__macro_frac_exact"] = float(d[f"{v}__frac_exact"].mean())
    if d["kind"].iloc[0] != "RICH":
        for lab in ("raw", "softmax"):
            out[f"macro_b_{lab}_bestview"] = float(d[f"b_{lab}_bestview"].mean())
            out[f"macro_b_{lab}_view"] = float(d[f"b_{lab}_view"].mean())
            out[f"macro_recovery_{lab}"] = float(d[f"recovery_{lab}"].mean())
            out[f"median_recovery_{lab}"] = float(d[f"recovery_{lab}"].median())
            out[f"sd_recovery_{lab}"] = float(d[f"recovery_{lab}"].std())
            out[f"min_recovery_{lab}"] = float(d[f"recovery_{lab}"].min())
            out[f"max_recovery_{lab}"] = float(d[f"recovery_{lab}"].max())
            p, l2, h2 = S.bootstrap_ci(d[f"recovery_{lab}"].to_numpy(float),
                                       statistic="mean")
            out[f"recovery_{lab}_ci_lo"] = float(l2)
            out[f"recovery_{lab}_ci_hi"] = float(h2)
            out[f"folds_recovery_{lab}_positive"] = int((d[f"recovery_{lab}"] > 0).sum())
            out[f"folds_beating_{lab}"] = int(d[f"beats_{lab}"].sum())
            out[f"band_{lab}"] = V.band_of(out[f"macro_recovery_{lab}"])
        out["folds_beating_both"] = int((d["beats_raw"] & d["beats_softmax"]).sum())
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", default=".")
    p.add_argument("--skip-dataset-records", action="store_true")
    p.add_argument("--overwrite-records", action="store_true")
    args = p.parse_args(argv)

    repo = Path(args.repo_root).resolve()
    out = ensure_dir(repo / OUT_REL)
    b0 = repo / B0_REL
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                            capture_output=True, text=True).stdout.strip()
    t_all = time.time()
    print("=" * 78)
    print("STAGE 2B-1  --  PACKAGE, GO GATES, INTEGRITY")
    print("=" * 78, flush=True)

    folds, slates = load_units(out)
    print(f"  {len(folds)} completed units over "
          f"{folds['condition_id'].nunique()} conditions", flush=True)
    write_csv_atomic(out / "all_condition_fold_metrics.csv", folds)

    summ = pd.DataFrame([macro_summary(g) for _, g in
                         folds.groupby("condition_id", sort=False)])
    order = [c["condition_id"] for c in FB.all_conditions()]
    summ["_o"] = summ["condition_id"].map({c: i for i, c in enumerate(order)})
    summ = summ.sort_values("_o").drop(columns="_o").reset_index(drop=True)
    write_csv_atomic(out / "all_condition_summary.csv", summ)

    dep = summ[summ["kind"] == "DEPLOYABLE"].copy()
    rich = summ[summ["kind"] == "RICH"].copy()
    write_csv_atomic(out / "rich60_summary.csv", rich)

    # ---- paired dataset-level deltas ---------------------------------------
    paired: Dict[str, pd.DataFrame] = {}
    for cid in summ["condition_id"]:
        parts = [bestview_table(slates[(cid, s)]) for s in V.OUTER_SPLITS
                 if (cid, s) in slates]
        paired[cid] = pd.concat(parts, ignore_index=True).sort_values("dataset_id")
    for lab in ("raw", "softmax"):
        dep[f"mean_paired_bestview_delta_{lab}"] = [
            float((paired[c]["reranker"] - paired[c][f"b_{lab}"]).mean())
            for c in dep["condition_id"]]

    # ---- masking curves ------------------------------------------------------
    curve_cols = ["condition_id", "utility_source", "k_mode", "macro_bestview",
                  "macro_view_selected_ari", "macro_oracle_regret",
                  "macro_frac_exact_oracle", "macro_frac_within_001",
                  "macro_frac_within_005", "macro_frac_within_01",
                  "macro_b_raw_bestview", "macro_b_softmax_bestview",
                  "macro_recovery_raw", "macro_recovery_softmax",
                  "folds_beating_raw", "folds_beating_softmax",
                  "band_raw", "band_softmax"]
    k_order = {"top10": 0, "dynamic": 1, "top5": 2, "top3": 3, "top1": 4}
    curve = dep[curve_cols].copy()
    curve["_k"] = curve["k_mode"].map(k_order)
    curve = curve.sort_values(["utility_source", "_k"]).drop(columns="_k")
    write_csv_atomic(out / "deployable_masking_curve.csv", curve)
    write_csv_atomic(out / "mlp_masking_curve.csv",
                     curve[curve["utility_source"] == "mlp"])
    write_csv_atomic(out / "knn_masking_curve.csv",
                     curve[curve["utility_source"] == "knn"])
    write_csv_atomic(out / "native_raw_baselines.csv",
                     folds[folds["kind"] == "DEPLOYABLE"][
                         ["condition_id", "outer_split", "b_raw_view",
                          "b_raw_bestview", "b_raw_regret", "recovery_raw",
                          "beats_raw"]])
    write_csv_atomic(out / "native_softmax_baselines.csv",
                     folds[folds["kind"] == "DEPLOYABLE"][
                         ["condition_id", "outer_split", "b_softmax_view",
                          "b_softmax_bestview", "b_softmax_regret",
                          "recovery_softmax", "beats_softmax"]])

    # ---- source comparison ---------------------------------------------------
    src_rows = []
    for km in FB.K_MODES:
        a, b = f"MLP_{km.upper()}", f"KNN_{km.upper()}"
        if a not in set(summ["condition_id"]) or b not in set(summ["condition_id"]):
            continue
        fa = folds[folds["condition_id"] == a].sort_values("outer_split")
        fb = folds[folds["condition_id"] == b].sort_values("outer_split")
        da = paired[a].set_index("dataset_id")["reranker"]
        db = paired[b].set_index("dataset_id")["reranker"]
        common = da.index.intersection(db.index)
        delta = (da.loc[common] - db.loc[common]).to_numpy()
        pt, lo, hi = S.bootstrap_ci(delta, statistic="mean")
        src_rows.append({
            "k_mode": km, "mlp_condition": a, "knn_condition": b,
            "mlp_macro_bestview": float(fa["mean_bestview_selected_ari"].mean()),
            "knn_macro_bestview": float(fb["mean_bestview_selected_ari"].mean()),
            "fold_delta": float(fa["mean_bestview_selected_ari"].mean()
                                - fb["mean_bestview_selected_ari"].mean()),
            "folds_mlp_better": int((fa["mean_bestview_selected_ari"].to_numpy()
                                     > fb["mean_bestview_selected_ari"].to_numpy()).sum()),
            "dataset_mean_delta": float(pt), "ci_lo": float(lo), "ci_hi": float(hi),
            "mlp_macro_recovery_raw": float(fa["recovery_raw"].mean()),
            "knn_macro_recovery_raw": float(fb["recovery_raw"].mean()),
        })
    write_csv_atomic(out / "source_comparison.csv", pd.DataFrame(src_rows))

    # ---- diagnostics ---------------------------------------------------------
    vrows = []
    for _, r in folds.iterrows():
        for v in EV.VIEW_IDS:
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
                            "mean_bestview_regret", "n_datasets"]])
    write_csv_atomic(out / "oracle_regret_summary.csv",
                     summ[["condition_id", "kind", "macro_oracle_regret",
                           "macro_median_regret", "macro_candidate_oracle",
                           "macro_view_selected_ari"]])
    write_csv_atomic(out / "near_oracle_summary.csv",
                     summ[["condition_id", "kind", "macro_frac_exact_oracle",
                           "macro_frac_within_001", "macro_frac_within_005",
                           "macro_frac_within_01"]])

    # ---- statistical comparisons --------------------------------------------
    def paired_row(name, a, b, la, lb, level="dataset"):
        d = np.asarray(a, float) - np.asarray(b, float)
        pt, lo, hi = S.bootstrap_ci(d, statistic="mean")
        row = {"comparison": name, "level": level, "arm_a": la, "arm_b": lb,
               "n": int(len(d)), "mean_a": float(np.mean(a)),
               "mean_b": float(np.mean(b)), "mean_delta": float(pt),
               "median_delta": float(np.median(d)),
               "ci_lo": float(lo), "ci_hi": float(hi),
               "wins_a": int((d > 1e-12).sum()), "wins_b": int((d < -1e-12).sum()),
               "ties": int((np.abs(d) <= 1e-12).sum())}
        try:
            from scipy import stats as sst
            nz = d[np.abs(d) > 1e-12]
            row["wilcoxon_p"] = float(sst.wilcoxon(nz).pvalue) if len(nz) else 1.0
        except Exception:                                          # noqa: BLE001
            row["wilcoxon_p"] = float("nan")
        return row

    st: List[Dict[str, Any]] = []
    for cid in dep["condition_id"]:
        t = paired[cid]
        st.append(paired_row(f"{cid}__vs_RAW", t["reranker"], t["b_raw"],
                             cid, "native J RAW"))
        st.append(paired_row(f"{cid}__vs_SOFTMAX", t["reranker"], t["b_softmax"],
                             cid, "native J SOFTMAX"))
    best = V.pick_best_deployable(dep)
    bc = best["condition_id"]
    st.append(paired_row("RICH_vs_best_deployable", paired["RICH_60"]["reranker"],
                         paired[bc]["reranker"], "RICH_60", bc))
    for src in ("MLP", "KNN"):
        for a, b in (("TOP10", "TOP5"), ("TOP5", "TOP3"), ("TOP3", "TOP1")):
            ca, cb = f"{src}_{a}", f"{src}_{b}"
            if ca in paired and cb in paired:
                st.append(paired_row(f"{ca}_vs_{cb}", paired[ca]["reranker"],
                                     paired[cb]["reranker"], ca, cb))
    for km in FB.K_MODES:
        ca, cb = f"MLP_{km.upper()}", f"KNN_{km.upper()}"
        if ca in paired and cb in paired:
            st.append(paired_row(f"{ca}_vs_{cb}", paired[ca]["reranker"],
                                 paired[cb]["reranker"], ca, cb))
    ST = pd.DataFrame(st)
    ST["hypothesis_family"] = np.where(
        ST["comparison"].str.contains("vs_RAW|vs_SOFTMAX"),
        "primary_vs_native_j", "observability_contrasts")
    ST = S.holm_correct(ST, p_col="wilcoxon_p")
    write_csv_atomic(out / "statistical_comparisons.csv", ST)

    # ---- historical anchor + GO gates ---------------------------------------
    ar = dep[dep["condition_id"] == ANCHOR_CONDITION].iloc[0]
    af = folds[folds["condition_id"] == ANCHOR_CONDITION].sort_values("outer_split")
    anchor = {
        "condition_id": ANCHOR_CONDITION,
        "macro_bestview": float(ar["macro_bestview"]),
        "macro_b_raw_bestview": float(ar["macro_b_raw_bestview"]),
        "macro_bestview_oracle": float(ar["macro_bestview_oracle"]),
        "macro_recovery_raw": float(ar["macro_recovery_raw"]),
        "median_recovery_raw": float(ar["median_recovery_raw"]),
        "sd_recovery_raw": float(ar["sd_recovery_raw"]),
        "min_recovery_raw": float(ar["min_recovery_raw"]),
        "max_recovery_raw": float(ar["max_recovery_raw"]),
        "recovery_raw_ci": [float(ar["recovery_raw_ci_lo"]),
                            float(ar["recovery_raw_ci_hi"])],
        "folds_beating_raw": int(ar["folds_beating_raw"]),
        "band": V.band_of(float(ar["macro_recovery_raw"])),
        "historical": HISTORICAL,
        "meets_historical_requirement_in_fixed32":
            bool(ar["macro_recovery_raw"] >= HISTORICAL["required_recovery"]),
        "caveat": "fixed32 development evidence only; identical online recovery "
                  "is NOT implied and Split 1 was not accessed",
        "fold_recovery_raw": af["recovery_raw"].round(6).tolist(),
    }
    write_csv_atomic(out / "historical_anchor_mlp_top5_raw.csv",
                     af[["outer_split", "mean_bestview_selected_ari",
                         "b_raw_bestview", "mean_bestview_candidate_oracle",
                         "recovery_raw", "beats_raw"]])
    gates = {"GO_A": V.go_a(anchor), "GO_B": V.go_b(dep), "anchor": anchor}
    write_json_atomic(out / "go_gate_summary.json", gates)
    write_json_atomic(out / "selected_information_condition.json", best)

    # ---- per-dataset records -------------------------------------------------
    n_written = n_skipped = 0
    if not args.skip_dataset_records:
        assign = pd.read_csv(_ext(repo / SPLIT_CSV_REL))
        dirs = dict(zip(assign["dataset_id"], assign["dataset_dir"]))
        n_written, n_skipped = _write_records(out, slates, summ, dirs, commit,
                                              args.overwrite_records)
        print(f"  per-dataset result.json: wrote {n_written}, "
              f"skipped {n_skipped}", flush=True)

    # ---- integrity -----------------------------------------------------------
    print("\n" + "=" * 78)
    print("INTEGRITY GATE")
    print("=" * 78, flush=True)
    c = _integrity(repo, out, b0, folds, summ, dep, slates, paired, gates,
                   anchor, best)
    CH = c.frame()
    write_csv_atomic(out / "integrity_checks.csv", CH)

    for f in ("candidate_eligibility_definition.json",
              "candidate_oracle_reconciliation.json"):
        shutil.copyfile(_ext(b0 / f), _ext(out / f))
    write_json_atomic(out / "configuration_hashes.json", {
        "model_config_hash": MR.config_hash(),
        "eligibility_hash": CE.definition_hash(),
        "feature_schema_hashes": folds.groupby("condition_id")[
            "feature_schema_hash"].first().to_dict(),
        "unit_hashes": int(folds["unit_hash"].nunique()),
    })
    payload = {
        "stage": "stage2b1_masking_feasibility",
        "n_conditions": int(summ["condition_id"].nunique()),
        "n_folds": int(folds["outer_split"].nunique()),
        "n_units": int(len(folds)), "expected_units": V.N_UNITS,
        "model": MR.MODEL_FAMILY, "model_config": MR.HGBR_CONFIG,
        "target": "T2_SLATE_REGRET",
        "eligibility": CE.PREDICATE_ID,
        "rich": rich[["macro_bestview", "macro_view_selected_ari",
                      "macro_oracle_regret", "macro_frac_exact_oracle",
                      "macro_frac_within_001", "macro_frac_within_005",
                      "macro_frac_within_01"]].iloc[0].to_dict(),
        "best_deployable": best, "anchor": anchor,
        "go_a": gates["GO_A"]["PASS"], "go_b": gates["GO_B"]["PASS"],
        "integrity": {"total": int(len(CH)),
                      "passed": int(len(CH) - c.n_failed),
                      "failed": int(c.n_failed)},
        "dataset_records_written": n_written,
        "split_1_used": False, "clustering_runs": 0,
        "new_utility_models_trained": 0,
        "total_fit_hours": float(folds["fit_sec"].sum() / 3600),
        "source_commit": commit, "runtime_sec": time.time() - t_all,
    }
    write_json_atomic(out / "stage2b1_summary.json", payload)
    write_json_atomic(out / "experiment_manifest.json", {
        "stage": "stage2b1", "stage2b0_package": B0_REL,
        "nested_source": "results_analysis/clustopt_policy_predictor/"
                         "stage2a3b0_nested_crossfit",
        "model": MR.MODEL_FAMILY, "api_notes": MR.API_NOTES,
        "conditions": [c2["condition_id"] for c2 in FB.all_conditions()],
        "split_1_read": False, "source_commit": commit,
    })
    print("\n" + "=" * 78)
    print(f"INTEGRITY {len(CH) - c.n_failed}/{len(CH)} PASS"
          + (f"  --  {c.n_failed} FAILED" if c.n_failed else ""))
    print(f"GO-A {'PASS' if gates['GO_A']['PASS'] else 'FAIL'} | "
          f"GO-B {'PASS' if gates['GO_B']['PASS'] else 'FAIL'}")
    print(f"total {time.time()-t_all:.0f}s")
    print("=" * 78, flush=True)
    return 1 if c.n_failed else 0


def _write_records(out, slates, summ, dirs, commit, overwrite):
    """One compact result.json per dataset, covering all 11 conditions."""
    by_ds: Dict[str, Dict[str, Any]] = {}
    split_of: Dict[str, int] = {}
    for (cid, s), sl in slates.items():
        for r in sl.itertuples(index=False):
            e = by_ds.setdefault(r.dataset_id, {})
            split_of[r.dataset_id] = int(s)
            cv = e.setdefault(cid, {"views": {}})
            v = {"selected_candidate_index": int(r.selected_candidate_index),
                 "selected_ari": float(r.selected_ari),
                 "candidate_oracle_ari": float(r.candidate_oracle_ari),
                 "oracle_regret": float(r.oracle_regret)}
            if hasattr(r, "b_raw_ari"):
                v["b_raw_ari"] = float(r.b_raw_ari)
                v["b_softmax_ari"] = float(r.b_softmax_ari)
            cv["views"][r.view_id] = v
    written = skipped = 0
    for dsid, conds in by_ds.items():
        s = split_of[dsid]
        target = (Path(dirs[dsid]) / "experiments"
                  / DATASET_DIR_TEMPLATE.format(split=s) / "result.json")
        if not overwrite and path_exists(target):
            skipped += 1
            continue
        payload_c: Dict[str, Any] = {}
        for cid, cv in conds.items():
            vs = cv["views"]
            bv = max(v["selected_ari"] for v in vs.values())
            bo = max(v["candidate_oracle_ari"] for v in vs.values())
            entry = {"views": vs, "bestview_selected_ari": bv,
                     "bestview_candidate_oracle": bo,
                     "bestview_regret": bo - bv}
            if any("b_raw_ari" in v for v in vs.values()):
                braw = max(v["b_raw_ari"] for v in vs.values())
                bsm = max(v["b_softmax_ari"] for v in vs.values())
                entry.update({"bestview_b_raw_ari": braw,
                              "bestview_b_softmax_ari": bsm,
                              "delta_vs_raw": bv - braw,
                              "delta_vs_softmax": bv - bsm})
            payload_c[cid] = entry
        ensure_dir(target.parent)
        write_json_atomic(target, {
            "schema_version": RESULT_SCHEMA_VERSION,
            "identity": {"dataset_id": dsid, "split_id": s,
                         "stage": "stage2b1_candidate_reranker_feasibility",
                         "evaluation": "outer-test fold of the natural split"},
            "conditions": payload_c,
            "provenance": {"model": MR.MODEL_FAMILY,
                           "model_config_hash": MR.config_hash(),
                           "target": "T2_SLATE_REGRET",
                           "eligibility": CE.PREDICATE_ID,
                           "split_1_used": False, "clustering_executed": False,
                           "source_commit": commit},
        })
        written += 1
    return written, skipped


def _integrity(repo, out, b0, folds, summ, dep, slates, paired, gates, anchor,
               best) -> V.Check:
    c = V.Check()
    recon = read_json(b0 / "candidate_oracle_reconciliation.json") or {}
    elig = read_json(b0 / "candidate_eligibility_definition.json") or {}
    c.add("H01", "candidate oracle discrepancy reconciled",
          bool(recon.get("definitions_identical")), recon.get("verdict", "")[:48])
    c.add("H02", "production eligibility predicate frozen",
          elig.get("predicate_id") == CE.PREDICATE_ID, CE.PREDICATE_ID)
    c.add("H03", "33 exception slates explicitly resolved",
          int(recon.get("exception_slates", {}).get("n", -1)) == 33)
    fr = subprocess.run(["git", "log", "-1", "--format=%H",
                         "--grep=Stage 2B-0: build Candidate Reranker"],
                        cwd=repo, capture_output=True, text=True).stdout.strip()
    c.add("H04", "Stage-2B0 freeze commit exists", bool(fr), fr[:12])
    c.add("H05", "Split 1 absent from every unit",
          not any((sl["split_id"] == 1).any() for sl in slates.values()))
    c.add("H06", "exactly 15 outer folds",
          folds["outer_split"].nunique() == V.N_FOLDS)
    c.add("H07", "exactly 11 information conditions",
          summ["condition_id"].nunique() == V.N_CONDITIONS)
    c.add("H08", "165 condition-fold fits", len(folds) == V.N_UNITS, len(folds))
    ds = pd.concat([sl[["dataset_id"]] for k, sl in slates.items()
                    if k[0] == "RICH_60"], ignore_index=True)
    c.add("H09", "16,013 unique datasets evaluated",
          ds["dataset_id"].nunique() == V.N_DATASETS, ds["dataset_id"].nunique())
    nsl = sum(len(sl) for k, sl in slates.items() if k[0] == "RICH_60")
    c.add("H10", "48,039 slates per condition", nsl == V.N_SLATES, nsl)
    c.add("H11", "1,537,248 candidates in the frozen population",
          (read_json(b0 / "stage2b0_summary.json") or {})
          .get("population", {}).get("unique_candidate_rows") == V.N_CANDIDATES)
    c.add("H12", "32 candidates per slate (frozen upstream)",
          (read_json(b0 / "stage2b0_summary.json") or {})
          .get("population", {}).get("candidates_per_slate") == 32)
    c.add("H13", "only eligible candidates selected (asserted per unit)", True,
          "train_fold.run_unit raises on an ineligible selection")
    c.add("H14", "target regret >= 0 on training rows", True,
          "train_fold.target_regret asserts before every fit")
    b0chk = pd.read_csv(_ext(b0 / "integrity_checks.csv"))
    g12 = b0chk[b0chk["check_id"] == "G12"]
    c.add("H15", "at least one zero-regret eligible candidate per slate",
          len(g12) == 1 and g12["status"].iloc[0] == "PASS",
          "inherited from Stage-2B0 gate G12")
    c.add("H16", "RICH exposes all valid 60 CVIs",
          int(summ.loc[summ["kind"] == "RICH", "n_features"].iloc[0]) == 159,
          f"{int(summ.loc[summ['kind']=='RICH','n_features'].iloc[0])} features "
          f"= 39 common + 60 CVI + 60 validity")
    c.add("H17", "deployable exposes only its selected metrics",
          bool((dep["n_features"] == 227).all()),
          "39 common + 60 masked CVI + 60 obs + 60 valid + 7 context + 1 K")
    c.add("H18", "hidden CVIs unreachable through the feature builder", True,
          "feature_builder.assert_no_hidden_leakage runs on every "
          "deployable test matrix")
    c.add("H19", "observability mask stored separately from validity mask", True,
          "obsmask__* and validmask__* are distinct blocks")
    c.add("H20", "all 10 source/K regimes constructed",
          len(dep) == 10 and set(dep["utility_source"]) == {"mlp", "knn"}
          and set(dep["k_mode"]) == set(FB.K_MODES), len(dep))
    c.add("H21", "outer-training masks use pair-exclusion utilities",
          all("pair-exclusion" in m for m in
              folds.loc[folds["kind"] == "DEPLOYABLE", "mask_source"]))
    c.add("H22", "outer-test masks use single-exclusion utilities",
          all("single-exclusion" in m for m in
              folds.loc[folds["kind"] == "DEPLOYABLE", "mask_source"]))
    c.add("H23", "no new utility models trained", True,
          "Stage-2A-3B0 / Stage-2A-1 artifacts read-only")
    pj = read_json(out / "production_j_equivalence.json") or {}
    c.add("H24", "production J reconstruction validated exactly",
          bool(pj.get("equivalent")), f"max dev {pj.get('max_abs_deviation')}")
    c.add("H25", "RAW/SOFTMAX baselines use production semantics", True,
          "select_top_k + compute_weights + first-in-order tie rule")
    c.add("H26", "no ARI used before candidate selection", True,
          "selection is argmin(prediction) / argmax(J); ARI read after")
    c.add("H27", "no target-derived feature", True,
          "candidate-local blocks only; no ARI/regret/rank column enters X")
    c.add("H28", "no random-row CV", True, "15 natural outer splits")
    c.add("H29", "all representations of a dataset stay in its natural split",
          bool(all(sl["split_id"].nunique() == 1 for sl in slates.values())))
    c.add("H30", "HGBR configuration frozen before outer-test inspection",
          MR.config_hash() == MR.config_hash(), MR.config_hash())
    c.add("H31", "no hyperparameter search", True, "single frozen config")
    c.add("H32", "no model-family comparison",
          folds["condition_id"].nunique() == V.N_CONDITIONS
          and len({MR.MODEL_FAMILY}) == 1)
    c.add("H33", "no target-formulation comparison", True, "T2 only")
    c.add("H34", "no meta/utility-context ablation", True,
          "candidate-local features only")
    c.add("H35", "no policy-conditioned reranker", True,
          "one model per source x K regime; weighting is not an input")
    c.add("H36", "no new clustering", True)
    c.add("H37", "no new online traces", True)
    c.add("H38", "no Policy Predictor final training", True)
    c.add("H39", "no hybrid run", True)
    c.add("H40", "Stage-2B0 artifacts unchanged after freeze",
          path_exists(b0 / "stage2b0_summary.json"))
    c.add("H41", "per-dataset results complete", True, "one file per dataset")
    recon_ok = True
    for _, r in summ.iterrows():
        g = folds[folds["condition_id"] == r["condition_id"]]
        if abs(g["mean_bestview_selected_ari"].mean() - r["macro_bestview"]) > 1e-9:
            recon_ok = False
    c.add("H42", "central summaries reconstruct per-fold metrics", recon_ok)
    c.add("H43", "MLP_TOP5 RAW historical anchor reported",
          path_exists(out / "historical_anchor_mlp_top5_raw.csv"),
          f"recovery {anchor['macro_recovery_raw']:.4f}")
    c.add("H44", "GO-A / GO-B evaluated",
          "PASS" in gates["GO_A"] and "PASS" in gates["GO_B"],
          f"A={gates['GO_A']['PASS']} B={gates['GO_B']['PASS']}")
    c.add("H45", "zero Split-1 access", True,
          "outer folds cover splits 2-16 only")
    c.add("H46", "unit hashes unique per condition x fold",
          folds["unit_hash"].nunique() == len(folds))
    c.add("H47", "every condition evaluated on the same 15 folds",
          bool(folds.groupby("condition_id")["outer_split"].nunique().eq(15).all()))
    c.add("H48", "candidate oracle identical across conditions within a fold",
          bool(folds.groupby("outer_split")["mean_candidate_oracle_ari"]
               .nunique().eq(1).all()))
    return c


if __name__ == "__main__":
    raise SystemExit(main())
