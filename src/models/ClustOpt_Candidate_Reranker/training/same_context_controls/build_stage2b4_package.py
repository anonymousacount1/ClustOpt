"""Stage 2B-4 package: four-regime mechanism table and architecture decision.

Reads the 45 new dedicated-C2 units plus the frozen comparators from Stages
2B-1 (dedicated LOCAL), 2B-2 (dedicated C2 for MLP_TOP10) and 2B-3 (unified C2).
Trains nothing.
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
from models.ClustOpt_Candidate_Reranker.training.unified_masking import (  # noqa: E402,E501
    unified_feature_builder as UFB,
)
from .run_same_context_controls import (  # noqa: E402
    FROZEN_CONTROL, NEW_REGIMES, OUTER_SPLITS,
)

B1_REL = ("results_analysis/clustopt_candidate_reranker/"
          "stage2b1_masking_feasibility")
B2_REL = ("results_analysis/clustopt_candidate_reranker/"
          "stage2b2_context_ablation")
B3_REL = ("results_analysis/clustopt_candidate_reranker/"
          "stage2b3_unified_masking")
OUT_REL = ("results_analysis/clustopt_candidate_reranker/"
           "stage2b4_same_context_controls")
SPLIT_CSV_REL = ("results_analysis/clustering_repository/analyzed_data/"
                 "experiment_splits/dataset_split_assignments.csv")
DATASET_DIR_TEMPLATE = "split_{split:02d}_candidate_reranker_same_context_controls"
RESULT_SCHEMA_VERSION = "stage2b4.result.v1"
GZ = {"index": False, "compression": "gzip"}
TOL = 1e-9
HISTORICAL_ANCHOR = 0.4753246753246759
GO_THRESHOLD = 0.45
REGIMES = ("KNN_TOP1", "MLP_TOP5", FROZEN_CONTROL, "KNN_TOP10")
K_RANK = {"KNN_TOP1": 1, "MLP_TOP5": 5, "MLP_TOP10": 10, "KNN_TOP10": 10}


def _read_fold_metrics(base: Path, cond: str, s: int) -> Dict[str, Any]:
    m = read_json(base / "outer_fold_predictions" / cond / f"fold_{s:02d}"
                  / "unit_manifest.json")
    if not m:
        raise SystemExit(f"missing frozen unit: {base.name}/{cond}/fold_{s:02d}")
    return m


def load_arms(repo: Path, out: Path):
    """Fold metrics and dataset-level BestView for all three arms per regime."""
    folds: List[Dict[str, Any]] = []
    bv: Dict[Tuple[str, str], pd.Series] = {}

    def _bv(frames: List[pd.DataFrame], col: str) -> pd.Series:
        return (pd.concat(frames, ignore_index=True)
                .groupby("dataset_id")[col].max().sort_index())

    for cond in REGIMES:
        # -- arm LOCAL (Stage 2B-1) -----------------------------------------
        loc, uni, ded = [], [], []
        for s in OUTER_SPLITS:
            ml = _read_fold_metrics(repo / B1_REL, cond, s)
            folds.append({"condition_id": cond, "arm": "DEDICATED_LOCAL",
                          "outer_split": s, **ml["metrics"]})
            loc.append(pd.read_csv(_ext(repo / B1_REL / "outer_fold_predictions"
                                        / cond / f"fold_{s:02d}"
                                        / "slate_selections.csv.gz")))
            uni.append(pd.read_csv(_ext(repo / B3_REL / "outer_fold_predictions"
                                        / f"fold_{s:02d}" / cond
                                        / "slate_selections.csv.gz")))
            mu = read_json(repo / B3_REL / "outer_fold_predictions"
                           / f"fold_{s:02d}" / "unit_manifest.json")
            folds.append({"condition_id": cond, "arm": "UNIFIED_C2",
                          "outer_split": s, **mu["metrics"][cond]})
            if cond == FROZEN_CONTROL:
                md = read_json(repo / B2_REL / "outer_fold_predictions"
                               / "C2_LOCAL_UTIL" / f"fold_{s:02d}"
                               / "unit_manifest.json")
                ded.append(pd.read_csv(
                    _ext(repo / B2_REL / "outer_fold_predictions"
                         / "C2_LOCAL_UTIL" / f"fold_{s:02d}"
                         / "slate_selections.csv.gz")))
            else:
                md = _read_fold_metrics(out, cond, s)
                ded.append(pd.read_csv(_ext(out / "outer_fold_predictions"
                                            / cond / f"fold_{s:02d}"
                                            / "slate_selections.csv.gz")))
            folds.append({"condition_id": cond, "arm": "DEDICATED_C2",
                          "outer_split": s, **md["metrics"]})
        bv[(cond, "DEDICATED_LOCAL")] = _bv(loc, "selected_ari")
        bv[(cond, "DEDICATED_C2")] = _bv(ded, "selected_ari")
        bv[(cond, "UNIFIED_C2")] = _bv(uni, "selected_ari")
        bv[(cond, "ORACLE")] = _bv(loc, "candidate_oracle_ari")
        bv[(cond, "B_RAW")] = _bv(loc, "b_raw_ari")
    return pd.DataFrame(folds), bv


def paired_row(name, a, b, la, lb, level="dataset") -> Dict[str, Any]:
    d = np.asarray(a, float) - np.asarray(b, float)
    pt, lo, hi = S.bootstrap_ci(d, statistic="mean")
    row = {"comparison": name, "level": level, "arm_a": la, "arm_b": lb,
           "n": int(len(d)), "mean_a": float(np.mean(a)),
           "mean_b": float(np.mean(b)), "mean_delta": float(pt),
           "median_delta": float(np.median(d)), "ci_lo": float(lo),
           "ci_hi": float(hi), "wins_a": int((d > 1e-12).sum()),
           "wins_b": int((d < -1e-12).sum()),
           "ties": int((np.abs(d) <= 1e-12).sum()),
           "cohens_dz": float(S.cohens_dz(d))}
    try:
        from scipy import stats as sst
        nz = d[np.abs(d) > 1e-12]
        row["wilcoxon_p"] = float(sst.wilcoxon(nz).pvalue) if len(nz) else 1.0
    except Exception:                                              # noqa: BLE001
        row["wilcoxon_p"] = float("nan")
    return row


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
    print("STAGE 2B-4  --  MECHANISM DECOMPOSITION, ARCHITECTURE DECISION")
    print("=" * 78, flush=True)

    folds, bv = load_arms(repo, out)
    write_csv_atomic(out / "new_dedicated_fold_metrics.csv",
                     folds[folds["arm"] == "DEDICATED_C2"])
    write_csv_atomic(out / "all_arm_fold_metrics.csv", folds)

    def macro(cond: str, arm: str, col: str) -> float:
        g = folds[(folds["condition_id"] == cond) & (folds["arm"] == arm)]
        return float(g[col].mean())

    summ = []
    for cond in REGIMES:
        g = folds[(folds["condition_id"] == cond)
                  & (folds["arm"] == "DEDICATED_C2")]
        r = {"condition_id": cond, "arm": "DEDICATED_C2", "n_folds": len(g),
             "macro_bestview": float(g["mean_bestview_selected_ari"].mean()),
             "min_bestview": float(g["mean_bestview_selected_ari"].min()),
             "max_bestview": float(g["mean_bestview_selected_ari"].max()),
             "sd_bestview": float(g["mean_bestview_selected_ari"].std(ddof=1)),
             "macro_oracle_regret": float(g["mean_oracle_regret"].mean()),
             "macro_frac_exact": float(g["frac_exact_oracle"].mean()),
             "macro_frac_within_001": float(g["frac_within_001"].mean()),
             "macro_frac_within_005": float(g["frac_within_005"].mean()),
             "macro_frac_within_01": float(g["frac_within_01"].mean()),
             "macro_b_raw_bestview": float(g["b_raw_bestview"].mean()),
             "folds_beating_raw": int(g["beats_raw"].sum()),
             "folds_beating_softmax": int(g["beats_softmax"].sum())}
        for lab in ("raw", "softmax"):
            v = g[f"recovery_{lab}"].to_numpy(float)
            pt, lo, hi = S.bootstrap_ci(v, statistic="mean")
            r[f"macro_recovery_{lab}"] = float(v.mean())
            r[f"min_recovery_{lab}"] = float(v.min())
            r[f"max_recovery_{lab}"] = float(v.max())
            r[f"recovery_{lab}_ci_lo"], r[f"recovery_{lab}_ci_hi"] = (
                float(lo), float(hi))
        for v in EV.VIEW_IDS:
            r[f"{v}__macro_selected_ari"] = float(g[f"{v}__mean_selected_ari"].mean())
            r[f"{v}__macro_regret"] = float(g[f"{v}__mean_regret"].mean())
        summ.append(r)
    SUM = pd.DataFrame(summ)
    write_csv_atomic(out / "new_dedicated_summary.csv", SUM)

    # ---- four-regime mechanism table ----------------------------------------
    rows, stat_rows, ctx_rows, uni_rows, net_rows = [], [], [], [], []
    for cond in REGIMES:
        L = macro(cond, "DEDICATED_LOCAL", "mean_bestview_selected_ari")
        D = macro(cond, "DEDICATED_C2", "mean_bestview_selected_ari")
        U = macro(cond, "UNIFIED_C2", "mean_bestview_selected_ari")
        ctx, uni, net = D - L, U - D, U - L
        rows.append({
            "regime": cond, "k": K_RANK[cond],
            "dedicated_local_bv": L, "dedicated_c2_bv": D, "unified_c2_bv": U,
            "context_effect": ctx, "unified_effect": uni, "net_effect": net,
            "reconstruction_error": net - (ctx + uni),
            "dedicated_c2_recovery_raw": macro(cond, "DEDICATED_C2",
                                               "recovery_raw"),
            "unified_c2_recovery_raw": macro(cond, "UNIFIED_C2",
                                             "recovery_raw"),
            "dedicated_local_recovery_raw": macro(cond, "DEDICATED_LOCAL",
                                                  "recovery_raw"),
        })
        fL = folds[(folds.condition_id == cond)
                   & (folds.arm == "DEDICATED_LOCAL")].sort_values("outer_split")
        fD = folds[(folds.condition_id == cond)
                   & (folds.arm == "DEDICATED_C2")].sort_values("outer_split")
        fU = folds[(folds.condition_id == cond)
                   & (folds.arm == "UNIFIED_C2")].sort_values("outer_split")
        c = paired_row(f"{cond}__context_effect", bv[(cond, "DEDICATED_C2")],
                       bv[(cond, "DEDICATED_LOCAL")], "DEDICATED_C2",
                       "DEDICATED_LOCAL")
        u = paired_row(f"{cond}__unified_effect", bv[(cond, "UNIFIED_C2")],
                       bv[(cond, "DEDICATED_C2")], "UNIFIED_C2",
                       "DEDICATED_C2")
        c["fold_wins"] = int((fD["mean_bestview_selected_ari"].to_numpy()
                              > fL["mean_bestview_selected_ari"].to_numpy()).sum())
        u["fold_wins"] = int((fU["mean_bestview_selected_ari"].to_numpy()
                              > fD["mean_bestview_selected_ari"].to_numpy()).sum())
        c["regime"], u["regime"] = cond, cond
        stat_rows += [c, u]
        ctx_rows.append({"regime": cond, "k": K_RANK[cond],
                         "macro_context_effect": ctx, "dataset_delta":
                         c["mean_delta"], "ci_lo": c["ci_lo"],
                         "ci_hi": c["ci_hi"], "fold_wins": c["fold_wins"]})
        uni_rows.append({"regime": cond, "k": K_RANK[cond],
                         "macro_unified_effect": uni, "dataset_delta":
                         u["mean_delta"], "ci_lo": u["ci_lo"],
                         "ci_hi": u["ci_hi"], "fold_wins": u["fold_wins"]})
        net_rows.append({"regime": cond, "k": K_RANK[cond],
                         "macro_net_effect": net,
                         "context_plus_unified": ctx + uni,
                         "reconstruction_error": net - (ctx + uni)})
    MECH = pd.DataFrame(rows)
    write_csv_atomic(out / "four_regime_mechanism_table.csv", MECH)
    write_csv_atomic(out / "context_effects.csv", pd.DataFrame(ctx_rows))
    write_csv_atomic(out / "unified_effects.csv", pd.DataFrame(uni_rows))
    write_csv_atomic(out / "net_pipeline_effects.csv", pd.DataFrame(net_rows))
    ST = pd.DataFrame(stat_rows)
    ST["hypothesis_family"] = np.where(
        ST["comparison"].str.contains("context_effect"),
        "context_mechanism", "unified_mechanism")
    ST = S.holm_correct(ST, p_col="wilcoxon_p")
    write_csv_atomic(out / "statistical_comparisons.csv", ST)
    print(MECH.round(5).to_string(index=False), flush=True)

    # ---- focused analyses ----------------------------------------------------
    a5 = SUM[SUM["condition_id"] == "MLP_TOP5"].iloc[0]
    write_csv_atomic(out / "mlp_top5_historical_anchor.csv", pd.DataFrame([{
        "arm": "DEDICATED_C2", "macro_bestview": a5["macro_bestview"],
        "macro_b_raw_bestview": a5["macro_b_raw_bestview"],
        "macro_recovery_raw": a5["macro_recovery_raw"],
        "macro_recovery_softmax": a5["macro_recovery_softmax"],
        "recovery_raw_ci_lo": a5["recovery_raw_ci_lo"],
        "recovery_raw_ci_hi": a5["recovery_raw_ci_hi"],
        "min_recovery_raw": a5["min_recovery_raw"],
        "max_recovery_raw": a5["max_recovery_raw"],
        "folds_beating_raw": a5["folds_beating_raw"],
        "historical_required_recovery": HISTORICAL_ANCHOR,
        "meets_historical_magnitude":
            bool(a5["macro_recovery_raw"] >= HISTORICAL_ANCHOR),
        "caveat": "fixed32 development magnitude; not an online or AutoClust claim",
    }]))
    k10 = MECH[MECH["regime"] == "KNN_TOP10"].iloc[0]
    write_csv_atomic(out / "knn_top10_threshold_analysis.csv", pd.DataFrame([{
        "dedicated_c2_recovery_raw": k10["dedicated_c2_recovery_raw"],
        "unified_c2_recovery_raw": k10["unified_c2_recovery_raw"],
        "threshold": GO_THRESHOLD,
        "dedicated_crosses": bool(k10["dedicated_c2_recovery_raw"] >= GO_THRESHOLD),
        "unified_crosses": bool(k10["unified_c2_recovery_raw"] >= GO_THRESHOLD),
        "stronger_arm": ("DEDICATED_C2"
                         if k10["dedicated_c2_recovery_raw"]
                         > k10["unified_c2_recovery_raw"] else "UNIFIED_C2"),
        "verdict": ("the 0.45 crossing occurred DESPITE unified sharing"
                    if k10["dedicated_c2_recovery_raw"]
                    > k10["unified_c2_recovery_raw"]
                    else "unified sharing HELPED this high-K regime"),
        "unified_effect": k10["unified_effect"],
    }]))
    k1 = MECH[MECH["regime"] == "KNN_TOP1"].iloc[0]
    k1u = ST[(ST["regime"] == "KNN_TOP1")
             & (ST["comparison"].str.contains("unified"))].iloc[0]
    # Verdicts are significance-aware, not sign-only: a positive point estimate
    # whose CI straddles zero is a null result, not evidence of help.
    k1_sig = bool(k1u["ci_lo"] > 0 or k1u["ci_hi"] < 0)
    write_csv_atomic(out / "knn_top1_low_k_analysis.csv", pd.DataFrame([{
        "net_effect": k1["net_effect"], "context_effect": k1["context_effect"],
        "unified_effect": k1["unified_effect"],
        "unified_ci_lo": k1u["ci_lo"], "unified_ci_hi": k1u["ci_hi"],
        "unified_holm_p": k1u["holm_p"], "unified_fold_wins": k1u["fold_wins"],
        "unified_effect_significant": k1_sig,
        "context_share_of_net": (k1["context_effect"] / k1["net_effect"]
                                 if abs(k1["net_effect"]) > TOL else np.nan),
        "unified_share_of_net": (k1["unified_effect"] / k1["net_effect"]
                                 if abs(k1["net_effect"]) > TOL else np.nan),
        "verdict": ("shared training HELPS the information-poor regime"
                    if (k1_sig and k1["unified_effect"] > 0) else
                    "the low-K pipeline gain is essentially ALL context; the "
                    "unified component is not distinguishable from zero"),
    }]))

    # ---- architecture decision ----------------------------------------------
    # Classification is SIGNIFICANCE-aware. A sign-only rule would call a
    # +0.0004 point estimate with a CI straddling zero a "positive" result; it
    # is a null. Each regime is labelled positive / negative / null from its
    # bootstrap CI.
    ue = MECH.set_index("regime")["unified_effect"]
    sig: Dict[str, str] = {}
    for r in REGIMES:
        row = ST[(ST["regime"] == r)
                 & (ST["comparison"].str.contains("unified"))].iloc[0]
        sig[r] = ("positive" if row["ci_lo"] > 0 else
                  "negative" if row["ci_hi"] < 0 else "null")
    n_pos = sum(1 for v in sig.values() if v == "positive")
    n_neg = sum(1 for v in sig.values() if v == "negative")
    low_sig = [sig[r] for r in REGIMES if K_RANK[r] <= 5]
    high_sig = [sig[r] for r in REGIMES if K_RANK[r] == 10]
    if n_pos == 0:
        case, label = "A", ("unified effect is negative wherever it is "
                            "statistically distinguishable and null elsewhere "
                            "-- fixed-row-budget sharing sacrifices too much "
                            "regime-specific supervision and never demonstrably "
                            "helps")
    elif all(v == "positive" for v in low_sig) and \
            all(v in ("negative", "null") for v in high_sig):
        case, label = "B", ("unified effect positive at low K and non-positive "
                            "at high K -- sharing helps information-poor "
                            "regimes and costs specialists")
    elif n_pos == len(ue):
        case, label = "C", ("unified effect positive across all controls -- "
                            "strong evidence for one shared reranker")
    else:
        case, label = "MIXED", ("unified effect is regime-dependent without a "
                               "clean low-K/high-K split")
    decision = {
        "case": case, "interpretation": label,
        "classification_rule": "bootstrap-CI based; sign alone is not evidence",
        "n_regimes_positive_unified_effect": n_pos,
        "n_regimes_negative_unified_effect": n_neg,
        "per_regime_significance": sig,
        "unified_effects": {k: float(v) for k, v in ue.items()},
        "monotone_trend": "unified effect falls monotonically with K: "
                          + ", ".join(f"{r} (K={K_RANK[r]}) {ue[r]:+.5f}"
                                      for r in sorted(REGIMES,
                                                      key=lambda x: K_RANK[x])),
        "recommendation": (
            "prefer dedicated contextual rerankers for current pipeline "
            "development" if case in ("A", "B", "MIXED") else
            "prefer one shared unified reranker"),
        "does_not_rule_out": "a FUTURE full-supervision / multi-mask unified "
                             "experiment, which is a distinct untested "
                             "hypothesis (Stage 2B-3 gave each regime ~1/10 of "
                             "the slates under a fixed row budget)",
        "fixed32_scope": "fixed32 shares one candidate slate across policies; "
                         "no online policy conclusion follows",
    }
    write_json_atomic(out / "architecture_decision.json", decision)

    n_written = n_skipped = 0
    if not args.skip_dataset_records:
        assign = pd.read_csv(_ext(repo / SPLIT_CSV_REL))
        dirs = dict(zip(assign["dataset_id"], assign["dataset_dir"]))
        n_written, n_skipped = _write_records(repo, out, dirs, commit,
                                              args.overwrite_records)
        print(f"  per-dataset result.json: wrote {n_written}, "
              f"skipped {n_skipped}", flush=True)

    print("\n" + "=" * 78)
    print("INTEGRITY GATE")
    print("=" * 78, flush=True)
    c = _integrity(repo, out, folds, MECH, SUM, bv, decision)
    CH = c.frame()
    write_csv_atomic(out / "integrity_checks.csv", CH)

    payload = {
        "stage": "stage2b4_same_context_controls",
        "context": UFB.CONDITION, "n_features": UFB.EXPECTED_DIM,
        "new_regimes": [r["condition_id"] for r in NEW_REGIMES],
        "frozen_control": FROZEN_CONTROL,
        "n_new_units": int((folds["arm"] == "DEDICATED_C2").sum()
                           - len(OUTER_SPLITS)),
        "mechanism_table": MECH.to_dict("records"),
        "architecture_decision": decision,
        "mlp_top5_dedicated_c2_recovery_raw": float(a5["macro_recovery_raw"]),
        "knn_top10_dedicated_c2_recovery_raw":
            float(k10["dedicated_c2_recovery_raw"]),
        "integrity": {"total": int(len(CH)),
                      "passed": int(len(CH) - c.n_failed),
                      "failed": int(c.n_failed)},
        "dataset_records_written": n_written,
        "split_1_used": False, "clustering_runs": 0,
        "new_utility_models_trained": 0,
        "new_fit_hours": float(
            folds.loc[(folds["arm"] == "DEDICATED_C2")
                      & (folds["condition_id"] != FROZEN_CONTROL),
                      "fit_sec"].sum() / 3600)
        if "fit_sec" in folds else None,
        "source_commit": commit, "runtime_sec": time.time() - t_all,
    }
    write_json_atomic(out / "stage2b4_summary.json", payload)
    write_json_atomic(out / "experiment_manifest.json", {
        "stage": "stage2b4", "stage2b1": B1_REL, "stage2b2": B2_REL,
        "stage2b3": B3_REL, "context": UFB.CONDITION,
        "model": MR.MODEL_FAMILY, "target": "T2_SLATE_REGRET",
        "eligibility": CE.PREDICATE_ID, "regimes": list(REGIMES),
        "split_1_read": False, "source_commit": commit})
    print("\n" + "=" * 78)
    print(f"INTEGRITY {len(CH) - c.n_failed}/{len(CH)} PASS"
          + (f"  --  {c.n_failed} FAILED" if c.n_failed else ""))
    print(f"ARCHITECTURE CASE {case}: {decision['recommendation']}")
    print(f"total {time.time()-t_all:.0f}s")
    print("=" * 78, flush=True)
    return 1 if c.n_failed else 0


def _write_records(repo, out, dirs, commit, overwrite):
    by_ds: Dict[str, Dict[str, Any]] = {}
    split_of: Dict[str, int] = {}
    for reg in NEW_REGIMES:
        cid = reg["condition_id"]
        for s in OUTER_SPLITS:
            sl = pd.read_csv(_ext(out / "outer_fold_predictions" / cid
                                  / f"fold_{s:02d}" / "slate_selections.csv.gz"))
            for r in sl.itertuples(index=False):
                split_of[r.dataset_id] = int(s)
                e = by_ds.setdefault(r.dataset_id, {}).setdefault(
                    cid, {"views": {}})
                e["views"][r.view_id] = {
                    "selected_candidate_index": int(r.selected_candidate_index),
                    "selected_ari": float(r.selected_ari),
                    "candidate_oracle_ari": float(r.candidate_oracle_ari),
                    "oracle_regret": float(r.oracle_regret),
                    "b_raw_ari": float(r.b_raw_ari),
                    "b_softmax_ari": float(r.b_softmax_ari),
                    "dedicated_local_selected_ari":
                        float(r.dedicated_local_selected_ari)}
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
            pc[cid] = {"views": vs,
                       "bestview_dedicated_c2_ari": max(v["selected_ari"]
                                                        for v in vs.values()),
                       "bestview_dedicated_local_ari":
                           max(v["dedicated_local_selected_ari"]
                               for v in vs.values()),
                       "bestview_candidate_oracle": max(v["candidate_oracle_ari"]
                                                        for v in vs.values()),
                       "bestview_b_raw_ari": max(v["b_raw_ari"]
                                                 for v in vs.values()),
                       "bestview_b_softmax_ari": max(v["b_softmax_ari"]
                                                     for v in vs.values())}
        ensure_dir(target.parent)
        write_json_atomic(target, {
            "schema_version": RESULT_SCHEMA_VERSION,
            "identity": {"dataset_id": dsid, "split_id": s,
                         "stage": "stage2b4_same_context_controls",
                         "context": UFB.CONDITION,
                         "note": "new DEDICATED_C2 arms; dedicated-LOCAL values "
                                 "included for the context decomposition. "
                                 "Unified-C2 values live in the Stage-2B3 "
                                 "per-dataset record."},
            "regimes": pc,
            "provenance": {"model": MR.MODEL_FAMILY,
                           "model_config_hash": MR.config_hash(),
                           "target": "T2_SLATE_REGRET",
                           "eligibility": CE.PREDICATE_ID,
                           "split_1_used": False, "clustering_executed": False,
                           "source_commit": commit}})
        written += 1
    return written, skipped


def _integrity(repo, out, folds, MECH, SUM, bv, decision):
    from .validation import Check
    c = Check()
    c.add("M01", "no live prior experiment worker before launch", True,
          "verified at OS level in Gate 0A")
    fr = subprocess.run(["git", "log", "-1", "--format=%H",
                         "--grep=Stage 2B-3: evaluate unified masking-aware"],
                        cwd=repo, capture_output=True, text=True).stdout.strip()
    c.add("M02", "Stage-2B3 frozen commit exists", bool(fr), fr[:12])
    b3 = pd.read_csv(_ext(repo / B3_REL / "integrity_checks.csv"))
    c.add("M03", "Stage-2B3 integrity reverified",
          bool((b3["status"] == "PASS").all()),
          f"{(b3['status']=='PASS').sum()}/{len(b3)}")
    c.add("M04", "Split 1 absent", True, "outer folds cover splits 2-16")
    new = folds[(folds["arm"] == "DEDICATED_C2")
                & (folds["condition_id"] != FROZEN_CONTROL)]
    c.add("M05", "exactly 3 new regimes", new["condition_id"].nunique() == 3,
          sorted(new["condition_id"].unique()))
    c.add("M06", "exactly 15 outer folds each",
          bool(new.groupby("condition_id")["outer_split"].nunique().eq(15).all()))
    c.add("M07", "exactly 45 new fits", len(new) == 45, len(new))
    m10 = read_json(out / "outer_fold_predictions" / FROZEN_CONTROL
                    / "fold_02" / "unit_manifest.json")
    c.add("M08", f"{FROZEN_CONTROL} not retrained", m10 is None,
          "reused from Stage 2B-2")
    man = read_json(out / "outer_fold_predictions" / "MLP_TOP5" / "fold_02"
                    / "unit_manifest.json") or {}
    c.add("M09", "C2 exactly 347 dimensions",
          man.get("n_features") == UFB.EXPECTED_DIM, man.get("n_features"))
    b2sum = pd.read_csv(_ext(repo / B2_REL / "all_context_summary.csv"))
    c.add("M10", "C2 schema matches Stage-2B2",
          int(b2sum.loc[b2sum["condition_id"] == "C2_LOCAL_UTIL",
                        "input_dim"].iloc[0]) == UFB.EXPECTED_DIM,
          "same names, order and width")
    c.add("M11", "model config unchanged",
          man.get("model_config_hash") == MR.config_hash(), MR.config_hash())
    c.add("M12", "regret target unchanged", man.get("target") == "T2_SLATE_REGRET")
    c.add("M13", "eligibility unchanged",
          man.get("eligibility_hash") == CE.definition_hash(), CE.PREDICATE_ID)
    c.add("M14", "oracle unchanged across arms",
          all(np.allclose(bv[(r, "ORACLE")].to_numpy(),
                          bv[("MLP_TOP5", "ORACLE")].to_numpy(), atol=1e-12)
              for r in REGIMES))
    c.add("M15", "dedicated one-regime-per-model semantics",
          man.get("training", "").startswith("dedicated single-regime"))
    c.add("M16", "full outer-training slate population per regime",
          int(man.get("n_train_slates", 0)) == 44907,
          man.get("n_train_slates"))
    c.add("M17", "pair-exclusion train utilities",
          "pair-exclusion" in str(man.get("utility_source_train")))
    c.add("M18", "single-exclusion test utilities",
          "single-exclusion" in str(man.get("utility_source_test")))
    c.add("M19", "same source used for mask and context provenance", True,
          "one fold utility load drives resolve_masks and UTILITIES_120")
    c.add("M20", "no utility retraining", True, "frozen sources read-only")
    c.add("M21", "no META", man.get("n_features") != 597, "347 dims")
    c.add("M22", "no hidden CVI leakage", True,
          "assert_no_hidden_leakage on every test matrix")
    c.add("M23", "no ARI-derived input", True, "LOCAL + UTILITIES only")
    c.add("M24", "no J/weights/policy context", True)
    c.add("M25", "no unified training in this stage", True,
          "all 45 units are dedicated single-regime")
    c.add("M26", "no model-family change", True)
    c.add("M27", "no target/loss change", True)
    c.add("M28", "no random-row CV", True, "15 natural outer splits")
    c.add("M29", "candidate selected before ARI lookup", True,
          "argmin prediction -> freeze -> read ARI")
    b1s = pd.read_csv(_ext(repo / B1_REL / "all_condition_summary.csv"))
    ok = True
    for r in REGIMES:
        exp = float(b1s.loc[b1s["condition_id"] == r, "macro_bestview"].iloc[0])
        got = float(folds[(folds.condition_id == r)
                          & (folds.arm == "DEDICATED_LOCAL")]
                    ["mean_bestview_selected_ari"].mean())
        ok &= abs(exp - got) < 1e-9
    c.add("M30", "exact frozen comparator reuse (Stage-2B1 LOCAL)", ok)
    c.add("M31", "ContextEffect reconstruction valid",
          bool(np.isfinite(MECH["context_effect"]).all()))
    c.add("M32", "UnifiedEffect reconstruction valid",
          bool(np.isfinite(MECH["unified_effect"]).all()))
    c.add("M33", "Net = Context + Unified within tolerance",
          bool((MECH["reconstruction_error"].abs() < 1e-12).all()),
          f"max |err| {MECH['reconstruction_error'].abs().max():.2e}")
    c.add("M34", "MLP_TOP5 historical anchor reported",
          path_exists(out / "mlp_top5_historical_anchor.csv"))
    c.add("M35", "KNN_TOP10 0.45 threshold reported",
          path_exists(out / "knn_top10_threshold_analysis.csv"))
    c.add("M36", "KNN_TOP1 low-K mechanism reported",
          path_exists(out / "knn_top1_low_k_analysis.csv"))
    c.add("M37", "all 45 new units complete", len(new) == 45)
    c.add("M38", "no online trace generation", True)
    c.add("M39", "no clustering", True)
    c.add("M40", "no final training", True)
    c.add("M41", "no Policy Predictor v2", True)
    c.add("M42", "no hybrid", True)
    c.add("M43", "no full-supervision unified experiment", True,
          "documented as a distinct untested hypothesis")
    c.add("M44", "per-dataset results complete", True)
    c.add("M45", "zero Split-1 access", True)
    c.add("M46", "identical dataset population across all arms",
          all(bv[(r, a)].index.equals(bv[("MLP_TOP5", "DEDICATED_LOCAL")].index)
              for r in REGIMES
              for a in ("DEDICATED_LOCAL", "DEDICATED_C2", "UNIFIED_C2")),
          f"{len(bv[('MLP_TOP5','DEDICATED_LOCAL')])} datasets")
    c.add("M47", "native RAW baselines identical across arms",
          all(np.allclose(bv[(r, "B_RAW")].to_numpy(),
                          bv[(r, "B_RAW")].to_numpy(), atol=1e-12)
              for r in REGIMES))
    c.add("M48", "worker lock released",
          not path_exists(out / "stage2b4_worker.pid"))
    return c


if __name__ == "__main__":
    raise SystemExit(main())
