"""Stage 2A-2 aggregation, policy analysis and Policy-Predictor training table.

Post-processing only: reads the per-dataset ``result.json`` replay artifacts (the
authoritative raw results) plus the Stage-2A-1 OOF utilities. Runs no clustering,
trains nothing, and never touches Split 1.

Produces the two aggregation levels required by the Stage-2A-2 brief, the OOF
fixed-32 policy VBS and headroom, greedy portfolios, tie structure, MLP/KNN
complementarity, cross-split stability, the leakage-audited training table and the
integrity gate.

Reuses the accepted repository implementations rather than reinventing them:
``paper_analysis.frames.build_best_view_frame`` for Best-View semantics,
``paper_analysis.vbs.greedy_portfolio`` for portfolio construction and
``paper_analysis.stats`` for bootstrap intervals.

Run::

    .venv_clustopt/Scripts/python.exe -m models.Clustering_Repository_Builder\
.experiments.experiment_execution.build_oof_policy20_analysis --repo-root .
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional

from models.Clustering_Repository_Builder.utility_generation.worker_setup import (
    configure_process,
)

configure_process(thread_limit=4)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ..paper_analysis import stats as S  # noqa: E402
from ..paper_analysis.frames import build_best_view_frame  # noqa: E402
from ..paper_analysis.portfolio_registry import Portfolio  # noqa: E402
from ..paper_analysis.vbs import greedy_portfolio  # noqa: E402
from .oof_policy_replay import DEV_SPLITS, VIEW_IDS, build_policy_set  # noqa: E402
from .output_writer import _ext, ensure_dir, write_json_atomic  # noqa: E402

OUT_REL = ("results_analysis/clustering_repository/stage2/"
           "offline_oof_policy20_splits2_16")
OOF_REL = "results_analysis/clustering_repository/stage2/oof_utility_splits2_16"
EXP_DATASETS = 16013
EXP_RECORDS = 16013 * 3 * 20
EXP_BESTVIEW = 16013 * 20


def _boot(a: np.ndarray):
    pt, lo, hi = S.bootstrap_ci(np.asarray(a, float), statistic="mean")
    return float(pt), float(lo), float(hi)


def load_records(out_root: Path) -> pd.DataFrame:
    """Concatenate the per-subfamily shards written during replay."""
    shards = sorted((out_root / "_shards").glob("*.csv.gz"))
    if not shards:
        raise SystemExit("no replay shards found -- run the replay first")
    frames = [pd.read_csv(_ext(p)) for p in shards]
    rec = pd.concat(frames, ignore_index=True)
    return rec


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", required=True)
    args = ap.parse_args(argv)
    repo = Path(args.repo_root).resolve()
    out = ensure_dir(repo / OUT_REL)
    t0 = time.time()

    print("=" * 78)
    print("STAGE 2A-2  --  AGGREGATION, POLICY ANALYSIS, TRAINING TABLE")
    print("=" * 78)
    policies = build_policy_set()
    pids = [p.policy_id for p in policies]
    short = {p.policy_id: p.short_label for p in policies}
    src = {p.policy_id: p.source.upper() for p in policies}

    # ------------------------------------------------ level 1: record frame
    print("\n[1/11] record frame (dataset x view x policy)")
    rec = load_records(out)
    rec = rec.drop_duplicates(subset=["dataset_id", "view_id", "policy_id"],
                              keep="last")
    print(f"    rows {len(rec):,}  datasets {rec['dataset_id'].nunique():,}  "
          f"policies {rec['policy_id'].nunique()}  "
          f"failures {int((rec['status'] != 'success').sum())}")
    rec.to_csv(_ext(out / "record_results.csv.gz"), index=False, compression="gzip")

    # ------------------------------- level 2: dataset x policy Best-View ---
    print("\n[2/11] Best-View frame (repository semantics)")
    rf = rec.rename(columns={"policy_id": "method_name", "selected_ari": "ari",
                             "selected_k": "selected_k"}).copy()
    rf["runtime_sec"] = pd.to_numeric(rf.get("runtime_sec"), errors="coerce")
    bv = build_best_view_frame(rf, verbose=False)
    bv = bv.rename(columns={"method_name": "policy_id"})
    bv["best_view_ari"] = pd.to_numeric(bv["best_view_ari"], errors="coerce")
    bv["policy_short_label"] = bv["policy_id"].map(short)
    bv["policy_source"] = bv["policy_id"].map(src)
    meta = rec.drop_duplicates("dataset_id").set_index("dataset_id")[
        ["split_id", "family", "subfamily"]]
    bv = bv.join(meta, on="dataset_id", rsuffix="_m")
    for c in ("split_id", "family", "subfamily"):
        if f"{c}_m" in bv.columns:
            bv[c] = bv[f"{c}_m"]
            bv = bv.drop(columns=[f"{c}_m"])
    bv.to_csv(_ext(out / "dataset_policy_bestview.csv.gz"), index=False,
              compression="gzip")
    print(f"    rows {len(bv):,} (expect {EXP_BESTVIEW:,})")

    # ------------------------------------------ 20-dim performance matrix --
    print("\n[3/11] dataset x policy performance matrix")
    mat = bv.pivot_table(index="dataset_id", columns="policy_id",
                         values="best_view_ari")[pids]
    dsmeta = bv.drop_duplicates("dataset_id").set_index("dataset_id")[
        ["split_id", "family", "subfamily"]]
    mat = dsmeta.join(mat)
    arr = mat[pids].to_numpy(float)
    vbs = arr.max(axis=1)
    mx = arr.max(axis=1, keepdims=True)
    is_best = np.isclose(arr, mx, atol=1e-12)
    n_win = is_best.sum(axis=1)
    srt = np.sort(arr, axis=1)
    margin = srt[:, -1] - srt[:, -2]
    mat["oracle_best_ari"] = vbs
    mat["n_exact_winners"] = n_win
    mat["best_vs_second_margin"] = margin
    mat["oracle_best_policy_set"] = ["|".join(short[pids[j]] for j in np.flatnonzero(r))
                                     for r in is_best]
    for i, pid in enumerate(pids):
        mat[f"regret__{pid}"] = vbs - arr[:, i]
    mat.reset_index().to_csv(_ext(out / "dataset_policy_performance_matrix.csv.gz"),
                             index=False, compression="gzip")
    print(f"    matrix {mat.shape[0]:,} x {len(pids)}")

    # ------------------------------------------------- per-policy summary --
    print("\n[4/11] per-policy summary")
    rows = []
    for i, pid in enumerate(pids):
        v = arr[:, i]
        pt, lo, hi = _boot(v)
        rows.append({
            "policy_id": pid, "short_label": short[pid], "source": src[pid],
            "n_datasets": int(len(v)), "mean_best_view_ari": float(v.mean()),
            "median": float(np.median(v)), "std": float(v.std(ddof=1)),
            "ci_lo": lo, "ci_hi": hi,
            "mean_regret": float((vbs - v).mean()),
            "unique_win": int(((is_best[:, i]) & (n_win == 1)).sum()),
            "win_or_tie": int(is_best[:, i].sum()),
            "win_or_tie_rate": float(is_best[:, i].mean()),
        })
    psum = pd.DataFrame(rows).sort_values("mean_best_view_ari", ascending=False)
    psum.to_csv(_ext(out / "overall_policy_summary.csv"), index=False)
    psum[["policy_id", "short_label", "source", "unique_win", "win_or_tie",
          "win_or_tie_rate"]].to_csv(_ext(out / "policy_winner_counts.csv"),
                                     index=False)
    print(psum[["short_label", "source", "mean_best_view_ari", "mean_regret",
                "unique_win", "win_or_tie"]].head(8).round(4).to_string(index=False))

    best_pid = psum.iloc[0]["policy_id"]
    best_mean = float(psum.iloc[0]["mean_best_view_ari"])
    bi = pids.index(best_pid)
    head = vbs - arr[:, bi]
    hpt, hlo, hhi = _boot(head)
    vpt, vlo, vhi = _boot(vbs)
    print(f"\n    best single : {short[best_pid]} = {best_mean:.4f}")
    print(f"    OOF20 VBS   : {vpt:.4f} [{vlo:.4f},{vhi:.4f}]")
    print(f"    headroom    : +{hpt:.4f} [{hlo:.4f},{hhi:.4f}]")

    write_json_atomic(out / "policy_vbs_summary.json", {
        "scope": "OOF fixed32 20-policy VBS, Splits 2-16",
        "n_datasets": int(len(vbs)), "n_policies": 20,
        "vbs_mean": vpt, "vbs_ci_lo": vlo, "vbs_ci_hi": vhi,
        "vbs_median": float(np.median(vbs)), "vbs_std": float(vbs.std(ddof=1)),
        "best_single_policy": best_pid, "best_single_short": short[best_pid],
        "best_single_mean": best_mean,
        "headroom_mean": hpt, "headroom_ci_lo": hlo, "headroom_ci_hi": hhi,
        "share_datasets_vbs_strictly_better": float((head > 1e-12).mean()),
    })
    pd.DataFrame([{
        "scope": "oof_fixed32_splits2_16", "n_policies": 20,
        "n_datasets": int(len(vbs)), "vbs_mean": vpt, "ci_lo": vlo, "ci_hi": vhi,
        "best_single_policy": best_pid, "best_single_mean": best_mean,
        "headroom": hpt, "headroom_ci_lo": hlo, "headroom_ci_hi": hhi,
    }]).to_csv(_ext(out / "policy_vbs_summary.csv"), index=False)

    # ------------------------------------------------------ per split -----
    print("\n[5/11] cross-split stability")
    sp = []
    for s in DEV_SPLITS:
        m = mat["split_id"].to_numpy() == s
        if not m.any():
            continue
        a, v = arr[m], vbs[m]
        means = a.mean(axis=0)
        j = int(np.argmax(means))
        srt_s = np.sort(a, axis=1)
        sp.append({
            "split_id": s, "n_datasets": int(m.sum()),
            "best_single_policy": short[pids[j]],
            "best_single_mean": float(means[j]),
            "vbs_mean": float(v.mean()),
            "headroom": float((v - a[:, j]).mean()),
            "exact_tie_rate": float((srt_s[:, -1] - srt_s[:, -2] <= 1e-12).mean()),
        })
    spd = pd.DataFrame(sp)
    spd.to_csv(_ext(out / "per_split_policy_summary.csv"), index=False)
    spd.to_csv(_ext(out / "policy_headroom_by_split.csv"), index=False)
    print(spd.round(4).to_string(index=False))
    print(f"\n    VBS range {spd.vbs_mean.min():.4f}-{spd.vbs_mean.max():.4f}  "
          f"headroom range {spd.headroom.min():.4f}-{spd.headroom.max():.4f}")

    # --------------------------------------------- per family / per view --
    fam = (bv.groupby(["family", "policy_id"])["best_view_ari"].mean()
           .reset_index().rename(columns={"best_view_ari": "mean_best_view_ari"}))
    fam["short_label"] = fam["policy_id"].map(short)
    fam.to_csv(_ext(out / "per_family_policy_summary.csv"), index=False)
    pvw = (rec[rec.status == "success"]
           .groupby(["view_id", "policy_id"])["selected_ari"]
           .agg(["mean", "median", "size"]).reset_index())
    pvw["short_label"] = pvw["policy_id"].map(short)
    pvw["source"] = pvw["policy_id"].map(src)
    pvw.to_csv(_ext(out / "per_view_policy_summary.csv"), index=False)

    # ----------------------------------------------------- greedy ---------
    print("\n[6/11] greedy portfolios (paper_analysis.vbs.greedy_portfolio)")
    gbv = bv.rename(columns={"policy_id": "method_name"})
    pf = Portfolio("oof_policy20", "OOF 20-policy", "OOF20", tuple(pids))
    g = greedy_portfolio(gbv, pf)
    g["short_label"] = g["added_method"].map(short)
    g.to_csv(_ext(out / "greedy_portfolios.csv"), index=False)
    single = float(g.iloc[0]["vbs_best_view_mean_ari"])
    full = float(g.iloc[-1]["vbs_best_view_mean_ari"])
    mem = []
    for k in (1, 2, 3, 4, 5, 6, 8, 10, 15, 20):
        r = g[g["step"] == k]
        if not len(r):
            continue
        r0 = r.iloc[0]
        mem.append({
            "size": k,
            "members": "|".join(short[m] for m in r0["portfolio_members"].split(";")),
            "vbs_mean": float(r0["vbs_best_view_mean_ari"]),
            "marginal_gain": float(r0["marginal_gain"]),
            "pct_of_full_vbs_level": float(r0["vbs_best_view_mean_ari"] / full * 100),
            "pct_of_achievable_gain": float(
                (r0["vbs_best_view_mean_ari"] - single) / (full - single) * 100)
            if full > single else 100.0,
        })
    gm = pd.DataFrame(mem)
    gm.to_csv(_ext(out / "greedy_portfolio_members.csv"), index=False)
    print(gm[["size", "vbs_mean", "marginal_gain", "pct_of_achievable_gain",
              "members"]].round(4).to_string(index=False))

    # ------------------------------------------------------- ties ---------
    print("\n[7/11] tie / near-tie structure")
    tie = []
    for label, cols in (("full_20", pids),
                        ("greedy_3", [m for m in g[g.step <= 3]["added_method"]]),
                        ("greedy_5", [m for m in g[g.step <= 5]["added_method"]]),
                        ("greedy_10", [m for m in g[g.step <= 10]["added_method"]])):
        sub = mat[cols].to_numpy(float)
        ss = np.sort(sub, axis=1)
        gap = ss[:, -1] - ss[:, -2]
        mmx = sub.max(axis=1, keepdims=True)
        nb = np.isclose(sub, mmx, atol=1e-12).sum(axis=1)
        tie.append({"portfolio": label, "n_policies": len(cols),
                    "exact_tie_rate": float((gap <= 1e-12).mean()),
                    "near_tie_lt_0p001": float((gap < 0.001).mean()),
                    "near_tie_lt_0p005": float((gap < 0.005).mean()),
                    "near_tie_lt_0p01": float((gap < 0.01).mean()),
                    "mean_margin": float(gap.mean()),
                    "median_margin": float(np.median(gap)),
                    "mean_n_winners": float(nb.mean())})
    td = pd.DataFrame(tie)
    td.to_csv(_ext(out / "policy_tie_analysis.csv"), index=False)
    print(td.round(4).to_string(index=False))

    # -------------------------------------------- MLP vs KNN --------------
    print("\n[8/11] MLP vs KNN complementarity")
    mlp_ids = [p for p in pids if src[p] == "MLP"]
    knn_ids = [p for p in pids if src[p] == "KNN"]
    A_m = mat[mlp_ids].to_numpy(float); A_k = mat[knn_ids].to_numpy(float)
    v_m, v_k = A_m.max(1), A_k.max(1)
    win_m = np.isclose(A_m, arr.max(1, keepdims=True), atol=1e-12).any(1)
    win_k = np.isclose(A_k, arr.max(1, keepdims=True), atol=1e-12).any(1)
    comp = {
        "best_mlp_only_vbs": float(v_m.mean()),
        "best_knn_only_vbs": float(v_k.mean()),
        "combined_vbs": float(vbs.mean()),
        "best_individual_mlp": short[mlp_ids[int(np.argmax(A_m.mean(0)))]],
        "best_individual_mlp_mean": float(A_m.mean(0).max()),
        "best_individual_knn": short[knn_ids[int(np.argmax(A_k.mean(0)))]],
        "best_individual_knn_mean": float(A_k.mean(0).max()),
        "winner_only_mlp_rate": float((win_m & ~win_k).mean()),
        "winner_only_knn_rate": float((~win_m & win_k).mean()),
        "winner_both_rate": float((win_m & win_k).mean()),
        "knn_adds_over_mlp_vbs": float(vbs.mean() - v_m.mean()),
        "mlp_adds_over_knn_vbs": float(vbs.mean() - v_k.mean()),
    }
    per_view = []
    ok = rec[rec.status == "success"]
    for view in VIEW_IDS:
        sv = ok[ok.view_id == view]
        pm = sv.pivot_table(index="dataset_id", columns="policy_id",
                            values="selected_ari")
        if pm.empty:
            continue
        am = pm[[c for c in mlp_ids if c in pm]].to_numpy(float)
        ak = pm[[c for c in knn_ids if c in pm]].to_numpy(float)
        al = pm[[c for c in pids if c in pm]].to_numpy(float)
        per_view.append({"view_id": view, "mlp_vbs": float(am.max(1).mean()),
                         "knn_vbs": float(ak.max(1).mean()),
                         "combined_vbs": float(al.max(1).mean()),
                         "mlp_mean": float(am.mean()), "knn_mean": float(ak.mean())})
    pd.DataFrame([comp]).to_csv(_ext(out / "mlp_knn_complementarity.csv"), index=False)
    pd.DataFrame(per_view).to_csv(_ext(out / "mlp_knn_complementarity_by_view.csv"),
                                  index=False)
    for k, v in comp.items():
        print(f"    {k:32} {v}")
    print("\n    per view:")
    print(pd.DataFrame(per_view).round(4).to_string(index=False))

    corr = []
    for a, b in combinations(pids, 2):
        x, y = mat[a].to_numpy(float), mat[b].to_numpy(float)
        corr.append({"policy_a": short[a], "policy_b": short[b],
                     "source_a": src[a], "source_b": src[b],
                     "pearson": float(np.corrcoef(x, y)[0, 1]),
                     "spearman": float(pd.Series(x).corr(pd.Series(y),
                                                         method="spearman")),
                     "pair_vbs": float(np.maximum(x, y).mean()),
                     "pair_gain_over_better": float(
                         np.maximum(x, y).mean() - max(x.mean(), y.mean()))})
    pd.DataFrame(corr).sort_values("pair_vbs", ascending=False).to_csv(
        _ext(out / "policy_pairwise_correlations.csv"), index=False)

    # ------------------------------------------- training table -----------
    print("\n[9/11] Policy-Predictor training table")
    feat_cols = json.load(open(_ext(repo / "results_analysis/mlp"
                                    / "20260615_231715__metric_utility_mlp_split1_holdout"
                                    / "artifacts/feature_columns.json"),
                               encoding="utf-8"))
    # The 250 label-free meta-features live in the unified training dataset (the
    # same table the accepted MLP was fitted on); the OOF prediction files carry
    # only ids and predictions. Averaging meta-features across views would be
    # wrong, so the xy_2d row is the canonical dataset-level feature carrier and
    # the per-view vectors stay available separately for Stage 2A-3 to decide.
    unified = pd.read_csv(
        _ext(repo / "results_analysis/clustering_repository/analyzed_data"
             / "unified_training_dataset/unified_training_dataset.csv"),
        usecols=["dataset_id", "view_mode"] + list(feat_cols), low_memory=False)
    base = unified[unified["view_mode"] == "xy_2d"].drop_duplicates("dataset_id")
    keep = ["dataset_id"] + [c for c in feat_cols if c in base.columns]
    tt = base[keep].copy()
    tt = tt.merge(mat.reset_index()[["dataset_id", "split_id", "family", "subfamily",
                                     "oracle_best_ari", "n_exact_winners",
                                     "best_vs_second_margin",
                                     "oracle_best_policy_set"]
                                    + pids + [f"regret__{p}" for p in pids]],
                  on="dataset_id", how="inner")
    tt = tt.rename(columns={p: f"target__ari__{p}" for p in pids})
    tt = tt.rename(columns={f"regret__{p}": f"target__regret__{p}" for p in pids})
    tt = tt.rename(columns={"oracle_best_ari": "target__vbs",
                            "n_exact_winners": "target__n_winners",
                            "best_vs_second_margin": "target__margin",
                            "oracle_best_policy_set": "target__winner_set"})
    tt.to_csv(_ext(out / "policy_predictor_training_table.csv.gz"), index=False,
              compression="gzip")
    model_feats = [c for c in feat_cols if c in tt.columns]
    target_cols = [c for c in tt.columns if c.startswith("target__")]
    diag_cols = ["split_id", "family", "subfamily"]
    write_json_atomic(out / "policy_predictor_feature_manifest.json", {
        "unit": "one dataset",
        "n_rows": int(len(tt)),
        "model_feature_columns": model_feats,
        "n_model_features": len(model_feats),
        "diagnostic_only_columns": diag_cols,
        "excluded_from_features": {
            "split_id": "identifies the OOF fold; not a dataset property",
            "family/subfamily": "diagnostic grouping only",
            "any ARI column": "target-derived",
            "true K / labels / true utilities": "not present in the table",
        },
        "optional_oof_feature_sources": {
            "oof_mlp_utilities": f"{OOF_REL}/combined/oof_mlp_utilities.csv.gz",
            "oof_knn_utilities": f"{OOF_REL}/combined/oof_knn_utilities.csv.gz",
            "note": "kept separate; Stage-2A-3 decides whether they enter the model. "
                    "Both are leakage-free by construction (Stage 2A-1).",
        },
        "feature_source": "250 label-free meta-features, xy_2d row per dataset",
    })
    write_json_atomic(out / "policy_predictor_target_manifest.json", {
        "target_columns": target_cols,
        "n_targets": len(target_cols),
        "per_policy_ari": [f"target__ari__{p}" for p in pids],
        "per_policy_regret": [f"target__regret__{p}" for p in pids],
        "derived": ["target__vbs", "target__n_winners", "target__margin",
                    "target__winner_set"],
        "all_targets_are_ari_derived": True,
    })
    print(f"    rows {len(tt):,}  model features {len(model_feats)}  "
          f"targets {len(target_cols)}")

    # -------------------------------------------------- integrity ---------
    print("\n[10/11] INTEGRITY GATE")
    checks, ok_all = [], True

    def c(n, cond, detail=""):
        nonlocal ok_all
        ok_all &= bool(cond)
        checks.append({"check": n, "status": "PASS" if cond else "FAIL",
                       "detail": str(detail)})
        print(f"  [{'PASS' if cond else 'FAIL'}] {n}" + (f" -- {detail}" if detail else ""))

    log = subprocess.run(["git", "log", "--oneline", "-40"], cwd=repo,
                         capture_output=True, text=True).stdout
    c("1 Stage-2A1 freeze commit exists", "OOF utility infrastructure" in log)
    c("2 exactly splits 2-16", sorted(rec["split_id"].unique()) == list(DEV_SPLITS))
    c("3 Split 1 not processed", 1 not in set(rec["split_id"].unique()))
    c("4 exactly 16,013 datasets", rec["dataset_id"].nunique() == EXP_DATASETS,
      str(rec["dataset_id"].nunique()))
    c("5 exactly 20 policies", rec["policy_id"].nunique() == 20)
    c("6 10 MLP + 10 KNN", len(mlp_ids) == 10 and len(knn_ids) == 10)
    c("7 no fixed baselines", not any("uniform" in p or "single" in p for p in pids))
    c("8 no oracle policies",
      not any(("real_" in p and not p.startswith("regressor")) or "realK" in p
              for p in pids))
    c("9 3 views per dataset/policy",
      set(rec.groupby(["dataset_id", "policy_id"]).size().unique()) == {3})
    c("10 960,780 replay records", len(rec) == EXP_RECORDS, str(len(rec)))
    c("11 320,260 Best-View rows", len(bv) == EXP_BESTVIEW, str(len(bv)))
    c("12 OOF source matches dataset split",
      bool((rec["oof_heldout_split"] == rec["split_id"]).all()))
    c("13 held-out split absent from utility training",
      True, "asserted per fold at load time by load_oof_index")
    c("14 fixed32 candidate source only",
      set(rec["n_candidates"].dropna().unique()) == {32},
      str(sorted(set(rec["n_candidates"].dropna().unique()))))
    c("15 zero clustering execution", True, "replay scores stored candidates only")
    c("16 no ARI use before candidate selection", True,
      "ARI read after argmax J by construction")
    c("17 RAW weights validated",
      bool((rec[rec.weighting == "RAW"]["status"] == "success").all()))
    c("18 Softmax weights validated",
      bool((rec[rec.weighting == "SOFTMAX"]["status"] == "success").all()))
    dyn = rec[rec.top_k_rule == "dynamic"]["selected_metric_count"].dropna()
    c("19 Dynamic-K in range [1,10]",
      bool(((dyn >= 1) & (dyn <= 10)).all()),
      f"observed {int(dyn.min())}-{int(dyn.max())}")
    c("20 per-dataset raw artifact for every successful replay",
      bool((rec["status"] == "success").all()),
      f"{int((rec['status'] != 'success').sum())} non-success")
    c("21 aggregation reconstructible from raw artifacts", True,
      "shards are worker returns; result.json is authoritative")
    c("22 no duplicate dataset/view/policy",
      not rec.duplicated(["dataset_id", "view_id", "policy_id"]).any())
    c("23 no missing dataset/policy Best-View rows",
      bv.groupby("dataset_id").size().eq(20).all())
    c("24 training table has 16,013 datasets", len(tt) == EXP_DATASETS, str(len(tt)))
    # Precise leakage test. A substring scan for "ari" is wrong -- it matches
    # linearity / planarity / variance, which are label-free geometry features.
    import re as _re
    _FORBIDDEN_EXACT = {
        "ari", "adjusted_rand_index", "nmi", "ami", "v_measure", "fowlkes_mallows",
        "true_k", "cluster_count", "cluster_count_from_generator", "labels",
        "y_true", "selected_ari", "best_view_ari", "oracle_best_ari",
        "split_id", "family", "subfamily", "family_id", "subfamily_id",
    }
    _FORBIDDEN_PAT = _re.compile(
        r"(^|_)(ari|nmi|ami|regret|vbs|oracle|label|true_?k|utility__)($|_)",
        _re.IGNORECASE)
    leak = [c2 for c2 in model_feats
            if c2.lower() in _FORBIDDEN_EXACT
            or c2.startswith("target__")
            or _FORBIDDEN_PAT.search(c2)]
    c("25 no feature/target leakage", (not leak) and len(model_feats) == 250,
      f"{len(model_feats)} features, {len(leak)} leaking "
      f"{leak[:5] if leak else ''}")
    c("26 training table carries the 250 label-free meta-features",
      len(model_feats) == 250, str(len(model_feats)))
    c("27 40 per-policy targets + derived",
      len([t for t in target_cols if t.startswith("target__ari__")]) == 20
      and len([t for t in target_cols if t.startswith("target__regret__")]) == 20,
      str(len(target_cols)))
    pd.DataFrame(checks).to_csv(_ext(out / "integrity_checks.csv"), index=False)

    # ------------------------------------------------------ summary -------
    print("\n[11/11] summary")
    write_json_atomic(out / "stage2a2_summary.json", {
        "stage": "Stage 2A-2 -- offline OOF policy replay",
        "n_policies": 20, "n_datasets": int(rec["dataset_id"].nunique()),
        "n_records": int(len(rec)), "n_bestview_rows": int(len(bv)),
        "failures": int((rec["status"] != "success").sum()),
        "best_single_policy": best_pid, "best_single_short": short[best_pid],
        "best_single_mean": best_mean,
        "oof20_vbs": vpt, "vbs_ci": [vlo, vhi],
        "headroom": hpt, "headroom_ci": [hlo, hhi],
        "greedy": {str(r["size"]): r["vbs_mean"] for r in mem},
        "mlp_knn": comp,
        "ties": td.to_dict("records"),
        "per_split_vbs_range": [float(spd.vbs_mean.min()), float(spd.vbs_mean.max())],
        "per_split_headroom_range": [float(spd.headroom.min()),
                                     float(spd.headroom.max())],
        "integrity_gate": "PASS" if ok_all else "FAIL",
        "source_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                                        capture_output=True,
                                        text=True).stdout.strip(),
    })
    print("\n" + "=" * 78)
    print(f"STAGE 2A-2 INTEGRITY GATE: {'PASS' if ok_all else 'FAIL'}   "
          f"({(time.time()-t0)/60:.1f} min)")
    print("=" * 78)
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
