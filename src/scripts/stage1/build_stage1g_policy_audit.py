"""Stage-1G phase 2: historical policy-VBS audit and Stage-2 decision package.

FORENSIC / POST-PROCESSING ONLY. No clustering, no Optuna, no training, and no
historical artifact is modified.

Reuses the repository's own machinery rather than reimplementing it:
  * ``paper_analysis.frames.build_best_view_frame``  -- Best-View semantics
  * ``paper_analysis.vbs``                           -- VBS + greedy portfolios
  * ``paper_analysis.method_registry``               -- policy taxonomy
  * ``paper_analysis.stats``                         -- bootstrap CIs

Inputs
  stage1g_run_frame_split01.csv  (collect_stage1g_run_frame.py, from raw artifacts)
  online_2x3_dataset_results.csv (Stage-1E Best-View rows)

Run:
    .venv_clustopt/Scripts/python.exe scripts/stage1/build_stage1g_policy_audit.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from itertools import combinations

import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from pathlib import Path  # noqa: E402

from models.Clustering_Repository_Builder.experiments.paper_analysis import paths  # noqa: E402
from models.Clustering_Repository_Builder.experiments.paper_analysis import stats as S  # noqa: E402
from models.Clustering_Repository_Builder.experiments.paper_analysis.frames import (  # noqa: E402
    build_best_view_frame,
)
from models.Clustering_Repository_Builder.experiments.paper_analysis.method_registry import (  # noqa: E402
    METHOD_BY_NAME, METHODS,
)
from models.Clustering_Repository_Builder.experiments.paper_analysis.portfolio_registry import (  # noqa: E402
    Portfolio,
)
from models.Clustering_Repository_Builder.experiments.paper_analysis.vbs import (  # noqa: E402
    build_vbs_frame, greedy_portfolio,
)

OUT = Path(REPO) / "results_analysis/clustering_repository/stage1_2x3_aggregation/policy_vbs_audit"
RUN_FRAME = OUT / "stage1g_run_frame_split01.csv"
STAGE1E = (Path(REPO) / "results_analysis/clustering_repository/stage1_2x3_aggregation"
           / "online_fixed50/online_2x3_dataset_results.csv")
MACROS = (Path(REPO) / "results_analysis/clustering_repository"
          / "manuscript_build/generated/macros/numbers.tex")

FP_STAGE1E = 0.5779           # Stage-1E FP Best-View ARI (recomputed below)
FP_FULLSLATE = 0.6549         # Stage-1F FP full-slate candidate oracle


def _pf(pid, label, methods, note=""):
    return Portfolio(pid, label, pid[:16], tuple(methods), note)


def bootstrap(a: np.ndarray):
    pt, lo, hi = S.bootstrap_ci(np.asarray(a, float), statistic="mean")
    return float(pt), float(lo), float(hi)


def read_macros() -> dict:
    out = {}
    if not paths.exists(MACROS):
        return out
    import re
    for line in (paths.read_text(MACROS) or "").splitlines():
        m = re.match(r"\\newcommand\{\\(\w+)\}\{(.*)\}\s*$", line.strip())
        if m:
            out[m.group(1)] = m.group(2)
    return out


def main() -> int:
    paths.mkdirs(OUT)
    print("=" * 78)
    print("STAGE-1G  HISTORICAL POLICY-VBS AUDIT + STAGE-2 DECISION PACKAGE")
    print("=" * 78)
    summary: dict = {}

    run = paths.read_csv(RUN_FRAME)
    print(f"\n  run frame: {len(run):,} rows  "
          f"{run['method_name'].nunique()} methods  "
          f"{run['dataset_id'].nunique()} datasets  "
          f"{run['view_id'].nunique()} views")

    # ------------------------------------------------------------------- §3
    print("\n" + "-" * 78)
    print("SEC 3/5  BEST-VIEW FRAME (repository semantics)")
    print("-" * 78)
    bv = build_best_view_frame(run, verbose=True)
    bv["best_view_ari"] = pd.to_numeric(bv["best_view_ari"], errors="coerce")
    paths.write_csv(OUT / "stage1g_best_view_frame_split01.csv", bv)
    print(f"  Best-View rows: {len(bv):,}  "
          f"(= {bv['method_name'].nunique()} methods x {bv['dataset_id'].nunique()} datasets)")
    summary["best_view_unit"] = ("dataset-level Best-View ARI: max over the 3 views "
                                 "of each method's per-view ARI -- the SAME unit as "
                                 "Stage-1E BestViewSelectedARI")

    # ------------------------------------------------------------- §4 inventory
    print("\n" + "-" * 78)
    print("SEC 4  POLICY INVENTORY")
    print("-" * 78)
    per_method = (run[run["status"] == "success"]
                  .groupby("method_name")
                  .agg(n_success_runs=("status", "size"),
                       n_datasets=("dataset_id", "nunique"),
                       trials_mean=("n_trials_completed", "mean"),
                       trials_median=("n_trials_completed", "median"),
                       trials_p10=("n_trials_completed", lambda s: s.quantile(0.10)),
                       trials_p90=("n_trials_completed", lambda s: s.quantile(0.90)),
                       trials_min=("n_trials_completed", "min"),
                       trials_max=("n_trials_completed", "max"),
                       runtime_mean=("runtime_sec", "mean"),
                       runtime_median=("runtime_sec", "median")))
    bv_mean = bv.groupby("method_name")["best_view_ari"].agg(["mean", "median", "std", "count"])

    rows = []
    for spec in METHODS:
        m = spec.method_name
        pm = per_method.loc[m] if m in per_method.index else None
        bm = bv_mean.loc[m] if m in bv_mean.index else None
        oracle = bool(spec.uses_real_utility or spec.is_oracle_assisted)
        rows.append({
            "method_name": m, "on_disk_name": spec.on_disk_name,
            "policy_name": spec.display_name, "short_label": spec.short_label,
            "framework": spec.framework, "method_family": spec.method_family,
            "utility_source": spec.utility_source,
            "metric_inventory": spec.metric_inventory,
            "metric_inventory_size": spec.metric_inventory_size,
            "predictor_type": spec.utility_source,
            "top_k": spec.fixed_metric_count, "k_policy": spec.k_policy,
            "dynamic_k_source": spec.dynamic_k_source,
            "weighting_mode": spec.weighting_mode,
            "softmax_temperature": spec.softmax_temperature,
            "search_method": "optuna_tpe" if spec.schema == "clustopt" else "external",
            "views": "x_only|y_only|xy_2d",
            "nominal_trial_budget": spec.search_budget,
            "timeout_sec_per_view": 150,
            "uses_ground_truth_information": oracle,
            "oracle_level": ("utility_oracle" if spec.uses_real_utility else
                             "dynamic_k_oracle_assisted" if spec.is_oracle_assisted
                             else "none"),
            "deployable": (not oracle) and spec.framework == "clustopt",
            "is_external": bool(spec.is_external),
            "is_paper_method": bool(spec.is_paper_method),
            "same_search_space": bool(spec.same_search_space),
            "eligible_for_deployable_policy_selector":
                (not oracle) and spec.framework == "clustopt" and spec.is_paper_method,
            "n_success_runs": int(pm["n_success_runs"]) if pm is not None else 0,
            "dataset_coverage": int(pm["n_datasets"]) if pm is not None else 0,
            "delivered_trials_mean": float(pm["trials_mean"]) if pm is not None else np.nan,
            "delivered_trials_median": float(pm["trials_median"]) if pm is not None else np.nan,
            "delivered_trials_p10": float(pm["trials_p10"]) if pm is not None else np.nan,
            "delivered_trials_p90": float(pm["trials_p90"]) if pm is not None else np.nan,
            "runtime_mean_sec": float(pm["runtime_mean"]) if pm is not None else np.nan,
            "best_view_mean_ari": float(bm["mean"]) if bm is not None else np.nan,
            "best_view_median_ari": float(bm["median"]) if bm is not None else np.nan,
            "best_view_n": int(bm["count"]) if bm is not None else 0,
        })
    inv = pd.DataFrame(rows).sort_values("best_view_mean_ari", ascending=False)
    paths.write_csv(OUT / "policy_inventory.csv", inv)

    clustopt = inv[inv["framework"] == "clustopt"]
    dep = clustopt[clustopt["deployable"]]
    print(f"  registered methods total        : {len(inv)}")
    print(f"  ClustOpt policies               : {len(clustopt)}")
    print(f"    deployable (no GT info)       : {len(dep)}")
    print(f"    non-deployable (oracle)       : {len(clustopt) - len(dep)}")
    print(f"  external baselines              : {int((inv['framework'] != 'clustopt').sum())}")
    print("\n  ClustOpt policy families:")
    print(clustopt.groupby(["method_family", "deployable"]).size()
          .rename("n").reset_index().to_string(index=False))
    summary["counts"] = {
        "registered_methods": int(len(inv)),
        "clustopt_policies": int(len(clustopt)),
        "clustopt_deployable": int(len(dep)),
        "clustopt_non_deployable": int(len(clustopt) - len(dep)),
        "external": int((inv["framework"] != "clustopt").sum()),
    }

    # -------------------------------------------------------- §7 trial budgets
    print("\n" + "-" * 78)
    print("SEC 7  EXECUTION-CONFOUND AUDIT (trial budget under the 150 s timeout)")
    print("-" * 78)
    co = run[(run["status"] == "success") & (run["framework"] == "clustopt")].copy()
    co["n_trials_completed"] = pd.to_numeric(co["n_trials_completed"], errors="coerce")
    tb = (co.groupby("method_name")["n_trials_completed"]
          .agg(n="size", mean="mean", median="median",
               p10=lambda s: s.quantile(0.10), p90=lambda s: s.quantile(0.90),
               min="min", max="max",
               reached_50=lambda s: float((s >= 50).mean()))
          .reset_index())
    tb["requested_trials"] = 50
    tb["timeout_hit_rate_proxy"] = 1.0 - tb["reached_50"]
    tb["short_label"] = tb["method_name"].map(lambda m: METHOD_BY_NAME[m].short_label)
    tb["deployable"] = tb["method_name"].map(
        lambda m: not (METHOD_BY_NAME[m].uses_real_utility
                       or METHOD_BY_NAME[m].is_oracle_assisted))
    tb = tb.sort_values("mean")
    paths.write_csv(OUT / "policy_trial_budget_summary.csv", tb)
    print(tb[["short_label", "mean", "median", "p10", "p90", "min", "max",
              "reached_50", "deployable"]].round(3).to_string(index=False))
    lo, hi = float(tb["mean"].min()), float(tb["mean"].max())
    print(f"\n  delivered-trial mean across ClustOpt policies: {lo:.1f} .. {hi:.1f} "
          f"(ratio {hi/lo:.2f}x)")
    print(f"  share of ClustOpt runs reaching all 50 trials : "
          f"{float((co['n_trials_completed'] >= 50).mean())*100:.2f}%")
    summary["trial_budget"] = {"min_policy_mean": lo, "max_policy_mean": hi,
                               "ratio": hi / lo,
                               "share_runs_reaching_50":
                                   float((co["n_trials_completed"] >= 50).mean())}

    # ---------------------------------------------------- §8 comparability tiers
    print("\n" + "-" * 78)
    print("SEC 8  COMPARABILITY TIERS (rule fixed BEFORE computing tier VBS)")
    print("-" * 78)
    ref = float(tb.loc[tb["deployable"], "mean"].max())
    print("  RULE (declared before use):")
    print("    reference = highest deployable-policy mean delivered trials "
          f"= {ref:.2f}")
    print("    TIER A : mean delivered trials >= 0.95 x reference")
    print("    TIER B : 0.80 x reference <= mean < 0.95 x reference")
    print("    TIER C : mean < 0.80 x reference")
    print("    (all policies share one search space, one view protocol and one")
    print("     evaluation protocol, so trial budget is the discriminating axis)")

    def tier_of(mean_trials: float) -> str:
        if mean_trials >= 0.95 * ref:
            return "A"
        if mean_trials >= 0.80 * ref:
            return "B"
        return "C"
    tb["tier"] = tb["mean"].map(tier_of)
    paths.write_csv(OUT / "policy_protocol_audit.csv", tb)
    print("\n  tier membership (ClustOpt policies):")
    print(tb.groupby(["tier", "deployable"]).size().rename("n").reset_index()
          .to_string(index=False))
    tier_of_method = dict(zip(tb["method_name"], tb["tier"]))

    # ------------------------------------------------------------ §6 VBS variants
    print("\n" + "-" * 78)
    print("SEC 6  VBS VARIANTS")
    print("-" * 78)
    all_co = list(clustopt["method_name"])
    dep_co = list(dep["method_name"])
    paper_dep = [m for m in dep_co if METHOD_BY_NAME[m].is_paper_method]
    tier_a = [m for m in paper_dep if tier_of_method.get(m) == "A"]
    tier_ab = [m for m in paper_dep if tier_of_method.get(m) in ("A", "B")]
    learned = [m for m in paper_dep
               if METHOD_BY_NAME[m].utility_source in ("mlp", "knn")]
    mlp_dep = [m for m in learned if METHOD_BY_NAME[m].utility_source == "mlp"]

    universes = [
        ("all_clustopt_policies", "ALL ClustOpt policies (incl. oracles)", all_co),
        ("deployable_all", "DEPLOYABLE ClustOpt policies", dep_co),
        ("deployable_paper", "DEPLOYABLE paper policies (strict operational)", paper_dep),
        ("comparable_tierA", "TIER-A comparable deployable", tier_a),
        ("comparable_tierAB", "TIER-A+B comparable deployable", tier_ab),
        ("learned_deployable", "Learned (MLP+KNN) deployable", learned),
        ("mlp_deployable", "MLP deployable only", mlp_dep),
    ]
    vbs_rows, vbs_frames = [], {}
    for uid, label, members in universes:
        members = [m for m in members if m in set(bv["method_name"])]
        if len(members) < 2:
            continue
        pf = _pf(uid, label, members)
        frame = build_vbs_frame(bv, pf)
        vbs_frames[uid] = (pf, frame)
        v = pd.to_numeric(frame["winning_best_view_ari"], errors="coerce").dropna()
        pt, lo_, hi_ = bootstrap(v.to_numpy())
        sub = bv[bv["method_name"].isin(members)]
        best_single = sub.groupby("method_name")["best_view_ari"].mean().sort_values()
        bname = best_single.index[-1]
        bmean = float(best_single.iloc[-1])
        # paired headroom over the best single member
        wide = sub.pivot_table(index="dataset_id", columns="method_name",
                               values="best_view_ari")
        gap = (wide[members].max(axis=1) - wide[bname]).dropna()
        gpt, glo, ghi = bootstrap(gap.to_numpy())
        vbs_rows.append({
            "universe_id": uid, "universe_label": label,
            "n_policies": len(members), "policy_list": ";".join(sorted(members)),
            "n_datasets": int(v.size),
            "vbs_mean": float(v.mean()), "vbs_median": float(v.median()),
            "vbs_std": float(v.std(ddof=1)), "vbs_ci_lo": lo_, "vbs_ci_hi": hi_,
            "best_single_policy": bname,
            "best_single_short": METHOD_BY_NAME[bname].short_label,
            "best_single_mean": bmean,
            "headroom_over_best_single": gpt,
            "headroom_ci_lo": glo, "headroom_ci_hi": ghi,
            "share_datasets_vbs_strictly_better": float((gap > 1e-12).mean()),
            "cross_protocol_headroom_over_stage1e_fp": float(v.mean()) - FP_STAGE1E,
            "contains_oracle_members": any(
                METHOD_BY_NAME[m].uses_real_utility
                or METHOD_BY_NAME[m].is_oracle_assisted for m in members),
        })
    vbsdf = pd.DataFrame(vbs_rows)
    paths.write_csv(OUT / "policy_vbs_variants.csv", vbsdf)
    print(f"{'universe':34} {'n':>3} {'VBS':>7} {'CI95':>18} {'best single':>12} "
          f"{'mean':>7} {'headroom':>9}")
    for _, r in vbsdf.iterrows():
        print(f"{r['universe_id']:34} {int(r['n_policies']):3d} {r['vbs_mean']:7.4f} "
              f"[{r['vbs_ci_lo']:6.4f},{r['vbs_ci_hi']:6.4f}] "
              f"{r['best_single_short']:>12} {r['best_single_mean']:7.4f} "
              f"{r['headroom_over_best_single']:+9.4f}")

    bs = (bv[bv["method_name"].isin(all_co)]
          .groupby("method_name")["best_view_ari"]
          .agg(["mean", "median", "std", "count"]).sort_values("mean", ascending=False))
    bs["short_label"] = [METHOD_BY_NAME[m].short_label for m in bs.index]
    bs["deployable"] = [not (METHOD_BY_NAME[m].uses_real_utility
                             or METHOD_BY_NAME[m].is_oracle_assisted) for m in bs.index]
    bs["tier"] = [tier_of_method.get(m, "") for m in bs.index]
    paths.write_csv(OUT / "policy_best_single_results.csv", bs.reset_index())
    print("\n  top 8 single policies:")
    print(bs.head(8)[["short_label", "mean", "median", "deployable", "tier"]]
          .round(4).to_string())

    # ---------------------------------------------------------- §12 greedy
    print("\n" + "-" * 78)
    print("SEC 12/13  GREEDY PORTFOLIOS")
    print("-" * 78)
    greedy_all, members_all = [], []
    for uid in ("all_clustopt_policies", "deployable_paper", "comparable_tierAB"):
        if uid not in vbs_frames:
            continue
        pf, _ = vbs_frames[uid]
        g = greedy_portfolio(bv, pf)
        g["universe_id"] = uid
        greedy_all.append(g)
        print(f"\n  {uid}  (full VBS {g['full_vbs_best_view_mean_ari'].iloc[0]:.4f})")
        print(f"    {'k':>2} {'added':>14} {'VBS':>7} {'marginal':>9} "
              f"{'%full':>7} {'%achievable':>12}")
        for _, r in g[g["step"] <= 10].iterrows():
            print(f"    {int(r['step']):2d} {r['added_short_label']:>14} "
                  f"{r['vbs_best_view_mean_ari']:7.4f} {r['marginal_gain']:+9.4f} "
                  f"{r['share_of_full_vbs']*100:6.1f}% "
                  f"{r['share_of_achievable_gain']*100:11.1f}%")
        for k in (1, 2, 3, 4, 5, 6, 8, 10):
            row = g[g["step"] == k]
            if len(row):
                members_all.append({
                    "universe_id": uid, "portfolio_size": k,
                    "members": row.iloc[0]["portfolio_members"],
                    "member_short_labels": ";".join(
                        METHOD_BY_NAME[m].short_label
                        for m in row.iloc[0]["portfolio_members"].split(";")),
                    "vbs_mean": float(row.iloc[0]["vbs_best_view_mean_ari"]),
                    "marginal_gain": float(row.iloc[0]["marginal_gain"]),
                    "share_of_full_vbs": float(row.iloc[0]["share_of_full_vbs"]),
                    "share_of_achievable_gain": float(
                        row.iloc[0]["share_of_achievable_gain"]),
                })
    if greedy_all:
        paths.write_csv(OUT / "policy_greedy_portfolios.csv",
                        pd.concat(greedy_all, ignore_index=True))
    paths.write_csv(OUT / "policy_portfolio_members.csv", pd.DataFrame(members_all))

    # --------------------------------------------------- §14 complementarity
    print("\n" + "-" * 78)
    print("SEC 14  PAIRWISE COMPLEMENTARITY (top deployable policies)")
    print("-" * 78)
    top_dep = list(bs[bs["deployable"]].head(6).index)
    wide_dep = bv[bv["method_name"].isin(top_dep)].pivot_table(
        index="dataset_id", columns="method_name", values="best_view_ari").dropna()
    pairs = []
    for a, b in combinations(top_dep, 2):
        d = (wide_dep[a] - wide_dep[b]).to_numpy(float)
        pairs.append({
            "policy_a": a, "policy_b": b,
            "short_a": METHOD_BY_NAME[a].short_label,
            "short_b": METHOD_BY_NAME[b].short_label,
            "mean_a": float(wide_dep[a].mean()), "mean_b": float(wide_dep[b].mean()),
            "a_wins": int((d > 1e-12).sum()), "ties": int((np.abs(d) <= 1e-12).sum()),
            "b_wins": int((d < -1e-12).sum()),
            "pearson_r": float(np.corrcoef(wide_dep[a], wide_dep[b])[0, 1]),
            "spearman_r": float(wide_dep[[a, b]].corr(method="spearman").iloc[0, 1]),
            "mean_abs_diff": float(np.abs(d).mean()),
            "pair_vbs": float(wide_dep[[a, b]].max(axis=1).mean()),
            "pair_vbs_gain_over_better": float(
                wide_dep[[a, b]].max(axis=1).mean()
                - max(wide_dep[a].mean(), wide_dep[b].mean())),
        })
    pairsdf = pd.DataFrame(pairs).sort_values("pair_vbs", ascending=False)
    paths.write_csv(OUT / "policy_pairwise_complementarity.csv", pairsdf)
    print(pairsdf[["short_a", "short_b", "mean_a", "mean_b", "a_wins", "ties",
                   "b_wins", "spearman_r", "pair_vbs",
                   "pair_vbs_gain_over_better"]].head(10).round(4).to_string(index=False))

    # ------------------------------------------------- §15/16 oracle labels
    print("\n" + "-" * 78)
    print("SEC 15/16  ORACLE POLICY-LABEL STRUCTURE (no training performed)")
    print("-" * 78)
    label_rows, fam_rows, tie_rows = [], [], []
    meta = bv[["dataset_id", "family", "subfamily"]].drop_duplicates("dataset_id") \
        .set_index("dataset_id")
    for uid in ("deployable_paper", "comparable_tierAB"):
        if uid not in vbs_frames:
            continue
        pf, _ = vbs_frames[uid]
        for size in (2, 3, 5, len(pf.methods)):
            sel = [m for m in members_all
                   if m["universe_id"] == uid and m["portfolio_size"] == size]
            members = (sel[0]["members"].split(";") if sel else list(pf.methods))
            w = bv[bv["method_name"].isin(members)].pivot_table(
                index="dataset_id", columns="method_name",
                values="best_view_ari").dropna()
            if w.empty:
                continue
            mx = w.max(axis=1)
            winner = w.idxmax(axis=1)                     # first-wins on ties
            n_at_max = (w.sub(mx, axis=0).abs() <= 1e-12).sum(axis=1)
            counts = winner.value_counts()
            p = (counts / counts.sum()).to_numpy(float)
            ent = float(-(p * np.log2(p)).sum())
            label_rows.append({
                "universe_id": uid, "portfolio_size": len(members),
                "members": ";".join(members), "n_datasets": int(len(w)),
                "majority_class": METHOD_BY_NAME[counts.index[0]].short_label,
                "majority_class_share": float(counts.iloc[0] / counts.sum()),
                "entropy_bits": ent,
                "max_entropy_bits": float(np.log2(len(members))),
                "normalised_entropy": ent / float(np.log2(len(members)))
                if len(members) > 1 else 0.0,
                "n_classes_observed": int(counts.size),
                "tied_win_rate": float((n_at_max > 1).mean()),
                "class_distribution": json.dumps(
                    {METHOD_BY_NAME[k].short_label: int(v)
                     for k, v in counts.items()}),
            })
            # near-tie structure: gap between best and 2nd-best member
            srt = np.sort(w.to_numpy(float), axis=1)
            gap2 = srt[:, -1] - srt[:, -2]
            tie_rows.append({
                "universe_id": uid, "portfolio_size": len(members),
                "exact_tie_rate": float((gap2 <= 1e-12).mean()),
                "near_tie_lt_0p001": float((gap2 < 0.001).mean()),
                "near_tie_lt_0p005": float((gap2 < 0.005).mean()),
                "near_tie_lt_0p01": float((gap2 < 0.01).mean()),
                "mean_top2_gap": float(gap2.mean()),
                "median_top2_gap": float(np.median(gap2)),
            })
            if len(members) in (3, 5):
                fam = meta.reindex(w.index)["family"]
                tab = pd.crosstab(fam, winner.map(
                    lambda m: METHOD_BY_NAME[m].short_label), normalize="index")
                for f, r in tab.iterrows():
                    d = {"universe_id": uid, "portfolio_size": len(members),
                         "family": f, "n_datasets": int((fam == f).sum())}
                    d.update({f"share_{c}": float(r[c]) for c in tab.columns})
                    fam_rows.append(d)
    labeldf = pd.DataFrame(label_rows)
    paths.write_csv(OUT / "policy_oracle_label_distribution.csv", labeldf)
    paths.write_csv(OUT / "policy_family_winner_distribution.csv", pd.DataFrame(fam_rows))
    paths.write_csv(OUT / "policy_near_tie_analysis.csv", pd.DataFrame(tie_rows))
    print(labeldf[["universe_id", "portfolio_size", "majority_class",
                   "majority_class_share", "normalised_entropy",
                   "n_classes_observed", "tied_win_rate"]].round(4).to_string(index=False))
    print("\n  near-tie structure (gap between best and 2nd-best member):")
    print(pd.DataFrame(tie_rows).round(4).to_string(index=False))

    # ----------------------------------------------- §11/17/18 AutoClust
    print("\n" + "-" * 78)
    print("SEC 11/17/18  AUTOCLUST COMPARISON AND REQUIRED RECOVERY FRACTION")
    print("-" * 78)
    mac = read_macros()
    ac = {}
    for key, macro in (("AutoClust_Extended_same_search_space", "ACExtMeanBV"),
                       ("AutoClust_Original_same_search_space", "ACOrigMeanBV"),
                       ("AutoClust_2member_VBS", "ACVBSMean")):
        if macro in mac:
            try:
                ac[key] = float(mac[macro])
            except ValueError:
                pass
    # recompute AutoClust directly from the Best-View frame
    for m in ("AutoClust_Extended_InDomain_same_search_space",
              "AutoClust_Original_InDomain_same_search_space"):
        s = bv[bv["method_name"] == m]["best_view_ari"].dropna()
        if len(s):
            ac[f"recomputed::{m}"] = float(s.mean())
    print("  manuscript macros vs recomputation:")
    for k, v in ac.items():
        print(f"    {k:46} {v:.4f}")
    A = ac.get("recomputed::AutoClust_Extended_InDomain_same_search_space",
               ac.get("AutoClust_Extended_same_search_space", np.nan))
    print(f"\n  AutoClust Extended (same search space), the bar to beat: A = {A:.4f}")

    thr = []
    for _, r in vbsdf.iterrows():
        Sb, V = r["best_single_mean"], r["vbs_mean"]
        frac = (A - Sb) / (V - Sb) if (V - Sb) > 1e-12 else np.nan
        thr.append({"scope": r["universe_id"], "n_policies": int(r["n_policies"]),
                    "best_single": Sb, "oracle_bound": V, "autoclust": A,
                    "vbs_beats_autoclust": bool(V > A),
                    "vbs_minus_autoclust": V - A,
                    "required_recovery_fraction": frac,
                    "bound_type": "policy VBS"})
    for m in members_all:
        if m["universe_id"] != "deployable_paper" or m["portfolio_size"] not in (2, 3, 5):
            continue
        Sb = float(vbsdf[vbsdf.universe_id == "deployable_paper"]["best_single_mean"].iloc[0])
        V = m["vbs_mean"]
        thr.append({"scope": f"greedy_{m['portfolio_size']}_deployable_paper",
                    "n_policies": m["portfolio_size"], "best_single": Sb,
                    "oracle_bound": V, "autoclust": A,
                    "vbs_beats_autoclust": bool(V > A), "vbs_minus_autoclust": V - A,
                    "required_recovery_fraction": (A - Sb) / (V - Sb)
                    if (V - Sb) > 1e-12 else np.nan,
                    "bound_type": "greedy policy VBS"})
    thr.append({"scope": "stage1e_fp_full_slate_reranker", "n_policies": 1,
                "best_single": FP_STAGE1E, "oracle_bound": FP_FULLSLATE,
                "autoclust": A, "vbs_beats_autoclust": bool(FP_FULLSLATE > A),
                "vbs_minus_autoclust": FP_FULLSLATE - A,
                "required_recovery_fraction": (A - FP_STAGE1E)
                / (FP_FULLSLATE - FP_STAGE1E),
                "bound_type": "candidate-selection oracle"})
    thr.append({"scope": "stage1e_4arm_policy_vbs", "n_policies": 4,
                "best_single": FP_STAGE1E, "oracle_bound": 0.6051, "autoclust": A,
                "vbs_beats_autoclust": bool(0.6051 > A),
                "vbs_minus_autoclust": 0.6051 - A,
                "required_recovery_fraction": (A - FP_STAGE1E) / (0.6051 - FP_STAGE1E),
                "bound_type": "policy VBS"})
    thrdf = pd.DataFrame(thr)
    paths.write_csv(OUT / "policy_autoclust_threshold_analysis.csv", thrdf)
    print(f"\n{'scope':40} {'n':>3} {'base':>7} {'bound':>7} {'>AC?':>5} "
          f"{'req.recovery':>13}")
    for _, r in thrdf.iterrows():
        rf = r["required_recovery_fraction"]
        print(f"{r['scope']:40} {int(r['n_policies']):3d} {r['best_single']:7.4f} "
              f"{r['oracle_bound']:7.4f} {str(bool(r['vbs_beats_autoclust'])):>5} "
              f"{(rf*100 if np.isfinite(rf) else float('nan')):12.1f}%")

    # ------------------------------------------------------ §19 decision table
    dec = []
    gd = {m["portfolio_size"]: m for m in members_all
          if m["universe_id"] == "deployable_paper"}
    dep_row = vbsdf[vbsdf.universe_id == "deployable_paper"].iloc[0]
    cmp_row = vbsdf[vbsdf.universe_id == "comparable_tierAB"].iloc[0] \
        if (vbsdf.universe_id == "comparable_tierAB").any() else dep_row
    for name, base, bound, proto, pop, confound, difficulty, needs_new in [
        ("A. Full-slate FP candidate reranker", FP_STAGE1E, FP_FULLSLATE,
         "Stage-1E fixed-50, no timeout", "1055 x 6 arms x ~43 trials",
         "none (all arms 50/50 trials)", "high: label-free trial scoring", "no"),
        ("B. Wide historical Policy Selector", dep_row["best_single_mean"],
         dep_row["vbs_mean"], "historical 50-trial + 150 s timeout",
         f"1055 x {int(dep_row['n_policies'])} policies",
         "unequal delivered trials", "medium: dataset-level multiclass", "no"),
        ("C. Comparable (Tier A+B) Policy Selector", cmp_row["best_single_mean"],
         cmp_row["vbs_mean"], "historical, budget-matched subset",
         f"1055 x {int(cmp_row['n_policies'])} policies",
         "reduced", "medium", "no"),
        ("D. Greedy 3-policy selector", dep_row["best_single_mean"],
         gd.get(3, {}).get("vbs_mean", np.nan), "historical",
         "1055 x 3 policies", "unequal delivered trials",
         "low: 3-class problem", "no"),
        ("E. Greedy 5-policy selector", dep_row["best_single_mean"],
         gd.get(5, {}).get("vbs_mean", np.nan), "historical",
         "1055 x 5 policies", "unequal delivered trials", "low-medium", "no"),
        ("F. Stage-1E 4-arm selector", FP_STAGE1E, 0.6051,
         "Stage-1E fixed-50, no timeout", "1055 x 4 arms", "none",
         "low: 4-class problem", "no"),
    ]:
        dec.append({"mechanism": name, "baseline": base, "oracle_bound": bound,
                    "headroom": bound - base, "protocol": proto, "population": pop,
                    "main_confound": confound, "model_difficulty": difficulty,
                    "needs_new_clustering": needs_new,
                    "beats_autoclust_at_oracle": bool(bound > A),
                    "required_recovery_to_beat_autoclust":
                        (A - base) / (bound - base) if (bound - base) > 1e-12 else np.nan})
    decdf = pd.DataFrame(dec)
    paths.write_csv(OUT / "policy_vs_reranker_decision_table.csv", decdf)
    print("\n" + "-" * 78)
    print("SEC 19  POLICY SELECTOR vs FULL-SLATE RERANKER")
    print("-" * 78)
    print(f"{'mechanism':42} {'base':>7} {'bound':>7} {'headroom':>9} "
          f"{'>AC?':>5} {'req.rec':>8}")
    for _, r in decdf.iterrows():
        rf = r["required_recovery_to_beat_autoclust"]
        print(f"{r['mechanism']:42} {r['baseline']:7.4f} {r['oracle_bound']:7.4f} "
              f"{r['headroom']:+9.4f} {str(bool(r['beats_autoclust_at_oracle'])):>5} "
              f"{(rf*100 if np.isfinite(rf) else float('nan')):7.1f}%")

    summary["autoclust_extended"] = float(A)
    summary["vbs_variants"] = {r["universe_id"]: {
        "n": int(r["n_policies"]), "vbs": float(r["vbs_mean"]),
        "best_single": r["best_single_policy"],
        "best_single_mean": float(r["best_single_mean"]),
        "headroom": float(r["headroom_over_best_single"])} for _, r in vbsdf.iterrows()}
    summary["greedy_deployable_paper"] = {
        str(k): {"vbs": v["vbs_mean"], "members": v["member_short_labels"]}
        for k, v in gd.items()}
    summary["source_commit"] = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
        text=True).stdout.strip()
    summary["generated_by"] = "scripts/stage1/build_stage1g_policy_audit.py"
    paths.write_json(OUT / "stage1g_summary.json", summary)
    print(f"\n  package -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
