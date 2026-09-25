"""Stage 4 -- runtime analysis from persisted timing records only.

No experiment is run and no scientific output is modified. Every number is read
from the frozen per-arm runtime block written at execution time.

Frozen semantics (runtime_contract_version idb_stage4_runtime_v1):

    method_runtime_equivalent = full physical_trace_runtime + arm_scoring_runtime

The full physical cost is charged to EVERY arm that consumed the trace; it is
never divided among them. Sharing a physical execution was an implementation
optimisation, so the deduplicated ``actual_wall_clock_runtime`` is reported
separately and never substituted for the equivalent.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
_REPO = _ROOT.parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "models"))
warnings.filterwarnings("ignore")

RESULTS = _REPO / "results_analysis" / "independent_domain_benchmark"
CFG = _ROOT / "configs"
VIEWS = ("x_only", "y_only", "xy_2d")
SOURCES = ("fcps", "sklearn_shapes", "real_pca")

FIELDS = ("method_runtime_equivalent", "physical_trace_runtime",
          "arm_scoring_runtime", "actual_wall_clock_runtime",
          "preparation_runtime", "model_init_runtime",
          "metafeature_runtime", "persistence_runtime")

FAMILY_OF = {"ClustOpt": "ClustOpt", "AutoClust": "AutoClust", "ML2DAC": "ML2DAC"}


def short(mid: str) -> str:
    return mid.split("_")[2]


def fam(mid: str) -> str:
    return mid.split("_")[1]


def stats_block(s: pd.Series, prefix: str = "") -> Dict[str, Any]:
    s = s.dropna()
    if s.empty:
        return {}
    return {prefix + "mean": s.mean(), prefix + "median": s.median(),
            prefix + "std": s.std(ddof=1), prefix + "p90": s.quantile(0.90),
            prefix + "p95": s.quantile(0.95), prefix + "min": s.min(),
            prefix + "max": s.max(), prefix + "n": int(s.size)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args(argv)

    run_dir = RESULTS / args.run_id
    out = run_dir / "scientific_review"
    figd = out / "figure_data"
    out.mkdir(parents=True, exist_ok=True)
    figd.mkdir(parents=True, exist_ok=True)

    man = json.loads((CFG / "dataset_manifest.json").read_text(encoding="utf-8"))
    entries = {e["dataset_id"]: e for e in man["datasets"] if e.get("accepted", True)}
    proto = json.loads((CFG / "benchmark_protocol.json").read_text(encoding="utf-8"))
    trace_groups = proto["seed_protocol"]["trace_groups"]
    arm_to_group = {}
    for g, arms in trace_groups.items():
        for a in arms:
            arm_to_group[a] = g

    # ------------------------------------------------ read persisted timings
    rows: List[Dict[str, Any]] = []
    contract = set()
    for f in glob.glob(str(run_dir / "scientific" / "*.json")):
        rec = json.loads(Path(f).read_text(encoding="utf-8"))
        c, p = rec["content"], rec["provenance"]
        rt = c.get("runtime") or {}
        contract.add(rt.get("runtime_contract_version"))
        ds = p["dataset_id"] if "dataset_id" in p else Path(f).stem.split("__")[0]
        vw = p.get("view_id") or Path(f).stem.split("__")[1]
        r = {"dataset_id": ds, "view_id": vw, "method_id": p["method_id"],
             "short": short(p["method_id"]), "family": fam(p["method_id"]),
             "source": entries[ds]["source"], "status": c["status"],
             "n_rows": entries[ds].get("n_rows")}
        for k in FIELDS:
            r[k] = rt.get(k)
        r["trace_group"] = arm_to_group.get(r["short"])
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(out / "runtime_per_arm_records.csv.gz", index=False)

    EQ = "method_runtime_equivalent"
    # identity check of the frozen definition, on the data itself
    # the runtime block is persisted rounded to 6 decimals, so the identity can
    # only be checked to that precision; the residual is bounded by 1e-6 by
    # construction, not by any disagreement about the definition
    chk = (df[EQ] - (df["physical_trace_runtime"] + df["arm_scoring_runtime"])).abs()
    identity_ok = bool((chk <= 1.5e-6).all())
    identity_max_residual = float(chk.max())

    # which decomposition fields carry information at all
    availability = {k: {"non_null": int(df[k].notna().sum()),
                        "non_zero": int((df[k].fillna(0) != 0).sum()),
                        "sum": float(df[k].fillna(0).sum())} for k in FIELDS}

    # ---------------------------------------------- 2. per-method summaries
    meth = []
    for mid, g in df.groupby("method_id"):
        r = {"method_id": mid, "short": short(mid), "family": fam(mid),
             "trace_group": arm_to_group.get(short(mid))}
        r.update(stats_block(g[EQ], "pooled_"))
        for v in VIEWS:
            r.update(stats_block(g[g["view_id"] == v][EQ], "%s_" % v))
        r.update(stats_block(g["physical_trace_runtime"], "phys_"))
        r.update(stats_block(g["arm_scoring_runtime"], "scoring_"))
        r["scoring_share_of_equivalent"] = float(
            g["arm_scoring_runtime"].sum() / g[EQ].sum())
        meth.append(r)
    msum = pd.DataFrame(meth).sort_values("pooled_median")
    msum.to_csv(out / "runtime_method_summary.csv", index=False)

    byview = []
    for (mid, v), g in df.groupby(["method_id", "view_id"]):
        r = {"method_id": mid, "short": short(mid), "view_id": v}
        r.update(stats_block(g[EQ]))
        byview.append(r)
    pd.DataFrame(byview).to_csv(out / "runtime_by_view.csv", index=False)

    bysrc = []
    for (mid, s), g in df.groupby(["method_id", "source"]):
        r = {"method_id": mid, "short": short(mid), "source": s}
        r.update(stats_block(g[EQ]))
        bysrc.append(r)
    pd.DataFrame(bysrc).to_csv(out / "runtime_by_source.csv", index=False)

    # --------------------------------- 3. three-view BestView execution cost
    piv = df.pivot_table(index=["dataset_id", "method_id"], columns="view_id",
                         values=EQ, aggfunc="first").reset_index()
    piv["three_view_runtime"] = piv[list(VIEWS)].sum(axis=1)
    piv["source"] = piv["dataset_id"].map(lambda d: entries[d]["source"])
    piv["short"] = piv["method_id"].map(short)
    piv.to_csv(out / "runtime_three_view_per_dataset.csv", index=False)
    tv = []
    for mid, g in piv.groupby("method_id"):
        r = {"method_id": mid, "short": short(mid), "family": fam(mid),
             "cost_definition": ("cost of obtaining the oracle BestView across the "
                                 "three pre-specified views; NOT single-view "
                                 "deployment runtime")}
        r.update(stats_block(g["three_view_runtime"], "three_view_"))
        for s in SOURCES:
            r["three_view_median_%s" % s] = g[g["source"] == s][
                "three_view_runtime"].median()
        tv.append(r)
    tvs = pd.DataFrame(tv).sort_values("three_view_median")
    tvs.to_csv(out / "runtime_three_view_bestview_cost.csv", index=False)

    # ------------------------------------------- 4. xy_2d deployment-style
    m_ari = pd.read_csv(run_dir / "phase_c" / "method_ari.csv.gz")
    xy = m_ari[(m_ari["view_id"] == "xy_2d") & (m_ari["status"] == "success")]
    xy_rt = df[df["view_id"] == "xy_2d"]
    dep = []
    for mid, g in xy_rt.groupby("method_id"):
        a = xy[xy["method_id"] == mid]["ari"]
        dep.append({"method_id": mid, "short": short(mid), "family": fam(mid),
                    "xy_2d_ari_mean": a.mean(), "xy_2d_ari_median": a.median(),
                    "xy_2d_runtime_median": g[EQ].median(),
                    "xy_2d_runtime_p90": g[EQ].quantile(0.90),
                    "xy_2d_runtime_mean": g[EQ].mean()})
    dep = pd.DataFrame(dep)
    fastest = dep["xy_2d_runtime_median"].min()
    dep["relative_to_fastest_arm"] = dep["xy_2d_runtime_median"] / fastest
    dep["relative_to_fastest_in_family"] = dep.groupby("family")[
        "xy_2d_runtime_median"].transform(lambda s: s / s.min())
    dep.sort_values("xy_2d_ari_mean", ascending=False).to_csv(
        out / "runtime_xy2d_deployment.csv", index=False)

    # ------------------------------------------------- 5. reranker ablation
    bestv = (m_ari[m_ari["status"] == "success"].dropna(subset=["ari"])
             .sort_values("ari", ascending=False)
             .groupby(["dataset_id", "method_id"], as_index=False).first())
    bp = bestv.pivot(index="dataset_id", columns="method_id", values="ari")
    C = {k: "IDB_ClustOpt_%s" % v for k, v in {
        "C0": "C0_MLP_TOP5_RAW_noR", "C1": "C1_KNN_TOP10_RAW_noR",
        "C3": "C3_MLP_TOP5_RAW_R", "C4": "C4_KNN_TOP10_RAW_R"}.items()}
    tvp = piv.pivot(index="dataset_id", columns="method_id",
                    values="three_view_runtime")
    xyp = df[df["view_id"] == "xy_2d"].pivot(index="dataset_id",
                                             columns="method_id", values=EQ)
    rer = []
    for lab, a, b in (("C3-C0 (MLP Top-5)", C["C3"], C["C0"]),
                      ("C4-C1 (KNN Top-10)", C["C4"], C["C1"])):
        for d in tvp.index:
            rer.append({
                "ablation": lab, "dataset_id": d,
                "source": entries[d]["source"],
                "runtime_delta_three_view": float(tvp.loc[d, a] - tvp.loc[d, b]),
                "runtime_ratio_three_view": float(tvp.loc[d, a] / tvp.loc[d, b]),
                "runtime_delta_xy_2d": float(xyp.loc[d, a] - xyp.loc[d, b]),
                "runtime_ratio_xy_2d": float(xyp.loc[d, a] / xyp.loc[d, b]),
                "ari_delta_bestview": float(bp.loc[d, a] - bp.loc[d, b])})
    rr = pd.DataFrame(rer)
    rr.to_csv(out / "runtime_reranker_ablation.csv", index=False)
    rsum = []
    for lab, g in rr.groupby("ablation"):
        r = {"ablation": lab, "scope": "overall", "n": len(g),
             "runtime_delta_mean": g["runtime_delta_three_view"].mean(),
             "runtime_delta_median": g["runtime_delta_three_view"].median(),
             "runtime_ratio_median": g["runtime_ratio_three_view"].median(),
             "xy_runtime_delta_median": g["runtime_delta_xy_2d"].median(),
             "xy_runtime_ratio_median": g["runtime_ratio_xy_2d"].median(),
             "ari_delta_mean": g["ari_delta_bestview"].mean(),
             "ari_delta_median": g["ari_delta_bestview"].median()}
        rsum.append(r)
        for s in SOURCES:
            gg = g[g["source"] == s]
            rsum.append({"ablation": lab, "scope": s, "n": len(gg),
                         "runtime_delta_mean": gg["runtime_delta_three_view"].mean(),
                         "runtime_delta_median": gg["runtime_delta_three_view"].median(),
                         "runtime_ratio_median": gg["runtime_ratio_three_view"].median(),
                         "xy_runtime_delta_median": gg["runtime_delta_xy_2d"].median(),
                         "xy_runtime_ratio_median": gg["runtime_ratio_xy_2d"].median(),
                         "ari_delta_mean": gg["ari_delta_bestview"].mean(),
                         "ari_delta_median": gg["ari_delta_bestview"].median()})
    pd.DataFrame(rsum).to_csv(out / "runtime_reranker_summary.csv", index=False)

    # ------------------------------------------------ 6. where time is spent
    decomp = []
    for mid, g in df.groupby("method_id"):
        tot = g[EQ].sum()
        decomp.append({
            "method_id": mid, "short": short(mid), "family": fam(mid),
            "trace_group": arm_to_group.get(short(mid)),
            "physical_search_share": float(g["physical_trace_runtime"].sum() / tot),
            "scoring_share": float(g["arm_scoring_runtime"].sum() / tot),
            "physical_median_s": g["physical_trace_runtime"].median(),
            "scoring_median_s": g["arm_scoring_runtime"].median(),
            "metafeature_median_s": g["metafeature_runtime"].median(),
            "model_init_median_s": g["model_init_runtime"].median(),
            "decomposition_note": (
                "only physical_trace_runtime and arm_scoring_runtime carry "
                "non-zero values in the persisted records; metafeature and "
                "model-init components are recorded as 0.0 and are NOT inferred")})
    pd.DataFrame(decomp).sort_values("scoring_share", ascending=False).to_csv(
        out / "runtime_cost_decomposition.csv", index=False)

    # -------------------------------------------------- 7. trade-off + Pareto
    bv_mean = bestv.groupby("method_id")["ari"].mean()
    tv_med = piv.groupby("method_id")["three_view_runtime"].median()
    tv_p90 = piv.groupby("method_id")["three_view_runtime"].quantile(0.90)
    tr = pd.DataFrame({"method_id": bv_mean.index,
                       "bestview_ari_mean": bv_mean.values})
    tr["short"] = tr["method_id"].map(short)
    tr["family"] = tr["method_id"].map(fam)
    tr["three_view_runtime_median"] = tr["method_id"].map(tv_med)
    tr["three_view_runtime_p90"] = tr["method_id"].map(tv_p90)
    tr["relative_to_fastest_arm"] = (tr["three_view_runtime_median"]
                                     / tr["three_view_runtime_median"].min())
    tr["relative_to_fastest_in_family"] = tr.groupby("family")[
        "three_view_runtime_median"].transform(lambda s: s / s.min())

    def pareto(frame, ari_col, rt_col):
        keep = []
        for i, r in frame.iterrows():
            dominated = ((frame[ari_col] >= r[ari_col])
                         & (frame[rt_col] <= r[rt_col])
                         & ((frame[ari_col] > r[ari_col])
                            | (frame[rt_col] < r[rt_col]))).any()
            keep.append(not dominated)
        return keep

    tr["pareto_efficient"] = pareto(tr, "bestview_ari_mean",
                                    "three_view_runtime_median")
    tr.sort_values("bestview_ari_mean", ascending=False).to_csv(
        out / "runtime_performance_tradeoff.csv", index=False)

    dep2 = dep.copy()
    dep2["pareto_efficient"] = pareto(dep2, "xy_2d_ari_mean",
                                      "xy_2d_runtime_median")
    dep2.sort_values("xy_2d_ari_mean", ascending=False).to_csv(
        out / "runtime_xy2d_tradeoff.csv", index=False)

    # --------------------------------- 8. physical sharing vs scientific cost
    share = []
    for (ds, vw, grp), g in df[df["trace_group"].notna()].groupby(
            ["dataset_id", "view_id", "trace_group"]):
        wall = g["actual_wall_clock_runtime"].iloc[0]
        share.append({"dataset_id": ds, "view_id": vw, "trace_group": grp,
                      "arms_in_group": len(g),
                      "sum_method_runtime_equivalent": float(g[EQ].sum()),
                      "deduplicated_actual_wall_clock": float(wall),
                      "wall_clock_saved_by_reuse": float(g[EQ].sum() - wall)})
    sh = pd.DataFrame(share)
    sh.to_csv(out / "runtime_physical_sharing.csv", index=False)
    gsum = sh.groupby("trace_group", as_index=False).agg(
        units=("dataset_id", "count"), arms_in_group=("arms_in_group", "max"),
        sum_equivalent_s=("sum_method_runtime_equivalent", "sum"),
        actual_wall_clock_s=("deduplicated_actual_wall_clock", "sum"),
        saved_s=("wall_clock_saved_by_reuse", "sum"))
    gsum["saved_fraction_of_equivalent"] = gsum["saved_s"] / gsum["sum_equivalent_s"]
    gsum.to_csv(out / "runtime_physical_sharing_summary.csv", index=False)

    # ------------------------------------------------------- figure data
    tr.to_csv(figd / "fig_bestview_ari_vs_three_view_runtime.csv", index=False)
    dep2.to_csv(figd / "fig_xy2d_ari_vs_runtime.csv", index=False)
    rr.to_csv(figd / "fig_reranker_ari_delta_vs_runtime_delta.csv", index=False)

    meta = {
        "run_id": args.run_id,
        "runtime_contract_version": sorted(x for x in contract if x),
        "frozen_definition": ("method_runtime_equivalent = full "
                              "physical_trace_runtime + arm_scoring_runtime"),
        "identity_holds_on_all_records": identity_ok,
        "identity_max_residual_s": identity_max_residual,
        "identity_tolerance_note": ("records are persisted rounded to 6 decimals; the residual is bounded by that rounding"),
        "shared_cost_not_divided": True,
        "records": int(len(df)),
        "field_availability": availability,
        "total_sum_method_runtime_equivalent_s": float(df[EQ].sum()),
        "total_deduplicated_wall_clock_s": float(
            sh["deduplicated_actual_wall_clock"].sum()),
        "total_saved_by_physical_reuse_s": float(sh["wall_clock_saved_by_reuse"].sum()),
    }
    (out / "runtime_provenance.json").write_text(
        json.dumps(meta, indent=2, default=str), encoding="utf-8")

    print("  records %d | identity holds: %s" % (len(df), identity_ok))
    print("  fastest xy_2d median: %s" % dep.nsmallest(1, "xy_2d_runtime_median")[
        ["short", "xy_2d_runtime_median"]].to_dict("records"))
    print("  fastest three-view median: %s" % tvs.head(1)[
        ["short", "three_view_median"]].to_dict("records"))
    print("  pareto (bestview): %s" % tr[tr["pareto_efficient"]]["short"].tolist())
    print("  pareto (xy_2d):    %s" % dep2[dep2["pareto_efficient"]]["short"].tolist())
    print("  equivalent total %.1f s vs deduplicated wall clock %.1f s (saved %.1f s)"
          % (meta["total_sum_method_runtime_equivalent_s"],
             meta["total_deduplicated_wall_clock_s"],
             meta["total_saved_by_physical_reuse_s"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
