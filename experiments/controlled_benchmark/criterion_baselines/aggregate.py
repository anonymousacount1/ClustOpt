"""Seeded criterion-baseline sweep (internal label E1; paper Tables 1, 16 and 17). Included for transparency; it expects the regenerated controlled repository.

E1 FINAL -- full Split-1 population analysis (1,055 datasets).

Manuscript-grade. Population-weighted: every Split-1 dataset counts once, so
subfamilies contribute in proportion to their true size (5-20 datasets), unlike
the early stratified sample which weighted all subfamilies equally.

Produces
  e1_final_arm_summary.csv
  e1_final_paired_contrasts.csv
  e1_final_family_contrasts.csv
  e1_final_hypotheses.json
  e1_final_sample_vs_full.csv
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE.parent / "p0_e8_e13_factual_audit"))
from _stats_common import paired_block, row_bootstrap_ci  # noqa: E402

AN = REPO / "results_analysis" / "clustering_repository" / "analyzed_data"
VIEWS = ("x_only", "y_only", "xy_2d")
E1_SPLIT = "split_01_e1_conditioning"
S1_SPLIT = "split_01_online_fixed50"
E1_ARMS = ("e1_c4_seed", "e1_meanprof", "e1_core3", "e1_best1", "e1_random")
REUSED = {"stage1_full60_uniform_fixed50": S1_SPLIT,
          "stage1_full60_mlp_top5_raw_fixed50": S1_SPLIT,
          "stage1_full60_oracle_top5_raw_fixed50": S1_SPLIT}
LABEL = {
    "e1_c4_seed": "E1-C4-SEED (conditioned: KNN predicted profile)",
    "e1_meanprof": "E1-MEANPROF (dev mean profile)",
    "e1_core3": "E1-CORE3 (static generalist core)",
    "e1_best1": "E1-BEST1 (dev best single index)",
    "e1_random": "E1-RANDOM (no criterion)",
    "stage1_full60_uniform_fixed50": "E1-UNIF60 (uniform over FULL60, reused)",
    "stage1_full60_mlp_top5_raw_fixed50": "Stage-1 MLP top-5 (conditioned, reused)",
    "stage1_full60_oracle_top5_raw_fixed50": "Stage-1 ORACLE top-5 (ceiling, reused)",
}
REF = "e1_c4_seed"
MANDATORY = ("e1_meanprof", "e1_core3", "e1_best1", "e1_random",
             "stage1_full60_uniform_fixed50")


def LP(p) -> str:
    return "\\\\?\\" + os.path.abspath(str(p))


def rj(p) -> Optional[dict]:
    try:
        with open(LP(p), "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def load() -> pd.DataFrame:
    sa = pd.read_csv(AN / "experiment_splits" / "dataset_split_assignments.csv")
    sa = sa[pd.to_numeric(sa["split_id"], errors="coerce") == 1]
    rows: List[Dict] = []
    for _, r in sa.iterrows():
        ds = Path(r["dataset_dir"])
        for arm in list(E1_ARMS) + list(REUSED):
            split = REUSED.get(arm, E1_SPLIT)
            for v in VIEWS:
                base = ds / "experiments" / split / arm / v
                st = rj(base / "status.json")
                if not st or st.get("status") != "success":
                    continue
                br = rj(base / "best_result.json") or {}
                seed = br.get("search_seed")
                oracle, rnd = np.nan, np.nan
                try:
                    with open(LP(base / "clustopt_results.csv"), "r",
                              encoding="utf-8") as fh:
                        cdf = pd.read_csv(fh)
                    ok = cdf
                    if "valid" in cdf.columns:
                        ok = cdf[cdf["valid"].astype(str).str.lower()
                                 .isin(["true", "1"])]
                    ari = pd.to_numeric(ok.get("ARI"), errors="coerce").dropna()
                    if ari.size:
                        oracle = float(ari.max())
                        if arm == "e1_random" and seed is not None:
                            rng = np.random.default_rng(int(seed))
                            rnd = float(ari.to_numpy()[rng.integers(0, ari.size)])
                except Exception:
                    pass
                ep = float(st["ari"]) if st.get("ari") is not None else np.nan
                rows.append({
                    "dataset_id": ds.name, "family_id": r["family_id"],
                    "subfamily_id": r["subfamily_id"], "view_id": v, "arm": arm,
                    "endpoint_ari": (rnd if arm == "e1_random" else ep),
                    "slate_oracle_ari": oracle,
                    "runtime_sec": st.get("runtime_sec"),
                })
    return pd.DataFrame(rows)


def main() -> int:
    runs = load()
    cnt = runs.groupby(["dataset_id", "arm"])["view_id"].nunique().unstack()
    arms = [a for a in list(E1_ARMS) + list(REUSED) if a in cnt.columns]
    complete = cnt[(cnt[arms] == 3).all(axis=1)].index.tolist()
    runs = runs[runs.dataset_id.isin(complete)].copy()
    meta = runs.drop_duplicates("dataset_id").set_index("dataset_id")

    ds = (runs.groupby(["dataset_id", "arm"])
          .agg(endpoint=("endpoint_ari", "max"),
               oracle=("slate_oracle_ari", "max"),
               runtime_sec_3view=("runtime_sec", "sum")).reset_index())
    xy = (runs[runs.view_id == "xy_2d"]
          .set_index(["dataset_id", "arm"])["endpoint_ari"]
          .rename("xy_endpoint").reset_index())
    ds = ds.merge(xy, on=["dataset_id", "arm"], how="left")
    ds["gap"] = ds.oracle - ds.endpoint
    for c in ("family_id", "subfamily_id"):
        ds[c] = ds.dataset_id.map(meta[c])
    ds.to_csv(HERE / "e1_final_dataset_results.csv", index=False)

    P = ds.pivot(index="dataset_id", columns="arm", values="endpoint")
    O = ds.pivot(index="dataset_id", columns="arm", values="oracle")
    X = ds.pivot(index="dataset_id", columns="arm", values="xy_endpoint")
    fam = meta["family_id"]

    srows = []
    for a in P.columns:
        e, o = P[a].dropna(), O[a].dropna()
        ci = row_bootstrap_ci(e.to_numpy())
        srows.append({
            "arm": a, "label": LABEL.get(a, a), "n_datasets": int(e.size),
            "mean_endpoint_bestview": float(e.mean()),
            "ci_lo": ci["lo"], "ci_hi": ci["hi"],
            "median_endpoint": float(e.median()),
            "mean_endpoint_xy": float(X[a].dropna().mean()),
            "mean_slate_oracle": float(o.mean()),
            "mean_endpoint_to_oracle_gap": float((o - e).mean()),
            "total_runtime_hours": float(
                ds.loc[ds.arm == a, "runtime_sec_3view"].sum() / 3600),
            "is_mandatory_comparator": a in MANDATORY,
        })
    summ = pd.DataFrame(srows).sort_values("mean_endpoint_bestview", ascending=False)
    summ.to_csv(HERE / "e1_final_arm_summary.csv", index=False)

    crows = []
    for a in P.columns:
        if a == REF:
            continue
        for unit, F in (("bestview_endpoint", P), ("xy_endpoint", X),
                        ("slate_oracle", O)):
            s = F[[REF, a]].dropna()
            b = paired_block(s[REF].to_numpy(), s[a].to_numpy())
            b.update({"baseline_arm": a, "baseline_label": LABEL.get(a, a),
                      "unit": unit, "is_mandatory": a in MANDATORY})
            crows.append(b)
    con = pd.DataFrame(crows)
    front = ["baseline_arm", "baseline_label", "unit", "is_mandatory", "n",
             "mean_a", "mean_b", "mean_delta", "median_delta", "ci_lo", "ci_hi",
             "wins_eps", "ties_eps", "losses_eps", "p_value", "rank_biserial"]
    con = con[[c for c in front if c in con.columns]]
    con.to_csv(HERE / "e1_final_paired_contrasts.csv", index=False)

    frows = []
    for a in P.columns:
        if a == REF:
            continue
        for f, idx in P.groupby(fam).groups.items():
            s = P.loc[idx, [REF, a]].dropna()
            so = O.loc[idx, [REF, a]].dropna()
            if s.empty:
                continue
            b = paired_block(s[REF].to_numpy(), s[a].to_numpy())
            bo = paired_block(so[REF].to_numpy(), so[a].to_numpy())
            frows.append({
                "family_id": f, "baseline_arm": a,
                "baseline_label": LABEL.get(a, a), "n": b["n"],
                "c4_mean": b["mean_a"], "baseline_mean": b["mean_b"],
                "delta_endpoint": b["mean_delta"],
                "ci_lo": b["ci_lo"], "ci_hi": b["ci_hi"],
                "interval_excludes_zero": bool(b["ci_lo"] * b["ci_hi"] > 0),
                "wins_eps": b["wins_eps"], "ties_eps": b["ties_eps"],
                "losses_eps": b["losses_eps"], "p_value": b["p_value"],
                "rank_biserial": b["rank_biserial"],
                "c4_oracle": bo["mean_a"], "baseline_oracle": bo["mean_b"],
                "delta_oracle_steering": bo["mean_delta"],
                "delta_selection": b["mean_delta"] - bo["mean_delta"],
            })
    famdf = pd.DataFrame(frows)
    famdf.to_csv(HERE / "e1_final_family_contrasts.csv", index=False)

    def blk(a, F):
        s = F[[REF, a]].dropna()
        return paired_block(s[REF].to_numpy(), s[a].to_numpy())

    H: Dict[str, object] = {"n_datasets": len(complete),
                            "population_weighted": True}
    A = {}
    for a in ("e1_meanprof", "e1_core3"):
        sub_f = famdf[famdf.baseline_arm == a]
        ov = blk(a, P)
        arcs = sub_f[sub_f.family_id == "arcs_circles_rings"]["delta_endpoint"]
        non = sub_f[sub_f.family_id != "arcs_circles_rings"]["delta_endpoint"]
        A[a] = {
            "overall_delta": ov["mean_delta"], "ci": [ov["ci_lo"], ov["ci_hi"]],
            "p_value": ov["p_value"], "rank_biserial": ov["rank_biserial"],
            "families_c4_leads": int((sub_f.delta_endpoint > 0).sum()),
            "families_total": int(len(sub_f)),
            "families_c4_leads_significantly": int(
                ((sub_f.delta_endpoint > 0) & sub_f.interval_excludes_zero).sum()),
            "families_baseline_leads_significantly": int(
                ((sub_f.delta_endpoint < 0) & sub_f.interval_excludes_zero).sum()),
            "median_family_delta": float(sub_f.delta_endpoint.median()),
            "arcs_delta": float(arcs.iloc[0]) if len(arcs) else None,
            "mean_delta_excluding_arcs": float(non.mean()) if len(non) else None,
        }
    H["A_across_families"] = A

    om = summ.set_index("arm")["mean_slate_oracle"]
    em = summ.set_index("arm")["mean_endpoint_bestview"]
    H["B_oracle_similarity"] = {
        "oracle_mean_by_arm": {k: float(v) for k, v in om.items()},
        "endpoint_mean_by_arm": {k: float(v) for k, v in em.items()},
        "oracle_spread": float(om.max() - om.min()),
        "endpoint_spread": float(em.max() - em.min()),
        "ratio": float((em.max() - em.min()) / (om.max() - om.min())),
        "paired_oracle_delta_vs_c4": {
            a: {"delta": blk(a, O)["mean_delta"],
                "ci": [blk(a, O)["ci_lo"], blk(a, O)["ci_hi"]],
                "excludes_zero": bool(blk(a, O)["ci_lo"] * blk(a, O)["ci_hi"] > 0)}
            for a in P.columns if a != REF},
    }

    C = {}
    for a in P.columns:
        if a == REF:
            continue
        be, bo = blk(a, P), blk(a, O)
        tot, steer = be["mean_delta"], bo["mean_delta"]
        sel = tot - steer
        C[a] = {"label": LABEL.get(a, a), "total_endpoint_delta": tot,
                "steering_oracle_delta": steer, "selection_delta": sel,
                "steering_share": (steer / tot) if tot else None,
                "selection_share": (sel / tot) if tot else None,
                "c4_gap": float(summ.loc[summ.arm == REF,
                                         "mean_endpoint_to_oracle_gap"].iloc[0]),
                "baseline_gap": float(summ.loc[summ.arm == a,
                                               "mean_endpoint_to_oracle_gap"].iloc[0])}
    H["C_steering_vs_selection"] = {
        "identity": "endpoint_delta = oracle_delta (steering) + selection_delta",
        "per_baseline": C}
    (HERE / "e1_final_hypotheses.json").write_text(
        json.dumps(H, indent=2, default=str), encoding="utf-8")

    # ---- sample vs full comparison ----
    try:
        sm = pd.read_csv(HERE / "e1_sample_arm_summary.csv").set_index("arm")
        sc = pd.read_csv(HERE / "e1_sample_paired_contrasts.csv")
        sc = sc[sc.unit == "bestview_endpoint"].set_index("baseline_arm")
        fc = con[con.unit == "bestview_endpoint"].set_index("baseline_arm")
        rows = []
        for a in summ.arm:
            rows.append({
                "arm": a, "label": LABEL.get(a, a),
                "sample_mean": float(sm.loc[a, "mean_endpoint_bestview"])
                if a in sm.index else None,
                "full_mean": float(summ.loc[summ.arm == a,
                                            "mean_endpoint_bestview"].iloc[0]),
                "sample_delta_vs_c4": float(sc.loc[a, "mean_delta"])
                if a in sc.index else None,
                "full_delta_vs_c4": float(fc.loc[a, "mean_delta"])
                if a in fc.index else None,
            })
        svf = pd.DataFrame(rows)
        svf["mean_shift"] = svf.full_mean - svf.sample_mean
        svf["delta_shift"] = svf.full_delta_vs_c4 - svf.sample_delta_vs_c4
        svf.to_csv(HERE / "e1_final_sample_vs_full.csv", index=False)
    except Exception as exc:
        print("sample-vs-full comparison skipped:", exc)

    pd.set_option("display.width", 250)
    print("datasets: %d" % len(complete))
    print()
    print(summ[["label", "mean_endpoint_bestview", "ci_lo", "ci_hi",
                "mean_endpoint_xy", "mean_slate_oracle",
                "mean_endpoint_to_oracle_gap"]].to_string(index=False))
    print()
    print(json.dumps(H["B_oracle_similarity"]["oracle_spread"], indent=1),
          "oracle spread |", json.dumps(H["B_oracle_similarity"]["endpoint_spread"]),
          "endpoint spread | ratio",
          round(H["B_oracle_similarity"]["ratio"], 1))
    print()
    for k, v in C.items():
        print("%-36s total=%+.5f steering=%+.5f (%.1f%%) selection=%+.5f (%.1f%%)"
              % (k, v["total_endpoint_delta"], v["steering_oracle_delta"],
                 100 * v["steering_share"], v["selection_delta"],
                 100 * v["selection_share"]))
    print()
    print(json.dumps(A, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
