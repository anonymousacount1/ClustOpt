"""Stage-1D: New-46 / per-metric attribution for the offline 2x3 result.

Explains WHY FO > FP: which metrics carry real oracle value, which the Full-60
predictor misses, and whether the failure is under-prediction, rank error or
competition against Head-14.

Pure analysis. No clustering, no Optuna, no training, no modification of any
Stage-1C output. Predicted utility vectors are recomputed by the PRODUCTION
predictor (inference only) because only the selected Top-5 were persisted.

Outputs (results_analysis/.../stage1_2x3_aggregation/offline_fixed32/):
    stage1d_metric_attribution.csv
    stage1d_new46_oracle_misses.csv
    stage1d_family_metric_attribution.csv
    stage1d_view_metric_attribution.csv
    stage1d_metric_set_overlap.csv
    stage1d_tie_plateau_analysis.csv
    stage1d_group_prediction_quality.csv
    stage1d_topk_composition.csv
    stage1d_summary.json

Run:
    .venv_clustopt/Scripts/python.exe scripts/stage1/build_stage1d_attribution.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from pathlib import Path  # noqa: E402

from models.Clustering_Repository_Builder.experiments.experiment_execution.config import (  # noqa: E402
    ExperimentExecutionConfig,
)
from models.Clustering_Repository_Builder.experiments.experiment_execution.output_writer import (  # noqa: E402
    read_json, write_json_atomic,
)
from models.Clustering_Repository_Builder.experiments.experiment_execution.run_offline_2x3_all_subfamilies import (  # noqa: E402
    discover_subfamilies, load_offline_records,
)

OUT = Path(REPO) / "results_analysis/clustering_repository/stage1_2x3_aggregation/offline_fixed32"
DOCS = Path(REPO) / "docs/internal_reports/stage1"
FULL60_DIR = "results_analysis/mlp/20260615_231715__metric_utility_mlp_split1_holdout"
HEAD14_DIR = "results_analysis/mlp/20260823_195830__metric_utility_mlp_head14_split1_holdout"
VIEWS = ("x_only", "y_only", "xy_2d")
TOP_K = 5


def main() -> int:
    print("=" * 78); print("STAGE-1D  NEW-46 / PER-METRIC ATTRIBUTION"); print("=" * 78)

    # ---------------------------------------------------- A1 partition
    from models.metric_utility_mlp.head_groups import (
        METRIC_HEAD_GROUPS, resolve_target_group,
    )
    head14 = resolve_target_group("head14")
    full60 = [m for ms in METRIC_HEAD_GROUPS.values() for m in ms]
    new46 = [m for m in full60 if m not in set(head14)]
    group_of = {m: g for g, ms in METRIC_HEAD_GROUPS.items() for m in ms}
    assert len(head14) == 14 and len(new46) == 46 and len(full60) == 60
    assert not (set(head14) & set(new46)) and set(head14) | set(new46) == set(full60)
    print(f"  partition OK: Head-14={len(head14)} New-46={len(new46)} Full-60={len(full60)}")

    # registry-key partition (pattern_base / image_base) for A8
    part = pd.read_csv(DOCS / "metric_partition.csv")
    cat_of = dict(zip(part["metric"], part["category_dir"]))

    # ---------------------------------------------------- load offline records
    print("\n[1/6] loading Stage-1C offline records ...", flush=True)
    analyzed = Path(REPO) / "results_analysis/clustering_repository/analyzed_data"
    split_csv = analyzed / "experiment_splits/dataset_split_assignments.csv"
    frames = []
    subs = discover_subfamilies(analyzed)
    for sub in subs:
        cfg = ExperimentExecutionConfig(
            repo_root=Path(REPO), subfamily_dir=sub, split_assignments=split_csv,
            split_id=1, split_dir_suffix="offline_fixed32",
            method_set="stage1_2x3", max_workers=8)
        df = load_offline_records(cfg)
        if not df.empty:
            frames.append(df)
    rec = pd.concat(frames, ignore_index=True)
    print(f"  records: {len(rec):,}")

    # dataset dir lookup
    ds_dir = {}
    for sub in subs:
        for d in sub.iterdir():
            if d.is_dir():
                ds_dir[d.name] = d

    keys = sorted({(r.dataset_id, r.view_id) for r in rec.itertuples()})
    print(f"  dataset/view keys: {len(keys):,}")

    # ---------------------------------------------------- predictions + truth
    print("\n[2/6] recomputing predicted utilities (production predictor, inference only)")
    from models.ClustOpt.cluster_validity_indices.dynamic_metric_selection.regressor_predictor import (
        get_regressor_predictor,
    )
    p60 = get_regressor_predictor(FULL60_DIR)
    p14 = get_regressor_predictor(HEAD14_DIR)
    feat_cols = list(p60.feature_columns)

    uni = pd.read_csv(
        analyzed / "unified_training_dataset/unified_training_dataset.csv",
        usecols=["dataset_id", "view_mode"] + feat_cols)
    uni = uni.set_index(["dataset_id", "view_mode"])

    true_u, pred60, pred14 = {}, {}, {}
    for n, (dsid, view) in enumerate(keys, 1):
        uv = read_json(ds_dir[dsid] / "utility" / view / "utility_vector.json") or {}
        us = uv.get("utilities", {})
        true_u[(dsid, view)] = {k.replace("utility__", ""): float(v) for k, v in us.items()}
        x = uni.loc[(dsid, view), feat_cols].to_numpy(dtype=np.float64)
        pred60[(dsid, view)] = p60.predict(x)
        pred14[(dsid, view)] = p14.predict(x)
        if n % 500 == 0:
            print(f"    {n:,}/{len(keys):,}", flush=True)

    # ---------------------------------------------------- selections
    print("\n[3/6] per-metric attribution")
    sel = {}
    for r in rec.itertuples():
        sel[(r.dataset_id, r.view_id, r.arm)] = set(str(r.selected_metrics).split("|"))
    ari = {(r.dataset_id, r.view_id, r.arm): r.ari for r in rec.itertuples()}
    fam_of = {r.dataset_id: r.family for r in rec.itertuples()}

    acc = {m: defaultdict(float) for m in full60}
    for m in full60:
        acc[m]["_n"] = 0
    fam_rows, view_rows = [], []
    fam_acc = defaultdict(lambda: defaultdict(float))
    view_acc = defaultdict(lambda: defaultdict(float))
    overlap_rows = []

    for (dsid, view) in keys:
        tu = true_u[(dsid, view)]
        pu = pred60[(dsid, view)]
        fam = fam_of[dsid]
        t_rank = {m: i for i, m in enumerate(sorted(tu, key=lambda x: -tu[x]), 1)}
        p_rank = {m: i for i, m in enumerate(sorted(pu, key=lambda x: -pu[x]), 1)}
        t_top5 = set(sorted(tu, key=lambda x: -tu[x])[:TOP_K])
        p_top5 = set(sorted(pu, key=lambda x: -pu[x])[:TOP_K])
        fo = sel.get((dsid, view, "FO"), set())
        fp = sel.get((dsid, view, "FP"), set())
        d_ari = (ari.get((dsid, view, "FO")) or 0) - (ari.get((dsid, view, "FP")) or 0)

        overlap_rows.append({
            "dataset_id": dsid, "view_id": view, "family": fam,
            "overlap": len(fo & fp), "exact_match": int(fo == fp),
            "fo_only": "|".join(sorted(fo - fp)), "fp_only": "|".join(sorted(fp - fo)),
            "fo_only_new46": len((fo - fp) & set(new46)),
            "fo_minus_fp_ari": d_ari,
            "same_candidate": int(
                rec[(rec.dataset_id == dsid) & (rec.view_id == view)
                    & (rec.arm == "FO")]["selected_algorithm"].iloc[0]
                == rec[(rec.dataset_id == dsid) & (rec.view_id == view)
                       & (rec.arm == "FP")]["selected_algorithm"].iloc[0]),
        })

        for m in full60:
            a = acc[m]
            a["_n"] += 1
            a["true_sum"] += tu.get(m, np.nan)
            a["pred_sum"] += pu.get(m, np.nan)
            a["abs_err"] += abs(tu.get(m, 0) - pu.get(m, 0))
            a["sq_err"] += (tu.get(m, 0) - pu.get(m, 0)) ** 2
            a["t_rank"] += t_rank.get(m, 60)
            a["p_rank"] += p_rank.get(m, 60)
            a["abs_rank_err"] += abs(t_rank.get(m, 60) - p_rank.get(m, 60))
            a["true_top5"] += m in t_top5
            a["pred_top5"] += m in p_top5
            a["fo_sel"] += m in fo
            a["fp_sel"] += m in fp
            miss = (m in fo) and (m not in fp)
            a["miss"] += miss
            a["fpos"] += (m in fp) and (m not in fo)
            if miss:
                a["miss_true_u"] += tu.get(m, 0)
                a["miss_pred_u"] += pu.get(m, 0)
                a["miss_t_rank"] += t_rank.get(m, 60)
                a["miss_p_rank"] += p_rank.get(m, 60)
                a["miss_dari"] += d_ari
                a["miss_win"] += d_ari > 1e-12
                a["miss_loss"] += d_ari < -1e-12
                fam_acc[(m, fam)]["miss"] += 1
                fam_acc[(m, fam)]["dari"] += d_ari
                view_acc[(m, view)]["miss"] += 1
                view_acc[(m, view)]["dari"] += d_ari
            fam_acc[(m, fam)]["fo"] += m in fo
            view_acc[(m, view)]["fo"] += m in fo

    rows = []
    for m in full60:
        a = acc[m]; n = a["_n"]
        miss = a["miss"]
        rows.append({
            "metric": m,
            "partition": "HEAD_14" if m in set(head14) else "NEW_46",
            "head_group": group_of.get(m, "?"),
            "registry_category": cat_of.get(m, "?"),
            "true_utility_mean": a["true_sum"] / n,
            "pred_utility_mean": a["pred_sum"] / n,
            "prediction_mae": a["abs_err"] / n,
            "prediction_rmse": (a["sq_err"] / n) ** 0.5,
            "mean_true_rank": a["t_rank"] / n,
            "mean_pred_rank": a["p_rank"] / n,
            "mean_abs_rank_error": a["abs_rank_err"] / n,
            "true_top5_rate": a["true_top5"] / n,
            "pred_top5_rate": a["pred_top5"] / n,
            "oracle_selection_rate": a["fo_sel"] / n,
            "pred_selection_rate": a["fp_sel"] / n,
            "oracle_miss_count": int(miss),
            "oracle_miss_rate": miss / n,
            "false_positive_count": int(a["fpos"]),
            "selection_recall": (a["fo_sel"] - miss) / a["fo_sel"] if a["fo_sel"] else np.nan,
            "selection_precision": (a["fp_sel"] - a["fpos"]) / a["fp_sel"] if a["fp_sel"] else np.nan,
            "true_utility_when_missed": a["miss_true_u"] / miss if miss else np.nan,
            "pred_utility_when_missed": a["miss_pred_u"] / miss if miss else np.nan,
            "underprediction_when_missed": (a["miss_true_u"] - a["miss_pred_u"]) / miss if miss else np.nan,
            "true_rank_when_missed": a["miss_t_rank"] / miss if miss else np.nan,
            "pred_rank_when_missed": a["miss_p_rank"] / miss if miss else np.nan,
            "assoc_FO_minus_FP_ari_when_missed": a["miss_dari"] / miss if miss else np.nan,
            "miss_win": int(a["miss_win"]), "miss_loss": int(a["miss_loss"]),
            "opportunity_score": (a["miss_dari"] / n) if miss else 0.0,
        })
    attrib = pd.DataFrame(rows)
    attrib.to_csv(OUT / "stage1d_metric_attribution.csv", index=False)

    new46_tbl = attrib[attrib.partition == "NEW_46"].sort_values(
        "opportunity_score", ascending=False)
    new46_tbl.to_csv(OUT / "stage1d_new46_oracle_misses.csv", index=False)

    pd.DataFrame([{"metric": m, "family": f, **dict(v)}
                  for (m, f), v in fam_acc.items()]).to_csv(
        OUT / "stage1d_family_metric_attribution.csv", index=False)
    pd.DataFrame([{"metric": m, "view_id": vw, **dict(v)}
                  for (m, vw), v in view_acc.items()]).to_csv(
        OUT / "stage1d_view_metric_attribution.csv", index=False)
    ov = pd.DataFrame(overlap_rows)
    ov.to_csv(OUT / "stage1d_metric_set_overlap.csv", index=False)

    # ---------------------------------------------------- A6 composition
    print("\n[4/6] Top-5 composition (New-46 count) by arm")
    comp_rows = []
    for arm in ("FP", "FO"):
        sub = rec[rec.arm == arm]
        for r in sub.itertuples():
            s = set(str(r.selected_metrics).split("|"))
            comp_rows.append({"arm": arm, "n_new46": len(s & set(new46)),
                              "ari": r.ari, "regret": r.ari_regret,
                              "ceiling_hit": bool(r.ceiling_hit)})
    comp = pd.DataFrame(comp_rows)
    comp_summary = (comp.groupby(["arm", "n_new46"])
                    .agg(N=("ari", "size"), mean_ari=("ari", "mean"),
                         mean_regret=("regret", "mean"),
                         ceiling_hit_rate=("ceiling_hit", "mean"))
                    .reset_index())
    comp_summary.to_csv(OUT / "stage1d_topk_composition.csv", index=False)

    # ---------------------------------------------------- A8 group quality
    print("[5/6] prediction quality by metric group")
    grp = (attrib.groupby("partition")
           .agg(n_metrics=("metric", "size"), mae=("prediction_mae", "mean"),
                rmse=("prediction_rmse", "mean"),
                abs_rank_err=("mean_abs_rank_error", "mean"),
                true_top5_rate=("true_top5_rate", "mean"),
                pred_top5_rate=("pred_top5_rate", "mean"),
                oracle_sel=("oracle_selection_rate", "mean"),
                pred_sel=("pred_selection_rate", "mean")).reset_index())
    grp2 = (attrib.groupby("registry_category")
            .agg(n_metrics=("metric", "size"), mae=("prediction_mae", "mean"),
                 abs_rank_err=("mean_abs_rank_error", "mean"),
                 oracle_sel=("oracle_selection_rate", "mean"),
                 pred_sel=("pred_selection_rate", "mean"),
                 miss_rate=("oracle_miss_rate", "mean")).reset_index())
    pd.concat([grp.rename(columns={"partition": "group"}),
               grp2.rename(columns={"registry_category": "group"})],
              ignore_index=True).to_csv(
        OUT / "stage1d_group_prediction_quality.csv", index=False)

    # ---------------------------------------------------- A12 tie plateau
    print("[6/6] tie-plateau analysis (full 1,055-dataset population)")
    tie = rec[pd.to_numeric(rec["n_tied_at_max"], errors="coerce") > 1].copy()
    tie["spread"] = (pd.to_numeric(tie["tie_ari_max"], errors="coerce")
                     - pd.to_numeric(tie["tie_ari_min"], errors="coerce"))
    tie["headroom"] = (pd.to_numeric(tie["tie_ari_max"], errors="coerce")
                       - pd.to_numeric(tie["ari"], errors="coerce"))
    tie_rows = []
    for keyname, grp_ in (("overall", [("all", tie)]),
                          ("arm", list(tie.groupby("arm"))),
                          ("view", list(tie.groupby("view_id"))),
                          ("family", list(tie.groupby("family")))):
        for name, g in grp_:
            if len(g) == 0:
                continue
            tie_rows.append({
                "scope": keyname, "key": name, "n_tied_records": len(g),
                "mean_tie_size": float(pd.to_numeric(g["n_tied_at_max"]).mean()),
                "max_tie_size": int(pd.to_numeric(g["n_tied_at_max"]).max()),
                "pct_spread_zero": float((g["spread"] <= 1e-12).mean()),
                "pct_spread_positive": float((g["spread"] > 1e-12).mean()),
                "mean_positive_spread": float(g.loc[g["spread"] > 1e-12, "spread"].mean())
                if (g["spread"] > 1e-12).any() else 0.0,
                "max_spread": float(g["spread"].max()),
                "mean_headroom_all_ties": float(g["headroom"].mean()),
                "mean_headroom_when_positive": float(
                    g.loc[g["spread"] > 1e-12, "headroom"].mean())
                if (g["spread"] > 1e-12).any() else 0.0,
            })
    pd.DataFrame(tie_rows).to_csv(OUT / "stage1d_tie_plateau_analysis.csv", index=False)

    # per-record potential gain if ties were broken perfectly (descriptive)
    total_headroom = float(tie["headroom"].sum())
    n_rec = len(rec)

    summary = {
        "population": {"records": int(n_rec), "dataset_view_keys": len(keys),
                       "datasets": int(rec.dataset_id.nunique())},
        "partition": {"head14": len(head14), "new46": len(new46), "full60": len(full60)},
        "fo_fp_overlap": {
            "mean_overlap": float(ov["overlap"].mean()),
            "exact_match_rate": float(ov["exact_match"].mean()),
            "mean_fo_only_new46": float(ov["fo_only_new46"].mean()),
        },
        "tie_plateau": {
            "tied_records": int(len(tie)),
            "tie_rate": float(len(tie) / n_rec),
            "pct_spread_zero": float((tie["spread"] <= 1e-12).mean()),
            "pct_spread_positive": float((tie["spread"] > 1e-12).mean()),
            "mean_headroom_per_tied_record": float(tie["headroom"].mean()),
            "total_headroom_ari_units": total_headroom,
            "mean_headroom_per_record_overall": total_headroom / n_rec,
        },
        "group_quality": grp.to_dict("records"),
        "top_new46_opportunities": new46_tbl.head(15)[
            ["metric", "registry_category", "oracle_selection_rate",
             "pred_selection_rate", "oracle_miss_rate", "true_utility_when_missed",
             "pred_utility_when_missed", "underprediction_when_missed",
             "assoc_FO_minus_FP_ari_when_missed", "opportunity_score"]
        ].to_dict("records"),
    }
    write_json_atomic(OUT / "stage1d_summary.json", summary)

    # ------------------------------------------------------------- print
    print("\n" + "=" * 78); print("PREDICTION QUALITY: Head-14 vs New-46"); print("=" * 78)
    print(grp.round(4).to_string(index=False))
    print("\nby registry category:")
    print(grp2.round(4).to_string(index=False))

    print("\n" + "=" * 78); print("TOP 15 NEW-46 OPPORTUNITIES (by opportunity score)")
    print("=" * 78)
    cols = ["metric", "registry_category", "oracle_selection_rate", "pred_selection_rate",
            "oracle_miss_rate", "true_utility_when_missed", "pred_utility_when_missed",
            "assoc_FO_minus_FP_ari_when_missed"]
    print(new46_tbl.head(15)[cols].round(4).to_string(index=False))

    print("\n" + "=" * 78); print("FO vs FP TOP-5 SET OVERLAP"); print("=" * 78)
    print(f"  mean overlap        : {ov['overlap'].mean():.3f} / 5")
    print(f"  exact match rate    : {ov['exact_match'].mean():.4f}")
    print(f"  mean FO-only New-46 : {ov['fo_only_new46'].mean():.3f}")

    print("\n" + "=" * 78); print("TOP-5 COMPOSITION (# New-46 metrics)"); print("=" * 78)
    print(comp_summary.round(4).to_string(index=False))

    print("\n" + "=" * 78); print("TIE PLATEAU"); print("=" * 78)
    t = summary["tie_plateau"]
    print(f"  tied records            : {t['tied_records']:,} / {n_rec:,} "
          f"({t['tie_rate']:.4f})")
    print(f"  ties with ARI spread 0  : {t['pct_spread_zero']:.4f}")
    print(f"  ties with spread > 0    : {t['pct_spread_positive']:.4f}")
    print(f"  mean headroom per tie   : {t['mean_headroom_per_tied_record']:.4f} ARI")
    print(f"  mean headroom / record  : {t['mean_headroom_per_record_overall']:.4f} ARI")
    print(f"\n  package -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
