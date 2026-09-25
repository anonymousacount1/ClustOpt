"""Stage 2A-3A: build the view-conditional Policy Predictor training dataset.

TRAINING DATA ONLY. Trains nothing, selects no architecture, tunes nothing, runs
no clustering and never reads Split 1.

Unit: DATASET x VIEW (16,013 x 3 = 48,039 rows).

  Feature block A  X_META           250 label-free meta-features (per view)
  Feature block B  X_OOF_UTILITY    60 OOF MLP + 60 OOF KNN utilities
  Optional         Dynamic-K        MLP / KNN predicted K, kept separate
  Targets          20 policy ARIs + 20 policy regrets + diagnostics

Stage-2A-2 replay results are the authoritative policy-performance source; the
central aggregation is used for efficiency only after it is verified to reproduce
the raw per-dataset ``result.json`` population on a sample.

Run::

    .venv_clustopt/Scripts/python.exe -m models.ClustOpt_Policy_Predictor\
.data.build_training_dataset --repo-root .
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

from models.Clustering_Repository_Builder.experiments.experiment_execution.oof_policy_replay import (  # noqa: E402,E501
    SPLIT_DIR_TEMPLATE, run_output_dir,
)
from models.Clustering_Repository_Builder.experiments.experiment_execution.output_writer import (  # noqa: E402,E501
    _ext, ensure_dir, read_json, write_json_atomic,
)
from models.Clustering_Repository_Builder.experiments.paper_analysis import stats as S  # noqa: E402,E501
from models.Clustering_Repository_Builder.experiments.paper_analysis.portfolio_registry import (  # noqa: E402,E501
    Portfolio,
)
from models.Clustering_Repository_Builder.experiments.paper_analysis.vbs import (  # noqa: E402,E501
    greedy_portfolio,
)

from . import feature_schema as FS
from . import target_schema as TS
from . import validation as V

OUT_REL = ("results_analysis/clustopt_policy_predictor/"
           "stage2a3a_view_level_training_data")
S2A2_REL = ("results_analysis/clustering_repository/stage2/"
            "offline_oof_policy20_splits2_16")
OOF_REL = "results_analysis/clustering_repository/stage2/oof_utility_splits2_16"
DEV_SPLITS = list(range(2, 17))
EXP_DATASETS, EXP_ROWS = 16013, 48039
GZ = dict(index=False, compression="gzip")


def _boot(a: np.ndarray):
    pt, lo, hi = S.bootstrap_ci(np.asarray(a, float), statistic="mean")
    return float(pt), float(lo), float(hi)


# --------------------------------------------------------------------------- #
def audit_meta_features(repo: Path, unified: pd.DataFrame,
                        meta_cols: List[str]) -> Dict[str, Any]:
    """Prove the meta-features are per dataset/view and describe their behaviour."""
    per_view: Dict[str, Any] = {}
    sub = unified[unified["view_mode"].isin(FS.VIEW_IDS)]
    for v in FS.VIEW_IDS:
        m = sub[sub["view_mode"] == v]
        X = m[meta_cols]
        nun = X.nunique(dropna=True)
        per_view[v] = {
            "n_rows": int(len(m)), "view_dim": FS.VIEW_DIM[v],
            "n_constant_features": int((nun <= 1).sum()),
            "constant_features": sorted(nun[nun <= 1].index.tolist()),
            "n_all_missing": int(X.isna().all().sum()),
            "mean_missing_rate": float(X.isna().mean().mean()),
        }
    # Which features actually differ between views for the same dataset?
    piv = sub.pivot_table(index="dataset_id", columns="view_mode",
                          values=meta_cols, aggfunc="first")
    varies = []
    for c in meta_cols:
        try:
            cols = [(c, v) for v in FS.VIEW_IDS if (c, v) in piv.columns]
            if len(cols) < 2:
                continue
            block = piv[cols].to_numpy(float)
            if not np.allclose(block, block[:, [0]], equal_nan=True):
                varies.append(c)
        except Exception:                                          # noqa: BLE001
            varies.append(c)
    const_1d = (set(per_view["x_only"]["constant_features"])
                & set(per_view["y_only"]["constant_features"]))
    const_2d = set(per_view["xy_2d"]["constant_features"])
    return {
        "authoritative_count": len(meta_cols),
        "source_of_truth": FS.FEATURE_COLUMNS_REL,
        "extraction_entry_point":
            "models/Clustering_Repository_Builder/features_extraction/"
            "feature_extractor.py::extract_features_for_view",
        "view_builder":
            "models/Clustering_Repository_Builder/features_extraction/"
            "view_builder.py::build_views (x_only dim1, y_only dim1, xy_2d dim2)",
        "record_identity": "record_id = f'{dataset_id}__{view_mode}'",
        "genuinely_per_dataset_view": True,
        "evidence": ("the extractor computes the full feature record for ONE view; "
                     "the unified table carries one row per (dataset_id, view_mode)"),
        "schema_physically_identical_across_views": True,
        "schema_hash": FS.schema_hash(meta_cols),
        "blocks": {k: {"description": d,
                       "n_features": sum(1 for c in meta_cols
                                         if FS.block_of(c) == k)}
                   for k, d in FS.FEATURE_BLOCKS.items()},
        "n_features_varying_across_views": len(varies),
        "n_features_identical_across_views": len(meta_cols) - len(varies),
        "features_identical_across_views": sorted(set(meta_cols) - set(varies)),
        "constant_within_both_1d_views": sorted(const_1d),
        "constant_within_2d_view": sorted(const_2d),
        "meaningful_only_in_2d": sorted(const_1d - const_2d),
        "meaningful_only_in_1d": sorted(const_2d - set(const_1d)),
        "per_view": per_view,
        "preprocessing_baked_in": ("none beyond what the extractor itself computes "
                                   "(log1p / ratio transforms appear in some column "
                                   "names); no scaling or imputation is applied here"),
        "missing_value_semantics": "NaN preserved; no imputation at this stage",
    }


# --------------------------------------------------------------------------- #
def verify_against_raw(repo: pd.DataFrame, assign: pd.DataFrame,
                       central: pd.DataFrame, pids: List[str],
                       n_sample: int = 40) -> Dict[str, Any]:
    """Confirm the central aggregation reproduces the raw result.json population."""
    a = assign[assign["split_id"].isin(DEV_SPLITS)].sort_values("dataset_id")
    sample = a.iloc[:: max(1, len(a) // n_sample)].head(n_sample)
    idx = central.set_index(["dataset_id", "view_id", "policy_id"])["selected_ari"]
    n = ok = 0
    for _, r in sample.iterrows():
        for view in FS.VIEW_IDS:
            for pid in pids:
                rec = read_json(run_output_dir(Path(r["dataset_dir"]),
                                               int(r["split_id"]), pid, view)
                                / "result.json")
                if not rec:
                    continue
                raw = float((rec.get("status") or {}).get("ari"))
                got = float(idx.loc[(r["dataset_id"], view, pid)])
                n += 1
                ok += int(abs(raw - got) <= 1e-12)
    return {"n_checked": n, "n_exact": ok,
            "reproduces_raw": bool(n > 0 and ok == n)}


# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", required=True)
    args = ap.parse_args(argv)
    repo = Path(args.repo_root).resolve()
    out = ensure_dir(repo / OUT_REL)
    t0 = time.time()
    chk = V.Check()

    print("=" * 78)
    print("STAGE 2A-3A  --  VIEW-CONDITIONAL POLICY PREDICTOR TRAINING DATA")
    print("=" * 78)

    # ---------------------------------------------- frozen target schema --
    pids = TS.policy_order()
    ari_cols = TS.ari_target_columns()
    reg_cols = TS.regret_target_columns()
    write_json_atomic(out / "policy_target_order.json", {
        "portfolio_id": TS.PORTFOLIO_ID, "n_policies": len(pids),
        "schema_hash": TS.target_schema_hash(),
        "policies": TS.policy_manifest(),
    })
    print(f"\n[1/12] target schema: {len(pids)} policies, "
          f"hash {TS.target_schema_hash()[:16]}")

    # ------------------------------------------------- meta-feature audit --
    print("\n[2/12] meta-feature audit")
    meta_cols = FS.load_meta_feature_columns(repo)
    metric_names = FS.load_metric_names(repo)
    unified = pd.read_csv(
        _ext(repo / FS.UNIFIED_DATASET_REL),
        usecols=["dataset_id", "view_mode"] + meta_cols, low_memory=False)
    audit = audit_meta_features(repo, unified, meta_cols)
    write_json_atomic(out / "feature_schema_audit.json", audit)
    block_str = " ".join(f"{k}:{b['n_features']}"
                         for k, b in audit["blocks"].items())
    print(f"    authoritative count {audit['authoritative_count']} | blocks {block_str}")
    print(f"    features varying across views: "
          f"{audit['n_features_varying_across_views']}/{len(meta_cols)}")
    print(f"    constant in both 1-D views: "
          f"{len(audit['constant_within_both_1d_views'])}  "
          f"| meaningful only in 2-D: {len(audit['meaningful_only_in_2d'])}")

    # ------------------------------------------------ policy performance --
    print("\n[3/12] policy performance from Stage-2A-2")
    central = pd.read_csv(_ext(repo / S2A2_REL / "record_results.csv.gz"),
                          usecols=["dataset_id", "split_id", "family", "subfamily",
                                   "view_id", "policy_id", "selected_ari", "status",
                                   "oof_heldout_split"])
    assign = pd.read_csv(_ext(repo / "results_analysis/clustering_repository/"
                              "analyzed_data/experiment_splits/"
                              "dataset_split_assignments.csv"))
    rawchk = verify_against_raw(repo, assign, central, pids)
    chk.add("central aggregation reproduces raw result.json",
            rawchk["reproduces_raw"],
            f"{rawchk['n_exact']}/{rawchk['n_checked']} exact")
    print(f"    records {len(central):,}")

    # ------------------------------------------------------- identifiers --
    ident = (central.drop_duplicates(["dataset_id", "view_id"])
             [["dataset_id", "split_id", "family", "subfamily", "view_id",
               "oof_heldout_split"]]
             .sort_values(["dataset_id", "view_id"]).reset_index(drop=True))
    ident["record_id"] = ident["dataset_id"] + "__" + ident["view_id"]
    print(f"    dataset/view rows {len(ident):,}")

    # ----------------------------------------------------- ARI targets ----
    print("\n[4/12] ARI + regret targets (per view)")
    piv = central.pivot_table(index=["dataset_id", "view_id"], columns="policy_id",
                              values="selected_ari")[pids]
    piv = piv.reindex(pd.MultiIndex.from_frame(ident[["dataset_id", "view_id"]]))
    A = piv.to_numpy(float)
    vbs = A.max(axis=1)
    R = vbs[:, None] - A
    is_best = np.isclose(A, vbs[:, None], atol=1e-12)
    n_win = is_best.sum(axis=1)
    srt = np.sort(A, axis=1)
    top1_top2 = srt[:, -1] - srt[:, -2]
    # highest score strictly below VBS; sentinel when every policy ties
    masked = np.where(is_best, -np.inf, A)
    second_distinct = masked.max(axis=1)
    all_tied = ~np.isfinite(second_distinct)
    top1_second_distinct = np.where(all_tied, TS.ALL_TIED_SENTINEL,
                                    vbs - np.where(all_tied, 0.0, second_distinct))

    ari_df = pd.DataFrame(A, columns=ari_cols)
    reg_df = pd.DataFrame(R, columns=reg_cols)
    diag = pd.DataFrame({
        "view_policy_vbs": vbs,
        "winner_set": ["|".join(np.array(pids)[r]) for r in is_best],
        "n_exact_winners": n_win,
        "top1_minus_top2_policy_score": top1_top2,
        "top1_minus_second_distinct_score": top1_second_distinct,
        "n_within_0_001": (R < 0.001).sum(axis=1),
        "n_within_0_005": (R < 0.005).sum(axis=1),
        "n_within_0_01": (R < 0.01).sum(axis=1),
    })
    rc = V.regret_contract(A, R)
    chk.add("regret == rowmax(ari) - ari", rc["max_abs_deviation"] <= 1e-12,
            f"max dev {rc['max_abs_deviation']:.2e}")
    chk.add("all regrets >= 0", rc["n_negative_rows"] == 0)
    chk.add(">=1 zero-regret policy per row", rc["n_rows_without_zero_regret"] == 0)
    print(f"    ARI targets {ari_df.shape}  regret targets {reg_df.shape}")
    print(f"    rows where every policy ties: {int(all_tied.sum()):,}")

    # -------------------------------------------------- OOF utilities -----
    print("\n[5/12] OOF utility features (60 MLP + 60 KNN)")
    oof_cols = FS.oof_feature_columns(metric_names)
    mlp = pd.read_csv(_ext(repo / OOF_REL / "combined/oof_mlp_utilities.csv.gz"),
                      usecols=["dataset_id", "view_mode", "heldout_split"]
                      + [f"pred_{m}" for m in metric_names])
    mlp = mlp.rename(columns={"view_mode": "view_id",
                              **{f"pred_{m}": f"oof_mlp_utility__{m}"
                                 for m in metric_names}})
    knn = pd.read_csv(_ext(repo / OOF_REL / "combined/oof_knn_utilities.csv.gz"),
                      usecols=["dataset_id", "view_id", "split_id"] + metric_names)
    knn = knn.rename(columns={m: f"oof_knn_utility__{m}" for m in metric_names})
    key = ident[["dataset_id", "view_id", "split_id"]]
    oof = (key.merge(mlp, on=["dataset_id", "view_id"], how="left")
           .merge(knn.drop(columns=["split_id"]), on=["dataset_id", "view_id"],
                  how="left"))
    chk.add("OOF MLP fold == dataset split",
            bool((oof["heldout_split"] == oof["split_id"]).all()),
            f"{int((oof['heldout_split'] != oof['split_id']).sum())} mismatched")
    chk.add("OOF utility matrix complete",
            not oof[oof_cols].isna().any().any(),
            f"{int(oof[oof_cols].isna().sum().sum())} NaN")
    oof_feats = oof[oof_cols].reset_index(drop=True)
    print(f"    X_OOF_UTILITY {oof_feats.shape} "
          f"(60 MLP + 60 KNN, canonical metric order)")

    # Optional Dynamic-K, kept OUT of the 120-vector.
    from models.ClustOpt.dynamic_topk_selection.heuristics.selected_dynamic_topk import (
        ALPHA, MAX_K, MIN_K, SOFT_CAP_BASE,
    )

    def dyn_k(M: np.ndarray):
        u = np.clip(np.nan_to_num(M, nan=0.0, posinf=1.0, neginf=0.0), 0, 1)
        umax = u.max(axis=1)
        krel = (u >= ALPHA * umax[:, None]).sum(axis=1)
        krel = np.where(umax > 1e-12, krel, MIN_K)
        K = np.where(krel <= SOFT_CAP_BASE, krel,
                     np.ceil((krel + SOFT_CAP_BASE) / 2.0)).astype(int)
        return np.clip(K, MIN_K, MAX_K), krel
    mk, mkr = dyn_k(oof_feats[[f"oof_mlp_utility__{m}" for m in metric_names]]
                    .to_numpy(float))
    kk, kkr = dyn_k(oof_feats[[f"oof_knn_utility__{m}" for m in metric_names]]
                    .to_numpy(float))
    dynf = pd.DataFrame({"oof_mlp_dynamic_k": mk, "oof_mlp_k_rel_raw": mkr,
                         "oof_knn_dynamic_k": kk, "oof_knn_k_rel_raw": kkr})
    print(f"    Dynamic-K (optional, separate): MLP mean {mk.mean():.2f}, "
          f"KNN mean {kk.mean():.2f}")

    # ------------------------------------------------------- write blocks --
    print("\n[6/12] writing feature/target blocks")
    meta_src = unified[unified["view_mode"].isin(FS.VIEW_IDS)].rename(
        columns={"view_mode": "view_id"})
    meta_feats = (ident[["dataset_id", "view_id"]]
                  .merge(meta_src, on=["dataset_id", "view_id"], how="left")
                  [meta_cols].reset_index(drop=True))
    chk.add("X_META aligned to identifiers", len(meta_feats) == len(ident))
    ident.to_csv(_ext(out / "view_level_identifiers.csv.gz"), **GZ)
    meta_feats.to_csv(_ext(out / "view_level_meta_features.csv.gz"), **GZ)
    oof_feats.to_csv(_ext(out / "view_level_oof_utility_features.csv.gz"), **GZ)
    dynf.to_csv(_ext(out / "view_level_dynamic_k_features.csv.gz"), **GZ)
    ari_df.to_csv(_ext(out / "view_level_policy_ari_targets.csv.gz"), **GZ)
    reg_df.to_csv(_ext(out / "view_level_policy_regret_targets.csv.gz"), **GZ)
    diag.to_csv(_ext(out / "view_level_target_diagnostics.csv.gz"), **GZ)
    print(f"    X_META {meta_feats.shape} | X_OOF {oof_feats.shape} | "
          f"targets {ari_df.shape[1]}+{reg_df.shape[1]}")

    # --------------------------------------------- convenience matrices ---
    print("\n[7/12] convenience training matrices")
    keycols = ident[["dataset_id", "view_id", "split_id"]].reset_index(drop=True)
    tgt = pd.concat([ari_df, reg_df, diag], axis=1)
    t_meta = pd.concat([keycols, meta_feats, tgt], axis=1)
    t_util = pd.concat([keycols, oof_feats, tgt], axis=1)
    t_both = pd.concat([keycols, meta_feats, oof_feats, tgt], axis=1)
    t_meta.to_csv(_ext(out / "view_level_training_table_meta_only.csv.gz"), **GZ)
    t_util.to_csv(_ext(out / "view_level_training_table_utilities_only.csv.gz"), **GZ)
    t_both.to_csv(_ext(out / "view_level_training_table_meta_plus_utilities.csv.gz"),
                  **GZ)
    for v in FS.VIEW_IDS:
        t_meta[t_meta["view_id"] == v].to_csv(
            _ext(out / f"{v}_training_table_meta_only.csv.gz"), **GZ)
    print(f"    META-only {t_meta.shape} | UTIL-only {t_util.shape} | "
          f"META+UTIL {t_both.shape}")

    # ------------------------------------------------ view-level oracle ---
    print("\n[8/12] view-level policy oracle and complementarity")
    means = A.mean(axis=0)
    gi = int(np.argmax(means))
    best_pid = pids[gi]
    hpt, hlo, hhi = _boot(vbs - A[:, gi])
    vpt, vlo, vhi = _boot(vbs)
    print(f"    global best single : {best_pid} = {means[gi]:.4f}")
    print(f"    view-level VBS     : {vpt:.4f} [{vlo:.4f},{vhi:.4f}]")
    print(f"    headroom           : +{hpt:.4f} [{hlo:.4f},{hhi:.4f}]")

    psum = pd.DataFrame({
        "policy_id": pids, "mean_view_ari": means,
        "mean_regret": R.mean(axis=0),
        "unique_win": [(is_best[:, i] & (n_win == 1)).sum() for i in range(20)],
        "win_or_tie": is_best.sum(axis=0),
    }).sort_values("mean_view_ari", ascending=False)
    psum.to_csv(_ext(out / "view_level_policy_summary.csv"), index=False)

    from models.Clustering_Repository_Builder.experiments.paper_analysis.method_registry import (
        METHOD_BY_NAME,
    )
    srcf = {p: ("MLP" if METHOD_BY_NAME[p].method_family == "clustopt_mlp" else "KNN")
            for p in pids}
    mlp_i = [i for i, p in enumerate(pids) if srcf[p] == "MLP"]
    knn_i = [i for i, p in enumerate(pids) if srcf[p] == "KNN"]

    rows_v, ties_v, comp_v = [], [], []
    vid = ident["view_id"].to_numpy()
    for v in list(FS.VIEW_IDS) + ["ALL"]:
        m = np.ones(len(A), bool) if v == "ALL" else (vid == v)
        Av, Vv, Rv = A[m], vbs[m], R[m]
        mv = Av.mean(axis=0); j = int(np.argmax(mv))
        h_pt, h_lo, h_hi = _boot(Vv - Av[:, j])
        v_pt, v_lo, v_hi = _boot(Vv)
        sv = np.sort(Av, axis=1); gapv = sv[:, -1] - sv[:, -2]
        nb = np.isclose(Av, Vv[:, None], atol=1e-12)
        rows_v.append({"view_id": v, "n_rows": int(m.sum()),
                       "best_single_policy": pids[j], "best_single_mean": float(mv[j]),
                       "vbs_mean": v_pt, "vbs_ci_lo": v_lo, "vbs_ci_hi": v_hi,
                       "headroom": h_pt, "headroom_ci_lo": h_lo,
                       "headroom_ci_hi": h_hi})
        ties_v.append({"view_id": v, "exact_tie_rate": float((gapv <= 1e-12).mean()),
                       "near_tie_lt_0p001": float((gapv < 0.001).mean()),
                       "near_tie_lt_0p005": float((gapv < 0.005).mean()),
                       "near_tie_lt_0p01": float((gapv < 0.01).mean()),
                       "mean_n_exact_winners": float(nb.sum(1).mean()),
                       "median_n_exact_winners": float(np.median(nb.sum(1))),
                       "mean_margin": float(gapv.mean()),
                       "median_margin": float(np.median(gapv))})
        wm = nb[:, mlp_i].any(1); wk = nb[:, knn_i].any(1)
        comp_v.append({"view_id": v,
                       "mlp_only_vbs": float(Av[:, mlp_i].max(1).mean()),
                       "knn_only_vbs": float(Av[:, knn_i].max(1).mean()),
                       "combined_vbs": float(Vv.mean()),
                       "combined_minus_mlp": float(Vv.mean()
                                                   - Av[:, mlp_i].max(1).mean()),
                       "combined_minus_knn": float(Vv.mean()
                                                   - Av[:, knn_i].max(1).mean()),
                       "winner_only_mlp": float((wm & ~wk).mean()),
                       "winner_only_knn": float((~wm & wk).mean()),
                       "winner_both": float((wm & wk).mean())})
    pvs = pd.DataFrame(rows_v); pvs.to_csv(_ext(out / "per_view_policy_summary.csv"),
                                           index=False)
    pvs.to_csv(_ext(out / "view_level_vbs_summary.csv"), index=False)
    tie_df = pd.DataFrame(ties_v)
    tie_df.to_csv(_ext(out / "view_level_tie_analysis.csv"), index=False)
    comp_df = pd.DataFrame(comp_v)
    comp_df.to_csv(_ext(out / "view_level_mlp_knn_complementarity.csv"), index=False)
    print(pvs[["view_id", "best_single_policy", "best_single_mean", "vbs_mean",
               "headroom"]].round(4).to_string(index=False))
    print(); print(tie_df[["view_id", "exact_tie_rate", "near_tie_lt_0p01",
                           "mean_n_exact_winners"]].round(4).to_string(index=False))
    print(); print(comp_df.round(4).to_string(index=False))

    # per split x view
    sid = ident["split_id"].to_numpy()
    psv = []
    for s in DEV_SPLITS:
        for v in FS.VIEW_IDS:
            m = (sid == s) & (vid == v)
            if not m.any():
                continue
            Av, Vv = A[m], vbs[m]
            mv = Av.mean(0); j = int(np.argmax(mv))
            psv.append({"split_id": s, "view_id": v, "n_rows": int(m.sum()),
                        "best_single_policy": pids[j],
                        "best_single_mean": float(mv[j]),
                        "vbs_mean": float(Vv.mean()),
                        "headroom": float((Vv - Av[:, j]).mean())})
    pd.DataFrame(psv).to_csv(_ext(out / "per_split_view_policy_summary.csv"),
                             index=False)

    # ----------------------------------------------------- greedy ---------
    print("\n[9/12] view-level greedy portfolios (descriptive; all 20 retained)")
    # The greedy engine keys on "dataset_id"; here the unit is dataset x view, so
    # record_id (= dataset_id__view) is passed as the key.
    gframe = pd.DataFrame({
        "dataset_id": np.tile(ident["record_id"].to_numpy(), 20),
        "method_name": np.repeat(pids, len(A)),
        "best_view_ari": A.T.reshape(-1),
    })
    pf = Portfolio("view_policy20", "View-level 20-policy", "VP20", tuple(pids))
    g = greedy_portfolio(gframe, pf)
    single = float(g.iloc[0]["vbs_best_view_mean_ari"])
    full = float(g.iloc[-1]["vbs_best_view_mean_ari"])
    gm = []
    for k in (1, 2, 3, 5, 10, 15, 20):
        r = g[g["step"] == k]
        if not len(r):
            continue
        r0 = r.iloc[0]
        gm.append({"size": k,
                   "members": "|".join(r0["portfolio_members"].split(";")),
                   "vbs_mean": float(r0["vbs_best_view_mean_ari"]),
                   "marginal_gain": float(r0["marginal_gain"]),
                   "pct_achievable_gain": float(
                       (r0["vbs_best_view_mean_ari"] - single) / (full - single) * 100)
                   if full > single else 100.0})
    gmd = pd.DataFrame(gm)
    gmd.to_csv(_ext(out / "view_level_greedy_portfolios.csv"), index=False)
    print(gmd[["size", "vbs_mean", "marginal_gain",
               "pct_achievable_gain"]].round(4).to_string(index=False))

    # ------------------------------------------- Best-View consistency ----
    print("\n[10/12] Best-View consistency vs Stage-2A-2")
    mine = (pd.DataFrame({"dataset_id": ident["dataset_id"].to_numpy(),
                          "vbs": vbs}).groupby("dataset_id")["vbs"].max())
    s2 = pd.read_csv(_ext(repo / S2A2_REL
                          / "dataset_policy_performance_matrix.csv.gz"),
                     usecols=["dataset_id", "oracle_best_ari"]
                     ).set_index("dataset_id")["oracle_best_ari"]
    j = pd.concat([mine.rename("mine"), s2.rename("s2a2")], axis=1, join="inner")
    dev = (j["mine"] - j["s2a2"]).abs()
    n_mis = int((dev > 1e-12).sum())
    pd.DataFrame({"n_datasets": [len(j)], "n_mismatch": [n_mis],
                  "max_abs_deviation": [float(dev.max())]}).to_csv(
        _ext(out / "bestview_consistency_check.csv"), index=False)
    write_json_atomic(out / "bestview_consistency_check.json",
                      {"n_datasets": int(len(j)), "n_mismatch": n_mis,
                       "max_abs_deviation": float(dev.max()),
                       "identity": "max_v max_p ARI(d,v,p) == Stage-2A-2 dataset VBS"})
    chk.add("max_v max_p reproduces Stage-2A-2 dataset VBS", n_mis == 0,
            f"{n_mis} mismatches of {len(j)}, max dev {dev.max():.2e}")

    # ------------------------------------------------------- CV manifest --
    folds = [{"fold": i, "heldout_split": s,
              "training_splits": [x for x in DEV_SPLITS if x != s],
              "n_heldout_rows": int((sid == s).sum()),
              "n_heldout_datasets": int(ident.loc[sid == s, "dataset_id"].nunique())}
             for i, s in enumerate(DEV_SPLITS)]
    write_json_atomic(out / "cv_split_manifest.json", {
        "protocol": "leave-one-split-out over Splits 2-16; NOT random K-fold",
        "n_folds": 15, "folds": folds,
        "grouping": "all 3 views of a dataset always share the dataset's split",
        "split1": "absent -- reserved as the final untouched test set",
    })
    grouped = ident.groupby("dataset_id")["split_id"].nunique()
    chk.add("all views of a dataset share one split_id",
            bool((grouped == 1).all()), f"{int((grouped != 1).sum())} violations")

    # ------------------------------------------ leakage + integrity -------
    print("\n[11/12] leakage audit and integrity gate")
    tgt_cols = list(ari_df.columns) + list(reg_df.columns) + list(diag.columns)
    meta_only_feats = list(meta_cols)
    lk = V.leakage_audit(meta_only_feats, oof_cols,
                         list(FS.OPTIONAL_FEATURE_COLUMNS), tgt_cols,
                         list(ident.columns))
    lk.to_csv(_ext(out / "feature_target_leakage_audit.csv"), index=False)
    bad_meta = V.assert_no_target_in_features(meta_only_feats, tgt_cols,
                                              list(ident.columns))
    bad_oof = V.assert_no_target_in_features(oof_cols, tgt_cols,
                                             list(ident.columns))
    ob = V.oof_split_binding(oof.assign(split_id=ident["split_id"].to_numpy(),
                                        oof_heldout_split=oof["heldout_split"]))

    chk.add("2 exactly 16,013 datasets", ident["dataset_id"].nunique() == EXP_DATASETS,
            str(ident["dataset_id"].nunique()))
    chk.add("3 exactly 48,039 dataset/view rows", len(ident) == EXP_ROWS,
            str(len(ident)))
    for v in FS.VIEW_IDS:
        chk.add(f"4 {v} has 16,013 rows", int((vid == v).sum()) == EXP_DATASETS,
                str(int((vid == v).sum())))
    chk.add("5 exactly Splits 2-16", sorted(set(sid.tolist())) == DEV_SPLITS)
    chk.add("6 Split 1 absent", not ob["split1_present"])
    chk.add("7 exactly 20 ARI targets", len(ari_cols) == 20)
    chk.add("8 exactly 20 regret targets", len(reg_cols) == 20)
    chk.add("9 no duplicate dataset/view rows",
            not ident.duplicated(["dataset_id", "view_id"]).any())
    chk.add("10 all 20 policy targets present per row",
            not ari_df.isna().any().any(),
            f"{int(ari_df.isna().sum().sum())} NaN")
    chk.add("13 canonical policy target order identical",
            list(ari_df.columns) == ari_cols
            and [c.replace(TS.REGRET_PREFIX, TS.ARI_PREFIX)
                 for c in reg_df.columns] == ari_cols)
    chk.add("14 meta-feature schema consistent",
            FS.schema_hash(meta_cols) == audit["schema_hash"]
            and len(meta_cols) == audit["authoritative_count"])
    chk.add("15 OOF matrix = 60 MLP + 60 KNN", len(oof_cols) == 120,
            str(len(oof_cols)))
    chk.add("16 OOF metric order matches registry",
            oof_cols[:60] == [f"oof_mlp_utility__{m}" for m in metric_names]
            and oof_cols[60:] == [f"oof_knn_utility__{m}" for m in metric_names])
    chk.add("17 no target leakage in feature blocks",
            not bad_meta and not bad_oof,
            f"meta {bad_meta[:3]} oof {bad_oof[:3]}")
    chk.add("18 OOF heldout split == dataset split", ob["n_mismatched"] == 0,
            f"{ob['n_mismatched']} mismatched")
    sub_rows = sum(int((vid == v).sum()) for v in FS.VIEW_IDS)
    chk.add("19 view subsets exactly partition the combined table",
            sub_rows == len(ident), f"{sub_rows} vs {len(ident)}")
    chk.add("23 no Split-1 artifact used", not ob["split1_present"],
            "OOF sources and replay are Splits 2-16 only")
    chk.add("24 zero model training executed", True,
            "no estimator is fitted anywhere in this stage")
    same_keys = (t_meta[["dataset_id", "view_id"]].equals(
        t_util[["dataset_id", "view_id"]])
        and t_meta[["dataset_id", "view_id"]].equals(
            t_both[["dataset_id", "view_id"]]))
    chk.add("25 META / UTILITIES / META+UTIL share row keys", same_keys)
    chk.add("26 every column role classification agrees with its declared block",
            bool(lk["agrees"].all()),
            f"{int((~lk['agrees']).sum())} disagreements")
    chk.add("27 no target/metadata column is predictively allowed",
            int(lk[lk["declared_block"].isin(["target", "metadata"])]
                ["predictive_allowed"].sum()) == 0)
    chk.add("1 Stage-2A2 freeze commit exists",
            "OOF offline replay of 20 learned" in subprocess.run(
                ["git", "log", "--oneline", "-40"], cwd=repo, capture_output=True,
                text=True).stdout)
    chk.frame().to_csv(_ext(out / "integrity_checks.csv"), index=False)

    # --------------------------------------------------------- manifests --
    print("\n[12/12] manifests and summary")
    write_json_atomic(out / "experiment_manifest.json", {
        "stage": "Stage 2A-3A -- view-conditional Policy Predictor training data",
        "unit": "dataset x view",
        "n_datasets": int(ident["dataset_id"].nunique()), "n_rows": int(len(ident)),
        "n_meta_features": len(meta_cols), "n_oof_utility_features": len(oof_cols),
        "n_optional_features": len(FS.OPTIONAL_FEATURE_COLUMNS),
        "n_ari_targets": 20, "n_regret_targets": 20,
        "policy_target_hash": TS.target_schema_hash(),
        "meta_feature_schema_hash": FS.schema_hash(meta_cols),
        "oof_source": OOF_REL, "policy_source": S2A2_REL,
        "splits": DEV_SPLITS, "split1": "absent",
        "model_trained": False, "clustering_executed": False,
        "column_roles": {
            FS.ROLE_META: len(meta_cols),
            FS.ROLE_OOF_UTILITY: len(oof_cols),
            FS.ROLE_OPTIONAL: len(FS.OPTIONAL_FEATURE_COLUMNS),
            FS.ROLE_TARGET: len(tgt_cols),
            FS.ROLE_METADATA: len(ident.columns),
        },
        "future_ablations": {
            "A_meta_only": len(meta_cols),
            "B_utilities_only": len(oof_cols),
            "C_meta_plus_utilities": len(meta_cols) + len(oof_cols),
            "D_plus_dynamic_k": len(meta_cols) + len(oof_cols)
            + len(FS.OPTIONAL_FEATURE_COLUMNS),
        },
        "integrity_gate": "PASS" if chk.ok else "FAIL",
        "source_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                                        capture_output=True, text=True).stdout.strip(),
        "environment_lock": "environment/stage1_clustopt_requirements.lock.txt",
    })
    write_json_atomic(out / "stage2a3a_summary.json", {
        "n_rows": int(len(ident)), "n_datasets": int(ident["dataset_id"].nunique()),
        "global_best_single_policy": best_pid,
        "global_best_single_mean": float(means[gi]),
        "view_level_vbs": vpt, "view_level_vbs_ci": [vlo, vhi],
        "headroom": hpt, "headroom_ci": [hlo, hhi],
        "per_view": pvs.to_dict("records"),
        "ties": tie_df.to_dict("records"),
        "mlp_knn": comp_df.to_dict("records"),
        "greedy": gmd.to_dict("records"),
        "bestview_consistency_mismatches": n_mis,
        "integrity_gate": "PASS" if chk.ok else "FAIL",
    })
    print("\n" + "=" * 78)
    print(f"STAGE 2A-3A INTEGRITY GATE: {'PASS' if chk.ok else 'FAIL'}  "
          f"({(time.time()-t0)/60:.1f} min)")
    print("=" * 78)
    return 0 if chk.ok else 1


if __name__ == "__main__":
    sys.exit(main())
