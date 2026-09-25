"""Stage 2C-1: how Candidate Reranking changed the policy-selection problem.

Computes the three baselines that need NO v2 training -- Fixed Policy + Reranker,
Policy Predictor v1 + Reranker, VBS + Reranker -- and contrasts the original
Stage-2A policy landscape with the reranker-aware one. Nothing is fitted here;
v1's selections are read verbatim from its frozen Stage-2A OOF predictions.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from models.Clustering_Repository_Builder.experiments.experiment_execution.output_writer import (  # noqa: E402,E501
    _ext, ensure_dir, write_csv_atomic, write_json_atomic,
)
from models.Clustering_Repository_Builder.experiments.paper_analysis import (  # noqa: E402,E501
    stats as S,
)
from models.ClustOpt_Candidate_Reranker.training.final_models import (  # noqa: E402,E501
    policy_mapping as PM,
)
from models.ClustOpt_Policy_Predictor.stage2c import regime_inventory as RI  # noqa: E402,E501

S2A_DATA = ("results_analysis/clustopt_policy_predictor/"
            "stage2a3a_view_level_training_data")
V1_REL = ("results_analysis/clustopt_policy_predictor/stage2a3b_model_selection/"
          "outer_fold_predictions/EXTRATREES__UNIFIED__META_UTILITIES__CENTERED")
V1_CONFIG = "EXTRATREES__UNIFIED__META_UTILITIES__CENTERED"
KEY = ["dataset_id", "view_id"]


def bestview(frame: pd.DataFrame, col: str) -> pd.Series:
    return frame.groupby("dataset_id")[col].max()


def load_original(repo: Path, pol_ids: Sequence[str]) -> pd.DataFrame:
    ids = pd.read_csv(_ext(repo / S2A_DATA / "view_level_identifiers.csv.gz"))
    ari = pd.read_csv(_ext(repo / S2A_DATA
                           / "view_level_policy_ari_targets.csv.gz"))
    o = pd.concat([ids[["dataset_id", "split_id", "view_id"]], ari], axis=1)
    o = o.rename(columns={"policy_ari__%s" % p: "orig__%s" % p
                          for p in pol_ids})
    return o[KEY + ["split_id"] + ["orig__%s" % p for p in pol_ids]]


def load_v1(repo: Path) -> pd.DataFrame:
    frames = []
    for s in RI.OUTER_SPLITS:
        f = repo / V1_REL / ("fold_%02d_predictions.csv.gz" % s)
        d = pd.read_csv(_ext(f))
        frames.append(d[["dataset_id", "split_id", "view_id",
                         "selected_policy_index", "selected_ari"]])
    v1 = pd.concat(frames, ignore_index=True)
    return v1.rename(columns={"selected_ari": "v1_original_selected_ari"})


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", default=".")
    args = p.parse_args(argv)
    repo = Path(args.repo_root).resolve()
    out = ensure_dir(repo / RI.STAGE2C_REL)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                            capture_output=True, text=True).stdout.strip()
    t0 = time.time()
    print("=" * 78)
    print("STAGE 2C-1  POLICY LANDSCAPE, FIXED+R, v1+R, VBS+R")
    print("=" * 78, flush=True)

    order = json.loads((repo / S2A_DATA / "policy_target_order.json")
                       .read_text(encoding="utf-8"))
    policies = sorted(order["policies"], key=lambda x: x["index"])
    pol_ids = [p["policy_id"] for p in policies]
    mapping = pd.DataFrame(PM.build_mapping()).set_index("policy_id")
    regime_of = mapping["reranker_id"].to_dict()

    new = pd.read_csv(_ext(out / "reranker_aware_targets_20policy.csv.gz"))
    old = load_original(repo, pol_ids)
    df = old.merge(new, on=KEY + ["split_id"], how="inner",
                   validate="one_to_one")
    if len(df) != len(new):
        raise AssertionError("original/reranked row alignment failed")
    O = df[["orig__%s" % p for p in pol_ids]].to_numpy(float)
    R = df[["target__ari__%s" % p for p in pol_ids]].to_numpy(float)
    print("  aligned %d dataset/view rows x 20 policies" % len(df), flush=True)

    uniq = [i for i, p in enumerate(pol_ids)
            if mapping.loc[p, "weighting_mode"] == "RAW"]           # 10 regimes
    reg_labels = [regime_of[pol_ids[i]] for i in uniq]

    # ---- 10. landscape -----------------------------------------------------
    ow, rw = O.argmax(1), R.argmax(1)
    ow10, rw10 = O[:, uniq].argmax(1), R[:, uniq].argmax(1)
    rows = []
    for j, pid in enumerate(pol_ids):
        d = df[["dataset_id"]].copy()
        d["o"], d["r"] = O[:, j], R[:, j]
        rows.append({
            "policy_id": pid, "regime_id": regime_of[pid],
            "weighting_mode": mapping.loc[pid, "weighting_mode"],
            "original_view_mean": float(O[:, j].mean()),
            "reranked_view_mean": float(R[:, j].mean()),
            "delta_view_mean": float(R[:, j].mean() - O[:, j].mean()),
            "original_bestview_mean": float(bestview(d, "o").mean()),
            "reranked_bestview_mean": float(bestview(d, "r").mean()),
            "original_winner_count": int((ow == j).sum()),
            "reranked_winner_count": int((rw == j).sum()),
        })
    LAND = pd.DataFrame(rows)
    write_csv_atomic(out / "original_vs_reranked_policy_landscape.csv", LAND)

    corr_row = np.array([np.corrcoef(O[i], R[i])[0, 1]
                         if O[i].std() > 0 and R[i].std() > 0 else np.nan
                         for i in range(len(O))])
    spread = pd.DataFrame([
        {"landscape": "original", "n_policies": 20,
         "mean_row_std": float(O.std(1).mean()),
         "mean_row_range": float((O.max(1) - O.min(1)).mean()),
         "mean_policy_mean": float(O.mean())},
        {"landscape": "reranked", "n_policies": 20,
         "mean_row_std": float(R.std(1).mean()),
         "mean_row_range": float((R.max(1) - R.min(1)).mean()),
         "mean_policy_mean": float(R.mean())},
        {"landscape": "original", "n_policies": 10,
         "mean_row_std": float(O[:, uniq].std(1).mean()),
         "mean_row_range": float((O[:, uniq].max(1) - O[:, uniq].min(1)).mean()),
         "mean_policy_mean": float(O[:, uniq].mean())},
        {"landscape": "reranked", "n_policies": 10,
         "mean_row_std": float(R[:, uniq].std(1).mean()),
         "mean_row_range": float((R[:, uniq].max(1) - R[:, uniq].min(1)).mean()),
         "mean_policy_mean": float(R[:, uniq].mean())}])
    write_csv_atomic(out / "policy_target_spread.csv", spread)

    def tie_stats(M: np.ndarray, label: str, n: int) -> Dict[str, Any]:
        srt = np.sort(M, axis=1)
        gap = srt[:, -1] - srt[:, -2]
        rng = M.max(1) - M.min(1)
        return {"landscape": label, "n_policies": n,
                "all_tied_rate": float((rng <= 1e-12).mean()),
                "exact_top_tie_rate": float((gap <= 1e-12).mean()),
                "near_tie_0.001": float((gap < 0.001).mean()),
                "near_tie_0.005": float((gap < 0.005).mean()),
                "near_tie_0.01": float((gap < 0.01).mean()),
                "mean_top2_gap": float(gap.mean())}
    TIES = pd.DataFrame([tie_stats(O, "original", 20), tie_stats(R, "reranked", 20),
                         tie_stats(O[:, uniq], "original", 10),
                         tie_stats(R[:, uniq], "reranked", 10)])
    write_csv_atomic(out / "policy_tie_analysis.csv", TIES)

    src_o = np.array([mapping.loc[pol_ids[i], "utility_source"] for i in ow])
    src_r = np.array([mapping.loc[pol_ids[i], "utility_source"] for i in rw])
    k_o = np.array([mapping.loc[pol_ids[i], "k_mode"] for i in ow])
    k_r = np.array([mapping.loc[pol_ids[i], "k_mode"] for i in rw])
    SW = pd.DataFrame({
        "dataset_id": df["dataset_id"], "view_id": df["view_id"],
        "original_winner": [pol_ids[i] for i in ow],
        "reranked_winner": [pol_ids[i] for i in rw],
        "original_regime": [reg_labels[i] for i in ow10],
        "reranked_regime": [reg_labels[i] for i in rw10],
        "winner_changed": ow != rw, "regime_changed": ow10 != rw10,
        "source_changed": src_o != src_r, "k_changed": k_o != k_r})
    write_csv_atomic(out / "policy_winner_switches.csv", SW)
    switch = {
        "winner_switch_fraction_20policy": float((ow != rw).mean()),
        "regime_switch_fraction_10regime": float((ow10 != rw10).mean()),
        "source_switch_fraction": float((src_o != src_r).mean()),
        "k_switch_fraction": float((k_o != k_r).mean()),
        "mean_row_correlation": float(np.nanmean(corr_row)),
        "median_row_correlation": float(np.nanmedian(corr_row)),
        "global_original_winner": pol_ids[int(np.bincount(ow, minlength=20).argmax())],
        "global_reranked_winner": pol_ids[int(np.bincount(rw, minlength=20).argmax())],
        "original_source_shares": pd.Series(src_o).value_counts(normalize=True).to_dict(),
        "reranked_source_shares": pd.Series(src_r).value_counts(normalize=True).to_dict(),
        "original_k_shares": pd.Series(k_o).value_counts(normalize=True).to_dict(),
        "reranked_k_shares": pd.Series(k_r).value_counts(normalize=True).to_dict(),
    }
    print("  winner switch %.4f (20-policy) | regime switch %.4f (10-regime) | "
          "row corr %.4f" % (switch["winner_switch_fraction_20policy"],
                             switch["regime_switch_fraction_10regime"],
                             switch["mean_row_correlation"]), flush=True)

    # ---- 11. fixed policy + reranker --------------------------------------
    fx = []
    for j, pid in enumerate(pol_ids):
        d = df[["dataset_id"]].copy()
        d["r"] = R[:, j]
        bv = bestview(d, "r")
        fx.append({"policy_id": pid, "policy_index": j,
                   "regime_id": regime_of[pid],
                   "weighting_mode": mapping.loc[pid, "weighting_mode"],
                   "view_mean_ari": float(R[:, j].mean()),
                   "bestview_mean_ari": float(bv.mean()),
                   "n_datasets": int(len(bv))})
    # RAW and SOFTMAX are exact duplicates here, so the winner is a genuine tie.
    # Break it with the frozen Stage-2A policy order (lowest index), never with
    # an arbitrary sort.
    FX = pd.DataFrame(fx).sort_values(["bestview_mean_ari", "policy_index"],
                                      ascending=[False, True])
    write_csv_atomic(out / "fixed_policy_plus_reranker.csv", FX)
    best_fixed = FX.iloc[0]
    bf_idx = pol_ids.index(best_fixed["policy_id"])
    print("  best fixed + reranker: %s (%s) BestView %.7f"
          % (best_fixed["policy_id"], best_fixed["regime_id"],
             best_fixed["bestview_mean_ari"]), flush=True)

    # ---- 12. VBS + reranker ------------------------------------------------
    d = df[["dataset_id"]].copy()
    d["po"] = R.max(1)
    vbs = bestview(d, "po")
    vv, vlo, vhi = S.bootstrap_ci(vbs.to_numpy(), statistic="mean")
    d["bf"] = R[:, bf_idx]
    bf_bv = bestview(d, "bf")
    vbs_summary = {
        "reranked_policy_oracle_view_mean": float(R.max(1).mean()),
        "reranked_vbs_bestview_mean": float(vbs.mean()),
        "ci": [float(vlo), float(vhi)],
        "best_fixed_policy": str(best_fixed["policy_id"]),
        "best_fixed_bestview_mean": float(bf_bv.mean()),
        "headroom_over_best_fixed": float(vbs.mean() - bf_bv.mean()),
        "n_datasets": int(len(vbs)),
        "note": "development-only (Splits 2-16 OOF); not comparable to the "
                "Split-1 online population",
    }
    write_json_atomic(out / "reranked_vbs_summary.json", vbs_summary)
    print("  VBS + reranker %.7f [%.5f, %.5f] | headroom over fixed %+.7f"
          % (vbs.mean(), vlo, vhi, vbs_summary["headroom_over_best_fixed"]),
          flush=True)

    # ---- 13. Policy Predictor v1 + reranker --------------------------------
    v1 = load_v1(repo)
    m = df[KEY + ["split_id"]].copy()
    m["row"] = np.arange(len(df))
    v1 = v1.merge(m, on=KEY + ["split_id"], how="inner", validate="one_to_one")
    if len(v1) != len(df):
        raise AssertionError("v1 OOF coverage %d != %d rows" % (len(v1), len(df)))
    sel = v1["selected_policy_index"].to_numpy(int)
    rows_idx = v1["row"].to_numpy(int)
    v1["v1_reranked_ari"] = R[rows_idx, sel]
    v1["fixed_reranked_ari"] = R[rows_idx, bf_idx]
    v1["oracle_reranked_ari"] = R[rows_idx].max(1)
    g = v1.groupby("dataset_id")[["v1_reranked_ari", "fixed_reranked_ari",
                                  "oracle_reranked_ari"]].max()
    dv = g["v1_reranked_ari"] - g["fixed_reranked_ari"]
    pt, lo, hi = S.bootstrap_ci(dv.to_numpy(), statistic="mean")
    head = float(g["oracle_reranked_ari"].mean() - g["fixed_reranked_ari"].mean())
    v1_summary = {
        "v1_config": V1_CONFIG,
        "v1_retrained": False,
        "v1_selection_source": "frozen Stage-2A OOF predictions "
                               "(selected_policy_index), unchanged",
        "v1_plus_reranker_bestview": float(g["v1_reranked_ari"].mean()),
        "best_fixed_plus_reranker_bestview": float(g["fixed_reranked_ari"].mean()),
        "delta_v1_minus_fixed": float(pt), "ci": [float(lo), float(hi)],
        "datasets_v1_better": int((dv > 1e-12).sum()),
        "datasets_v1_worse": int((dv < -1e-12).sum()),
        "datasets_tied": int((np.abs(dv) <= 1e-12).sum()),
        "reranked_vbs_bestview": float(vbs.mean()),
        "headroom_fixed_to_vbs": head,
        "headroom_recovery_of_v1": float(pt / head) if head > 1e-12 else float("nan"),
        "v1_original_bestview_reference": float(
            v1.groupby("dataset_id")["v1_original_selected_ari"].max().mean()),
        "n_datasets": int(len(g)),
    }
    write_json_atomic(out / "v1_plus_reranker_summary.json", v1_summary)
    print("  v1 + reranker %.7f vs fixed %.7f -> delta %+.7f [%.5f, %.5f] "
          "(recovery %.4f)" % (v1_summary["v1_plus_reranker_bestview"],
                               v1_summary["best_fixed_plus_reranker_bestview"],
                               pt, lo, hi, v1_summary["headroom_recovery_of_v1"]),
          flush=True)

    # ---- 14. four-level decomposition (C absent by design) -----------------
    FOUR = pd.DataFrame([
        {"level": "A", "name": "Fixed Policy + Reranker",
         "definition": "one global policy for every dataset/view",
         "policy_source": str(best_fixed["policy_id"]),
         "bestview_mean": float(g["fixed_reranked_ari"].mean()),
         "status": "COMPUTED"},
        {"level": "B", "name": "Policy Predictor v1 + Reranker",
         "definition": "frozen Stage-2A predictor chooses the policy; the "
                       "reranker-aware outcome is used",
         "policy_source": V1_CONFIG,
         "bestview_mean": float(g["v1_reranked_ari"].mean()),
         "status": "COMPUTED"},
        {"level": "C", "name": "Policy Predictor v2 + Reranker",
         "definition": "predictor trained directly on reranker-aware targets",
         "policy_source": "not trained",
         "bestview_mean": np.nan, "status": "NOT_RUN"},
        {"level": "D", "name": "VBS + Reranker",
         "definition": "oracle policy choice on true reranker-aware outcomes",
         "policy_source": "oracle",
         "bestview_mean": float(vbs.mean()), "status": "COMPUTED"},
    ])
    write_csv_atomic(out / "four_level_decomposition_pre_v2.csv", FOUR)
    print("\n" + FOUR[["level", "name", "bestview_mean", "status"]]
          .to_string(index=False), flush=True)

    # ---- 17. frozen v2 protocol (declaration only) -------------------------
    sel_cfg = json.loads((repo / "results_analysis/clustopt_policy_predictor/"
                          "stage2a3b_model_selection/selected_configuration.json")
                         .read_text(encoding="utf-8"))
    write_json_atomic(out / "future_v2_frozen_protocol.json", {
        "status": "DECLARED_NOT_TRAINED",
        "config_id": sel_cfg["config_id"],
        "model_family": sel_cfg["model_family"],
        "view_arch": sel_cfg["view_arch"],
        "feature_set": sel_cfg["feature_set"],
        "input_dim": sel_cfg["input_dim"],
        "output_dim": sel_cfg["output_dim"],
        "model_config": sel_cfg["model_config"],
        "seed": sel_cfg["seed"],
        "selection_rule": sel_cfg["selection_rule"],
        "policy_order_hash": sel_cfg["policy_order_hash"],
        "target_old": "CENTERED original policy-selected ARI",
        "target_new": "CENTERED reranker-assisted policy ARI "
                      "(Stage-2C single-exclusion OOF)",
        "only_intended_difference": "the supervision; architecture, features, "
                                    "hyperparameters, seed, policy order and "
                                    "tie-breaking are unchanged",
        "training_population": "Splits 2-16 only; Split 1 forbidden",
    })

    write_json_atomic(out / "stage2c01_summary.json", {
        "landscape": switch, "spread": spread.to_dict("records"),
        "ties": TIES.to_dict("records"),
        "best_fixed_plus_reranker": FX.iloc[0].to_dict(),
        "v1_plus_reranker": v1_summary, "vbs_plus_reranker": vbs_summary,
        "four_level": FOUR.to_dict("records"),
        "source_commit": commit, "runtime_sec": time.time() - t0})
    print("\n  done (%.0fs)" % (time.time() - t0), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
