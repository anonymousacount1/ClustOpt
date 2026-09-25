"""Stage 2C-4A: baselines, LOSO evaluation, headroom and runtime accounting.

Seven weighting-mode policies are evaluated on the SAME paired online data, so
every comparison is exactly paired and no arm ever ran a different search:

    A  current_tiebreak      the mode the frozen OOF-v2 Policy Predictor's
                             20-way argmax already implies for the row
    B  always_raw
    C  always_softmax
    D  loso_global_fixed     one mode, chosen on the 14 training splits only
    E  loso_per_regime_fixed one mode per regime, training splits only
    F  learned_selector      LOSO out-of-fold ExtraTrees prediction, threshold 0
    G  weighting_oracle      max(y_raw, y_softmax); DIAGNOSTIC ONLY

The headline is dataset-level BestView ARI over the three views. Primary
comparison: F vs E. Secondary: F vs A. The oracle bounds the achievable
headroom and is never presented as a result.

Runtime accounting keeps the deployment boundary explicit. The Stage-2C4P
reranker refitting cost (~4.7 h of offline infrastructure) is excluded: it is a
one-off cost of MAKING the OOF rerankers scoreable, not a per-run deployment
cost. Inclusive timers are never summed with their own components.
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
    _ext, ensure_dir, read_json, write_csv_atomic, write_json_atomic,
)
from models.Clustering_Repository_Builder.experiments.paper_analysis import stats as ST  # noqa: E402,E501
from models.ClustOpt_Candidate_Reranker.training.final_models import (  # noqa: E402,E501
    policy_mapping as PM,
)
from models.ClustOpt_Policy_Predictor.stage2c import stage2c4_common as C  # noqa: E402,E501

SCREEN_REL = "results_analysis/clustopt_policy_predictor/stage2c_v2_screening"
ARMS: Tuple[Tuple[str, str], ...] = (
    ("current_tiebreak", "A. current Stage-2C3 tie-break"),
    ("always_raw", "B. always RAW"),
    ("always_softmax", "C. always SOFTMAX"),
    ("loso_global_fixed", "D. LOSO train-selected global fixed mode"),
    ("loso_per_regime_fixed", "E. LOSO train-selected per-regime fixed mode"),
    ("learned_selector", "F. learned ExtraTrees selector"),
    ("weighting_oracle", "G. weighting-mode oracle (diagnostic)"),
)
TOL = 1e-12


# ------------------------------------------------------------------- helpers
def current_tiebreak_modes(repo: Path) -> pd.DataFrame:
    """The weighting mode the frozen OOF-v2 20-way argmax already implies.

    Read from the Stage-2C2A row-level OOF folds, i.e. the same artefact the
    frozen sample's regime column came from. This is a DEVELOPMENT fact; no
    Split-1 selection is consulted.
    """
    mapping = {r["policy_id"]: ("RAW" if r["weighting_mode"] == "RAW"
                                else "SOFTMAX") for r in PM.build_mapping()}
    frames = []
    for f in sorted((repo / SCREEN_REL).glob("rowlevel_fold_*.csv.gz")):
        frames.append(pd.read_csv(_ext(f), usecols=[
            "dataset_id", "view_id", "outer_split", "v2_selected_policy",
            "v2_selected_regime"]))
    d = pd.concat(frames, ignore_index=True)
    d["current_tiebreak"] = d["v2_selected_policy"].map(mapping)
    if d["current_tiebreak"].isna().any():
        raise AssertionError("unmapped v2-selected policy")
    d["row_id"] = d["dataset_id"].astype(str) + "|" + d["view_id"].astype(str)
    return d[["row_id", "v2_selected_policy", "v2_selected_regime",
              "current_tiebreak"]]


def realize(T: pd.DataFrame, modes: Sequence[str]) -> np.ndarray:
    return np.where(np.asarray(modes) == "RAW",
                    T["y_raw"].to_numpy(float), T["y_softmax"].to_numpy(float))


def dataset_level(rows: pd.DataFrame, arms: Sequence[str]) -> pd.DataFrame:
    """One row per (arm, dataset): BestView ARI over the three views."""
    out = []
    for a in arms:
        g = (rows.groupby(["dataset_id", "split_id", "family_id",
                           "subfamily_id"])["ari__%s" % a].max()
             .reset_index().rename(columns={"ari__%s" % a: "best_view_ari"}))
        g["method_name"] = a
        out.append(g)
    return pd.concat(out, ignore_index=True)


def compare(dl: pd.DataFrame, a: str, b: str, *, label: str,
            primary: bool = False) -> Dict[str, Any]:
    row = ST.paired_test(dl, ST.Hypothesis(
        hypothesis_id=label, family="stage2c4a_weighting_mode",
        method_a=a, method_b=b, direction="two-sided", is_primary=primary,
        rationale="paired on identical online slates"))
    va, vb, ids = ST.paired_values(dl, a, b)
    d = va - vb
    row.update({"wins": int((d > TOL).sum()), "losses": int((d < -TOL).sum()),
                "ties": int((np.abs(d) <= TOL).sum())})
    return row


def positive_splits(rows: pd.DataFrame, a: str, b: str) -> Tuple[int, int, List[float]]:
    per = []
    for s, g in rows.groupby("split_id"):
        dl = dataset_level(g, [a, b])
        va, vb, _ = ST.paired_values(dl, a, b)
        per.append(float(np.mean(va - vb)) if len(va) else float("nan"))
    ok = int(sum(1 for x in per if np.isfinite(x) and x > TOL))
    return ok, len(per), per


def breakdown(rows: pd.DataFrame, by: str) -> pd.DataFrame:
    g = rows.groupby(by)
    out = g.apply(lambda d: pd.Series({
        "n_rows": int(len(d)),
        "n_datasets": int(d["dataset_id"].nunique()),
        "raw_wins": int((d["delta_mode"] > TOL).sum()),
        "softmax_wins": int((d["delta_mode"] < -TOL).sum()),
        "ties": int((d["delta_mode"].abs() <= TOL).sum()),
        "mean_delta_mode": float(d["delta_mode"].mean()),
        "median_delta_mode": float(d["delta_mode"].median()),
        "mean_abs_delta_mode": float(d["delta_mode"].abs().mean()),
        "mean_y_raw": float(d["y_raw"].mean()),
        "mean_y_softmax": float(d["y_softmax"].mean()),
        "mean_ari_current": float(d["ari__current_tiebreak"].mean()),
        "mean_ari_per_regime_fixed": float(d["ari__loso_per_regime_fixed"].mean()),
        "mean_ari_learned": float(d["ari__learned_selector"].mean()),
        "mean_ari_oracle": float(d["ari__weighting_oracle"].mean()),
        "mae_predicted_delta": float(np.mean(np.abs(d["predicted_delta"]
                                                    - d["delta_mode"]))),
        "nontie_sign_accuracy": _sign_acc(d),
    }), include_groups=False).reset_index()
    return out


def _sign_acc(d: pd.DataFrame) -> float:
    nz = d[d["delta_mode"].abs() > TOL]
    if not len(nz):
        return float("nan")
    return float((np.sign(nz["predicted_delta"]) == np.sign(nz["delta_mode"])).mean())


def _corr(a: np.ndarray, b: np.ndarray, method: str) -> float:
    s = pd.Series(a)
    return float(s.corr(pd.Series(b), method=method)) if s.std() > 0 else float("nan")


# ---------------------------------------------------------------------- main
def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", default=".")
    args = p.parse_args(argv)
    repo = Path(args.repo_root).resolve()
    out = ensure_dir(repo / C.OUT_REL)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                            capture_output=True, text=True).stdout.strip()
    t_all = time.time()
    print("=" * 78)
    print("STAGE 2C-4A  BASELINES, LOSO EVALUATION, HEADROOM, RUNTIME")
    print("=" * 78, flush=True)

    T = pd.read_csv(_ext(out / "raw_softmax_online_targets.csv.gz"))
    P = pd.read_csv(_ext(out / "loso_row_predictions.csv.gz"))
    CT = current_tiebreak_modes(repo)
    R = T.merge(P[["row_id", "predicted_delta", "learned_mode",
                   "loso_global_fixed_mode", "loso_per_regime_fixed_mode",
                   "outer_split"]], on="row_id", how="inner")
    R = R.merge(CT, on="row_id", how="left")
    if len(R) != len(T) or R["current_tiebreak"].isna().any():
        raise SystemExit("row alignment failed (%d/%d, %d unmapped)"
                         % (len(R), len(T), int(R["current_tiebreak"].isna().sum())))
    if (R["v2_selected_regime"] != R["regime_id"]).any():
        raise SystemExit("frozen OOF-v2 regime disagrees with the sample")

    # ---- arm realisations --------------------------------------------------
    R["ari__current_tiebreak"] = realize(R, R["current_tiebreak"])
    R["ari__always_raw"] = R["y_raw"]
    R["ari__always_softmax"] = R["y_softmax"]
    R["ari__loso_global_fixed"] = realize(R, R["loso_global_fixed_mode"])
    R["ari__loso_per_regime_fixed"] = realize(R, R["loso_per_regime_fixed_mode"])
    R["ari__learned_selector"] = realize(R, R["learned_mode"])
    R["ari__weighting_oracle"] = np.maximum(R["y_raw"], R["y_softmax"])
    arms = [a for a, _ in ARMS]
    R.to_csv(_ext(out / "per_dataset_dev_results.csv.gz"), **C.GZ)

    DL = dataset_level(R, arms)
    base_rows = []
    for a, desc in ARMS:
        g = DL[DL["method_name"] == a]
        v = g["best_view_ari"].to_numpy(float)
        rv = R["ari__%s" % a].to_numpy(float)
        base_rows.append({
            "arm": a, "description": desc,
            "n_datasets": int(len(v)), "n_rows": int(len(rv)),
            "bestview_mean_ari": float(v.mean()),
            "bestview_median_ari": float(np.median(v)),
            "bestview_std_ari": float(v.std(ddof=1)),
            "view_level_mean_ari": float(rv.mean()),
            "view_level_median_ari": float(np.median(rv)),
            "share_raw": float(np.mean(_modes_of(R, a) == "RAW")),
            "diagnostic_only": a == "weighting_oracle"})
    B = pd.DataFrame(base_rows)
    write_csv_atomic(out / "development_baselines.csv", B)

    # The frozen OOF-v2 argmax turns out to pick a RAW policy on EVERY
    # development row, so arm A is not merely close to arm B -- it is the same
    # assignment. Recorded from the full 48,039-row OOF table, not just the
    # sample, because it changes how the secondary comparison must be read.
    ALL_CT = current_tiebreak_modes(repo)
    ct_fact = {
        "development_rows_scored": int(len(ALL_CT)),
        "share_raw_all_development": float((ALL_CT["current_tiebreak"]
                                            == "RAW").mean()),
        "share_raw_in_sample": float((R["current_tiebreak"] == "RAW").mean()),
        "current_tiebreak_is_always_raw": bool(
            (ALL_CT["current_tiebreak"] == "RAW").all()),
        "consequence": ("arm A (current Stage-2C3 tie-break) and arm B (always "
                        "RAW) are the identical assignment on development; the "
                        "secondary comparison is therefore also learned vs "
                        "always-RAW"),
    }
    write_json_atomic(out / "current_tiebreak_fact.json", ct_fact)
    print("  current Stage-2C3 tie-break picks RAW on %.1f%% of %d development "
          "rows" % (100 * ct_fact["share_raw_all_development"],
                    ct_fact["development_rows_scored"]), flush=True)
    print("\n  BestView mean ARI by arm:", flush=True)
    for _, r in B.iterrows():
        print("    %-24s %.6f  (view-level %.6f)%s"
              % (r["arm"], r["bestview_mean_ari"], r["view_level_mean_ari"],
                 "  [diagnostic]" if r["diagnostic_only"] else ""), flush=True)

    # ---- primary and secondary comparisons --------------------------------
    comps = []
    for a, b, label, prim in (
            ("learned_selector", "loso_per_regime_fixed",
             "PRIMARY_learned_vs_per_regime_fixed", True),
            ("learned_selector", "current_tiebreak",
             "SECONDARY_learned_vs_current_tiebreak", False),
            ("learned_selector", "loso_global_fixed",
             "learned_vs_global_fixed", False),
            ("learned_selector", "always_raw", "learned_vs_always_raw", False),
            ("learned_selector", "always_softmax",
             "learned_vs_always_softmax", False),
            ("loso_per_regime_fixed", "current_tiebreak",
             "per_regime_fixed_vs_current", False),
            ("weighting_oracle", "current_tiebreak",
             "HEADROOM_oracle_vs_current", False),
            ("weighting_oracle", "learned_selector",
             "oracle_vs_learned", False),
            ("weighting_oracle", "loso_per_regime_fixed",
             "oracle_vs_per_regime_fixed", False)):
        row = compare(DL, a, b, label=label, primary=prim)
        ok, tot, per = positive_splits(R, a, b)
        row["positive_outer_splits"] = ok
        row["n_outer_splits"] = tot
        row["per_split_mean_delta"] = json.dumps([None if not np.isfinite(x)
                                                  else round(x, 8) for x in per])
        comps.append(row)
        # view-level companion
        rowv = compare(_view_level(R), a, b, label=label + "__view_level")
        rowv["positive_outer_splits"] = None
        rowv["n_outer_splits"] = None
        rowv["per_split_mean_delta"] = None
        comps.append(rowv)
    CMP = pd.DataFrame(comps)
    write_csv_atomic(out / "development_baseline_comparisons.csv", CMP)

    def pick(label: str) -> Dict[str, Any]:
        return CMP[CMP["hypothesis_id"] == label].iloc[0].to_dict()

    pri, sec = pick("PRIMARY_learned_vs_per_regime_fixed"), \
        pick("SECONDARY_learned_vs_current_tiebreak")
    print("\n  PRIMARY  learned vs per-regime fixed : delta %+.6f "
          "CI [%+.6f, %+.6f] p=%.4g  W/L/T %d/%d/%d  splits %d/%d"
          % (pri["mean_delta"], pri["mean_delta_ci_low"],
             pri["mean_delta_ci_high"], pri["wilcoxon_p"], pri["wins"],
             pri["losses"], pri["ties"], pri["positive_outer_splits"],
             pri["n_outer_splits"]), flush=True)
    print("  SECONDARY learned vs current tie-break: delta %+.6f "
          "CI [%+.6f, %+.6f] p=%.4g  W/L/T %d/%d/%d  splits %d/%d"
          % (sec["mean_delta"], sec["mean_delta_ci_low"],
             sec["mean_delta_ci_high"], sec["wilcoxon_p"], sec["wins"],
             sec["losses"], sec["ties"], sec["positive_outer_splits"],
             sec["n_outer_splits"]), flush=True)

    # ---- headroom ----------------------------------------------------------
    def bv(a: str) -> float:
        return float(DL[DL["method_name"] == a]["best_view_ari"].mean())
    cur, ora = bv("current_tiebreak"), bv("weighting_oracle")
    head = ora - cur
    rec = lambda a: ((bv(a) - cur) / head) if abs(head) > 1e-12 else float("nan")  # noqa: E731
    HD = {
        "definition": "aggregate dataset-level BestView; ratios are computed on "
                      "the aggregate means, never averaged row-wise",
        "current_tiebreak_bestview": cur,
        "weighting_oracle_bestview": ora,
        "headroom": head,
        "always_raw_bestview": bv("always_raw"),
        "always_softmax_bestview": bv("always_softmax"),
        "global_fixed_bestview": bv("loso_global_fixed"),
        "per_regime_fixed_bestview": bv("loso_per_regime_fixed"),
        "learned_bestview": bv("learned_selector"),
        "recovery_global_fixed": rec("loso_global_fixed"),
        "recovery_per_regime_fixed": rec("loso_per_regime_fixed"),
        "recovery_learned": rec("learned_selector"),
        "oracle_is_diagnostic_only": True,
    }
    write_json_atomic(out / "weighting_mode_oracle.json", HD)
    print("  headroom current -> oracle %+.6f | recovery per-regime %.3f, "
          "learned %.3f" % (head, HD["recovery_per_regime_fixed"],
                            HD["recovery_learned"]), flush=True)

    # ---- predictability ----------------------------------------------------
    y, yp = R["delta_mode"].to_numpy(float), R["predicted_delta"].to_numpy(float)
    nz = np.abs(y) > TOL
    per_regime = breakdown(R, "regime_id")
    pr_ok = per_regime[per_regime["nontie_sign_accuracy"].notna()]
    DIAG = {
        "n_rows": int(len(R)), "n_datasets": int(R["dataset_id"].nunique()),
        "mae_delta": float(np.mean(np.abs(yp - y))),
        "rmse_delta": float(np.sqrt(np.mean((yp - y) ** 2))),
        "mae_baseline_predict_zero": float(np.mean(np.abs(y))),
        "pearson": _corr(yp, y, "pearson"), "spearman": _corr(yp, y, "spearman"),
        "n_nontied_rows": int(nz.sum()),
        "nontie_sign_accuracy": (float((np.sign(yp[nz]) == np.sign(y[nz])).mean())
                                 if nz.any() else float("nan")),
        "share_predicted_raw": float(np.mean(R["learned_mode"] == "RAW")),
        "share_exact_zero_predictions": float(np.mean(np.abs(yp) <= TOL)),
        "tie_break_mode": "RAW",
        "strongest_predictable_regime": (
            pr_ok.sort_values("nontie_sign_accuracy", ascending=False)
            .iloc[0][["regime_id", "nontie_sign_accuracy", "n_rows"]].to_dict()
            if len(pr_ok) else None),
        "weakest_predictable_regime": (
            pr_ok.sort_values("nontie_sign_accuracy")
            .iloc[0][["regime_id", "nontie_sign_accuracy", "n_rows"]].to_dict()
            if len(pr_ok) else None),
        "structural_tie_note": ("K=1 regimes are structural ties: with a single "
                                "selected CVI, raw-normalised and softmax "
                                "weights are both exactly 1.0, so RAW and "
                                "SOFTMAX are the same policy"),
        "n_rows_k1": int((R["actual_k"] == 1).sum()),
        "n_rows_k1_tied": int(((R["actual_k"] == 1)
                               & (R["delta_mode"].abs() <= TOL)).sum()),
    }
    write_json_atomic(out / "selector_diagnostics.json", DIAG)
    print("  predictability: MAE %.6f (predict-zero %.6f) | pearson %.3f "
          "spearman %.3f | non-tie sign acc %.3f on %d rows"
          % (DIAG["mae_delta"], DIAG["mae_baseline_predict_zero"],
             DIAG["pearson"], DIAG["spearman"], DIAG["nontie_sign_accuracy"],
             DIAG["n_nontied_rows"]), flush=True)

    # ---- breakdowns --------------------------------------------------------
    write_csv_atomic(out / "regime_analysis.csv", per_regime)
    write_csv_atomic(out / "view_analysis.csv", breakdown(R, "view_id"))
    write_csv_atomic(out / "family_analysis.csv", breakdown(R, "family_id"))
    sub = breakdown(R, "subfamily_id")
    sub["sample_supports_breakdown"] = sub["n_rows"] >= 6
    write_csv_atomic(out / "subfamily_analysis.csv", sub)

    # ---- target distribution ----------------------------------------------
    write_json_atomic(out / "target_distribution.json", {
        "raw_wins": int((y > TOL).sum()), "softmax_wins": int((y < -TOL).sum()),
        "ties": int((np.abs(y) <= TOL).sum()),
        "mean_delta_mode": float(y.mean()), "median_delta_mode": float(np.median(y)),
        "mean_abs_delta_mode": float(np.abs(y).mean()),
        "max_abs_delta_mode": float(np.abs(y).max()),
        "strongest_regime_effect": per_regime.sort_values(
            "mean_abs_delta_mode", ascending=False).iloc[0][
                ["regime_id", "mean_abs_delta_mode", "mean_delta_mode",
                 "n_rows"]].to_dict(),
        "by_k_mode": R.groupby("k_mode")["delta_mode"].agg(
            ["size", "mean", "median", lambda s: float(s.abs().mean())]
        ).rename(columns={"<lambda_0>": "mean_abs"}).to_dict("index"),
    })

    # ---- runtime accounting -----------------------------------------------
    ex = read_json(out / "online_pair_execution_summary.json") or {}
    F = pd.read_csv(_ext(out / "loso_fold_results.csv"))
    man = read_json(out / "final_weighting_selector_manifest.json") or {}
    search = pd.concat([T["raw_search_sec"], T["softmax_search_sec"]]).astype(float)
    cvi = pd.concat([T["raw_cvi_eval_sec"], T["softmax_cvi_eval_sec"]]).astype(float)
    build = pd.concat([T["raw_feature_build_ms"],
                       T["softmax_feature_build_ms"]]).astype(float)
    pred = pd.concat([T["raw_reranker_predict_ms"],
                      T["softmax_reranker_predict_ms"]]).astype(float)
    sel = pd.concat([T["raw_reranker_select_ms"],
                     T["softmax_reranker_select_ms"]]).astype(float)
    k = pd.concat([T["actual_k"], T["actual_k"]]).astype(float)
    RT = {
        "n_searches": int(len(search)),
        "search_sec": {"mean": float(search.mean()),
                       "median": float(search.median()),
                       "p95": float(search.quantile(0.95)),
                       "max": float(search.max()),
                       "total_hours": float(search.sum()) / 3600.0},
        "selected_cvi": {"mean_count": float(k.mean()),
                         "min_count": float(k.min()), "max_count": float(k.max()),
                         "mean_cvi_eval_sec": float(cvi.mean()),
                         "median_cvi_eval_sec": float(cvi.median()),
                         "note": "cvi_eval_sec is a profiler total measured "
                                 "INSIDE search_sec; the two are never summed"},
        "online_c2_feature_build_ms": {"mean": float(build.mean()),
                                       "median": float(build.median()),
                                       "p95": float(build.quantile(0.95))},
        "reranker_predict_ms": {"mean": float(pred.mean()),
                                "median": float(pred.median()),
                                "p95": float(pred.quantile(0.95))},
        "reranker_select_ms": {"mean": float(sel.mean()),
                               "median": float(sel.median())},
        "reranker_total_incremental_ms": float((build + pred + sel).mean()),
        "reranker_model_load_ms_mean": float(T["reranker_model_load_ms"].mean()),
        "selector_predict_ms_per_row": float(F["predict_ms_per_row"].mean()),
        "selector_fit_sec_final": man.get("fit_sec"),
        "batch_wall_clock_hours": ex.get("cumulative_wall_clock_hours"),
        "batch_cpu_hours_search": ex.get("cpu_hours_search"),
        "workers": ex.get("workers"),
        "accounted_deployment_runtime_definition": (
            "per (dataset, view): ClustOpt search_sec (which already contains "
            "clustering, the SELECTED CVIs and the objective) + online C2 "
            "feature build + reranker predict + reranker argmin + selector "
            "predict. Utility prediction is a lookup here because U_{-s} is "
            "precomputed, so no utility-prediction time is claimed."),
        "excluded_from_deployment_runtime": [
            "Stage-2C4P OOF reranker refitting (~4.7 h, one-off offline "
            "infrastructure, not a per-run cost)",
            "LOSO diagnostic fits",
            "dataset loading, view building and meta-feature extraction "
            "(outside the historical ClustOpt runtime boundary)"],
        "extra_cvis_introduced_by_the_selector": 0,
        "extra_trials_introduced_by_the_selector": 0,
    }
    RT["accounted_deployment_ms_per_view"] = (
        float(search.mean()) * 1000.0 + RT["reranker_total_incremental_ms"]
        + RT["selector_predict_ms_per_row"])
    write_json_atomic(out / "runtime_summary.json", RT)

    write_json_atomic(out / "runtime_component_schema.json", {
        "components": [
            {"name": "utility_load", "phase": "pre-search", "unit": "ms",
             "inclusive": False, "measured": False,
             "note": "U_{-s} is a precomputed OOF prediction read from disk; "
                     "no model inference happens per run"},
            {"name": "policy_predictor_predict", "phase": "pre-search",
             "unit": "ms", "inclusive": False, "measured": "stage2c3",
             "note": "the regime is already frozen upstream of this stage"},
            {"name": "selector_predict", "phase": "pre-search", "unit": "ms",
             "inclusive": False, "measured": True,
             "source": "loso_fold_results.predict_ms_per_row"},
            {"name": "search_sec", "phase": "search", "unit": "s",
             "inclusive": True,
             "contains": ["clustering fit/predict", "selected-CVI evaluation",
                          "objective aggregation", "per-trial logging"],
             "source": "timing_summary.runtime_sec"},
            {"name": "cvi_eval_sec", "phase": "search", "unit": "s",
             "inclusive": False, "contained_by": "search_sec",
             "source": "timing_summary.cvi_eval_sec"},
            {"name": "fit_predict_sec", "phase": "search", "unit": "s",
             "inclusive": False, "contained_by": "search_sec",
             "source": "timing_summary.fit_predict_sec"},
            {"name": "online_c2_feature_build", "phase": "reranker",
             "unit": "ms", "inclusive": False, "measured": True},
            {"name": "reranker_predict", "phase": "reranker", "unit": "ms",
             "inclusive": False, "measured": True},
            {"name": "reranker_select", "phase": "reranker", "unit": "ms",
             "inclusive": False, "measured": True},
            {"name": "reranker_model_load", "phase": "reranker", "unit": "ms",
             "inclusive": False, "measured": True,
             "note": "amortised once per (split, regime), not per run"},
        ],
        "sum_rule": "deployment total = search_sec + reranker components + "
                    "selector_predict; cvi_eval_sec and fit_predict_sec are "
                    "NEVER added because search_sec already contains them",
    })
    write_json_atomic(out / "runtime_boundary_audit.json", {
        "double_counted_components": [],
        "inclusive_timers": ["search_sec"],
        "components_inside_search_sec": ["cvi_eval_sec", "fit_predict_sec"],
        "checked": "cvi_eval_sec <= search_sec on every run",
        "cvi_within_search_violations": int((cvi > search + 1e-6).sum()),
        "stage2c4p_cost_included_in_deployment_runtime": False,
        "stage2c4p_cost_hours": 4.7,
        "end_to_end_runtime_available": False,
        "end_to_end_reason": ("the ClustOpt runtime boundary is search-only "
                              "(clustopt_execution starts the timer immediately "
                              "before search); dataset loading, view building "
                              "and meta-feature extraction are outside it and "
                              "were not instrumented per run"),
        "selector_adds_cvis": False, "selector_adds_trials": False,
    })

    # ---- summary -----------------------------------------------------------
    summ = {
        "stage": C.STAGE, "sample_hash": C.FROZEN_SAMPLE_HASH,
        "n_rows": int(len(R)), "n_datasets": int(R["dataset_id"].nunique()),
        "n_splits": int(R["split_id"].nunique()),
        "baselines": {r["arm"]: r["bestview_mean_ari"] for r in base_rows},
        "primary": {k: pri[k] for k in
                    ("mean_a", "mean_b", "mean_delta", "median_delta",
                     "mean_delta_ci_low", "mean_delta_ci_high", "wilcoxon_p",
                     "wins", "losses", "ties", "positive_outer_splits",
                     "n_outer_splits", "n_paired")},
        "secondary": {k: sec[k] for k in
                      ("mean_a", "mean_b", "mean_delta", "median_delta",
                       "mean_delta_ci_low", "mean_delta_ci_high", "wilcoxon_p",
                       "wins", "losses", "ties", "positive_outer_splits",
                       "n_outer_splits", "n_paired")},
        "headroom": HD, "predictability": DIAG, "runtime": RT,
        "current_tiebreak_fact": ct_fact,
        "split_1_used": False, "source_commit": commit,
        "runtime_sec": time.time() - t_all,
    }
    write_json_atomic(out / "stage2c4a_summary.json", summ)
    print("\n  analysis written in %.1fs" % (time.time() - t_all), flush=True)
    return 0


def _modes_of(R: pd.DataFrame, arm: str) -> np.ndarray:
    col = {"current_tiebreak": "current_tiebreak",
           "loso_global_fixed": "loso_global_fixed_mode",
           "loso_per_regime_fixed": "loso_per_regime_fixed_mode",
           "learned_selector": "learned_mode"}.get(arm)
    if col:
        return R[col].to_numpy()
    if arm == "always_raw":
        return np.full(len(R), "RAW", dtype=object)
    if arm == "always_softmax":
        return np.full(len(R), "SOFTMAX", dtype=object)
    return np.where(R["y_raw"].to_numpy(float) >= R["y_softmax"].to_numpy(float),
                    "RAW", "SOFTMAX")


def _view_level(R: pd.DataFrame) -> pd.DataFrame:
    """Same shape as ``dataset_level`` but one entry per (dataset, view)."""
    out = []
    for a, _ in ARMS:
        g = R[["dataset_id", "view_id", "split_id", "family_id", "subfamily_id",
               "ari__%s" % a]].copy()
        g["dataset_id"] = g["dataset_id"].astype(str) + "|" + g["view_id"]
        g = g.rename(columns={"ari__%s" % a: "best_view_ari"})
        g["method_name"] = a
        out.append(g)
    return pd.concat(out, ignore_index=True)


if __name__ == "__main__":
    raise SystemExit(main())
