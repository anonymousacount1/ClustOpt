"""Stage 2C-4A: the split-grouped LOSO diagnostic and the ONE final selector.

The learned decision is a single scalar per row: predict ``delta_mode = y_raw -
y_softmax`` from PRE-SEARCH information only, then take RAW when the prediction
is positive and SOFTMAX when it is negative, with the frozen policy-order
tie-break at exactly zero. There is no model comparison, no hyperparameter
search, no threshold tuning and no seed sweep -- one estimator with the frozen
Stage-2A configuration.

Features are the frozen Policy-Predictor representation for the row (META 250 +
UTILITIES 120 + view one-hot 3 = 373) plus the 10-dim one-hot of the regime the
frozen OOF-v2 Policy Predictor already chose, giving 383. Utilities are the
single-exclusion U_{-s} block, so a row's own split never contributed to its own
features. Every outcome-derived quantity -- ARI, candidate counts, delivered
trials, timeout, post-search CVI values, trajectory data -- is excluded by
construction and asserted against the column names.

Fifteen LOSO folds are diagnostic. The frozen artefacts are fitted once at the
end on ALL sampled development rows.
"""
from __future__ import annotations

import argparse
import json
import re
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
from models.ClustOpt_Policy_Predictor.training import feature_sets as FS  # noqa: E402,E501
from models.ClustOpt_Policy_Predictor.stage2c import stage2c4_common as C  # noqa: E402,E501

#: The single frozen estimator. Identical to the Stage-2A EXTRATREES entry.
ESTIMATOR: Dict[str, Any] = {
    "family": "EXTRATREES", "estimator": "sklearn.ensemble.ExtraTreesRegressor",
    "n_estimators": 400, "min_samples_leaf": 8, "max_features": "sqrt",
    "random_state": 42,
}
#: Frozen policy order used to break an exact zero prediction. RAW comes first in
#: the Stage-2A policy order for every regime, so a zero prediction chooses RAW.
TIE_BREAK_MODE = "RAW"
BASE_DIM = 373
REGIME_DIM = 10
TOTAL_DIM = BASE_DIM + REGIME_DIM
#: Whole-word tokens that would betray an outcome-derived feature. Matched
#: against the ``_``-separated tokens of a column name, never as substrings --
#: "ari" must not fire on "linearity", and "variance" is a legitimate meta
#: feature.
FORBIDDEN_TOKENS = ("ari", "delta", "trials", "timeout", "oracle", "regret",
                    "outcome", "reranked", "softmax", "raw")
#: Exact outcome columns of the target table; none may appear as a feature.
FORBIDDEN_EXACT = ("y_raw", "y_softmax", "delta_mode", "raw_selected_index",
                   "softmax_selected_index", "raw_visited_oracle_ari",
                   "softmax_visited_oracle_ari", "raw_delivered_trials",
                   "softmax_delivered_trials", "raw_search_sec",
                   "softmax_search_sec")


def build_estimator():
    """The frozen estimator. ``n_jobs`` is a resource knob, not a model choice:
    an ExtraTrees fit with a fixed ``random_state`` is identical at any n_jobs."""
    import os

    from sklearn.ensemble import ExtraTreesRegressor
    return ExtraTreesRegressor(
        n_estimators=ESTIMATOR["n_estimators"],
        min_samples_leaf=ESTIMATOR["min_samples_leaf"],
        max_features=ESTIMATOR["max_features"],
        random_state=ESTIMATOR["random_state"],
        n_jobs=max(1, min(4, (os.cpu_count() or 2) // 2)))


def estimator_config_hash() -> str:
    import hashlib
    return hashlib.sha256(json.dumps(ESTIMATOR, sort_keys=True,
                                     separators=(",", ":")).encode()
                          ).hexdigest()[:16]


# ------------------------------------------------------------------ features
def assert_pre_search_only(names: Sequence[str], meta_cols: Sequence[str],
                           util_cols: Sequence[str]) -> None:
    """Structural whitelist plus a whole-word blacklist.

    The whitelist is the real guarantee: the feature names must be EXACTLY the
    frozen Policy-Predictor META block, the frozen single-exclusion UTILITIES
    block, the view one-hot and the regime one-hot -- nothing else can enter.
    The blacklist is a second, independent check on the same names.
    """
    expected = (list(meta_cols) + list(util_cols)
                + ["view__x_only", "view__y_only", "view__xy_2d"]
                + ["regime__%s" % r for r in C.RERANKER_IDS])
    if list(names) != expected:
        raise AssertionError("feature names are not the frozen whitelist "
                             "(%d vs %d)" % (len(names), len(expected)))
    if not all(c.startswith(("oof_mlp_utility__", "oof_knn_utility__"))
               for c in util_cols):
        raise AssertionError("utility block is not the OOF utility schema")
    lower = [n.lower() for n in names]
    bad = [n for n, l in zip(names, lower)
           if (l in FORBIDDEN_EXACT
               or set(re.split(r"[^a-z0-9]+", l)) & set(FORBIDDEN_TOKENS))]
    bad = [n for n in bad if not n.startswith(("oof_mlp_utility__",
                                               "oof_knn_utility__"))]
    if bad:
        raise AssertionError("outcome-derived feature(s) present: %s" % bad[:5])


def build_features(repo: Path, T: pd.DataFrame
                   ) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    """383-dim PRE-SEARCH design matrix aligned to ``T``.

    The META and UTILITIES blocks come from the row's own outer fold, i.e. the
    test side of ``outer_{s}``, which is the single-exclusion U_{-s} source.
    """
    root = repo / C.NESTED_REL
    blocks: List[np.ndarray] = []
    names: List[str] = []
    keep: List[int] = []
    order: List[int] = []
    for s, g in T.groupby("split_id"):
        d = root / "outer_folds" / ("outer_%02d" % int(s))
        ident = pd.read_csv(_ext(d / "test_identifiers.csv.gz"))
        meta = pd.read_csv(_ext(d / "test_meta_features.csv.gz"))
        util = pd.read_csv(_ext(d / "test_single_oof_utility_features.csv.gz"))
        if (ident["split_id"] != int(s)).any():
            raise AssertionError("outer fold %s carries other splits" % s)
        pos = {(a, b): i for i, (a, b) in
               enumerate(zip(ident["dataset_id"], ident["view_id"]))}
        idx = [pos[(r.dataset_id, r.view_id)] for r in g.itertuples()]
        meta_cols, util_cols = list(meta.columns), list(util.columns)
        X, n_cont, cols = FS.build_X(meta.iloc[idx].reset_index(drop=True),
                                     util.iloc[idx].reset_index(drop=True),
                                     "META_UTILITIES",
                                     g["view_id"].reset_index(drop=True),
                                     "UNIFIED")
        if X.shape[1] != BASE_DIM:
            raise AssertionError("base features %d != %d" % (X.shape[1], BASE_DIM))
        names = cols
        blocks.append(X)
        order.extend(list(g.index))
    Xb = np.vstack(blocks)
    T2 = T.loc[order].reset_index(drop=True)
    oh = np.zeros((len(T2), REGIME_DIM), dtype=np.float32)
    for i, rid in enumerate(T2["regime_id"]):
        oh[i, C.RERANKER_IDS.index(rid)] = 1.0
    X = np.hstack([Xb, oh]).astype(np.float32)
    names = list(names) + ["regime__%s" % r for r in C.RERANKER_IDS]
    if X.shape[1] != TOTAL_DIM:
        raise AssertionError("feature width %d != %d" % (X.shape[1], TOTAL_DIM))
    assert_pre_search_only(names, meta_cols, util_cols)
    if not np.isfinite(X).all():
        raise AssertionError("non-finite feature values")
    return X, names, T2


# ------------------------------------------------------------------ policies
def mode_from_delta(pred: np.ndarray) -> np.ndarray:
    """>0 -> RAW, <0 -> SOFTMAX, ==0 -> frozen policy-order tie-break."""
    out = np.where(pred > 0.0, "RAW", np.where(pred < 0.0, "SOFTMAX",
                                               TIE_BREAK_MODE))
    return out.astype(object)


def fixed_mode(delta: np.ndarray) -> str:
    """Train-only choice between always-RAW and always-SOFTMAX."""
    m = float(np.mean(delta)) if len(delta) else 0.0
    return "RAW" if m > 0 else ("SOFTMAX" if m < 0 else TIE_BREAK_MODE)


def per_regime_modes(T: pd.DataFrame) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for rid in C.RERANKER_IDS:
        g = T[T["regime_id"] == rid]
        out[rid] = fixed_mode(g["delta_mode"].to_numpy(float)) if len(g) \
            else TIE_BREAK_MODE
    return out


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
    print("STAGE 2C-4A  WEIGHTING-MODE SELECTOR: LOSO + FINAL FIT")
    print("=" * 78, flush=True)
    import joblib

    C.load_sample(repo)
    tf = out / "raw_softmax_online_targets.csv.gz"
    T = pd.read_csv(_ext(tf))
    print("  %d target rows | %d datasets | %d splits"
          % (len(T), T["dataset_id"].nunique(), T["split_id"].nunique()),
          flush=True)

    X, names, T = build_features(repo, T)
    y = T["delta_mode"].to_numpy(float)
    print("  features %s (%d base + %d regime one-hot); no outcome inputs"
          % (X.shape, BASE_DIM, REGIME_DIM), flush=True)

    # ---- LOSO grouped by split -------------------------------------------
    folds: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []
    for s in sorted(T["split_id"].unique()):
        te = (T["split_id"] == s).to_numpy()
        tr = ~te
        t0 = time.perf_counter()
        m = build_estimator().fit(X[tr], y[tr])
        fit_sec = time.perf_counter() - t0
        t0 = time.perf_counter()
        pred = np.asarray(m.predict(X[te]), dtype=float)
        pred_sec = time.perf_counter() - t0
        gmode = fixed_mode(y[tr])
        rmodes = per_regime_modes(T[tr])
        sub = T[te].reset_index(drop=True)
        for i, (_, r) in enumerate(sub.iterrows()):
            rows.append({
                "outer_split": int(s), "row_id": r["row_id"],
                "dataset_id": r["dataset_id"], "view_id": r["view_id"],
                "family_id": r["family_id"], "subfamily_id": r["subfamily_id"],
                "regime_id": r["regime_id"], "k_mode": r["k_mode"],
                "utility_source": r["utility_source"], "actual_k": r["actual_k"],
                "y_raw": r["y_raw"], "y_softmax": r["y_softmax"],
                "delta_mode": r["delta_mode"],
                "predicted_delta": float(pred[i]),
                "learned_mode": mode_from_delta(np.array([pred[i]]))[0],
                "loso_global_fixed_mode": gmode,
                "loso_per_regime_fixed_mode": rmodes[r["regime_id"]]})
        folds.append({
            "outer_split": int(s), "n_train_rows": int(tr.sum()),
            "n_test_rows": int(te.sum()),
            "n_train_datasets": int(T[tr]["dataset_id"].nunique()),
            "n_test_datasets": int(sub["dataset_id"].nunique()),
            "fit_sec": fit_sec, "predict_sec": pred_sec,
            "predict_ms_per_row": pred_sec * 1000.0 / max(int(te.sum()), 1),
            "train_global_fixed_mode": gmode,
            "train_per_regime_modes": json.dumps(rmodes),
            "train_mean_delta": float(np.mean(y[tr])),
            "test_mean_delta": float(np.mean(y[te])),
            "pred_mean": float(pred.mean()), "pred_std": float(pred.std()),
            "mae": float(np.mean(np.abs(pred - y[te]))),
            "rmse": float(np.sqrt(np.mean((pred - y[te]) ** 2)))})
        print("   fold s%02d: train %5d / test %4d | fit %.1fs | MAE %.5f"
              % (s, tr.sum(), te.sum(), fit_sec, folds[-1]["mae"]), flush=True)

    F = pd.DataFrame(folds)
    P = pd.DataFrame(rows)
    write_csv_atomic(out / "loso_fold_results.csv", F)
    P.to_csv(_ext(out / "loso_row_predictions.csv.gz"), **C.GZ)

    # ---- ONE final fit on ALL sampled development rows --------------------
    t0 = time.perf_counter()
    final = build_estimator().fit(X, y)
    final_fit_sec = time.perf_counter() - t0
    mf = out / "final_weighting_selector.joblib"
    joblib.dump(final, _ext(mf), compress=3)

    gmode = fixed_mode(y)
    rmodes = per_regime_modes(T)
    counts = T.groupby("regime_id")["delta_mode"].agg(
        ["size", "mean", "median"]).to_dict("index")
    write_json_atomic(out / "development_global_mode.json", {
        "stage": C.STAGE, "mode": gmode,
        "rule": "sign of the mean delta_mode over ALL sampled development rows",
        "mean_delta_mode": float(np.mean(y)),
        "median_delta_mode": float(np.median(y)),
        "n_rows": int(len(y)), "tie_break": TIE_BREAK_MODE,
        "split_1_used": False, "source_commit": commit})
    write_json_atomic(out / "development_per_regime_modes.json", {
        "stage": C.STAGE, "modes": rmodes,
        "rule": "sign of the mean delta_mode within the regime, all sampled "
                "development rows",
        "support": {k: {"n_rows": int(v["size"]),
                        "mean_delta_mode": float(v["mean"]),
                        "median_delta_mode": float(v["median"])}
                    for k, v in counts.items()},
        "tie_break": TIE_BREAK_MODE, "split_1_used": False,
        "source_commit": commit})

    fhash = C.sha_frame(pd.DataFrame(X, columns=names))
    man = {
        "stage": C.STAGE, "artifact": "final_weighting_selector.joblib",
        "estimator": ESTIMATOR,
        "estimator_config_hash": estimator_config_hash(),
        "model_search_performed": False, "threshold_tuned": False,
        "seed_sweep_performed": False, "classifier_alternative": False,
        "decision_rule": "predicted_delta > 0 -> RAW; < 0 -> SOFTMAX; == 0 -> "
                         "frozen policy order (%s)" % TIE_BREAK_MODE,
        "threshold": 0.0,
        "target": "delta_mode = y_raw - y_softmax (continuous)",
        "feature_schema": {
            "n_features": TOTAL_DIM, "base": BASE_DIM,
            "base_source": "Policy Predictor META_UTILITIES + UNIFIED view "
                           "one-hot (feature_sets.build_X)",
            "regime_one_hot": REGIME_DIM,
            "regime_order": list(C.RERANKER_IDS),
            "utility_block": "Stage-2A1 single-exclusion U_{-s}",
            "pre_search_only": True,
            "forbidden_inputs_absent": list(FORBIDDEN_TOKENS)},
        "feature_schema_hash": fhash,
        "feature_names_hash": C.sha_frame(pd.DataFrame({"n": names})),
        "target_hash": C.sha_frame(pd.DataFrame({"y": y})),
        "sample_hash": C.FROZEN_SAMPLE_HASH,
        "target_file_hash": C.sha_file(tf),
        "model_file_sha256_16": C.sha_file(mf),
        "model_bytes": int(Path(_ext(mf)).stat().st_size),
        "n_train_rows": int(len(y)),
        "n_train_datasets": int(T["dataset_id"].nunique()),
        "training_splits": sorted(int(s) for s in T["split_id"].unique()),
        "fit_sec": final_fit_sec,
        "loso_folds": int(len(F)), "loso_grouping": "outer split",
        "split_1_used": False,
        "stage2c4p_commit": C.STAGE2C4P_COMMIT,
        "source_commit": commit, "runtime_sec": time.time() - t_all,
    }
    write_json_atomic(out / "final_weighting_selector_manifest.json", man)
    print("\n  final selector fitted on %d rows in %.1fs -> %s"
          % (len(y), final_fit_sec, man["model_file_sha256_16"]), flush=True)
    print("  development global mode %s | per-regime %s"
          % (gmode, json.dumps(rmodes)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
