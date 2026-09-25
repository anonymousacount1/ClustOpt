"""Stage 2C-3: final v2 training and the frozen Split-1 online replay.

Two phases, and the separation is structural rather than a matter of discipline:

* ``--phase select`` trains the final v1 and v2 models on Splits 2-16, picks the
  development fixed policy, builds the Split-1 label-free feature matrix, runs
  inference, and writes ``pre_outcome_selection_manifest.json`` with hashes of
  every frozen artifact. It never opens a Stage-2B5 outcome file.
* ``--phase replay`` refuses to start unless that manifest exists and its hashes
  still validate. Only then does it read the frozen Stage-2B5 online matrix and
  look up the ARI of the already-chosen policies.

The replay is a lookup: no clustering, no Optuna, no CVI computation, no
Candidate-Reranker inference. Split 1 never enters training.
"""
from __future__ import annotations

import argparse
import hashlib
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
    _ext, ensure_dir, path_exists, read_json, write_csv_atomic, write_json_atomic,
)
from models.ClustOpt_Candidate_Reranker.training.final_models import (  # noqa: E402,E501
    policy_mapping as PM,
)
from models.ClustOpt_Policy_Predictor.data import feature_schema as FSCH  # noqa: E402,E501
from models.ClustOpt_Policy_Predictor.stage2c import regime_inventory as RI  # noqa: E402,E501
from models.ClustOpt_Policy_Predictor.training import feature_sets as FSET  # noqa: E402,E501
from models.ClustOpt_Policy_Predictor.training import model_registry as MR  # noqa: E402,E501
from models.ClustOpt_Policy_Predictor.training import target_transforms as TT  # noqa: E402,E501
from models.ClustOpt_Policy_Predictor.training import train_fold as TFOLD  # noqa: E402,E501

OUT_REL = ("results_analysis/clustopt_policy_predictor/"
           "stage2c_final_v2_split1_online_replay")
S2A_DATA = ("results_analysis/clustopt_policy_predictor/"
            "stage2a3a_view_level_training_data")
B5_REL = ("results_analysis/clustering_repository/"
          "candidate_reranker_split1_online_final")
KNN_CTX = ("results_analysis/clustopt_candidate_reranker/"
           "stage2b5a_final_models_and_split1_preparation/"
           "split1_knn_utility_context.csv.gz")
MLP_PRED = ("results_analysis/mlp/20260615_231715__metric_utility_mlp_"
            "split1_holdout/artifacts/all_fold_test_predictions.csv")
UNIFIED = ("results_analysis/clustering_repository/analyzed_data/"
           "unified_training_dataset/unified_training_dataset.csv")
SPLIT_CSV = ("results_analysis/clustering_repository/analyzed_data/"
             "experiment_splits/dataset_split_assignments.csv")
AC_BV = B5_REL + "/autoclust_bestview.csv.gz"
FROZEN = {"feature_set": "META_UTILITIES", "target": "CENTERED",
          "view_arch": "UNIFIED", "family": "EXTRATREES"}
GZ = {"index": False, "compression": "gzip"}
KEY = ["dataset_id", "view_id"]
VIEW_IDS = FSET.VIEW_IDS


def sha_frame(df: pd.DataFrame) -> str:
    return hashlib.sha256(
        pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest()[:16]


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(_ext(p), "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def dev_matrices(repo: Path, pol_ids: Sequence[str]
                 ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Development identifiers, META block, 120-dim utilities, and both targets."""
    ids = pd.read_csv(_ext(repo / S2A_DATA / "view_level_identifiers.csv.gz"))
    meta = pd.read_csv(_ext(repo / S2A_DATA / "view_level_meta_features.csv.gz"))
    util = pd.read_csv(_ext(repo / S2A_DATA
                            / "view_level_oof_utility_features.csv.gz"))
    orig = pd.read_csv(_ext(repo / S2A_DATA
                            / "view_level_policy_ari_targets.csv.gz"))
    orig.columns = pol_ids
    rer = pd.read_csv(_ext(repo / RI.STAGE2C_REL
                           / "reranker_aware_targets_20policy.csv.gz"))
    rer = (ids[KEY].merge(rer, on=KEY, how="left", validate="one_to_one")
           [["target__ari__%s" % p for p in pol_ids]])
    if rer.isna().any().any():
        raise AssertionError("reranker-aware targets missing for development rows")
    return ids, meta, util, (orig.to_numpy(np.float64),
                             rer.to_numpy(np.float64))


def build_X(meta: pd.DataFrame, util: pd.DataFrame, views: pd.Series
            ) -> Tuple[np.ndarray, int]:
    X, n_cont, _ = FSET.build_X(meta, util, "META_UTILITIES", views, "UNIFIED")
    return X, n_cont


def split1_features(repo: Path, meta_cols: Sequence[str],
                    metric_names: Sequence[str]) -> Tuple[pd.DataFrame,
                                                          pd.DataFrame,
                                                          pd.DataFrame]:
    """Split-1 identifiers, META block and the 120-dim utility block.

    Label-free by construction: META comes from the frozen repository
    meta-feature table and the utilities from final predictors trained on Splits
    2-16 only. No ARI, no policy outcome, no Stage-2B5 information.
    """
    assign = pd.read_csv(_ext(repo / SPLIT_CSV))
    s1 = assign[assign["split_id"] == 1][["dataset_id"]].drop_duplicates()
    knn = pd.read_csv(_ext(repo / KNN_CTX))
    knn = knn[knn["dataset_id"].isin(set(s1["dataset_id"]))]
    ids = knn[KEY].copy().sort_values(KEY).reset_index(drop=True)

    uni = pd.read_csv(_ext(repo / UNIFIED),
                      usecols=["dataset_id", "record_id"] + list(meta_cols))
    a3 = pd.read_csv(_ext(repo / S2A_DATA / "view_level_identifiers.csv.gz"),
                     usecols=["record_id", "view_id"])
    suffix = {}
    for _, r in a3.drop_duplicates("view_id").iterrows():
        suffix[r["view_id"]] = str(r["record_id"]).rsplit("__", 1)[-1]
    if set(suffix) != set(VIEW_IDS):
        raise AssertionError("could not learn the record_id view convention")
    uni["view_id"] = uni["record_id"].astype(str).str.rsplit(
        "__", n=1).str[-1].map({v: k for k, v in suffix.items()})
    meta = ids.merge(uni[KEY + list(meta_cols)], on=KEY, how="left",
                     validate="one_to_one")
    if meta[list(meta_cols)].isna().any().any():
        raise AssertionError("Split-1 META features incomplete")

    mlp = pd.read_csv(_ext(repo / MLP_PRED))
    vm = {"1d_x": "x_only", "1d_y": "y_only", "2d": "xy_2d"}
    col = "view_mode" if "view_mode" in mlp.columns else "view_dim"
    mlp["view_id"] = mlp[col].astype(str).map(vm).fillna(mlp[col].astype(str))
    pre = [c for c in mlp.columns if c.startswith("pred__")] or \
          [c for c in mlp.columns if c.startswith("pred_")]
    if len(pre) != len(metric_names):
        cand = {c.split("__", 1)[-1]: c for c in mlp.columns if "__" in c}
        pre = [cand[m] for m in metric_names if m in cand]
    if len(pre) != len(metric_names):
        raise AssertionError("Split-1 MLP utility columns not resolvable "
                             "(found %d of %d)" % (len(pre), len(metric_names)))
    mlp = mlp[KEY + pre].rename(
        columns={c: "oof_mlp_utility__%s" % m for c, m in zip(pre, metric_names)})
    knn = knn.rename(columns={"knn_utility__%s" % m: "oof_knn_utility__%s" % m
                              for m in metric_names})
    util = ids.merge(mlp, on=KEY, how="left", validate="one_to_one").merge(
        knn[KEY + ["oof_knn_utility__%s" % m for m in metric_names]],
        on=KEY, how="left", validate="one_to_one")
    cols = FSCH.oof_feature_columns(metric_names)
    if util[cols].isna().any().any():
        raise AssertionError("Split-1 utility features incomplete")
    return ids, meta[list(meta_cols)], util[cols]


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", default=".")
    p.add_argument("--phase", choices=("select", "replay"), required=True)
    args = p.parse_args(argv)
    repo = Path(args.repo_root).resolve()
    out = ensure_dir(repo / OUT_REL)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                            capture_output=True, text=True).stdout.strip()
    t0 = time.time()
    order = json.loads((repo / S2A_DATA / "policy_target_order.json")
                       .read_text(encoding="utf-8"))
    pol_ids = [q["policy_id"] for q in sorted(order["policies"],
                                              key=lambda x: x["index"])]
    mapping = pd.DataFrame(PM.build_mapping()).set_index("policy_id")

    if args.phase == "select":
        print("=" * 78)
        print("STAGE 2C-3  PHASE 1  FINAL TRAINING AND PRE-OUTCOME FREEZE")
        print("=" * 78, flush=True)
        meta_cols = FSCH.load_meta_feature_columns(repo)
        metric_names = FSCH.load_metric_names(repo)

        ids, meta, util, (Y_ORIG, Y_RER) = dev_matrices(repo, pol_ids)
        if (ids["split_id"] == 1).any():
            raise SystemExit("Split 1 present in the development population")
        Xd, n_cont = build_X(meta, util, ids["view_id"])
        print("  development: %d rows x %d features | splits %s"
              % (Xd.shape[0], Xd.shape[1],
                 sorted(ids["split_id"].unique())), flush=True)

        models = {}
        for name, Y in (("v2", Y_RER), ("v1", Y_ORIG)):
            y = TT.transform(Y, "CENTERED")
            t = time.perf_counter()
            m = MR.build_extratrees().fit(Xd, y)
            fit_sec = time.perf_counter() - t
            models[name] = m
            write_json_atomic(out / ("final_%s_model_manifest.json" % name), {
                "config_id": TFOLD.config_id(**FROZEN),
                "config_hash": TFOLD.config_hash(
                    FROZEN["feature_set"], FROZEN["target"],
                    FROZEN["view_arch"], FROZEN["family"]),
                "model_config": MR.config_of("EXTRATREES"),
                "seed": MR.GLOBAL_SEED,
                "target_semantics": ("CENTERED reranker-aware (Stage-2C1 "
                                     "single-exclusion OOF)" if name == "v2"
                                     else "CENTERED original policy ARI "
                                          "(Stage-2A semantics)"),
                "training_splits": sorted(int(s) for s in
                                          ids["split_id"].unique()),
                "n_train_rows": int(Xd.shape[0]),
                "input_dim": int(Xd.shape[1]), "output_dim": 20,
                "fit_sec": fit_sec, "split_1_used": False,
                "reused_existing_final_model": False,
                "source_commit": commit})
            print("  final %s fitted in %.0fs" % (name, fit_sec), flush=True)

        # ---- development-selected fixed policy (no Split-1 information) ----
        rer = pd.DataFrame(Y_RER, columns=pol_ids)
        rer["dataset_id"] = ids["dataset_id"].to_numpy()
        bvm = {p: rer.groupby("dataset_id")[p].max().mean() for p in pol_ids}
        best_regime, best_val = None, -np.inf
        for rid, g in mapping.groupby("reranker_id"):
            v = float(np.mean([bvm[p] for p in g.index]))
            if v > best_val:
                best_regime, best_val = rid, v
        cand = [p for p in pol_ids if mapping.loc[p, "reranker_id"] == best_regime]
        fixed_policy = min(cand, key=pol_ids.index)      # frozen policy order
        write_json_atomic(out / "development_selected_fixed_policy.json", {
            "policy_id": fixed_policy, "policy_index": pol_ids.index(fixed_policy),
            "regime_id": best_regime,
            "development_bestview_mean": float(best_val),
            "selection_population": "Splits 2-16 Stage-2C1 reranker-aware OOF "
                                    "targets; dataset-level BestView mean",
            "tie_rule": "frozen Stage-2A policy order, lowest index",
            "split_1_used": False,
            "note": "this is the deployable Fixed+R baseline; the Stage-2B5 "
                    "'best fixed on Split 1' is hindsight and is reported "
                    "separately as a diagnostic"})
        print("  development fixed policy: %s (%s) dev BestView %.7f"
              % (fixed_policy, best_regime, best_val), flush=True)

        # ---- Split-1 label-free features -----------------------------------
        s1_ids, s1_meta, s1_util = split1_features(repo, meta_cols, metric_names)
        X1, _ = build_X(s1_meta, s1_util, s1_ids["view_id"])
        if len(s1_ids) != 3165:
            raise AssertionError("Split-1 rows = %d, expected 3165" % len(s1_ids))
        fhash = sha_frame(pd.concat([s1_ids.reset_index(drop=True),
                                     s1_meta.reset_index(drop=True),
                                     s1_util.reset_index(drop=True)], axis=1))
        write_json_atomic(out / "split1_feature_manifest.json", {
            "n_rows": int(len(s1_ids)),
            "n_datasets": int(s1_ids["dataset_id"].nunique()),
            "n_views": int(s1_ids["view_id"].nunique()),
            "input_dim": int(X1.shape[1]),
            "meta_source": UNIFIED, "meta_columns": len(meta_cols),
            "mlp_utility_source": MLP_PRED,
            "knn_utility_source": KNN_CTX,
            "utility_training_splits": "2-16 only",
            "contains_ari": False, "contains_policy_outcome": False,
            "contains_stage2b5_information": False,
            "feature_hash": fhash, "source_commit": commit})

        # ---- inference and FREEZE ------------------------------------------
        sel_hashes = {}
        for name in ("v1", "v2"):
            t = time.perf_counter()
            P = np.asarray(models[name].predict(X1), dtype=np.float64)
            pred_sec = time.perf_counter() - t
            t = time.perf_counter()
            sel = TT.select_policy(P, "CENTERED")
            argmax_sec = time.perf_counter() - t
            srt = np.sort(P, axis=1)
            tie = np.isclose(srt[:, -1], srt[:, -2], atol=0.0)
            frame = pd.DataFrame({
                "dataset_id": s1_ids["dataset_id"].to_numpy(),
                "view_id": s1_ids["view_id"].to_numpy(),
                "selected_policy_index": sel,
                "selected_policy_id": [pol_ids[i] for i in sel],
                "selected_regime_id": [mapping.loc[pol_ids[i], "reranker_id"]
                                       for i in sel],
                "weighting_mode": [mapping.loc[pol_ids[i], "weighting_mode"]
                                   for i in sel],
                "raw_softmax_tie": tie,
                "score_vector_hash": [hashlib.sha256(
                    P[i].tobytes()).hexdigest()[:16] for i in range(len(P))]})
            f = out / ("split1_%s_policy_selections.csv.gz" % name)
            frame.to_csv(_ext(f), **GZ)
            sel_hashes[name] = sha_file(f)
            write_json_atomic(out / ("split1_%s_inference_runtime.json" % name), {
                "n_rows": int(len(frame)),
                "predict_sec": pred_sec, "argmax_sec": argmax_sec,
                "predict_ms_per_row": pred_sec * 1000.0 / len(frame),
                "argmax_ms_per_row": argmax_sec * 1000.0 / len(frame),
                "excludes": "outcome lookup, ARI evaluation, reporting"})
            print("  %s selections frozen: %d rows, %d ties, predict %.2fs"
                  % (name, len(frame), int(tie.sum()), pred_sec), flush=True)

        man = {
            "phase": "PRE_OUTCOME_FREEZE",
            "final_v2_model_manifest_hash": sha_file(
                out / "final_v2_model_manifest.json"),
            "final_v1_model_manifest_hash": sha_file(
                out / "final_v1_model_manifest.json"),
            "split1_feature_hash": fhash,
            "split1_v1_selections_hash": sel_hashes["v1"],
            "split1_v2_selections_hash": sel_hashes["v2"],
            "development_fixed_policy_hash": sha_file(
                out / "development_selected_fixed_policy.json"),
            "stage2b5_outcomes_read": False,
            "assertion": "no Stage-2B5 outcome file was opened during phase 1",
            "source_commit": commit, "runtime_sec": time.time() - t0}
        write_json_atomic(out / "pre_outcome_selection_manifest.json", man)
        print("\n  PRE-OUTCOME FREEZE WRITTEN. Selections are now immutable.",
              flush=True)
        print("  %s" % json.dumps({k: v for k, v in man.items()
                                   if k.endswith("hash")}), flush=True)
        return 0

    # ================= PHASE 2: replay ======================================
    print("=" * 78)
    print("STAGE 2C-3  PHASE 2  FROZEN SPLIT-1 ONLINE REPLAY (LOOKUP ONLY)")
    print("=" * 78, flush=True)
    man = read_json(out / "pre_outcome_selection_manifest.json")
    if not man:
        raise SystemExit("pre-outcome freeze manifest missing; run --phase select")
    checks = {
        "split1_v1_selections_hash": sha_file(
            out / "split1_v1_policy_selections.csv.gz"),
        "split1_v2_selections_hash": sha_file(
            out / "split1_v2_policy_selections.csv.gz"),
        "development_fixed_policy_hash": sha_file(
            out / "development_selected_fixed_policy.json"),
        "final_v2_model_manifest_hash": sha_file(
            out / "final_v2_model_manifest.json"),
        "final_v1_model_manifest_hash": sha_file(
            out / "final_v1_model_manifest.json")}
    bad = {k: (man.get(k), v) for k, v in checks.items() if man.get(k) != v}
    if bad:
        raise SystemExit("frozen selections changed since the freeze: %s" % bad)
    print("  freeze manifest validates (%d hashes)" % len(checks), flush=True)
    write_json_atomic(out / "experiment_manifest.json", {
        "stage": "2C-3",
        "kind": "FROZEN DEVELOPMENT-INFORMED ONLINE REPLAY",
        "not_a": ["untouched final held-out validation of the hybrid"],
        "why_development_informed": "Stage-2B Split-1 results informed the "
                                    "decision to pursue reranker-aware policy "
                                    "selection and the knowledge of remaining "
                                    "policy headroom",
        "what_was_frozen_first": "v2 architecture, hyperparameters and target "
                                 "semantics were frozen before its own Split-1 "
                                 "policy outcomes were inspected",
        "replay_semantics": "selector chooses p, then lookup frozen "
                            "RerankedARI(d,v,p); no clustering, Optuna, CVI or "
                            "reranker inference",
        "pre_outcome_manifest": man, "source_commit": commit})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
