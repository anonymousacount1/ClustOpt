"""Stage 2A-3B0 phase 3: assemble the 15 outer-fold-clean model-selection packages.

For outer split ``s``:

  TEST  rows (split s)      -- reuse the Stage-2A-1 SINGLE-exclusion utilities and
                               the Stage-2A-2 / Stage-2A-3A single-exclusion policy
                               targets. Already valid: their utility source never
                               trained on s. Not regenerated.

  TRAIN rows (split t != s) -- use the PAIR-exclusion source {s,t}: utilities from a
                               model trained on S \\ {s,t}, and policy targets from
                               the pair-specific replay of that same source.

The 250 label-free meta-features are reused unchanged from Stage 2A-3A -- they do
not depend on cross-fitting. Only the learned utility features, the Dynamic-K values
and the policy targets differ between the nested and final datasets.

**Even a META-ONLY predictor needs the nested targets**: a policy ARI label records
which candidate a utility-driven policy selected, so a single-exclusion label for
split t may have been produced by a utility model that trained on outer test split
s. Reusing it would reintroduce the leakage this stage exists to remove.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from models.Clustering_Repository_Builder.experiments.experiment_execution.output_writer import (  # noqa: E501
    _ext, ensure_dir, path_exists, read_json, write_json_atomic,
)

from ..data import feature_schema as FS
from ..data import target_schema as TS
from .pair_utility_builder import SPLITS, VIEWS

S2A3A_REL = ("results_analysis/clustopt_policy_predictor/"
             "stage2a3a_view_level_training_data")
OUTER_SCHEMA_VERSION = "stage2a3b0.outer.v1"


def outer_dir(root: Path, s: int) -> Path:
    return root / "outer_folds" / f"outer_{s:02d}"


def outer_is_complete(root: Path, s: int) -> bool:
    man = read_json(outer_dir(root, s) / "manifest.json")
    if not man or man.get("schema_version") != OUTER_SCHEMA_VERSION:
        return False
    if int(man.get("outer_split", -1)) != s or man.get("status") != "complete":
        return False
    d = outer_dir(root, s)
    need = ("train_identifiers.csv.gz", "test_identifiers.csv.gz",
            "train_policy_ari_targets.csv.gz", "test_policy_ari_targets.csv.gz",
            "train_policy_regret_targets.csv.gz",
            "test_policy_regret_targets.csv.gz",
            "train_pair_utility_features.csv.gz",
            "test_single_oof_utility_features.csv.gz",
            "train_target_diagnostics.csv.gz", "test_target_diagnostics.csv.gz")
    return all(path_exists(d / f) for f in need)


def load_stage2a3a(repo: Path) -> Dict[str, pd.DataFrame]:
    """The frozen final-training package -- read-only, never modified here."""
    b = repo / S2A3A_REL
    return {
        "ident": pd.read_csv(_ext(b / "view_level_identifiers.csv.gz")),
        "meta": pd.read_csv(_ext(b / "view_level_meta_features.csv.gz")),
        "oof": pd.read_csv(_ext(b / "view_level_oof_utility_features.csv.gz")),
        "dyn": pd.read_csv(_ext(b / "view_level_dynamic_k_features.csv.gz")),
        "ari": pd.read_csv(_ext(b / "view_level_policy_ari_targets.csv.gz")),
        "reg": pd.read_csv(_ext(b / "view_level_policy_regret_targets.csv.gz")),
        "diag": pd.read_csv(_ext(b / "view_level_target_diagnostics.csv.gz")),
    }


def dynamic_k(M: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """The locked deterministic Dynamic Predict-K rule, applied to any utility set."""
    from models.ClustOpt.dynamic_topk_selection.heuristics.selected_dynamic_topk import (
        ALPHA, MAX_K, MIN_K, SOFT_CAP_BASE,
    )
    u = np.clip(np.nan_to_num(M, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    umax = u.max(axis=1)
    krel = (u >= ALPHA * umax[:, None]).sum(axis=1)
    krel = np.where(umax > 1e-12, krel, MIN_K)
    K = np.where(krel <= SOFT_CAP_BASE, krel,
                 np.ceil((krel + SOFT_CAP_BASE) / 2.0)).astype(int)
    return np.clip(K, MIN_K, MAX_K), krel


def build_outer_fold(repo: Path, root: Path, s: int, base: Dict[str, pd.DataFrame],
                     replay: pd.DataFrame, metric_names: Sequence[str],
                     commit: str) -> Dict[str, Any]:
    """One outer-fold package: nested-clean TRAIN, single-exclusion TEST."""
    t0 = time.time()
    d = ensure_dir(outer_dir(root, s))
    ident = base["ident"].reset_index(drop=True)
    key = ident[["dataset_id", "view_id"]]
    is_test = (ident["split_id"] == s).to_numpy()

    # ---------------- TEST side: reuse Stage-2A-1 / 2A-3A single-exclusion ----
    ti = ident[is_test].copy()
    ti["outer_split"] = s
    ti["row_split"] = s
    ti["source_kind"] = "single_exclusion"
    ti["source_exclusion_set"] = f"{{{s}}}"
    ti.to_csv(_ext(d / "test_identifiers.csv.gz"), index=False, compression="gzip")
    base["meta"][is_test].to_csv(_ext(d / "test_meta_features.csv.gz"),
                                 index=False, compression="gzip")
    base["oof"][is_test].to_csv(
        _ext(d / "test_single_oof_utility_features.csv.gz"), index=False,
        compression="gzip")
    base["dyn"][is_test].to_csv(_ext(d / "test_dynamic_k_features.csv.gz"),
                                index=False, compression="gzip")
    base["ari"][is_test].to_csv(_ext(d / "test_policy_ari_targets.csv.gz"),
                                index=False, compression="gzip")
    base["reg"][is_test].to_csv(_ext(d / "test_policy_regret_targets.csv.gz"),
                                index=False, compression="gzip")
    base["diag"][is_test].to_csv(_ext(d / "test_target_diagnostics.csv.gz"),
                                 index=False, compression="gzip")

    # ---------------- TRAIN side: pair-exclusion {s,t} for every t != s -------
    pair_key = replay["excluded_pair"].to_numpy()
    want = {f"{min(s, t):02d}_{max(s, t):02d}" for t in SPLITS if t != s}
    sel = replay[np.isin(pair_key, list(want))
                 & (replay["target_split"] != s)].copy()
    # each training row must come from the pair that excludes BOTH s and its split
    sel = sel[sel.apply(
        lambda r: {int(r["excluded_a"]), int(r["excluded_b"])}
        == {s, int(r["target_split"])}, axis=1)]
    sel = sel.sort_values(["dataset_id", "view_id"]).reset_index(drop=True)

    tr_ident = sel[["dataset_id", "view_id", "target_split", "excluded_pair"]].copy()
    tr_ident = tr_ident.rename(columns={"target_split": "row_split"})
    tr_ident["outer_split"] = s
    tr_ident["source_kind"] = "pair_exclusion"
    tr_ident["source_exclusion_set"] = tr_ident["row_split"].map(
        lambda t: f"{{{min(s, t)},{max(s, t)}}}")
    meta_idx = base["meta"].copy()
    meta_idx[["dataset_id", "view_id"]] = key
    tr_meta = (tr_ident[["dataset_id", "view_id"]]
               .merge(meta_idx, on=["dataset_id", "view_id"], how="left")
               .drop(columns=["dataset_id", "view_id"]))

    # Pair-specific utility features, canonical order. Only the ONE pair file per
    # training split is read (not all 14), and the 120 columns are assembled in a
    # single concat rather than by repeated insertion.
    from .pair_utility_builder import pair_dir as _pd
    ucols = FS.oof_feature_columns(metric_names)
    mlp_names = [f"oof_mlp_utility__{m}" for m in metric_names]
    knn_names = [f"oof_knn_utility__{m}" for m in metric_names]
    frames = []
    for t in SPLITS:
        if t == s:
            continue
        a, b = min(s, t), max(s, t)
        pdir = _pd(root, a, b)
        m_df = pd.read_csv(_ext(pdir / "mlp" / f"predictions_split_{t:02d}.csv.gz"))
        k_df = pd.read_csv(_ext(pdir / "knn" / f"predictions_split_{t:02d}.csv.gz"))
        key = m_df[["dataset_id", "view_id"]].reset_index(drop=True)
        mvals = pd.DataFrame(
            m_df[[f"pred_{m}" for m in metric_names]].to_numpy(np.float64),
            columns=mlp_names)
        k_df = key.merge(k_df, on=["dataset_id", "view_id"], how="left")
        kvals = pd.DataFrame(k_df[list(metric_names)].to_numpy(np.float64),
                             columns=knn_names)
        frames.append(pd.concat([key, mvals, kvals], axis=1))
    up = pd.concat(frames, ignore_index=True)
    tr_util = (tr_ident[["dataset_id", "view_id"]]
               .merge(up, on=["dataset_id", "view_id"], how="left"))[ucols]
    mk, mkr = dynamic_k(tr_util[[f"oof_mlp_utility__{m}"
                                 for m in metric_names]].to_numpy(float))
    kk, kkr = dynamic_k(tr_util[[f"oof_knn_utility__{m}"
                                 for m in metric_names]].to_numpy(float))
    tr_dyn = pd.DataFrame({"oof_mlp_dynamic_k": mk, "oof_mlp_k_rel_raw": mkr,
                           "oof_knn_dynamic_k": kk, "oof_knn_k_rel_raw": kkr})

    acols, rcols = TS.ari_target_columns(), TS.regret_target_columns()
    dcols = ["view_policy_vbs", "winner_set", "n_exact_winners",
             "top1_minus_top2_policy_score", "top1_minus_second_distinct_score",
             "n_within_0_001", "n_within_0_005", "n_within_0_01"]

    tr_ident.to_csv(_ext(d / "train_identifiers.csv.gz"), index=False,
                    compression="gzip")
    tr_meta.to_csv(_ext(d / "train_meta_features.csv.gz"), index=False,
                   compression="gzip")
    tr_util.to_csv(_ext(d / "train_pair_utility_features.csv.gz"), index=False,
                   compression="gzip")
    tr_dyn.to_csv(_ext(d / "train_dynamic_k_features.csv.gz"), index=False,
                  compression="gzip")
    sel[acols].to_csv(_ext(d / "train_policy_ari_targets.csv.gz"), index=False,
                      compression="gzip")
    sel[rcols].to_csv(_ext(d / "train_policy_regret_targets.csv.gz"), index=False,
                      compression="gzip")
    sel[dcols].to_csv(_ext(d / "train_target_diagnostics.csv.gz"), index=False,
                      compression="gzip")

    man = {
        "schema_version": OUTER_SCHEMA_VERSION, "outer_split": s,
        "n_test_rows": int(is_test.sum()), "n_train_rows": int(len(sel)),
        "n_test_datasets": int(ident.loc[is_test, "dataset_id"].nunique()),
        "n_train_datasets": int(sel["dataset_id"].nunique()),
        "test_source": "single_exclusion {%d}" % s,
        "train_source": "pair_exclusion {outer, row_split}",
        "n_meta_features": tr_meta.shape[1],
        "n_utility_features": tr_util.shape[1],
        "n_ari_targets": len(acols), "n_regret_targets": len(rcols),
        "policy_order_hash": TS.target_schema_hash(),
        "meta_schema_hash": FS.schema_hash(list(tr_meta.columns)),
        "status": "complete", "runtime_sec": time.time() - t0,
        "source_commit": commit,
    }
    write_json_atomic(d / "manifest.json", man)
    return man
