"""Stage 2C-2A: frozen reranker-aware Policy Predictor screening.

APPROXIMATE DEVELOPMENT SCREENING -- not an unbiased nested evaluation, not a
final development estimate, not a held-out result.

The Stage-2A outer-fold machinery is reused verbatim: ``feature_sets.load_fold``
supplies the nested input features (train rows carry the Stage-2A3B0
pair-exclusion utilities ``S \\ {s,t}``; test rows the Stage-2A1 single-exclusion
utilities ``S \\ {s}``), and ``train_fold.run_fold`` fits the frozen
``EXTRATREES__UNIFIED__META_UTILITIES__CENTERED`` winner. The ONLY thing this
module changes is the two target arrays, which are swapped for the Stage-2C1
reranker-aware OOF outcomes.

Residual approximation, isolated deliberately to one place: the training target
for a row in split ``t`` comes from ``R_{-t}``, a reranker that may have trained
on the outer-test split ``s``. Test-side targets come from ``R_{-s}`` and are
therefore OOF-clean with respect to the reranker. No feature-side shortcut is
taken anywhere.
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
from models.ClustOpt_Policy_Predictor.training import evaluate_fold as EF  # noqa: E402,E501
from models.ClustOpt_Policy_Predictor.training import feature_sets as FSET  # noqa: E402,E501
from models.ClustOpt_Policy_Predictor.training import model_registry as MR  # noqa: E402,E501
from models.ClustOpt_Policy_Predictor.training import train_fold as TFOLD  # noqa: E402,E501

NESTED_REL = ("results_analysis/clustopt_policy_predictor/"
              "stage2a3b0_nested_crossfit")
S2A_DATA = ("results_analysis/clustopt_policy_predictor/"
            "stage2a3a_view_level_training_data")
V1_REL = ("results_analysis/clustopt_policy_predictor/stage2a3b_model_selection/"
          "outer_fold_predictions/EXTRATREES__UNIFIED__META_UTILITIES__CENTERED")
OUT_REL = "results_analysis/clustopt_policy_predictor/stage2c_v2_screening"
FROZEN = {"feature_set": "META_UTILITIES", "target": "CENTERED",
          "view_arch": "UNIFIED", "family": "EXTRATREES"}
GZ = {"index": False, "compression": "gzip"}
KEY = ["dataset_id", "view_id"]


def reranked_targets(repo: Path, pol_ids: Sequence[str]) -> pd.DataFrame:
    t = pd.read_csv(_ext(repo / RI.STAGE2C_REL
                         / "reranker_aware_targets_20policy.csv.gz"))
    return t.set_index(KEY)[["target__ari__%s" % p for p in pol_ids]]


def align(target: pd.DataFrame, ident: pd.DataFrame) -> np.ndarray:
    """Reranked targets in the exact row order of a fold's identifier frame."""
    idx = pd.MultiIndex.from_arrays([ident["dataset_id"].to_numpy(),
                                     ident["view_id"].to_numpy()])
    got = target.reindex(idx)
    if got.isna().any().any():
        raise AssertionError("reranker-aware targets missing for %d rows"
                             % int(got.isna().any(axis=1).sum()))
    return got.to_numpy(np.float64)


def bv(ds: np.ndarray, a: np.ndarray) -> pd.Series:
    return pd.DataFrame({"d": ds, "a": a}).groupby("d")["a"].max()


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", default=".")
    p.add_argument("--splits", type=int, nargs="*", default=None)
    p.add_argument("--calibration", action="store_true",
                   help="also run the approximate-protocol calibration control")
    args = p.parse_args(argv)
    repo = Path(args.repo_root).resolve()
    root = repo / NESTED_REL
    out = ensure_dir(repo / OUT_REL)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                            capture_output=True, text=True).stdout.strip()
    t_all = time.time()
    print("=" * 78)
    print("STAGE 2C-2A  FROZEN RERANKER-AWARE POLICY PREDICTOR SCREENING")
    print("=" * 78, flush=True)
    print("  APPROXIMATE DEVELOPMENT SCREENING -- not a nested estimate",
          flush=True)

    order = json.loads((repo / S2A_DATA / "policy_target_order.json")
                       .read_text(encoding="utf-8"))
    pol_ids = [q["policy_id"] for q in sorted(order["policies"],
                                              key=lambda x: x["index"])]
    mapping = pd.DataFrame(PM.build_mapping()).set_index("policy_id")
    RER = reranked_targets(repo, pol_ids)
    splits = args.splits or list(RI.OUTER_SPLITS)
    if 1 in splits:
        raise SystemExit("Split 1 is forbidden")

    cfg_hash = TFOLD.config_hash(FROZEN["feature_set"], FROZEN["target"],
                                 FROZEN["view_arch"], FROZEN["family"])
    print("  config %s hash %s"
          % (TFOLD.config_id(**{k: FROZEN[k] for k in
                                ("feature_set", "target", "view_arch",
                                 "family")}), cfg_hash), flush=True)

    fold_rows: List[Dict[str, Any]] = []
    per_rows: List[pd.DataFrame] = []
    calib: List[Dict[str, Any]] = []

    for s in splits:
        t0 = time.time()
        F = FSET.load_fold(root, s)
        tri, tei = F["train_ident"], F["test_ident"]
        A_tr = align(RER, tri)
        A_te = align(RER, tei)

        # --- v2: identical pipeline, reranker-aware targets only -------------
        Fv2 = dict(F)
        Fv2["train_ari"] = pd.DataFrame(A_tr, columns=pol_ids)
        Fv2["test_ari"] = pd.DataFrame(A_te, columns=pol_ids)
        res = TFOLD.run_fold(root, s, fold=Fv2, **FROZEN)
        pr = res["predictions"]

        # --- A: fixed policy chosen on OUTER-TRAIN reranked targets only -----
        base = EF.fold_baselines(Fv2, pol_ids)
        fixed_idx = int(base["global_best_single_index"])
        ds_te = tei["dataset_id"].to_numpy()
        fixed_bv = bv(ds_te, A_te[:, fixed_idx])

        # --- B: frozen v1 selections re-scored on reranked outcomes ----------
        v1 = pd.read_csv(_ext(repo / V1_REL / ("fold_%02d_predictions.csv.gz" % s)))
        v1 = tei[KEY].merge(v1[KEY + ["selected_policy_index"]], on=KEY,
                            how="left", validate="one_to_one")
        if v1["selected_policy_index"].isna().any():
            raise AssertionError("fold %d: v1 OOF selections incomplete" % s)
        v1_sel = v1["selected_policy_index"].to_numpy(int)
        v1_ari = A_te[np.arange(len(v1_sel)), v1_sel]
        v1_bv = bv(ds_te, v1_ari)

        # --- C / D ------------------------------------------------------------
        v2_sel = pr["selected_policy_index"].to_numpy(int)
        v2_bv = bv(ds_te, pr["selected_ari"].to_numpy(float))
        vbs_bv = bv(ds_te, A_te.max(axis=1))

        idx = fixed_bv.index
        d_fix = (v2_bv - fixed_bv).reindex(idx)
        fold_rows.append({
            "outer_split": s, "n_test_datasets": int(len(idx)),
            "n_test_rows": int(len(tei)), "n_train_rows": int(len(tri)),
            "fixed_policy": pol_ids[fixed_idx],
            "fixed_regime": mapping.loc[pol_ids[fixed_idx], "reranker_id"],
            "A_fixed_bestview": float(fixed_bv.mean()),
            "B_v1_bestview": float(v1_bv.mean()),
            "C_v2_bestview": float(v2_bv.mean()),
            "D_vbs_bestview": float(vbs_bv.mean()),
            "delta_v2_minus_fixed": float(d_fix.mean()),
            "delta_v2_minus_v1": float((v2_bv - v1_bv).reindex(idx).mean()),
            "fit_sec": float(res["fit_info"][0].get("fit_sec", np.nan)),
            "fold_sec": time.time() - t0,
            "input_dim": res["input_dim"], "output_dim": res["output_dim"],
        })
        per = pd.DataFrame({
            "dataset_id": idx, "outer_split": s,
            "A_fixed": fixed_bv.reindex(idx).to_numpy(),
            "B_v1": v1_bv.reindex(idx).to_numpy(),
            "C_v2": v2_bv.reindex(idx).to_numpy(),
            "D_vbs": vbs_bv.reindex(idx).to_numpy()})
        per_rows.append(per)

        rowlvl = pd.DataFrame({
            "dataset_id": ds_te, "view_id": tei["view_id"].to_numpy(),
            "outer_split": s,
            "family": tei["family"].to_numpy() if "family" in tei else "",
            "subfamily": tei["subfamily"].to_numpy() if "subfamily" in tei else "",
            "v2_selected_policy_index": v2_sel,
            "v2_selected_policy": [pol_ids[i] for i in v2_sel],
            "v2_selected_regime": [mapping.loc[pol_ids[i], "reranker_id"]
                                   for i in v2_sel],
            "v2_selected_ari": pr["selected_ari"].to_numpy(float),
            "v1_selected_policy_index": v1_sel,
            "fixed_ari": A_te[:, fixed_idx],
            "v1_ari": v1_ari,
            "row_oracle_ari": A_te.max(axis=1),
            "oracle_regime": [mapping.loc[pol_ids[i], "reranker_id"]
                              for i in A_te.argmax(axis=1)]})
        rowlvl.to_csv(_ext(out / ("rowlevel_fold_%02d.csv.gz" % s)), **GZ)

        print("   [fold %02d] A=%.5f (%s) B=%.5f C=%.5f D=%.5f  dC-A=%+.5f "
              "(%.0fs)" % (s, fixed_bv.mean(),
                           mapping.loc[pol_ids[fixed_idx], "reranker_id"],
                           v1_bv.mean(), v2_bv.mean(), vbs_bv.mean(),
                           d_fix.mean(), time.time() - t0), flush=True)

        # --- 16. calibration control on the OLD problem ----------------------
        if args.calibration:
            single = pd.read_csv(_ext(repo / S2A_DATA
                                      / "view_level_policy_ari_targets.csv.gz"))
            sid = pd.read_csv(_ext(repo / S2A_DATA
                                   / "view_level_identifiers.csv.gz"))
            single = pd.concat([sid[KEY], single], axis=1).set_index(KEY)
            single.columns = pol_ids
            Fap = dict(F)
            Fap["train_ari"] = pd.DataFrame(align(single, tri), columns=pol_ids)
            Fap["test_ari"] = F["test_ari"]      # unchanged, as in the shortcut
            ra = TFOLD.run_fold(root, s, fold=Fap, **FROZEN)
            calib.append({
                "outer_split": s,
                "approx_old_v1_bestview": float(
                    bv(ds_te, ra["predictions"]["selected_ari"].to_numpy(float))
                    .mean())})

    FOLDS = pd.DataFrame(fold_rows)
    write_csv_atomic(out / "outer_fold_results.csv", FOLDS)
    PER = pd.concat(per_rows, ignore_index=True)
    PER.to_csv(_ext(out / "per_dataset_results.csv.gz"), **GZ)
    if calib:
        write_csv_atomic(out / "approx_calibration_folds.csv",
                         pd.DataFrame(calib))

    write_json_atomic(out / "experiment_manifest.json", {
        "stage": "2C-2A", "kind": "APPROXIMATE_DEVELOPMENT_SCREENING",
        "not_a": ["unbiased nested evaluation", "final development estimate",
                  "held-out final result"],
        "config_id": TFOLD.config_id(**{k: FROZEN[k] for k in
                                        ("feature_set", "target", "view_arch",
                                         "family")}),
        "config_hash": cfg_hash,
        "model_config": MR.config_of(FROZEN["family"]),
        "seed": MR.GLOBAL_SEED,
        "policy_order_hash": order["schema_hash"],
        "input_features": "Stage-2A3B0 nested: train pair-exclusion S\\{s,t}, "
                          "test single-exclusion S\\{s}",
        "targets": "Stage-2C1 reranker-aware CENTERED (single-exclusion OOF)",
        "known_approximation": "training targets for split t come from R_{-t}, "
                               "which may have trained on outer-test split s",
        "split_1_used": False, "n_outer_folds": len(FOLDS),
        "source_commit": commit, "runtime_sec": time.time() - t_all})
    print("\n  %d folds in %.0fs -> %s" % (len(FOLDS), time.time() - t_all, out),
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
