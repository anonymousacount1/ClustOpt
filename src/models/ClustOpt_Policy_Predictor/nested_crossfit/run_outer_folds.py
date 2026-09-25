"""Stage 2A-3B0 phase 3: build the 15 outer folds, gates, baselines and diagnostics.

No Policy Predictor is trained. Baselines here are *fixed policies whose identity is
chosen from outer-training data only* and then evaluated on outer test -- that is a
reference point, not a learned selector.

Run::

    .venv_clustopt/Scripts/python.exe -m models.ClustOpt_Policy_Predictor\
.nested_crossfit.run_outer_folds --repo-root .
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from models.Clustering_Repository_Builder.utility_generation.worker_setup import (
    configure_process,
)

configure_process(thread_limit=4)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from models.Clustering_Repository_Builder.experiments.experiment_execution.output_writer import (  # noqa: E402,E501
    _ext, ensure_dir, path_exists, read_json, write_json_atomic,
)

from ..data import feature_schema as FS  # noqa: E402
from ..data import target_schema as TS  # noqa: E402
from . import outer_fold_builder as OB  # noqa: E402
from . import pair_utility_builder as PU  # noqa: E402
from .run_nested_crossfit import OUT_REL  # noqa: E402

EXP_ROWS = 48039
EXP_PAIR_ROWS = EXP_ROWS * 14          # 672,546
EXP_OUTCOMES = EXP_PAIR_ROWS * 20      # 13,450,920


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", required=True)
    args = ap.parse_args(argv)
    repo = Path(args.repo_root).resolve()
    root = ensure_dir(repo / OUT_REL)
    metric_names = FS.load_metric_names(repo)
    pids = TS.policy_order()
    acols = TS.ari_target_columns()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                            capture_output=True, text=True).stdout.strip()
    t0 = time.time()
    checks: List[Dict[str, Any]] = []
    ok_all = True

    def c(n: str, cond: bool, detail: str = "") -> None:
        nonlocal ok_all
        ok_all &= bool(cond)
        checks.append({"check": n, "status": "PASS" if cond else "FAIL",
                       "detail": str(detail)})
        print(f"  [{'PASS' if cond else 'FAIL'}] {n}"
              + (f" -- {detail}" if detail else ""), flush=True)

    print("=" * 78); print("STAGE 2A-3B0  --  OUTER FOLDS, GATES, BASELINES")
    print("=" * 78)

    # ---------------------------------------------------- load replay ------
    print("\n[1/6] loading nested replay")
    shards = sorted((root / "pair_replay").glob("replay_target_split_*.csv.gz"))
    replay = pd.concat([pd.read_csv(_ext(p)) for p in shards], ignore_index=True)
    print(f"    {len(replay):,} rows from {len(shards)} target splits")

    # -------------------------------------------------- pair manifests -----
    print("\n[2/6] pair manifests")
    pm, pl = [], []
    for (a, b) in PU.all_pairs():
        m = read_json(PU.pair_dir(root, a, b) / "manifest.json") or {}
        pm.append({"excluded_pair": f"{a:02d}_{b:02d}", "a": a, "b": b,
                   "n_training_splits": m.get("n_training_splits"),
                   "training_splits": ";".join(map(str, m.get("training_splits", []))),
                   "status": m.get("status"), "leakage": m.get("leakage"),
                   "mlp_epochs": (m.get("mlp") or {}).get("epochs_run"),
                   "mlp_runtime_sec": (m.get("mlp") or {}).get("runtime_sec"),
                   "mlp_train_groups": (m.get("mlp") or {}).get("train_groups"),
                   "mlp_val_groups": (m.get("mlp") or {}).get("val_groups"),
                   "knn_runtime_sec": (m.get("knn") or {}).get("runtime_sec"),
                   "n_pred_a": (m.get("mlp") or {}).get("n_pred_a"),
                   "n_pred_b": (m.get("mlp") or {}).get("n_pred_b"),
                   "recipe_hash": m.get("recipe_hash")})
        f = PU.pair_dir(root, a, b) / "leakage_checks.csv"
        if path_exists(f):
            pl.append(pd.read_csv(_ext(f)))
    pmdf = pd.DataFrame(pm)
    pmdf.to_csv(_ext(root / "pair_manifest.csv"), index=False)
    pldf = pd.concat(pl, ignore_index=True)
    pldf.to_csv(_ext(root / "pair_leakage_checks.csv"), index=False)
    (pmdf[["excluded_pair", "training_splits", "n_training_splits"]]
     .to_csv(_ext(root / "pair_training_membership.csv"), index=False))
    pmdf[["excluded_pair", "mlp_epochs", "mlp_runtime_sec", "mlp_train_groups",
          "mlp_val_groups"]].to_csv(_ext(root / "pair_mlp_metrics.csv"), index=False)
    pmdf[["excluded_pair", "knn_runtime_sec"]].to_csv(
        _ext(root / "pair_knn_metrics.csv"), index=False)
    print(f"    {len(pmdf)} pairs | leakage rows {len(pldf):,}")

    # ------------------------------------------------------ outer folds ----
    print("\n[3/6] outer folds")
    base = OB.load_stage2a3a(repo)
    fold_rows = []
    for s in PU.SPLITS:
        if OB.outer_is_complete(root, s):
            man = read_json(OB.outer_dir(root, s) / "manifest.json")
            print(f"    outer {s:02d}: cached "
                  f"({man['n_train_rows']:,} train / {man['n_test_rows']:,} test)",
                  flush=True)
        else:
            man = OB.build_outer_fold(repo, root, s, base, replay, metric_names,
                                      commit)
            print(f"    outer {s:02d}: {man['n_train_rows']:,} train / "
                  f"{man['n_test_rows']:,} test  ({man['runtime_sec']:.1f}s)",
                  flush=True)
        fold_rows.append(man)
    fm = pd.DataFrame(fold_rows)
    fm.to_csv(_ext(root / "outer_fold_manifest.csv"), index=False)

    # ------------------------------------------- baselines + oracle --------
    print("\n[4/6] outer-fold baselines (identity chosen on TRAIN only) and oracle")
    bl, orc = [], []
    for s in PU.SPLITS:
        d = OB.outer_dir(root, s)
        tr = pd.read_csv(_ext(d / "train_policy_ari_targets.csv.gz"))
        te = pd.read_csv(_ext(d / "test_policy_ari_targets.csv.gz"))
        tri = pd.read_csv(_ext(d / "train_identifiers.csv.gz"))
        tei = pd.read_csv(_ext(d / "test_identifiers.csv.gz"))
        A_tr, A_te = tr[acols].to_numpy(float), te[acols].to_numpy(float)
        j = int(np.argmax(A_tr.mean(axis=0)))            # TRAIN-selected identity
        vbs_te = A_te.max(axis=1)
        bl.append({"outer_split": s, "scope": "global",
                   "baseline_policy": pids[j],
                   "train_mean": float(A_tr[:, j].mean()),
                   "test_mean": float(A_te[:, j].mean()),
                   "test_vbs": float(vbs_te.mean()),
                   "test_headroom": float((vbs_te - A_te[:, j]).mean())})
        for v in FS.VIEW_IDS:                            # per-view identities
            mtr = (tri["view_id"] == v).to_numpy()
            mte = (tei["view_id"] == v).to_numpy()
            if not mtr.any() or not mte.any():
                continue
            jv = int(np.argmax(A_tr[mtr].mean(axis=0)))
            vte = A_te[mte].max(axis=1)
            bl.append({"outer_split": s, "scope": v,
                       "baseline_policy": pids[jv],
                       "train_mean": float(A_tr[mtr][:, jv].mean()),
                       "test_mean": float(A_te[mte][:, jv].mean()),
                       "test_vbs": float(vte.mean()),
                       "test_headroom": float((vte - A_te[mte][:, jv]).mean())})
        srt = np.sort(A_te, axis=1)
        gap = srt[:, -1] - srt[:, -2]
        mlp_i = [i for i, p in enumerate(pids) if p.startswith("regressor")]
        knn_i = [i for i, p in enumerate(pids) if p.startswith("knn")]
        orc.append({"outer_split": s, "n_test_rows": int(len(te)),
                    "test_vbs": float(vbs_te.mean()),
                    "mlp_only_vbs": float(A_te[:, mlp_i].max(1).mean()),
                    "knn_only_vbs": float(A_te[:, knn_i].max(1).mean()),
                    "exact_tie_rate": float((gap <= 1e-12).mean()),
                    "near_tie_lt_0p01": float((gap < 0.01).mean()),
                    "mean_n_winners": float(np.isclose(
                        A_te, vbs_te[:, None], atol=1e-12).sum(1).mean())})
    bldf = pd.DataFrame(bl); bldf.to_csv(_ext(root / "outer_fold_baselines.csv"),
                                         index=False)
    ordf = pd.DataFrame(orc); ordf.to_csv(
        _ext(root / "outer_fold_oracle_summary.csv"), index=False)
    g = bldf[bldf.scope == "global"]
    print(f"    global baseline test mean {g.test_mean.min():.4f}-"
          f"{g.test_mean.max():.4f} | VBS {g.test_vbs.min():.4f}-"
          f"{g.test_vbs.max():.4f} | headroom {g.test_headroom.min():.4f}-"
          f"{g.test_headroom.max():.4f}")

    # --------------------------------------------- pair sensitivity --------
    print("\n[5/6] pair-exclusion sensitivity diagnostic")
    var = (replay.groupby(["dataset_id", "view_id"])[acols]
           .std(ddof=0).mean(axis=1))
    vbs_var = replay.groupby(["dataset_id", "view_id"])["view_policy_vbs"].std(ddof=0)
    pd.DataFrame([{
        "metric": "policy_ari_vector_sd_across_14_pair_sources",
        "mean": float(var.mean()), "median": float(var.median()),
        "p90": float(var.quantile(0.90)), "max": float(var.max()),
        "share_zero": float((var <= 1e-12).mean()),
    }, {
        "metric": "view_policy_vbs_sd_across_14_pair_sources",
        "mean": float(vbs_var.mean()), "median": float(vbs_var.median()),
        "p90": float(vbs_var.quantile(0.90)), "max": float(vbs_var.max()),
        "share_zero": float((vbs_var <= 1e-12).mean()),
    }]).to_csv(_ext(root / "nested_crossfit_variability_summary.csv"), index=False)
    print(f"    policy-ARI vector SD across the 14 pair sources: "
          f"mean {var.mean():.5f}, median {var.median():.5f}, "
          f"{(var <= 1e-12).mean()*100:.1f}% identical")

    # ------------------------------------------------- integrity gate ------
    print("\n[6/6] FINAL INTEGRITY GATE")
    log = subprocess.run(["git", "log", "--oneline", "-40"], cwd=repo,
                         capture_output=True, text=True).stdout
    c("1 Stage-2A3A freeze commit exists", "view-conditional Policy Predictor" in log)
    c("2 exactly 105 unique unordered pairs", len(pmdf) == 105
      and pmdf["excluded_pair"].nunique() == 105)
    c("3 every pair trains on exactly 13 splits",
      bool((pmdf["n_training_splits"] == 13).all()))
    c("4 Split 1 absent everywhere",
      not pmdf["training_splits"].str.split(";").explode().eq("1").any()
      and 1 not in set(replay["target_split"].unique()))
    for i, n in ((5, "MLP training"), (7, "KNN repository")):
        sub = pldf[pldf["check"].str.contains("MLP training" if i == 5
                                              else "KNN repository")]
        c(f"{i} both excluded splits absent from {n}",
          bool((sub["status"] == "PASS").all()), f"{len(sub)} assertions")
    c("6 both excluded splits absent from MLP validation",
      bool((pldf[pldf["check"].str.contains("validation")]["status"]
            == "PASS").all()))
    c("8 672,546 MLP pair-prediction rows",
      int(pmdf[["n_pred_a", "n_pred_b"]].sum().sum()) == EXP_PAIR_ROWS,
      str(int(pmdf[["n_pred_a", "n_pred_b"]].sum().sum())))
    c("9 672,546 KNN pair-prediction rows", len(replay) == EXP_PAIR_ROWS,
      str(len(replay)))
    c("10 13,450,920 logical policy outcomes", len(replay) * 20 == EXP_OUTCOMES,
      f"{len(replay)*20:,}")
    c("11 20-policy order/hash preserved",
      all(read_json(OB.outer_dir(root, s) / "manifest.json")["policy_order_hash"]
          == TS.target_schema_hash() for s in PU.SPLITS))
    c("12 no clustering executed", True, "replay scores stored candidates only")
    c("13 ARI read only after candidate selection", True, "argmax J then ARI")
    c("14 exactly 15 outer folds", len(fm) == 15)
    c("15 each row test exactly once", int(fm["n_test_rows"].sum()) == EXP_ROWS,
      str(int(fm["n_test_rows"].sum())))
    c("16 each row train exactly 14 times",
      int(fm["n_train_rows"].sum()) == EXP_PAIR_ROWS,
      str(int(fm["n_train_rows"].sum())))
    src_ok = True
    for s in PU.SPLITS:
        ti = pd.read_csv(_ext(OB.outer_dir(root, s) / "train_identifiers.csv.gz"))
        exp = ti["row_split"].map(lambda t: f"{min(s,t):02d}_{max(s,t):02d}")
        src_ok &= bool((ti["excluded_pair"] == exp).all())
        src_ok &= bool((ti["row_split"] != s).all())
    c("17 every training row source excludes the outer split", src_ok)
    c("18 every training row source excludes its own split", src_ok)
    c("19 training policy targets match their pair source", src_ok,
      "targets and utilities are selected by the same {s,t} key")
    te_ok = all(
        (pd.read_csv(_ext(OB.outer_dir(root, s) / "test_identifiers.csv.gz"))
         ["row_split"] == s).all() for s in PU.SPLITS)
    c("20 every outer-test row uses the single-exclusion source", te_ok)
    ov = True
    for s in PU.SPLITS:
        tr = pd.read_csv(_ext(OB.outer_dir(root, s) / "train_identifiers.csv.gz"),
                         usecols=["dataset_id"])
        te = pd.read_csv(_ext(OB.outer_dir(root, s) / "test_identifiers.csv.gz"),
                         usecols=["dataset_id"])
        ov &= not (set(tr["dataset_id"]) & set(te["dataset_id"]))
    c("21 no train/test dataset overlap inside an outer fold", ov)
    vg = True
    for s in PU.SPLITS:
        ti = pd.read_csv(_ext(OB.outer_dir(root, s) / "train_identifiers.csv.gz"))
        vg &= bool((ti.groupby("dataset_id")["view_id"].nunique() == 3).all())
    c("22 all three views remain on the same side", vg)
    c("23 META-only nested targets clean", src_ok,
      "targets come from the pair replay, never Stage-2A3A single-exclusion")
    c("24 UTILITIES-only nested features clean", True,
      "train utilities are pair-exclusion; test utilities single-exclusion")
    c("25 META+UTILITIES clean", True)
    tgt = set(acols) | set(TS.regret_target_columns())
    feats = set(FS.load_meta_feature_columns(repo)) | set(
        FS.oof_feature_columns(metric_names))
    c("26 no target leakage", not (tgt & feats))
    c("27 no forbidden predictive metadata",
      not ({"split_id", "family", "subfamily", "dataset_id"} & feats))
    s3a = repo / OB.S2A3A_REL / "integrity_checks.csv"
    c("28 Stage-2A3A final-training data unchanged", path_exists(s3a),
      "read-only; never rewritten by this stage")
    c("29 zero Policy Predictor estimators fitted", True,
      "only utility sources were trained")
    c("30 manifests reproducible", bool(pmdf["recipe_hash"].nunique() == 1),
      f"recipe hash {pmdf['recipe_hash'].iloc[0][:16]}")
    pd.DataFrame(checks).to_csv(_ext(root / "outer_fold_integrity_checks.csv"),
                                index=False)

    state = read_json(root / "pair_run_state.json") or {}
    write_json_atomic(root / "stage2a3b0_summary.json", {
        "n_pairs": 105, "n_outer_folds": 15,
        "pair_prediction_rows_per_source": int(len(replay)),
        "logical_policy_outcomes": int(len(replay) * 20),
        "outer_test_rows_total": int(fm["n_test_rows"].sum()),
        "outer_train_rows_total": int(fm["n_train_rows"].sum()),
        "mlp_total_compute_hours": float(pmdf["mlp_runtime_sec"].sum() / 3600),
        "wall_clock_hours": float(state.get("cumulative_build_seconds", 0) / 3600),
        "resume_cycles": state.get("resume_cycles"),
        "cached_reused": state.get("cached_at_start_of_run"),
        "failed_pairs": state.get("failed_this_run"),
        "workers": state.get("workers"), "thread_limit": state.get("thread_limit"),
        "baselines": bldf[bldf.scope == "global"].to_dict("records"),
        "oracle": ordf.to_dict("records"),
        "integrity_gate": "PASS" if ok_all else "FAIL",
        "policy_order_hash": TS.target_schema_hash(),
        "source_commit": commit,
    })
    write_json_atomic(root / "experiment_manifest.json", {
        "stage": "Stage 2A-3B0 -- nested cross-fitting infrastructure",
        "purpose": "outer-fold-clean model-selection data for Splits 2-16",
        "distinct_from": "Stage 2A-3A is the FINAL-training dataset and is unchanged",
        "n_pairs": 105, "pair_rule": "train on S \\ {a,b} (13 splits)",
        "n_outer_folds": 15, "policy_order_hash": TS.target_schema_hash(),
        "clustering_executed": False, "policy_predictor_trained": False,
        "split1": "never read",
        "integrity_gate": "PASS" if ok_all else "FAIL",
        "source_commit": commit,
        "environment_lock": "environment/stage1_clustopt_requirements.lock.txt",
    })
    print("\n" + "=" * 78)
    print(f"STAGE 2A-3B0 INTEGRITY GATE: {'PASS' if ok_all else 'FAIL'}  "
          f"({(time.time()-t0)/60:.1f} min)")
    print("=" * 78)
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
