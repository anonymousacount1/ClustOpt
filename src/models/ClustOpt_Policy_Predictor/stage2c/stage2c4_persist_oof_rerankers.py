"""Stage 2C-4P: persist the single-exclusion Candidate Rerankers R_{-s,r}.

Infrastructure only. The historical Stage-2B/2C units stored predictions on the
fixed-32 slate but not the estimator, so they cannot score a NEW online slate.
This module refits each of the 15 x 10 units through the *same* frozen path,
writes ``model.joblib`` beside the existing unit, and then validates the
reconstruction by reproducing that unit's historical selections exactly.

Nothing scientific changes: same population, features, target, estimator, seed
and eligibility. The only difference is that the fitted estimator is kept.
Historical files are never modified -- only new files are added.
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
    _ext, ensure_dir, read_json, write_csv_atomic, write_json_atomic,
)
from models.ClustOpt_Candidate_Reranker.data import candidate_eligibility as CE  # noqa: E402,E501
from models.ClustOpt_Candidate_Reranker.training import evaluate_fold as EV  # noqa: E402,E501
from models.ClustOpt_Candidate_Reranker.training import feature_builder as FB  # noqa: E402,E501
from models.ClustOpt_Candidate_Reranker.training import model_registry as MR  # noqa: E402,E501
from models.ClustOpt_Candidate_Reranker.training import train_fold as TF  # noqa: E402,E501
from models.ClustOpt_Candidate_Reranker.training.context_ablation.run_context_ablation import (  # noqa: E402,E501
    WorkerLock,
)
from models.ClustOpt_Candidate_Reranker.training.run_masking_feasibility import (  # noqa: E402,E501
    CandidateStore, resolve_masks, _memory_probes,
)
from models.ClustOpt_Candidate_Reranker.training.same_context_controls.run_same_context_controls import (  # noqa: E402,E501
    _fold_arrays,
)
from models.ClustOpt_Candidate_Reranker.training.unified_masking import (  # noqa: E402,E501
    unified_feature_builder as UFB,
)
from models.ClustOpt_Policy_Predictor.stage2c import regime_inventory as RI  # noqa: E402,E501

GLOBAL_REL = ("results_analysis/clustopt_policy_predictor/"
              "stage2c4_online_weighting_mode_dev")
LOCK_NAME = "stage2c4p_worker.pid"
MODEL_NAME = "model.joblib"
MODEL_MANIFEST = "model_manifest.json"


def regime_parts(cid: str) -> Dict[str, str]:
    src, km = cid.split("_", 1)
    return {"source": src.lower(), "k_mode": km.lower()}


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(_ext(p), "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def unit_persisted(d: Path, s: int, cid: str) -> bool:
    m = read_json(d / MODEL_MANIFEST)
    return bool(m and m.get("status") == "complete"
                and m.get("excluded_split") == s
                and m.get("regime_id") == cid
                and m.get("reproduction_passed")
                and m.get("model_config_hash") == MR.config_hash()
                and m.get("n_features") == UFB.EXPECTED_DIM
                and Path(_ext(d / MODEL_NAME)).exists())


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", default=".")
    p.add_argument("--splits", type=int, nargs="*", default=None)
    p.add_argument("--regimes", nargs="*", default=None)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args(argv)
    repo = Path(args.repo_root).resolve()
    glob_out = ensure_dir(repo / GLOBAL_REL)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                            capture_output=True, text=True).stdout.strip()
    import joblib

    print("=" * 78)
    print("STAGE 2C-4P  PERSIST OOF CANDIDATE RERANKERS R_{-s,r}")
    print("=" * 78, flush=True)
    print("  model %s | context %s (%d dims) | target T2_SLATE_REGRET | "
          "eligibility %s" % (MR.config_hash(), UFB.CONDITION,
                              UFB.EXPECTED_DIM, CE.definition_hash()),
          flush=True)

    inv = RI.build(repo)
    if int((inv["status"] != "EXISTS").sum()):
        raise SystemExit("historical OOF units incomplete; cannot reproduce")
    splits = args.splits or list(RI.OUTER_SPLITS)
    regimes = args.regimes or list(RI.RERANKER_IDS)
    if 1 in splits:
        raise SystemExit("Split 1 is forbidden")

    todo = inv[inv["split_id"].isin(splits) & inv["regime_id"].isin(regimes)]
    todo = todo.sort_values(["split_id", "regime_id"])
    if args.limit:
        todo = todo.head(args.limit)
    print("  %d candidate units (150 expected for the full matrix)" % len(todo),
          flush=True)

    rows: List[Dict[str, Any]] = []
    checks: List[Dict[str, Any]] = []
    with WorkerLock(glob_out / LOCK_NAME):
        store = CandidateStore(repo)
        mem, _sys = _memory_probes()
        t_all = time.time()
        n_fit = n_skip = n_fail = 0
        for s, grp in todo.groupby("split_id"):
            s = int(s)
            need = [r for _, r in grp.iterrows()
                    if not unit_persisted(Path(r["unit_dir"]), s, r["regime_id"])]
            n_skip += len(grp) - len(need)
            if not need:
                continue
            A = _fold_arrays(repo, store, s)
            fu = A["fu"]
            y_tr = TF.target_regret(store.ari[A["tr_rows"]],
                                    store.oracle_row[A["tr_rows"]])
            te_starts = np.cumsum(np.r_[0, store.slate_size[A["te_slates"]]])[:-1]
            te_sizes = store.slate_size[A["te_slates"]]
            elig_te = store.eligible[A["te_rows"]]
            ari_te = store.ari[A["te_rows"]]

            for r in need:
                cid = r["regime_id"]
                d = Path(r["unit_dir"])
                rp = regime_parts(cid)
                sx, km = rp["source"], rp["k_mode"]
                t0 = time.time()
                n_tr, n_te = len(A["tr_slates"]), len(A["te_slates"])
                mtr = resolve_masks(fu["train_util"][sx][A["tr_order"]],
                                    store.metric_names, km)
                mte = resolve_masks(fu["test_util"][sx][A["te_order"]],
                                    store.metric_names, km)
                sc_tr = np.full(n_tr, FB.SOURCES.index(sx), np.int64)
                kc_tr = np.full(n_tr, FB.K_MODES.index(km), np.int64)
                sc_te = np.full(n_te, FB.SOURCES.index(sx), np.int64)
                kc_te = np.full(n_te, FB.K_MODES.index(km), np.int64)
                X_tr, n_common = UFB.build_unified_X(
                    view_codes=store.view_code[A["tr_rows"]],
                    cat_rows=store.cat_row[A["tr_rows"]],
                    cat_matrix=store.cat_matrix,
                    partition=store.partition[A["tr_rows"]],
                    cvi=store.cvi[A["tr_rows"]],
                    cvi_valid=store.cvi_valid[A["tr_rows"]],
                    observability=mtr["mask"], actual_k=mtr["actual_k"],
                    source_code=sc_tr, kmode_code=kc_tr,
                    slate_pos=A["tr_slate_pos"], util=A["util_tr"])
                X_te, _ = UFB.build_unified_X(
                    view_codes=store.view_code[A["te_rows"]],
                    cat_rows=store.cat_row[A["te_rows"]],
                    cat_matrix=store.cat_matrix,
                    partition=store.partition[A["te_rows"]],
                    cvi=store.cvi[A["te_rows"]],
                    cvi_valid=store.cvi_valid[A["te_rows"]],
                    observability=mte["mask"], actual_k=mte["actual_k"],
                    source_code=sc_te, kmode_code=kc_te,
                    slate_pos=A["te_slate_pos"], util=A["util_te"])
                UFB.assert_no_hidden_leakage(X_te, mte["mask"],
                                             A["te_slate_pos"], n_common)

                pred, info = TF.fit_predict(X_tr, y_tr, X_te, return_model=True)
                model = info.pop("model")
                chosen = EV.select_per_slate(pred, elig_te, te_starts,
                                             te_sizes, minimise=True)
                sel_idx = store.candidate_index[A["te_rows"]][chosen]
                sel_ari = ari_te[chosen]
                n_train_rows = int(X_tr.shape[0])
                del X_tr, X_te

                # ---- historical reproduction gate --------------------------
                hist = pd.read_csv(_ext(d / "slate_selections.csv.gz"))
                same_idx = np.array_equal(
                    hist["selected_candidate_index"].to_numpy(), sel_idx)
                same_ari = bool(np.allclose(
                    hist["selected_ari"].to_numpy(float), sel_ari, atol=1e-12))
                rate = float((hist["selected_candidate_index"].to_numpy()
                              == sel_idx).mean())
                ok = bool(same_idx and same_ari)
                checks.append({
                    "excluded_split": s, "regime_id": cid,
                    "source_stage": r["source_stage"],
                    "n_slates": int(len(hist)),
                    "selection_reproduction_rate": rate,
                    "selected_index_identical": bool(same_idx),
                    "selected_ari_identical": same_ari,
                    "max_abs_ari_difference": float(np.max(np.abs(
                        hist["selected_ari"].to_numpy(float) - sel_ari))),
                    "n_eligible_candidates_hist": int(len(hist)),
                    "passed": ok})
                if not ok:
                    n_fail += 1
                    print("   [FAIL %-12s s%02d] selection reproduction %.6f"
                          % (cid, s, rate), flush=True)
                    continue

                joblib.dump(model, _ext(d / MODEL_NAME), compress=3)
                mfile = d / MODEL_NAME
                write_json_atomic(d / MODEL_MANIFEST, {
                    "status": "complete", "stage": "2C-4P",
                    "excluded_split": s, "regime_id": cid,
                    "utility_source": sx, "k_mode": km,
                    "training_splits": [x for x in RI.OUTER_SPLITS if x != s],
                    "training": "dedicated single-regime, full slate population",
                    "context": UFB.CONDITION,
                    "n_features": UFB.EXPECTED_DIM,
                    "n_train_rows": n_train_rows,
                    "n_train_slates": int(n_tr),
                    "feature_schema": "C2_LOCAL_UTIL/347 via "
                                      "unified_feature_builder.build_unified_X",
                    "model_config_hash": MR.config_hash(),
                    "model_family": MR.MODEL_FAMILY,
                    "target": "T2_SLATE_REGRET",
                    "eligibility_hash": CE.definition_hash(),
                    "utility_source_train": fu["train_source"],
                    "utility_source_test": fu["test_source"],
                    "random_seed": MR.HGBR_CONFIG.get("random_state", 42),
                    "model_file": MODEL_NAME,
                    "model_file_sha256_16": sha_file(mfile),
                    "model_bytes": int(mfile.stat().st_size),
                    "fit_sec": info["fit_sec"],
                    "predict_sec": info["predict_sec"],
                    "reproduction_passed": True,
                    "selection_reproduction_rate": rate,
                    "historical_source_stage": r["source_stage"],
                    "split_1_used": False,
                    "source_commit": commit,
                    "note": "refit through the frozen train_fold.fit_predict "
                            "path; historical unit files were not modified"})
                rows.append({
                    "excluded_split": s, "regime_id": cid,
                    "unit_dir": str(d), "source_stage": r["source_stage"],
                    "n_train_rows": n_train_rows,
                    "fit_sec": info["fit_sec"],
                    "predict_sec": info["predict_sec"],
                    "unit_sec": time.time() - t0,
                    "model_bytes": int(mfile.stat().st_size),
                    "model_sha": sha_file(mfile),
                    "rss_mb": mem()})
                n_fit += 1
                el = time.time() - t_all
                print("   [%-12s s%02d] %3d/%3d rows=%d fit=%.0fs repro=%.4f "
                      "elapsed %.1fh ETA %.1fh"
                      % (cid, s, n_fit, len(todo) - n_skip, n_train_rows,
                         info["fit_sec"], rate, el / 3600.0,
                         el / max(n_fit, 1) * (len(todo) - n_skip - n_fit)
                         / 3600.0), flush=True)
                if rows:
                    write_csv_atomic(glob_out / "oof_reranker_runtime_rows.csv",
                                     pd.DataFrame(rows))
                    write_csv_atomic(
                        glob_out / "oof_reranker_reproduction_checks.csv",
                        pd.DataFrame(checks))
            del A

        print("\n  %d fitted, %d reused, %d failed, %.1f h"
              % (n_fit, n_skip, n_fail, (time.time() - t_all) / 3600.0),
              flush=True)

    # ---- global inventory --------------------------------------------------
    inv2 = RI.build(repo)
    ready = []
    for _, r in inv2.iterrows():
        d = Path(r["unit_dir"]) if r["unit_dir"] else None
        m = read_json(d / MODEL_MANIFEST) if d else None
        ready.append({
            "excluded_split": int(r["split_id"]), "regime_id": r["regime_id"],
            "unit_dir": r["unit_dir"], "source_stage": r["source_stage"],
            "model_persisted": bool(m and m.get("status") == "complete"),
            "reproduction_passed": bool(m and m.get("reproduction_passed")),
            "model_sha": (m or {}).get("model_file_sha256_16"),
            "model_bytes": (m or {}).get("model_bytes"),
            "fit_sec": (m or {}).get("fit_sec")})
    INVD = pd.DataFrame(ready)
    write_csv_atomic(glob_out / "oof_reranker_model_inventory.csv", INVD)

    # Backfill the reproduction table from the per-unit manifests, which are the
    # authoritative record. A resumed run only re-checks the units it fitted, so
    # without this the aggregate would silently under-report units validated in
    # an earlier invocation.
    seen = {(int(c["excluded_split"]), c["regime_id"]) for c in checks}
    for _, r in inv2.iterrows():
        key = (int(r["split_id"]), r["regime_id"])
        if key in seen or not r["unit_dir"]:
            continue
        m = read_json(Path(r["unit_dir"]) / MODEL_MANIFEST)
        if not m:
            continue
        checks.append({
            "excluded_split": key[0], "regime_id": key[1],
            "source_stage": r["source_stage"],
            "n_slates": None,
            "selection_reproduction_rate": m.get("selection_reproduction_rate"),
            "selected_index_identical": bool(m.get("reproduction_passed")),
            "selected_ari_identical": bool(m.get("reproduction_passed")),
            "max_abs_ari_difference": 0.0,
            "n_eligible_candidates_hist": None,
            "passed": bool(m.get("reproduction_passed")),
            "provenance": "backfilled from unit model_manifest.json"})
    if checks:
        write_csv_atomic(glob_out / "oof_reranker_reproduction_checks.csv",
                         pd.DataFrame(checks))
    done = int(INVD["model_persisted"].sum())
    ft = INVD["fit_sec"].dropna()
    write_json_atomic(glob_out / "oof_reranker_persistence_summary.json", {
        "expected_units": 150, "persisted": done,
        "missing": int(150 - done),
        "reproduction_passed": int(INVD["reproduction_passed"].sum()),
        "splits": sorted(int(x) for x in INVD["excluded_split"].unique()),
        "regimes": sorted(INVD["regime_id"].unique().tolist()),
        "split_1_present": False,
        "model_config_hash": MR.config_hash(),
        "eligibility_hash": CE.definition_hash(),
        "context": UFB.CONDITION, "n_features": UFB.EXPECTED_DIM,
        "total_model_bytes": int(INVD["model_bytes"].dropna().sum()),
        "source_commit": commit})
    if len(ft):
        write_json_atomic(glob_out / "oof_reranker_runtime_summary.json", {
            "n_units": int(len(ft)), "mean_fit_sec": float(ft.mean()),
            "median_fit_sec": float(ft.median()),
            "p25_fit_sec": float(ft.quantile(.25)),
            "p75_fit_sec": float(ft.quantile(.75)),
            "p90_fit_sec": float(ft.quantile(.90)),
            "p95_fit_sec": float(ft.quantile(.95)),
            "total_fit_cpu_hours": float(ft.sum() / 3600.0),
            "cost_class": "TRAINING/EXPERIMENT INFRASTRUCTURE COST -- must NOT "
                          "be counted in ClustOpt deployment runtime"})
    print("  inventory: %d/150 persisted, %d reproduction-passed"
          % (done, int(INVD["reproduction_passed"].sum())), flush=True)
    return 0 if done == 150 else 1


if __name__ == "__main__":
    raise SystemExit(main())
