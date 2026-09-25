"""Stage 2C-1: fit the MISSING single-exclusion OOF rerankers.

Every reusable unit was already found by :mod:`regime_inventory`; only genuinely
missing (regime, held-out split) pairs are fitted here. The unit computation is
imported from the Stage-2B4 runner rather than reimplemented, so the frozen
Candidate-Reranker contract -- dedicated single-regime model, C2_LOCAL_UTIL
context, 347 features, HGBR ``8f488cecbc1d4ba5``, T2 slate regret,
``ELIGIBLE_CANDIDATE.v1`` -- is identical by construction.

Leakage: the model for held-out split ``s`` trains on ``{2..16} \\ {s}`` only;
training-row utilities are the Stage-2A3B0 pair-exclusion representation and
test-row utilities the Stage-2A1 single-exclusion representation, exactly as in
Stage 2B. Split 1 is refused outright. ARI is read only AFTER the candidate has
been chosen.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

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

B1_REL = ("results_analysis/clustopt_candidate_reranker/"
          "stage2b1_masking_feasibility")
GZ = {"index": False, "compression": "gzip"}
LOCK_NAME = "stage2c_worker.pid"


def regime_parts(cid: str) -> Dict[str, str]:
    src, km = cid.split("_", 1)
    return {"source": src.lower(), "k_mode": km.lower()}


def unit_dir(out: Path, cid: str, s: int) -> Path:
    return out / "outer_fold_predictions" / cid / ("fold_%02d" % s)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", default=".")
    p.add_argument("--preflight", action="store_true")
    p.add_argument("--regimes", nargs="*", default=None)
    p.add_argument("--splits", type=int, nargs="*", default=None)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args(argv)

    repo = Path(args.repo_root).resolve()
    out = ensure_dir(repo / RI.STAGE2C_REL)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                            capture_output=True, text=True).stdout.strip()
    print("=" * 78)
    print("STAGE 2C-1  MISSING SINGLE-EXCLUSION OOF RERANKERS")
    print("=" * 78, flush=True)
    print("  context %s (%d dims) | model %s | target T2_SLATE_REGRET"
          % (UFB.CONDITION, UFB.EXPECTED_DIM, MR.config_hash()), flush=True)

    inv = RI.build(repo)
    todo = inv[inv["status"] != "EXISTS"][["regime_id", "split_id"]]
    if args.regimes:
        todo = todo[todo["regime_id"].isin(args.regimes)]
    if args.splits:
        todo = todo[todo["split_id"].isin(args.splits)]
    if 1 in set(todo["split_id"]):
        raise SystemExit("Split 1 is forbidden")
    todo = todo.sort_values(["regime_id", "split_id"])
    if args.limit:
        todo = todo.head(args.limit)
    print("  %d units to fit (%d already reusable)"
          % (len(todo), int((inv["status"] == "EXISTS").sum())), flush=True)
    if todo.empty:
        print("  nothing to do", flush=True)
        return 0

    with WorkerLock(out / LOCK_NAME):
        store = CandidateStore(repo)
        mem, sysmem = _memory_probes()
        if args.preflight:
            s = int(todo.iloc[0]["split_id"])
            t0 = time.time()
            A = _fold_arrays(repo, store, s)
            pre = {"rss_mb": mem(), "system_mb": sysmem(),
                   "probe_split": s, "n_train_slates": int(len(A["tr_slates"])),
                   "n_train_rows_eligible": int(len(A["tr_rows"])),
                   "est_train_matrix_gb": float(len(A["tr_rows"])
                                                * UFB.EXPECTED_DIM * 4 / 2**30),
                   "fold_array_sec": time.time() - t0,
                   "units_to_fit": int(len(todo)),
                   "workers": 1,
                   "decision": "serial, one worker: a single training matrix is "
                               "~1.5 GB and two concurrent fits would exceed the "
                               "safe budget"}
            write_json_atomic(out / "resource_preflight.json", pre)
            print("  preflight: %s" % pre, flush=True)
            return 0

        write_csv_atomic(out / "configuration_registry.csv", pd.DataFrame([
            {"condition_id": cid, **regime_parts(cid),
             "input_dim": UFB.EXPECTED_DIM, "context": UFB.CONDITION,
             "training": "dedicated single-regime, full slate population",
             "model_family": MR.MODEL_FAMILY,
             "model_config_hash": MR.config_hash(),
             "target": "T2_SLATE_REGRET"} for cid in RI.RERANKER_IDS]))

        rows: List[Dict[str, Any]] = []
        t_all, n_done = time.time(), 0
        for s, grp in todo.groupby("split_id"):
            s = int(s)
            A = _fold_arrays(repo, store, s)
            fu = A["fu"]
            y_tr = TF.target_regret(store.ari[A["tr_rows"]],
                                    store.oracle_row[A["tr_rows"]])
            te_starts = np.cumsum(
                np.r_[0, store.slate_size[A["te_slates"]]])[:-1]
            te_sizes = store.slate_size[A["te_slates"]]
            ari_te = store.ari[A["te_rows"]]
            orc_te = store.oracle_row[A["te_rows"]]
            elig_te = store.eligible[A["te_rows"]]
            ds_te = store.dataset_id[A["te_rows"]][te_starts]
            vw_te = store.view_id[A["te_rows"]][te_starts]
            orc_sl = orc_te[te_starts]

            for cid in grp["regime_id"]:
                rp = regime_parts(cid)
                sx, km = rp["source"], rp["k_mode"]
                uhash = MR.unit_hash("DEDICATED_C2__%s" % cid, s, UFB.CONDITION,
                                     CE.definition_hash(),
                                     "%s / %s" % (fu["train_source"],
                                                  fu["test_source"]))
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
                UFB.assert_regime_identity(X_tr, A["tr_slate_pos"], sc_tr,
                                           kc_tr, n_common)
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
                res = TF.run_unit(condition={"kind": "DEPLOYABLE"}, X_tr=X_tr,
                                  y_tr=y_tr, X_te=X_te, eligible_te=elig_te,
                                  slate_starts=te_starts, slate_sizes=te_sizes,
                                  ari_te=ari_te, oracle_te=orc_te)
                n_train_rows = int(X_tr.shape[0])
                del X_tr, X_te

                m = EV.slate_metrics(res["selected_ari"], orc_sl, vw_te, ds_te)
                b1 = pd.read_csv(_ext(repo / B1_REL / "outer_fold_predictions"
                                      / cid / ("fold_%02d" % s)
                                      / "slate_selections.csv.gz"))
                if not np.array_equal(b1["dataset_id"].to_numpy(), ds_te) or \
                   not np.array_equal(b1["view_id"].to_numpy(), vw_te):
                    raise AssertionError("%s: frozen slate order differs" % cid)
                if not np.allclose(b1["candidate_oracle_ari"].to_numpy(),
                                   orc_sl, atol=1e-12):
                    raise AssertionError("%s: frozen oracle differs" % cid)
                for lab in ("raw", "softmax"):
                    bm = EV.slate_metrics(b1["b_%s_ari" % lab].to_numpy(float),
                                          orc_sl, vw_te, ds_te)
                    m["b_%s_view" % lab] = bm["mean_selected_ari"]
                    m["b_%s_bestview" % lab] = bm["mean_bestview_selected_ari"]
                    m["recovery_%s" % lab] = EV.recovery(
                        m["mean_bestview_selected_ari"],
                        bm["mean_bestview_selected_ari"],
                        m["mean_bestview_candidate_oracle"])
                dl = read_json(repo / B1_REL / "outer_fold_predictions" / cid
                               / ("fold_%02d" % s) / "unit_manifest.json")
                m["dedicated_local_bestview"] = float(
                    dl["metrics"]["mean_bestview_selected_ari"])

                d = ensure_dir(unit_dir(out, cid, s))
                pd.DataFrame({
                    "dataset_id": ds_te, "split_id": s, "view_id": vw_te,
                    "selected_candidate_index":
                        store.candidate_index[A["te_rows"]][res["chosen"]],
                    "selected_ari": res["selected_ari"],
                    "candidate_oracle_ari": orc_sl,
                    "oracle_regret": orc_sl - res["selected_ari"],
                    "b_raw_ari": b1["b_raw_ari"].to_numpy(),
                    "b_softmax_ari": b1["b_softmax_ari"].to_numpy(),
                    "dedicated_local_selected_ari": b1["selected_ari"].to_numpy(),
                }).to_csv(_ext(d / "slate_selections.csv.gz"), **GZ)
                write_json_atomic(d / "unit_manifest.json", {
                    "status": "complete", "unit_hash": uhash,
                    "condition_id": cid, "outer_split": s,
                    "utility_source": sx, "k_mode": km,
                    "training": "dedicated single-regime, full slate population",
                    "context": UFB.CONDITION, "n_features": UFB.EXPECTED_DIM,
                    "n_train_rows": n_train_rows, "n_train_slates": int(n_tr),
                    "eligibility_hash": CE.definition_hash(),
                    "utility_source_train": fu["train_source"],
                    "utility_source_test": fu["test_source"],
                    "model_config_hash": MR.config_hash(),
                    "target": "T2_SLATE_REGRET",
                    "fit_info": res["fit_info"], "metrics": m,
                    "leakage_checks_pass": True, "split_1_used": False,
                    "stage": "2C-1", "source_commit": commit,
                })
                rows.append({"regime_id": cid, "split_id": s,
                             "n_train_rows": n_train_rows,
                             "n_train_slates": int(n_tr),
                             "n_features": UFB.EXPECTED_DIM,
                             "unit_hash": uhash,
                             "fit_sec": res["fit_info"].get("fit_sec"),
                             "unit_sec": time.time() - t0,
                             "bestview": m["mean_bestview_selected_ari"],
                             "recovery_raw": m["recovery_raw"]})
                n_done += 1
                print("   [%-12s fold %02d] %3d/%3d rows=%d BV=%.4f rec=%+.4f "
                      "(%.0fs)" % (cid, s, n_done, len(todo), n_train_rows,
                                   m["mean_bestview_selected_ari"],
                                   m["recovery_raw"], time.time() - t0),
                      flush=True)
                if rows:
                    write_csv_atomic(out / "reranker_oof_training_summary.csv",
                                     pd.DataFrame(rows))
            del A

        print("\n  %d units fitted in %.0fs" % (n_done, time.time() - t_all),
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
