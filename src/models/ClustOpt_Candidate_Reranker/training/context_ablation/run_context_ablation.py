"""Stage 2B-2 orchestrator: 3 new context conditions x 15 outer folds = 45 fits.

C0 (LOCAL) is never retrained -- its Stage-2B1 ``MLP_TOP10`` units are reused and
their equivalence is validated rather than assumed.

Process safety (brief section 37): Stage 2B-1 lost time because shell-wrapper
restarts spawned duplicate Python workers -- a shell exiting does NOT mean its
worker died. This module therefore takes a PID lock and validates process
liveness before running. A stale PID file never blocks a resume; a live worker
always blocks a duplicate.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from models.Clustering_Repository_Builder.experiments.experiment_execution.output_writer import (  # noqa: E402,E501
    _ext, ensure_dir, path_exists, read_json, write_csv_atomic, write_json_atomic,
)
from models.ClustOpt_Candidate_Reranker.data import candidate_eligibility as CE  # noqa: E402,E501
from models.ClustOpt_Candidate_Reranker.data import slate_schema as SS  # noqa: E402,E501
from models.ClustOpt_Candidate_Reranker.training import evaluate_fold as EV  # noqa: E402,E501
from models.ClustOpt_Candidate_Reranker.training import model_registry as MR  # noqa: E402,E501
from models.ClustOpt_Candidate_Reranker.training import train_fold as TF  # noqa: E402,E501
from models.ClustOpt_Candidate_Reranker.training.run_masking_feasibility import (  # noqa: E402,E501
    CandidateStore, load_fold_utilities, resolve_masks, _memory_probes,
)
from . import context_feature_builder as CFB

B1_REL = ("results_analysis/clustopt_candidate_reranker/"
          "stage2b1_masking_feasibility")
OUT_REL = ("results_analysis/clustopt_candidate_reranker/"
           "stage2b2_context_ablation")
OUTER_SPLITS: Tuple[int, ...] = tuple(range(2, 17))
GZ = {"index": False, "compression": "gzip"}
LOCK_NAME = "stage2b2_worker.pid"


# ------------------------------------------------------------ process guard
def _pid_alive(pid: int) -> bool:
    try:
        import ctypes
        h = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))
        if not h:
            return False
        code = ctypes.c_ulong()
        ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(h)
        return code.value == 259                    # STILL_ACTIVE
    except Exception:                               # noqa: BLE001
        return False


class WorkerLock:
    """Single-worker guard keyed on PID liveness, not on file existence."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def __enter__(self) -> "WorkerLock":
        prior = read_json(self.path)
        if prior:
            pid = int(prior.get("pid", -1))
            if pid != os.getpid() and _pid_alive(pid):
                raise SystemExit(
                    f"a Stage-2B2 worker is already running (pid {pid}); "
                    f"refusing to start a duplicate")
            print(f"  stale worker lock (pid {prior.get('pid')}) cleared",
                  flush=True)
        write_json_atomic(self.path, {"pid": os.getpid(),
                                      "started": time.time()})
        return self

    def __exit__(self, *exc: Any) -> None:
        try:
            cur = read_json(self.path) or {}
            if int(cur.get("pid", -1)) == os.getpid():
                Path(_ext(self.path)).unlink(missing_ok=True)
        except Exception:                            # noqa: BLE001
            pass


# ---------------------------------------------------------------- artifacts
def unit_dir(out: Path, cid: str, s: int) -> Path:
    return out / "outer_fold_predictions" / cid / f"fold_{s:02d}"


def unit_is_complete(out: Path, cid: str, s: int, uhash: str) -> bool:
    m = read_json(unit_dir(out, cid, s) / "unit_manifest.json")
    return bool(m and m.get("status") == "complete"
                and m.get("unit_hash") == uhash
                and m.get("leakage_checks_pass"))


def load_c0_units(repo: Path) -> Dict[int, Dict[str, Any]]:
    """Stage-2B1 MLP_TOP10 units, reused verbatim as C0."""
    out: Dict[int, Dict[str, Any]] = {}
    for s in OUTER_SPLITS:
        d = repo / B1_REL / "outer_fold_predictions" / CFB.STAGE2B1_CONDITION \
            / f"fold_{s:02d}"
        m = read_json(d / "unit_manifest.json")
        if not m or m.get("status") != "complete":
            raise SystemExit(f"Stage-2B1 {CFB.STAGE2B1_CONDITION} fold {s} "
                             f"is missing or incomplete")
        out[s] = {"manifest": m,
                  "slates": pd.read_csv(_ext(d / "slate_selections.csv.gz"))}
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", default=".")
    p.add_argument("--preflight", action="store_true")
    p.add_argument("--conditions", nargs="*", default=None)
    p.add_argument("--splits", type=int, nargs="*", default=None)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)

    repo = Path(args.repo_root).resolve()
    out = ensure_dir(repo / OUT_REL)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                            capture_output=True, text=True).stdout.strip()

    print("=" * 78)
    print("STAGE 2B-2  --  CANDIDATE RERANKER DATASET-CONTEXT ABLATION")
    print("=" * 78, flush=True)
    print(f"  regime {CFB.REGIME_ID} | model {MR.config_hash()} | "
          f"target T2_SLATE_REGRET | eligibility {CE.PREDICATE_ID}", flush=True)

    with WorkerLock(out / LOCK_NAME):
        store = CandidateStore(repo)
        ctx = SS.load_slate_context(repo)
        meta_cols = list(ctx["meta"].columns)
        meta_all = ctx["meta"].to_numpy(np.float32)
        meta_key = (ctx["ident"]["dataset_id"].astype(str) + "|"
                    + ctx["ident"]["view_id"]).to_numpy()
        if meta_all.shape[1] != CFB.N_META:
            raise SystemExit(f"META block is {meta_all.shape[1]}, expected 250")

        conds = CFB.new_conditions()
        if args.conditions:
            conds = [c for c in conds if c["condition_id"] in args.conditions]
        splits = args.splits or list(OUTER_SPLITS)
        if 1 in splits:
            raise SystemExit("Split 1 is forbidden")

        write_csv_atomic(out / "configuration_registry.csv", pd.DataFrame([
            {**c, "blocks": "+".join(c["blocks"]) or "none",
             "input_dim": CFB.expected_dim(c["condition_id"]),
             "model_family": MR.MODEL_FAMILY,
             "model_config_hash": MR.config_hash(),
             "target": "T2_SLATE_REGRET", "regime": CFB.REGIME_ID}
            for c in CFB.all_conditions()]))

        if args.preflight:
            return _preflight(repo, store, out, meta_all, meta_key)

        c0 = load_c0_units(repo)
        t_all, n_done, n_skip = time.time(), 0, 0
        for s in splits:
            fu = load_fold_utilities(repo, s, store.metric_names)
            tr_slates = np.flatnonzero(store.slate_split != s)
            te_slates = np.flatnonzero(store.slate_split == s)
            key = store.slate_key.to_numpy()
            tr_pos = pd.Index(key[tr_slates]).get_indexer(fu["train_key"])
            te_pos = pd.Index(key[te_slates]).get_indexer(fu["test_key"])
            if (tr_pos < 0).any() or (te_pos < 0).any():
                raise AssertionError(f"fold {s}: utility identifiers misaligned")
            tr_order, te_order = np.argsort(tr_pos), np.argsort(te_pos)

            # Slate-level context, ordered to match the fold's slate lists.
            mi = pd.Index(meta_key)
            meta_tr = meta_all[mi.get_indexer(key[tr_slates])]
            meta_te = meta_all[mi.get_indexer(key[te_slates])]
            util_tr = np.hstack([fu["train_util"]["mlp"][tr_order],
                                 fu["train_util"]["knn"][tr_order]]
                                ).astype(np.float32)
            util_te = np.hstack([fu["test_util"]["mlp"][te_order],
                                 fu["test_util"]["knn"][te_order]]
                                ).astype(np.float32)
            CFB.assert_context_finite(meta_tr, util_tr)
            CFB.assert_context_finite(meta_te, util_te)

            mtr = resolve_masks(fu["train_util"][CFB.REGIME_SOURCE][tr_order],
                                store.metric_names, CFB.REGIME_K_MODE)
            mte = resolve_masks(fu["test_util"][CFB.REGIME_SOURCE][te_order],
                                store.metric_names, CFB.REGIME_K_MODE)

            tr_rows_all = store.rows_for_slates(tr_slates)
            tr_rows = tr_rows_all[store.eligible[tr_rows_all]]
            te_rows = store.rows_for_slates(te_slates)
            tr_slate_pos = pd.Series(np.arange(len(tr_slates)), index=tr_slates
                                     ).reindex(store.slate_of_row[tr_rows]
                                               ).to_numpy().astype(np.int64)
            te_slate_pos = np.repeat(np.arange(len(te_slates)), 32)
            y_tr = TF.target_regret(store.ari[tr_rows],
                                    store.oracle_row[tr_rows])
            te_starts = np.cumsum(np.r_[0, store.slate_size[te_slates]])[:-1]
            te_sizes = store.slate_size[te_slates]
            ari_te, orc_te = store.ari[te_rows], store.oracle_row[te_rows]
            elig_te = store.eligible[te_rows]

            util_cols = ([f"mlp__{m}" for m in store.metric_names]
                         + [f"knn__{m}" for m in store.metric_names])
            for cond in conds:
                cid = cond["condition_id"]
                names = CFB.feature_names(cid, store.metric_names,
                                          store.cat_names, meta_cols, util_cols)
                uhash = MR.unit_hash(cid, s, CFB.schema_hash(names),
                                     CE.definition_hash(),
                                     f"{fu['train_source']} / {fu['test_source']}")
                if not args.overwrite and unit_is_complete(out, cid, s, uhash):
                    n_skip += 1
                    continue
                t0 = time.time()
                X_tr, n_common = CFB.build_context_X(
                    cid, view_codes=store.view_code[tr_rows],
                    cat_rows=store.cat_row[tr_rows], cat_matrix=store.cat_matrix,
                    partition=store.partition[tr_rows], cvi=store.cvi[tr_rows],
                    cvi_valid=store.cvi_valid[tr_rows], observability=mtr["mask"],
                    actual_k=mtr["actual_k"], slate_pos=tr_slate_pos,
                    meta=meta_tr, util=util_tr)
                X_te, _ = CFB.build_context_X(
                    cid, view_codes=store.view_code[te_rows],
                    cat_rows=store.cat_row[te_rows], cat_matrix=store.cat_matrix,
                    partition=store.partition[te_rows], cvi=store.cvi[te_rows],
                    cvi_valid=store.cvi_valid[te_rows], observability=mte["mask"],
                    actual_k=mte["actual_k"], slate_pos=te_slate_pos,
                    meta=meta_te, util=util_te)
                CFB.assert_no_hidden_leakage(X_te, mte["mask"], te_slate_pos,
                                             n_common)
                res = TF.run_unit(condition={"kind": "DEPLOYABLE"}, X_tr=X_tr,
                                  y_tr=y_tr, X_te=X_te, eligible_te=elig_te,
                                  slate_starts=te_starts, slate_sizes=te_sizes,
                                  ari_te=ari_te, oracle_te=orc_te)
                del X_tr, X_te

                sl_ds = store.dataset_id[te_rows][te_starts]
                sl_vw = store.view_id[te_rows][te_starts]
                sl_orc = orc_te[te_starts]
                m = EV.slate_metrics(res["selected_ari"], sl_orc, sl_vw, sl_ds)
                base = _reuse_baselines(c0[s], sl_ds, sl_vw, sl_orc)
                m.update(base)
                m["recovery_raw"] = EV.recovery(
                    m["mean_bestview_selected_ari"], base["b_raw_bestview"],
                    m["mean_bestview_candidate_oracle"])
                m["recovery_softmax"] = EV.recovery(
                    m["mean_bestview_selected_ari"], base["b_softmax_bestview"],
                    m["mean_bestview_candidate_oracle"])
                m["beats_raw"] = bool(m["mean_bestview_selected_ari"]
                                      > base["b_raw_bestview"])
                m["beats_softmax"] = bool(m["mean_bestview_selected_ari"]
                                          > base["b_softmax_bestview"])
                _write_unit(out, cid, s, uhash, cond, res, m, names, store,
                            te_rows, te_starts, sl_ds, sl_vw, sl_orc, c0[s],
                            fu, commit)
                n_done += 1
                print(f"   [{cid:<20} fold {s:02d}] "
                      f"BV={m['mean_bestview_selected_ari']:.4f} "
                      f"reg={m['mean_oracle_regret']:.4f} "
                      f"recRAW={m['recovery_raw']:+.4f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
            del meta_tr, meta_te, util_tr, util_te

        print(f"\n  {n_done} units executed, {n_skip} skipped, "
              f"{time.time()-t_all:.0f}s", flush=True)
    return 0


def _reuse_baselines(c0_fold: Dict[str, Any], ds, vw, orc) -> Dict[str, float]:
    """Native RAW/SOFTMAX baselines taken from the Stage-2B1 C0 unit itself."""
    sl = c0_fold["slates"]
    if not np.array_equal(sl["dataset_id"].to_numpy(), ds) or \
       not np.array_equal(sl["view_id"].to_numpy(), vw):
        raise AssertionError("C0 slate ordering does not match this fold")
    if not np.allclose(sl["candidate_oracle_ari"].to_numpy(), orc, atol=1e-12):
        raise AssertionError("C0 candidate oracle differs from this fold's")
    out = {}
    for lab in ("raw", "softmax"):
        a = sl[f"b_{lab}_ari"].to_numpy(float)
        bm = EV.slate_metrics(a, orc, vw, ds)
        out[f"b_{lab}_view"] = bm["mean_selected_ari"]
        out[f"b_{lab}_bestview"] = bm["mean_bestview_selected_ari"]
        out[f"b_{lab}_regret"] = bm["mean_oracle_regret"]
    exp = c0_fold["manifest"]["metrics"]
    for lab in ("raw", "softmax"):
        if abs(out[f"b_{lab}_bestview"] - exp[f"b_{lab}_bestview"]) > 1e-9:
            raise AssertionError(f"reconstructed {lab} baseline differs from the "
                                 f"Stage-2B1 value")
    return out


def _write_unit(out, cid, s, uhash, cond, res, m, names, store, te_rows,
                te_starts, ds, vw, orc, c0_fold, fu, commit) -> None:
    d = ensure_dir(unit_dir(out, cid, s))
    sl0 = c0_fold["slates"]
    pd.DataFrame({
        "dataset_id": ds, "split_id": s, "view_id": vw,
        "selected_candidate_index": store.candidate_index[te_rows][res["chosen"]],
        "selected_ari": res["selected_ari"], "candidate_oracle_ari": orc,
        "oracle_regret": orc - res["selected_ari"],
        "c0_selected_ari": sl0["selected_ari"].to_numpy(),
        "b_raw_ari": sl0["b_raw_ari"].to_numpy(),
        "b_softmax_ari": sl0["b_softmax_ari"].to_numpy(),
    }).to_csv(_ext(d / "slate_selections.csv.gz"), **GZ)
    write_json_atomic(d / "unit_manifest.json", {
        "status": "complete", "unit_hash": uhash, "condition_id": cid,
        "outer_split": int(s), "blocks": list(cond["blocks"]),
        "regime": CFB.REGIME_ID, "feature_schema_hash": CFB.schema_hash(names),
        "n_features": len(names), "expected_dim": CFB.expected_dim(cid),
        "eligibility_hash": CE.definition_hash(),
        "utility_source_train": fu["train_source"],
        "utility_source_test": fu["test_source"],
        "model_config_hash": MR.config_hash(), "target": "T2_SLATE_REGRET",
        "fit_info": res["fit_info"], "metrics": m,
        "leakage_checks_pass": True, "split_1_used": False,
        "source_commit": commit,
    })


def _preflight(repo: Path, store: CandidateStore, out: Path,
               meta_all: np.ndarray, meta_key: np.ndarray) -> int:
    """Resource-only probe on ONE outer-training population. No test metrics."""
    import gc
    mem, sysmem = _memory_probes()
    s = 2
    fu = load_fold_utilities(repo, s, store.metric_names)
    tr_slates = np.flatnonzero(store.slate_split != s)
    key = store.slate_key.to_numpy()
    tr_order = np.argsort(pd.Index(key[tr_slates]).get_indexer(fu["train_key"]))
    meta_tr = meta_all[pd.Index(meta_key).get_indexer(key[tr_slates])]
    util_tr = np.hstack([fu["train_util"]["mlp"][tr_order],
                         fu["train_util"]["knn"][tr_order]]).astype(np.float32)
    mtr = resolve_masks(fu["train_util"][CFB.REGIME_SOURCE][tr_order],
                        store.metric_names, CFB.REGIME_K_MODE)
    tr_rows_all = store.rows_for_slates(tr_slates)
    tr_rows = tr_rows_all[store.eligible[tr_rows_all]]
    slate_pos = pd.Series(np.arange(len(tr_slates)), index=tr_slates
                          ).reindex(store.slate_of_row[tr_rows]
                                    ).to_numpy().astype(np.int64)
    y = TF.target_regret(store.ari[tr_rows], store.oracle_row[tr_rows])

    rows = []
    for cond in CFB.new_conditions():
        cid = cond["condition_id"]
        gc.collect()
        before = mem()
        t0 = time.time()
        X, _ = CFB.build_context_X(
            cid, view_codes=store.view_code[tr_rows],
            cat_rows=store.cat_row[tr_rows], cat_matrix=store.cat_matrix,
            partition=store.partition[tr_rows], cvi=store.cvi[tr_rows],
            cvi_valid=store.cvi_valid[tr_rows], observability=mtr["mask"],
            actual_k=mtr["actual_k"], slate_pos=slate_pos,
            meta=meta_tr, util=util_tr)
        build_sec = time.time() - t0
        after_build = mem()
        t1 = time.time()
        model = MR.build_model()
        model.fit(X, y)
        fit_sec = time.time() - t1
        peak = mem()
        t2 = time.time()
        model.predict(X[:100000])
        pred_est = (time.time() - t2) * (len(X) / 100000)
        rows.append({
            "condition_id": cid, "n_rows": int(len(X)),
            "n_features": int(X.shape[1]), "dtype": str(X.dtype),
            "input_matrix_mb": round(X.nbytes / 2 ** 20, 1),
            "rss_before_mb": round(before, 1),
            "rss_after_build_mb": round(after_build, 1),
            "rss_peak_fit_mb": round(peak, 1),
            "temporary_copy_overhead_mb": round(
                after_build - before - X.nbytes / 2 ** 20, 1),
            "fit_delta_mb": round(peak - after_build, 1),
            "build_sec": round(build_sec, 1), "fit_sec": round(fit_sec, 1),
            "predict_full_sec_est": round(pred_est, 1),
        })
        print(f"   {cid:<20} {X.shape} X={X.nbytes/2**20:.0f}MB "
              f"peak={peak:.0f}MB fit={fit_sec:.0f}s", flush=True)
        del X, model
        gc.collect()

    total, avail = sysmem()
    worst = max(r["rss_peak_fit_mb"] for r in rows)
    headroom = avail - (worst - rows[0]["rss_before_mb"])
    payload = {
        "model": MR.MODEL_FAMILY, "config": MR.HGBR_CONFIG,
        "regime": CFB.REGIME_ID, "probes": rows,
        "system_total_mb": round(total, 1),
        "system_available_mb": round(avail, 1),
        "worst_peak_rss_mb": worst,
        "estimated_headroom_mb": round(headroom, 1),
        "safe": bool(headroom > 300),
        "recommended_concurrency": 1,
        "float32_used": True, "feature_blocks_dropped": "none",
        "estimated_total_fit_hours": round(
            sum(r["fit_sec"] for r in rows) * 15 / 3600, 2),
    }
    write_json_atomic(out / "resource_preflight.json", payload)
    print(f"\n  worst peak {worst:.0f} MB | available {avail:.0f} MB | "
          f"headroom {headroom:.0f} MB | safe={payload['safe']} | "
          f"est {payload['estimated_total_fit_hours']} h", flush=True)
    if not payload["safe"]:
        print("\n  RESOURCE BLOCKER: C3 cannot run safely. Features were NOT "
              "dropped and no substitute model was chosen.", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
