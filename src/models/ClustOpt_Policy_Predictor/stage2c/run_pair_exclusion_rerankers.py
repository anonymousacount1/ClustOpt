"""Stage 2C-2B: fit the exact pair-exclusion rerankers R_{-(s,t)}.

For Policy-Predictor outer-test split ``s`` and a training-row split ``t``, the
reranker that generates that row's target must exclude BOTH splits. This module
fits those models for the three pre-registered outer splits.

ORDERED, NOT UNORDERED PAIRS. ``load_fold_utilities(repo, s)`` binds every
training row's utility features to the outer split: row ``u`` carries the
pair-exclusion source ``{s,u}``. So ``R_{-(2,9)}`` under outer split 2 and
``R_{-(9,2)}`` under outer split 9 are trained on the same rows but with
different feature sources, and they are NOT interchangeable. Deduplicating the
three symmetric pairs would force one direction to use utilities fitted with the
outer-test split included -- the very leakage this stage removes. The inventory
is therefore 3 x 14 = 42 ordered pairs x 10 regimes = 420 units.

Everything else is the frozen Stage-2B contract: dedicated single-regime HGBR
``8f488cecbc1d4ba5``, C2_LOCAL_UTIL / 347 dims, T2 slate regret,
``ELIGIBLE_CANDIDATE.v1``. No tuning. Split 1 is refused.
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

OUT_REL = ("results_analysis/clustopt_policy_predictor/"
           "stage2c_v2_sampled_exact_nesting")
OUTER_SPLITS: Tuple[int, ...] = (2, 9, 16)
ALL_SPLITS: Tuple[int, ...] = tuple(range(2, 17))
SLATE = 32
GZ = {"index": False, "compression": "gzip"}
LOCK_NAME = "stage2c2b_worker.pid"


def regime_parts(cid: str) -> Dict[str, str]:
    src, km = cid.split("_", 1)
    return {"source": src.lower(), "k_mode": km.lower()}


def pair_inventory() -> pd.DataFrame:
    rows = []
    for s in OUTER_SPLITS:
        for t in ALL_SPLITS:
            if t == s:
                continue
            for cid in RI.RERANKER_IDS:
                rows.append({"outer_split": s, "target_split": t,
                             "regime_id": cid,
                             "unit_id": "s%02d_t%02d_%s" % (s, t, cid),
                             "excludes": "{%d,%d}" % (s, t)})
    return pd.DataFrame(rows)


def inventory_hash(inv: pd.DataFrame) -> str:
    payload = json.dumps(sorted(inv["unit_id"].tolist()), separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def unit_dir(out: Path, s: int, t: int, cid: str) -> Path:
    return (out / "pair_predictions" / ("outer_%02d" % s)
            / ("target_%02d" % t) / cid)


def unit_complete(out: Path, s: int, t: int, cid: str, uhash: str) -> bool:
    m = read_json(unit_dir(out, s, t, cid) / "unit_manifest.json")
    return bool(m and m.get("status") == "complete"
                and m.get("unit_hash") == uhash
                and m.get("leakage_checks_pass")
                and not m.get("split_1_used")
                and m.get("n_features") == UFB.EXPECTED_DIM
                and m.get("model_config_hash") == MR.config_hash()
                and int(m.get("outer_split", -1)) == s
                and int(m.get("target_split", -1)) == t)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", default=".")
    p.add_argument("--preflight", action="store_true")
    p.add_argument("--outer", type=int, nargs="*", default=None)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args(argv)

    repo = Path(args.repo_root).resolve()
    out = ensure_dir(repo / OUT_REL)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                            capture_output=True, text=True).stdout.strip()
    INV = pair_inventory()
    ihash = inventory_hash(INV)
    write_csv_atomic(out / "pair_exclusion_inventory.csv", INV)
    write_json_atomic(out / "selected_outer_splits.json", {
        "outer_splits": list(OUTER_SPLITS),
        "selection_rule": "first / middle / last development split by ascending "
                          "split id; outcome-blind and fixed before any exact "
                          "nested result was observed",
        "ordered_pairs": int(INV[["outer_split", "target_split"]]
                             .drop_duplicates().shape[0]),
        "regimes": len(RI.RERANKER_IDS),
        "units": int(len(INV)),
        "why_ordered_not_unordered":
            "training-row utilities are bound to the outer split (row u uses "
            "pair source {s,u}), so R_{-(s,t)} and R_{-(t,s)} differ; "
            "deduplicating the 3 symmetric pairs would make one direction use "
            "utilities fitted WITH the outer-test split",
        "inventory_hash": ihash})

    print("=" * 78)
    print("STAGE 2C-2B  EXACT PAIR-EXCLUSION RERANKERS")
    print("=" * 78, flush=True)
    print("  outer splits %s | %d ordered pairs x %d regimes = %d units | "
          "inventory %s" % (list(OUTER_SPLITS),
                            INV[["outer_split", "target_split"]]
                            .drop_duplicates().shape[0],
                            len(RI.RERANKER_IDS), len(INV), ihash), flush=True)
    print("  model %s | context %s (%d dims) | target T2_SLATE_REGRET"
          % (MR.config_hash(), UFB.CONDITION, UFB.EXPECTED_DIM), flush=True)

    outers = args.outer or list(OUTER_SPLITS)
    if 1 in outers:
        raise SystemExit("Split 1 is forbidden")

    with WorkerLock(out / LOCK_NAME):
        store = CandidateStore(repo)
        mem, sysmem = _memory_probes()
        rows: List[Dict[str, Any]] = []
        t_all = time.time()
        n_done = n_skip = 0
        todo_total = int(len(INV[INV["outer_split"].isin(outers)]))

        for s in outers:
            A = _fold_arrays(repo, store, s)
            fu = A["fu"]
            tr_slate_split = store.slate_split[A["tr_slates"]]
            masks_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}

            if args.preflight:
                pre = {"rss_mb": mem(), "system_mb": sysmem(),
                       "outer_split": s,
                       "n_train_slates": int(len(A["tr_slates"])),
                       "est_train_matrix_gb": float(
                           len(A["tr_rows"]) * UFB.EXPECTED_DIM * 4 / 2**30),
                       "units_total": len(INV),
                       "decision": "worker count is decided by the concurrent "
                                   "preflight in preflight_workers.py"}
                write_json_atomic(out / "resource_preflight_arrays.json", pre)
                print("  preflight: %s" % pre, flush=True)
                return 0

            for t in [x for x in ALL_SPLITS if x != s]:
                pos_t = np.flatnonzero(tr_slate_split == t)
                t_slates = A["tr_slates"][pos_t]
                is_t = np.isin(A["tr_slate_pos"], pos_t)
                fit_rows = A["tr_rows"][~is_t]
                fit_pos = A["tr_slate_pos"][~is_t]
                sc_rows = store.rows_for_slates(t_slates)
                sc_pos = np.repeat(pos_t, SLATE)
                if len(sc_rows) != len(pos_t) * SLATE:
                    raise AssertionError("pair (%d,%d): ragged slates" % (s, t))
                # leakage assertions: neither split may appear in the fit set
                fit_splits = set(np.unique(tr_slate_split[np.unique(fit_pos)]))
                if s in fit_splits or t in fit_splits:
                    raise AssertionError("pair (%d,%d): excluded split leaked "
                                         "into the fit population" % (s, t))

                y_fit = TF.target_regret(store.ari[fit_rows],
                                         store.oracle_row[fit_rows])
                starts = np.arange(len(pos_t)) * SLATE
                sizes = np.full(len(pos_t), SLATE, dtype=np.int64)
                ari_sc = store.ari[sc_rows]
                orc_sc = store.oracle_row[sc_rows]
                elig_sc = store.eligible[sc_rows]
                ds_sc = store.dataset_id[sc_rows][starts]
                vw_sc = store.view_id[sc_rows][starts]
                orc_sl = orc_sc[starts]

                t_pair = time.time()
                n_reg = 0
                for cid in RI.RERANKER_IDS:
                    rp = regime_parts(cid)
                    sx, km = rp["source"], rp["k_mode"]
                    uhash = MR.unit_hash(
                        "PAIREXCL_C2__%s" % cid, s * 100 + t, UFB.CONDITION,
                        CE.definition_hash(),
                        "%s / excl{%d,%d}" % (fu["train_source"], s, t))
                    if unit_complete(out, s, t, cid, uhash):
                        n_skip += 1
                        n_reg += 1
                        continue
                    t0 = time.time()
                    key = (sx, km)
                    if key not in masks_cache:
                        masks_cache[key] = resolve_masks(
                            fu["train_util"][sx][A["tr_order"]],
                            store.metric_names, km)
                    M = masks_cache[key]
                    n_fit_sl = len(A["tr_slates"])
                    sc_tr = np.full(n_fit_sl, FB.SOURCES.index(sx), np.int64)
                    kc_tr = np.full(n_fit_sl, FB.K_MODES.index(km), np.int64)

                    X_fit, n_common = UFB.build_unified_X(
                        view_codes=store.view_code[fit_rows],
                        cat_rows=store.cat_row[fit_rows],
                        cat_matrix=store.cat_matrix,
                        partition=store.partition[fit_rows],
                        cvi=store.cvi[fit_rows], cvi_valid=store.cvi_valid[fit_rows],
                        observability=M["mask"], actual_k=M["actual_k"],
                        source_code=sc_tr, kmode_code=kc_tr,
                        slate_pos=fit_pos, util=A["util_tr"])
                    X_sc, _ = UFB.build_unified_X(
                        view_codes=store.view_code[sc_rows],
                        cat_rows=store.cat_row[sc_rows],
                        cat_matrix=store.cat_matrix,
                        partition=store.partition[sc_rows],
                        cvi=store.cvi[sc_rows], cvi_valid=store.cvi_valid[sc_rows],
                        observability=M["mask"], actual_k=M["actual_k"],
                        source_code=sc_tr, kmode_code=kc_tr,
                        slate_pos=sc_pos, util=A["util_tr"])
                    UFB.assert_no_hidden_leakage(X_sc, M["mask"], sc_pos, n_common)
                    res = TF.run_unit(condition={"kind": "DEPLOYABLE"},
                                      X_tr=X_fit, y_tr=y_fit, X_te=X_sc,
                                      eligible_te=elig_sc, slate_starts=starts,
                                      slate_sizes=sizes, ari_te=ari_sc,
                                      oracle_te=orc_sc)
                    n_fit_rows = int(X_fit.shape[0])
                    del X_fit, X_sc

                    m = EV.slate_metrics(res["selected_ari"], orc_sl, vw_sc, ds_sc)
                    d = ensure_dir(unit_dir(out, s, t, cid))
                    pd.DataFrame({
                        "dataset_id": ds_sc, "split_id": t, "view_id": vw_sc,
                        "selected_candidate_index":
                            store.candidate_index[sc_rows][res["chosen"]],
                        "selected_ari": res["selected_ari"],
                        "candidate_oracle_ari": orc_sl,
                        "oracle_regret": orc_sl - res["selected_ari"],
                    }).to_csv(_ext(d / "slate_selections.csv.gz"), **GZ)
                    write_json_atomic(d / "unit_manifest.json", {
                        "status": "complete", "unit_hash": uhash,
                        "condition_id": cid, "outer_split": s,
                        "target_split": t, "excludes": [s, t],
                        "utility_source": sx, "k_mode": km,
                        "training": "dedicated single-regime, pair-exclusion "
                                    "population",
                        "context": UFB.CONDITION,
                        "n_features": UFB.EXPECTED_DIM,
                        "n_train_rows": n_fit_rows,
                        "n_train_slates": int(len(A["tr_slates"]) - len(pos_t)),
                        "n_scored_slates": int(len(pos_t)),
                        "eligibility_hash": CE.definition_hash(),
                        "utility_source_train": fu["train_source"],
                        "model_config_hash": MR.config_hash(),
                        "target": "T2_SLATE_REGRET",
                        "fit_info": res["fit_info"], "metrics": m,
                        "leakage_checks_pass": True, "split_1_used": False,
                        "stage": "2C-2B", "inventory_hash": ihash,
                        "source_commit": commit})
                    rows.append({"outer_split": s, "target_split": t,
                                 "regime_id": cid, "unit_hash": uhash,
                                 "n_train_rows": n_fit_rows,
                                 "fit_sec": res["fit_info"].get("fit_sec"),
                                 "unit_sec": time.time() - t0,
                                 "bestview": m["mean_bestview_selected_ari"]})
                    n_done += 1
                    n_reg += 1
                if rows:
                    write_csv_atomic(out / "pair_model_training_summary.csv",
                                     pd.DataFrame(rows))
                el = time.time() - t_all
                rate = el / max(n_done, 1)
                print("   [outer %02d | target %02d] regimes=%2d/10 | %d/%d done "
                      "%d skip | %.0fs pair | elapsed %.1fh ETA %.1fh"
                      % (s, t, n_reg, n_done, todo_total, n_skip,
                         time.time() - t_pair, el / 3600.0,
                         rate * (todo_total - n_done - n_skip) / 3600.0),
                      flush=True)
            del A

        print("\n  %d units fitted, %d reused, %.1f h"
              % (n_done, n_skip, (time.time() - t_all) / 3600.0), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
