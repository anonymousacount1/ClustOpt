"""Stage 2C-4A: turn completed RAW/SOFTMAX twins into the continuous target.

For each twin the NEW online slate is re-represented in the frozen 347-dim C2
schema by :func:`online_feature_builder.build_online_X` -- the generic,
provenance-agnostic builder -- and scored by the persisted single-exclusion
Candidate Reranker ``R_{-s,r}`` for that row's split and OOF-v2 regime. The
reranked candidate's identity is frozen before its ARI is read, exactly as in
Stage-2B5.

    y_raw      = reranked online ARI under RAW
    y_softmax  = reranked online ARI under SOFTMAX
    delta_mode = y_raw - y_softmax          <- the stored continuous target

The target repository is never collapsed to a binary label; the sign is derived
downstream where it is needed.

Every scientific gate the twins must satisfy is evaluated here and written to
``online_target_gate_checks.csv``: the reranker excludes the row's split, the
utility is the OOF prediction U_{-s}, the C2 matrix is 347-wide with the 227/347
boundary intact, unobserved metrics are NaN, the twins share regime / K /
selected-CVI mask / search space / trials / timeout / seed, and the sparse-metric
property holds (the 60-dim utility vector must never cause 60 CVIs to be
computed).
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
from models.ClustOpt_Candidate_Reranker.data import candidate_schema as CS  # noqa: E402,E501
from models.ClustOpt_Candidate_Reranker.split1_preparation import (  # noqa: E402,E501
    online_feature_builder as OFB,
)
from models.ClustOpt_Candidate_Reranker.training import feature_builder as FB  # noqa: E402,E501
from models.ClustOpt_Candidate_Reranker.training import model_registry as MR  # noqa: E402,E501
from models.ClustOpt_Candidate_Reranker.training.unified_masking import (  # noqa: E402,E501
    unified_feature_builder as UFB,
)
from models.ClustOpt_Policy_Predictor.stage2c import stage2c4_common as C  # noqa: E402,E501

TIE_TOL = 1e-12
N_LOCAL = 227


def reranker_index(repo: Path) -> pd.DataFrame:
    f = repo / C.OUT_REL / "oof_reranker_model_inventory.csv"
    inv = pd.read_csv(_ext(f))
    bad = inv[~(inv["model_persisted"] & inv["reproduction_passed"])]
    if len(bad):
        raise SystemExit("%d OOF rerankers are not usable" % len(bad))
    return inv.set_index(["excluded_split", "regime_id"])


def load_reranker(unit_dir: Path, split_id: int, regime_id: str):
    import joblib
    m = read_json(unit_dir / "model_manifest.json") or {}
    if (m.get("status") != "complete" or int(m.get("excluded_split", -1)) != split_id
            or m.get("regime_id") != regime_id or not m.get("reproduction_passed")
            or m.get("n_features") != UFB.EXPECTED_DIM
            or m.get("model_config_hash") != MR.config_hash()
            or m.get("split_1_used")):
        raise SystemExit("reranker manifest mismatch at %s" % unit_dir)
    if split_id in list(m.get("training_splits") or []):
        raise SystemExit("reranker %s/%d trained on its own split"
                         % (regime_id, split_id))
    t0 = time.perf_counter()
    model = joblib.load(_ext(unit_dir / "model.joblib"))
    return model, m, (time.perf_counter() - t0) * 1000.0


def score_twin(*, log: pd.DataFrame, view_id: str, names: Sequence[str],
               util: np.ndarray, source: str, k_mode: str, actual_k: int,
               model) -> Dict[str, Any]:
    """Build C2, score the slate, freeze the choice, THEN read ARI."""
    t0 = time.perf_counter()
    X, n_common, diag = OFB.build_online_X(
        trial_log=log, view_id=view_id, metric_names=list(names),
        utility_vector=util, source=source, k_mode=k_mode,
        actual_k=int(actual_k))
    build_ms = (time.perf_counter() - t0) * 1000.0
    if X.shape[1] != UFB.EXPECTED_DIM:
        raise AssertionError("C2 width %d != %d" % (X.shape[1], UFB.EXPECTED_DIM))
    if n_common + 3 * OFB.N_METRICS + len(FB.SOURCES) + len(FB.K_MODES) + 1 != N_LOCAL:
        raise AssertionError("local block boundary is not 227")
    OFB.assert_no_hidden_exposure(X, n_common, list(names), log)

    elig = log["valid"].map(CS.is_valid_flag).to_numpy(bool)
    if not elig.any():
        raise AssertionError("no eligible candidate in the slate")

    t1 = time.perf_counter()
    pred = np.asarray(model.predict(X), dtype=np.float64)
    predict_ms = (time.perf_counter() - t1) * 1000.0

    t2 = time.perf_counter()
    idx = np.flatnonzero(elig)
    v = pred[idx]
    chosen = int(idx[np.abs(v - v.min()) <= TIE_TOL][0])
    select_ms = (time.perf_counter() - t2) * 1000.0

    ari = pd.to_numeric(log["ARI"], errors="coerce").to_numpy(float)
    eligible_ari = ari[elig]
    # --- utility block identity: the last 120 columns are the injected U_{-s}
    util_ok = bool(np.allclose(X[:, N_LOCAL:], util.astype(np.float32)[None, :],
                               atol=0, rtol=0))
    return {
        "n_visited_candidates": int(len(log)),
        "n_eligible_candidates": int(elig.sum()),
        "selected_index": chosen,
        "selected_algorithm": str(log["algorithm"].iloc[chosen]),
        "selected_ari": float(ari[chosen]),
        "visited_oracle_ari": float(np.nanmax(eligible_ari)),
        "n_observed_metrics": int(diag["n_observed_metrics"]),
        "observed_metrics": diag["observed_metrics"],
        "n_unique_configs": int(diag["n_unique_configs"]),
        "n_unseen_algorithm_candidates": int(diag["n_unseen_algorithm_candidates"]),
        "c2_dims": int(X.shape[1]), "c2_n_common": int(n_common),
        "c2_util_block_matches_injection": util_ok,
        "feature_build_ms": build_ms, "reranker_predict_ms": predict_ms,
        "reranker_select_ms": select_ms,
        "reranker_total_ms": build_ms + predict_ms + select_ms,
    }


def read_twin(run_dir: Path) -> Dict[str, Any]:
    return {
        "log": pd.read_csv(_ext(run_dir / CS.CANDIDATE_CSV)),
        "status": read_json(run_dir / "status.json") or {},
        "best": read_json(run_dir / "best_result.json") or {},
        "selected": read_json(run_dir / "selected_metrics.json") or {},
        "timing": read_json(run_dir / "timing_summary.json") or {},
        "phaseD": read_json(run_dir / "phaseD_metadata.json") or {},
        "config": read_json(run_dir / "config_snapshot.json") or {},
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", default=".")
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--rows", nargs="*", default=None)
    p.add_argument("--label", default="dev")
    p.add_argument("--strict", action="store_true",
                   help="fail on the first incomplete pair instead of skipping")
    args = p.parse_args(argv)
    repo = Path(args.repo_root).resolve()
    out = ensure_dir(repo / C.OUT_REL)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                            capture_output=True, text=True).stdout.strip()
    t_all = time.time()
    print("=" * 78)
    print("STAGE 2C-4A  BUILD ONLINE RAW/SOFTMAX TARGETS  [%s]" % args.label)
    print("=" * 78, flush=True)

    C.load_sample(repo)
    names = C.metric_names(repo)
    INV = pd.read_csv(_ext(out / "online_pair_inventory.csv"))
    INV = INV[INV["rank"] <= int(args.n)]
    if args.rows:
        INV = INV[INV["row_id"].isin(set(args.rows))]
    RINV = reranker_index(repo)

    targets: List[Dict[str, Any]] = []
    gates: List[Dict[str, Any]] = []
    skipped = 0
    for (s, rid), grp in INV.groupby(["split_id", "regime_id"]):
        s, rid = int(s), str(rid)
        unit = Path(RINV.loc[(s, rid), "unit_dir"])
        model, mman, load_ms = load_reranker(unit, s, rid)
        idx = C.load_oof_utilities(repo, s, names)
        src, km = C.source_of(rid), C.k_mode_of(rid)
        for row_id, pair in grp.groupby("row_id"):
            if set(pair["weighting_arm"]) != set(C.MODES):
                skipped += 1
                continue
            base = pair.iloc[0]
            twins, ok = {}, True
            for _, r in pair.iterrows():
                d = read_twin(Path(r["run_dir"]))
                if d["status"].get("status") != "success":
                    ok = False
                twins[r["weighting_arm"]] = (r, d)
            if not ok:
                skipped += 1
                if args.strict:
                    raise SystemExit("incomplete pair %s" % row_id)
                continue

            util = C.util120(idx, base["dataset_id"], base["view_id"])
            res: Dict[str, Dict[str, Any]] = {}
            for mode, (r, d) in twins.items():
                sel = list(d["selected"].get("selected_metrics") or [])
                res[mode] = score_twin(
                    log=d["log"], view_id=base["view_id"], names=names,
                    util=util, source=src, k_mode=km,
                    actual_k=len(sel) or 1, model=model)
                res[mode]["selected_metrics"] = sel
                res[mode]["record"] = (r, d)

            R, S = res["RAW"], res["SOFTMAX"]
            (rr, rd), (sr, sd) = R.pop("record"), S.pop("record")
            n_norm_raw = sum(1 for c in rd["log"].columns if c.endswith("_norm"))
            n_norm_sm = sum(1 for c in sd["log"].columns if c.endswith("_norm"))
            same_mask = R["selected_metrics"] == S["selected_metrics"]
            k_raw = len(R["selected_metrics"])

            g = {
                "row_id": row_id, "split_id": s, "regime_id": rid,
                "dataset_id": base["dataset_id"], "view_id": base["view_id"],
                "reranker_excludes_split": (s not in
                                            list(mman.get("training_splits") or [])),
                "reranker_regime_matches": mman.get("regime_id") == rid,
                "reranker_context_c2": mman.get("context") == UFB.CONDITION,
                "reranker_n_features_347": mman.get("n_features") == UFB.EXPECTED_DIM,
                "utility_is_oof_prediction": (rd["phaseD"].get("real_utility_path")
                                              == C.injected_utility_rel(
                                                  base["view_id"], src)),
                "utility_vector_hash": C.vector_hash(util),
                "c2_dims_raw": R["c2_dims"], "c2_dims_softmax": S["c2_dims"],
                "c2_dims_ok": (R["c2_dims"] == S["c2_dims"] == UFB.EXPECTED_DIM),
                "c2_local_block_227": (R["c2_n_common"] + 3 * 60 + 2 + 5 + 1
                                       == N_LOCAL),
                "c2_util_block_matches_injection": bool(
                    R["c2_util_block_matches_injection"]
                    and S["c2_util_block_matches_injection"]),
                "same_regime": True, "same_k": k_raw == len(S["selected_metrics"]),
                "actual_k": k_raw,
                "same_selected_cvi_mask": bool(same_mask),
                "same_search_space": (json.dumps(rd["config"].get("search_space"),
                                                 sort_keys=True)
                                      == json.dumps(sd["config"].get("search_space"),
                                                    sort_keys=True)),
                "same_requested_trials": (rd["best"].get("requested_trials")
                                          == sd["best"].get("requested_trials")
                                          == C.N_TRIALS),
                "same_timeout": ((rd["config"].get("search_algorithm") or {})
                                 .get("params", {}).get("timeout")
                                 == (sd["config"].get("search_algorithm") or {})
                                 .get("params", {}).get("timeout") == C.TIMEOUT_SEC),
                "same_seed": (rd["best"].get("search_seed")
                              == sd["best"].get("search_seed")
                              == int(base["search_seed"])),
                "only_weighting_differs": (
                    rd["phaseD"].get("weighting_mode") == "raw_normalized"
                    and sd["phaseD"].get("weighting_mode") == "softmax_t05"),
                "n_norm_columns_raw": n_norm_raw,
                "n_norm_columns_softmax": n_norm_sm,
                "accidental_additional_cvis_raw": n_norm_raw - k_raw,
                "accidental_additional_cvis_softmax": (n_norm_sm
                                                       - len(S["selected_metrics"])),
                "sparse_metric_gate": (n_norm_raw == k_raw
                                       and n_norm_sm == len(S["selected_metrics"])
                                       and k_raw < 60),
                "unobserved_metrics_are_nan": True,   # enforced by the builder
                "fallback_used_raw": bool(rd["selected"].get("used_fallback")),
                "fallback_used_softmax": bool(sd["selected"].get("used_fallback")),
                "split_1_touched": False,
            }
            g["all_gates_pass"] = bool(
                g["reranker_excludes_split"] and g["reranker_regime_matches"]
                and g["reranker_context_c2"] and g["reranker_n_features_347"]
                and g["utility_is_oof_prediction"] and g["c2_dims_ok"]
                and g["c2_local_block_227"] and g["c2_util_block_matches_injection"]
                and g["same_k"] and g["same_selected_cvi_mask"]
                and g["same_search_space"] and g["same_requested_trials"]
                and g["same_timeout"] and g["same_seed"]
                and g["only_weighting_differs"] and g["sparse_metric_gate"]
                and not g["fallback_used_raw"] and not g["fallback_used_softmax"])
            gates.append(g)

            targets.append({
                "row_id": row_id, "split_id": s, "rank": int(base["rank"]),
                "dataset_id": base["dataset_id"], "view_id": base["view_id"],
                "family_id": base["family_id"], "subfamily_id": base["subfamily_id"],
                "regime_id": rid, "utility_source": src, "k_mode": km,
                "actual_k": k_raw, "search_seed": int(base["search_seed"]),
                "selected_cvis": "|".join(R["selected_metrics"]),
                "y_raw": R["selected_ari"], "y_softmax": S["selected_ari"],
                "delta_mode": R["selected_ari"] - S["selected_ari"],
                "raw_original_ari": rd["status"].get("ari"),
                "softmax_original_ari": sd["status"].get("ari"),
                "raw_visited_oracle_ari": R["visited_oracle_ari"],
                "softmax_visited_oracle_ari": S["visited_oracle_ari"],
                "raw_n_visited": R["n_visited_candidates"],
                "softmax_n_visited": S["n_visited_candidates"],
                "raw_n_eligible": R["n_eligible_candidates"],
                "softmax_n_eligible": S["n_eligible_candidates"],
                "raw_selected_index": R["selected_index"],
                "softmax_selected_index": S["selected_index"],
                "raw_selected_algorithm": R["selected_algorithm"],
                "softmax_selected_algorithm": S["selected_algorithm"],
                "raw_search_sec": rd["timing"].get("runtime_sec"),
                "softmax_search_sec": sd["timing"].get("runtime_sec"),
                "raw_cvi_eval_sec": rd["timing"].get("cvi_eval_sec"),
                "softmax_cvi_eval_sec": sd["timing"].get("cvi_eval_sec"),
                "raw_delivered_trials": rd["best"].get("delivered_trials"),
                "softmax_delivered_trials": sd["best"].get("delivered_trials"),
                "raw_feature_build_ms": R["feature_build_ms"],
                "softmax_feature_build_ms": S["feature_build_ms"],
                "raw_reranker_predict_ms": R["reranker_predict_ms"],
                "softmax_reranker_predict_ms": S["reranker_predict_ms"],
                "raw_reranker_select_ms": R["reranker_select_ms"],
                "softmax_reranker_select_ms": S["reranker_select_ms"],
                "reranker_model_load_ms": load_ms,
                "raw_n_unseen_algorithm_candidates":
                    R["n_unseen_algorithm_candidates"],
                "softmax_n_unseen_algorithm_candidates":
                    S["n_unseen_algorithm_candidates"],
                "gates_pass": g["all_gates_pass"],
            })
        del model
        print("   split %02d %-12s : %d pairs" % (s, rid, len(grp) // 2),
              flush=True)

    T = pd.DataFrame(targets)
    G = pd.DataFrame(gates)
    if T.empty:
        raise SystemExit("no complete pairs to target")
    T = T.sort_values(["split_id", "rank", "dataset_id", "view_id"]
                      ).reset_index(drop=True)
    tf = out / ("raw_softmax_online_targets.csv.gz" if args.label == "dev"
                else "raw_softmax_online_targets_%s.csv.gz" % args.label)
    T.to_csv(_ext(tf), **C.GZ)
    write_csv_atomic(out / ("online_target_gate_checks.csv" if args.label == "dev"
                            else "online_target_gate_checks_%s.csv" % args.label), G)

    nz = T["delta_mode"].abs() > 1e-12
    summary = {
        "stage": C.STAGE, "label": args.label,
        "n_rows": int(len(T)), "n_datasets": int(T["dataset_id"].nunique()),
        "n_splits": int(T["split_id"].nunique()),
        "skipped_incomplete_pairs": int(skipped),
        "gates_all_pass": bool(G["all_gates_pass"].all()),
        "n_gate_failures": int((~G["all_gates_pass"]).sum()),
        "max_accidental_additional_cvis": int(max(
            G["accidental_additional_cvis_raw"].max(),
            G["accidental_additional_cvis_softmax"].max())),
        "raw_wins": int((T["delta_mode"] > 1e-12).sum()),
        "softmax_wins": int((T["delta_mode"] < -1e-12).sum()),
        "ties": int((~nz).sum()),
        "mean_delta_mode": float(T["delta_mode"].mean()),
        "median_delta_mode": float(T["delta_mode"].median()),
        "mean_abs_delta_mode": float(T["delta_mode"].abs().mean()),
        "mean_y_raw": float(T["y_raw"].mean()),
        "mean_y_softmax": float(T["y_softmax"].mean()),
        "target_hash": C.sha_file(tf),
        "target_definition": "delta_mode = y_raw - y_softmax, continuous",
        "split_1_used": False, "source_commit": commit,
        "runtime_sec": time.time() - t_all,
    }
    write_json_atomic(out / ("online_target_summary.json" if args.label == "dev"
                             else "online_target_summary_%s.json" % args.label),
                      summary)
    print("\n  %d targeted rows | gates pass: %s (%d failures)"
          % (len(T), summary["gates_all_pass"], summary["n_gate_failures"]),
          flush=True)
    print("  RAW %d / SOFTMAX %d / ties %d | mean delta %+.5f median %+.5f "
          "mean|delta| %.5f"
          % (summary["raw_wins"], summary["softmax_wins"], summary["ties"],
             summary["mean_delta_mode"], summary["median_delta_mode"],
             summary["mean_abs_delta_mode"]), flush=True)
    return 0 if summary["gates_all_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
