"""Stage 4C Phase C — open the offline ground-truth boundary.

Runs only after Phase A and Phase B are COMPLETE. This is the first and only
point in Stage 4C where ``y_true`` is loaded.

It computes, from persisted data alone:

* method ARI for every successful scientific arm (from the stored final labels);
* fixed32 candidate ARI for every valid bank candidate (from the stored labels);
* the canonical Stage-3B metric utility per dataset x view x metric.

It never calls method inference, search, scoring or bank construction: every
scientific execution path is stubbed to raise for the duration, so a Phase-C run
that needed to re-execute anything would fail loudly rather than quietly
recompute. A failed method output keeps its failure status and is never given
ARI = 0 or any other sentinel.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_REPO = _ROOT.parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "models"))
sys.path.insert(0, str(_REPO / "experiments/external_baselines/Unified_MKR"))
warnings.filterwarnings("ignore")
import logging                                                        # noqa: E402
logging.disable(logging.INFO)

from Independent_Domain_Benchmark.execution import hashing as H        # noqa: E402
from Independent_Domain_Benchmark.execution import store as ST         # noqa: E402

RESULTS = _REPO / "results_analysis" / "independent_domain_benchmark"
CFG = _ROOT / "configs"
VIEW_IDS = ("x_only", "y_only", "xy_2d")

# compute_metric_utility returns its score in "utility_score"; there is no
# "utility" column. Reading the wrong name silently yielded None for every
# metric, and the old broad except reported that as
# INSUFFICIENT_VALID_CANDIDATES -- an empty metric track that looked like
# sparse data rather than a bug.
UTILITY_VALUE_COLUMN = "utility_score"


class UtilityContractError(RuntimeError):
    "The canonical utility function did not return its documented shape."


def _driver():
    spec = importlib.util.spec_from_file_location(
        "r4c", str(_ROOT / "runners" / "run_stage4c_benchmark.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class _ExecutionGuard:
    """Make every scientific execution path raise for the duration of Phase C."""

    def __enter__(self):
        from Independent_Domain_Benchmark.analysis import fixed32_track as F32
        from Independent_Domain_Benchmark.methods import autoclust as AC
        from Independent_Domain_Benchmark.methods import autoclust_scoring as AS
        from Independent_Domain_Benchmark.methods import ml2dac as ML
        from Independent_Domain_Benchmark.methods import ml2dac_scoring as MS
        self.calls = {"n": 0}

        def boom(*a, **k):
            self.calls["n"] += 1
            raise AssertionError("scientific execution invoked during Phase C")

        # run_configuration is the actual clustering entry point: without it
        # stubbed the guard would let a fitted estimator slip past the adapter
        # layer it does stub.
        from unified_mkr import clustering as CL
        self._mods = [(AC, "build_autoclust_candidate_trace"),
                      (ML, "build_ml2dac_candidate_trace"),
                      (AS, "score_trace"), (MS, "score_trace"),
                      (F32, "build_fixed32_bank"),
                      (F32, "evaluate_bank_metrics"),
                      (CL, "run_configuration")]
        self._keep = [(m, n, getattr(m, n)) for m, n in self._mods]
        for m, n in self._mods:
            setattr(m, n, boom)
        return self

    def __exit__(self, *exc):
        for m, n, f in self._keep:
            setattr(m, n, f)
        return False


def _ground_truth(entry, D) -> np.ndarray:
    """Loaded here and nowhere else in Stage 4C."""
    _X2, y = D.materialise(entry)
    return np.asarray(y)


def main(argv=None) -> int:
    from sklearn.metrics import adjusted_rand_score
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--limit-datasets", type=int, default=0)
    args = ap.parse_args(argv)

    D = _driver()
    entries = [e for e in D._manifest()["datasets"] if e.get("accepted", True)]
    if args.limit_datasets:
        entries = entries[:args.limit_datasets]
    man = {e["dataset_id"]: e for e in entries}
    store = ST.ResultStore(RESULTS, args.run_id)
    registry = json.loads((CFG / "metric_analysis_registry.json")
                          .read_text(encoding="utf-8"))
    primary = {e["canonical_metric_id"]: e for e in registry["entries"]}

    method_rows: List[Dict[str, Any]] = []
    fixed32_rows: List[Dict[str, Any]] = []
    util_rows: List[Dict[str, Any]] = []
    stats = {"method_success": 0, "method_failed": 0, "method_ari": 0,
             "fixed32_candidates": 0, "fixed32_ari": 0,
             "utility_valid": 0, "utility_insufficient": 0,
             "utility_errors": 0, "success_without_usable_labels": 0,
             "execution_calls_during_phase_c": 0}
    unknown_metric_ids: set = set()
    utility_error_detail: List[str] = []

    t0 = time.perf_counter()
    with _ExecutionGuard() as guard:
        for ds in sorted(man):
            y = _ground_truth(man[ds], D)
            # ---- method track
            for v in VIEW_IDS:
                for k in store.keys(ST.Layer.SCIENTIFIC):
                    if not k.startswith("%s__%s__" % (ds, v)):
                        continue
                    rec = store.load(ST.Layer.SCIENTIFIC, k)
                    c, pr = rec["content"], rec["provenance"]
                    row = {"dataset_id": ds, "source": man[ds]["source"],
                           "view_id": v, "method_id": pr["method_id"],
                           "status": c["status"],
                           "failure_kind": c.get("failure_kind"),
                           "result_origin": c.get("result_origin"),
                           "selected_algorithm": c.get("selected_algorithm"),
                           "n_evaluations": c.get("n_evaluations"),
                           "method_runtime_equivalent":
                               (c.get("runtime") or {}).get("method_runtime_equivalent"),
                           "physical_trace_id": c.get("physical_trace_id"),
                           "ari": None}
                    if c["status"] == "success":
                        stats["method_success"] += 1
                        lab = c.get("final_labels")
                        if lab is not None and len(lab) == len(y):
                            row["ari"] = float(adjusted_rand_score(
                                y, np.asarray(lab, dtype=np.int64)))
                            stats["method_ari"] += 1
                        else:
                            # a success that cannot be scored is a defect, not a
                            # quiet blank: count it so it cannot pass unnoticed
                            stats["success_without_usable_labels"] += 1
                            row["ari_unavailable_reason"] = (
                                "no final_labels" if lab is None
                                else "length %d != %d" % (len(lab), len(y)))
                    else:
                        stats["method_failed"] += 1
                    method_rows.append(row)

            # ---- fixed32 track
            for v in VIEW_IDS:
                key = "fixed32__%s__%s" % (ds, v)
                brec = store.load(ST.Layer.PHYSICAL, key)
                mrec = store.load(ST.Layer.EVALUATION, "metrics__%s__%s" % (ds, v))
                if brec is None or mrec is None:
                    continue
                npz = RESULTS / args.run_id / "labels" / ("%s.npz" % key)
                labs = np.load(npz) if npz.is_file() else {}
                ari_by_idx: Dict[int, float] = {}
                for cand in brec["content"]["candidates"]:
                    stats["fixed32_candidates"] += 1
                    if cand["status"] != "valid":
                        continue
                    nm = "c%d" % cand["candidate_index"]
                    if nm not in labs:
                        continue
                    a = float(adjusted_rand_score(y, np.asarray(labs[nm])))
                    ari_by_idx[cand["candidate_index"]] = a
                    stats["fixed32_ari"] += 1
                    fixed32_rows.append({
                        "dataset_id": ds, "source": man[ds]["source"], "view_id": v,
                        "bank_id": brec["content"]["bank_id"],
                        "candidate_index": cand["candidate_index"],
                        "candidate_id": cand["candidate_id"],
                        "algorithm": cand["algorithm"], "ari": a})
                # ---- canonical utility on this bank
                by_cand: Dict[int, Dict[str, float]] = {}
                for r in mrec["content"]["rows"]:
                    if not r["primary"] or r["status"] != "valid":
                        continue
                    val = r["raw_metric_value"]
                    if val is None:
                        continue
                    e = primary[r["metric_id"]]
                    oriented = val if e["orientation"] == "MAXIMIZE" else -val
                    by_cand.setdefault(r["candidate_index"], {})[
                        "%s_norm" % r["metric_id"]] = oriented
                recs = [{"ARI": ari_by_idx[i], "valid": True, **cols}
                        for i, cols in sorted(by_cand.items()) if i in ari_by_idx]
                if not recs:
                    continue
                import pandas as pd
                import tempfile
                sys.path.insert(0, str(_REPO / "models" / "ClustOpt"
                                       / "external_evaluation"))
                from compute_metric_utility import compute_metric_utilities
                with tempfile.TemporaryDirectory() as td:
                    p = Path(td) / "u.csv"
                    pd.DataFrame(recs).to_csv(p, index=False)
                    try:
                        u = compute_metric_utilities(str(p))
                        missing_cols = [c for c in ("metric", UTILITY_VALUE_COLUMN)
                                        if c not in u.columns]
                        if missing_cols:
                            raise UtilityContractError(
                                "utility frame lacks %s; got %s"
                                % (missing_cols, sorted(u.columns)[:8]))
                        got = set()
                        for _, ur in u.iterrows():
                            mid = str(ur["metric"])
                            got.add(mid)
                            if mid not in primary:
                                unknown_metric_ids.add(mid)
                                continue
                            util_rows.append({
                                "dataset_id": ds, "source": man[ds]["source"],
                                "view_id": v, "metric_id": mid,
                                "metric_group": primary[mid]["group"],
                                "utility": float(ur[UTILITY_VALUE_COLUMN]),
                                "status": "valid"})
                            stats["utility_valid"] += 1
                        for mid in primary:
                            if mid not in got:
                                util_rows.append({
                                    "dataset_id": ds, "source": man[ds]["source"],
                                    "view_id": v, "metric_id": mid,
                                    "metric_group": primary[mid]["group"],
                                    "utility": None,
                                    "status": "INSUFFICIENT_VALID_CANDIDATES"})
                                stats["utility_insufficient"] += 1
                    except ValueError as exc:
                        # the documented sparsity guard: fewer than 10 usable
                        # candidates cannot support a rank-correlation utility
                        if "Not enough valid rows" not in str(exc) \
                                and "ARI column not found" not in str(exc):
                            raise
                        for mid in primary:
                            util_rows.append({
                                "dataset_id": ds, "source": man[ds]["source"],
                                "view_id": v, "metric_id": mid,
                                "metric_group": primary[mid]["group"],
                                "utility": None,
                                "status": "INSUFFICIENT_VALID_CANDIDATES",
                                "detail": str(exc)[:120]})
                            stats["utility_insufficient"] += 1
                    except Exception as exc:  # noqa: BLE001
                        # NOT sparsity. A defect here must stay visible instead
                        # of being reported as a thin candidate bank.
                        stats["utility_errors"] += 1
                        utility_error_detail.append(
                            "%s/%s: %s: %s" % (ds, v, type(exc).__name__,
                                               str(exc)[:160]))
                        for mid in primary:
                            util_rows.append({
                                "dataset_id": ds, "source": man[ds]["source"],
                                "view_id": v, "metric_id": mid,
                                "metric_group": primary[mid]["group"],
                                "utility": None,
                                "status": "UTILITY_COMPUTATION_ERROR",
                                "detail": "%s: %s" % (type(exc).__name__,
                                                      str(exc)[:120])})
            print("   [C] %-26s method_ari=%d f32_ari=%d util=%d"
                  % (ds, stats["method_ari"], stats["fixed32_ari"],
                     stats["utility_valid"]), flush=True)
        stats["execution_calls_during_phase_c"] = guard.calls["n"]

    out_dir = RESULTS / args.run_id / "phase_c"
    out_dir.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    pd.DataFrame(method_rows).to_csv(out_dir / "method_ari.csv.gz", index=False)
    pd.DataFrame(fixed32_rows).to_csv(out_dir / "fixed32_candidate_ari.csv.gz",
                                      index=False)
    pd.DataFrame(util_rows).to_csv(out_dir / "metric_utility.csv.gz", index=False)
    stats["wall_clock_seconds"] = round(time.perf_counter() - t0, 1)
    stats["run_id"] = args.run_id
    stats["unknown_metric_ids"] = sorted(unknown_metric_ids)
    stats["utility_error_detail"] = utility_error_detail[:20]
    stats["utility_value_column"] = UTILITY_VALUE_COLUMN
    stats["utility_source_hash"] = H.canonical_text_hash(
        (_REPO / "models/ClustOpt/external_evaluation/compute_metric_utility.py")
        .read_bytes())
    (out_dir / "phase_c_summary.json").write_text(
        json.dumps(stats, indent=2, default=str), encoding="utf-8")
    print("  method ARI %d | fixed32 ARI %d | utility valid %d / insufficient %d"
          " / errors %d | execution calls %d"
          % (stats["method_ari"], stats["fixed32_ari"], stats["utility_valid"],
             stats["utility_insufficient"], stats["utility_errors"],
             stats["execution_calls_during_phase_c"]))
    if unknown_metric_ids:
        print("  WARNING unknown metric ids from the utility function: %s"
              % sorted(unknown_metric_ids)[:6])
    if stats["success_without_usable_labels"]:
        print("  WARNING %d successful arms had no usable label vector"
              % stats["success_without_usable_labels"])
    if stats["utility_errors"]:
        print("  WARNING %d utility computation errors (NOT sparsity): %s"
              % (stats["utility_errors"], utility_error_detail[:3]))
    ok = (not stats["utility_errors"] and not unknown_metric_ids
          and not stats["success_without_usable_labels"])
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
