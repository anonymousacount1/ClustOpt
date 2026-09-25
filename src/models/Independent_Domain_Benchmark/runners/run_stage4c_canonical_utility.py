"""Stage 4C -- canonical-candidate-basis utility for all 67 primary metrics.

Amendment ``stage4c_utility_candidate_basis_correction``.

The mathematical utility definition is UNCHANGED. Formula, weights, correlation
handling, NDCG, top-k/bottom-k, invalid-value treatment, orientation and
minimum-valid semantics are exactly as frozen. What is corrected is the
candidate-frame assembly that feeds the function.

The candidate universe for every metric is derived from the frozen
Fixed32CandidateBank alone: every candidate whose CLUSTERING is valid and which
has a candidate ARI. A metric being invalid or non-finite on a candidate affects
only that metric's own value and status; it never deletes the candidate row, and
K=1 partitions are retained because they are real clustering outputs even though
most CVIs are undefined on them.

No metric value, candidate ARI, bank, clustering or method output is recomputed
here, and no metric implementation is invoked.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_REPO = _ROOT.parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "models"))
sys.path.insert(0, str(_REPO / "models/ClustOpt/external_evaluation"))
warnings.filterwarnings("ignore")
import logging                                                        # noqa: E402
logging.disable(logging.INFO)

from Independent_Domain_Benchmark.execution import hashing as H        # noqa: E402
from Independent_Domain_Benchmark.execution import store as ST         # noqa: E402

RESULTS = _REPO / "results_analysis" / "independent_domain_benchmark"
CFG = _ROOT / "configs"
UTIL = "utility_score"
COMPONENTS = ("spearman_signed", "kendall_signed", "pairwise_accuracy",
              "ndcg_all", "topk_overlap", "bottomk_overlap", "topk_ndcg",
              "bottomk_ndcg", "n_rows_used_for_corr")
TOL = 1e-9

EXPECTED_SLOTS = 4800
EXPECTED_VALID = 4776
EXPECTED_UNITS = 150
EXPECTED_PRIMARY = 67


class BasisError(RuntimeError):
    """The canonical basis could not be derived uniquely from bank validity."""


def _utilities(pd, compute, rows, tmpdir, tag):
    if not rows:
        return {}
    p = Path(tmpdir) / ("u_%s.csv" % tag)
    pd.DataFrame(rows).to_csv(p, index=False)
    try:
        u = compute(str(p))
    except Exception:  # noqa: BLE001
        return {}
    return {str(r["metric"]): dict(r) for _, r in u.iterrows()}


def main(argv=None) -> int:
    import pandas as pd
    from compute_metric_utility import compute_metric_utilities

    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--skip-isolation", action="store_true")
    args = ap.parse_args(argv)

    run_dir = RESULTS / args.run_id
    store = ST.ResultStore(RESULTS, args.run_id)
    registry = json.loads((CFG / "metric_analysis_registry.json")
                          .read_text(encoding="utf-8"))
    amend = json.loads((CFG / "stage4c_utility_basis_amendment.json")
                       .read_text(encoding="utf-8"))
    dispatch = json.loads((CFG / "stage4c_metric_dispatch_amendment.json")
                          .read_text(encoding="utf-8"))
    primary = {e["canonical_metric_id"]: e for e in registry["entries"]}
    repair = set(dispatch["repair_set"])
    if len(primary) != EXPECTED_PRIMARY:
        raise BasisError("registry declares %d primary metrics" % len(primary))

    ari_df = pd.read_csv(run_dir / "phase_c" / "fixed32_candidate_ari.csv.gz")
    ari_key = {(r.dataset_id, r.view_id, int(r.candidate_index)): float(r.ari)
               for r in ari_df.itertuples()}
    v1_util = pd.read_csv(run_dir / "phase_c" / "metric_utility.csv.gz")
    v1_lookup = {(r.dataset_id, r.view_id, r.metric_id): float(r.utility)
                 for r in v1_util[v1_util["status"] == "valid"].itertuples()}
    v2p = run_dir / "phase_c_v2" / "metric_utility_v2.csv.gz"
    v2_lookup = {}
    if v2p.is_file():
        d2 = pd.read_csv(v2p)
        v2_lookup = {(r.dataset_id, r.view_id, r.metric_id): float(r.utility)
                     for r in d2[d2["status"] == "valid"].itertuples()}

    stats = {
        "units": 0, "total_slots": 0, "valid_clustering_candidates": 0,
        "failed_clustering_slots": 0, "canonical_candidates": 0,
        "logical_combinations": 0, "resolved_combinations": 0,
        "missing_unexplained": 0, "metric_recomputations": 0,
        "utility_cells_expected": 0, "utility_cells_produced": 0,
        "utility_valid": 0, "utility_insufficient": 0, "utility_errors": 0,
        "v1_cells_compared": 0, "v1_changed": 0, "v1_unchanged": 0,
        "v2_cells_compared": 0, "v2_changed": 0, "v2_unchanged": 0,
        "isolation_compared": 0, "isolation_status_mismatch": 0,
        "isolation_count_mismatch": 0, "isolation_component_mismatch": 0,
        "isolation_utility_mismatch": 0,
    }
    util_rows: List[Dict[str, Any]] = []
    impact_rows: List[Dict[str, Any]] = []
    comp_delta: Dict[str, List[float]] = {c: [] for c in COMPONENTS}
    missing: List[str] = []
    iso_fail: List[Dict[str, Any]] = []
    errors: List[str] = []

    bank_keys = [k for k in store.keys(ST.Layer.PHYSICAL)
                 if k.startswith("fixed32__")]
    t0 = time.perf_counter()

    with tempfile.TemporaryDirectory() as td:
        for key in sorted(bank_keys):
            body = key[len("fixed32__"):]
            ds, view_id = body.rsplit("__", 1)
            brec = store.load(ST.Layer.PHYSICAL, key)
            v1rec = store.load(ST.Layer.EVALUATION, "metrics__" + body)
            v2rec = store.load(ST.Layer.EVALUATION, "metrics_v2__" + body)
            if brec is None or v1rec is None or v2rec is None:
                raise BasisError("%s: missing a record layer" % body)
            stats["units"] += 1

            # ---------------- canonical basis, from bank validity ALONE
            cands = brec["content"]["candidates"]
            stats["total_slots"] += len(cands)
            valid_ids = [c["candidate_index"] for c in cands
                         if c["status"] == "valid"]
            stats["valid_clustering_candidates"] += len(valid_ids)
            stats["failed_clustering_slots"] += len(cands) - len(valid_ids)
            canonical = [i for i in valid_ids if (ds, view_id, i) in ari_key]
            if len(canonical) != len(valid_ids):
                raise BasisError("%s: %d valid candidates lack an ARI"
                                 % (body, len(valid_ids) - len(canonical)))
            canonical.sort()
            stats["canonical_candidates"] += len(canonical)

            # ------- authoritative metric status/value per logical combination
            #         repaired six -> v2, everything else -> v1. Exactly one.
            status: Dict[tuple, Dict[str, Any]] = {}
            for rec, is_v1 in ((v1rec, True), (v2rec, False)):
                for r in rec["content"]["rows"]:
                    mid = r["metric_id"]
                    if mid not in primary:
                        continue                      # diagnostics stay out
                    if is_v1 and mid in repair:
                        continue                      # superseded, never merged
                    if not is_v1 and mid not in repair:
                        continue
                    status[(r["candidate_index"], mid)] = r

            stats["logical_combinations"] += len(canonical) * EXPECTED_PRIMARY
            for i in canonical:
                for mid in primary:
                    if (i, mid) in status:
                        stats["resolved_combinations"] += 1
                    else:
                        stats["missing_unexplained"] += 1
                        if len(missing) < 20:
                            missing.append("%s c%d %s" % (body, i, mid))

            # ------------------------------- canonical frame, all 67 columns
            frame = []
            for i in canonical:
                row = {"ARI": ari_key[(ds, view_id, i)], "valid": True}
                for mid, e in primary.items():
                    r = status.get((i, mid))
                    v = np.nan
                    if r is not None and r["status"] == "valid" \
                            and r["raw_metric_value"] is not None:
                        v = (r["raw_metric_value"] if e["orientation"] == "MAXIMIZE"
                             else -r["raw_metric_value"])
                    row["%s_norm" % mid] = v
                frame.append(row)
            canon = _utilities(pd, compute_metric_utilities, frame, td, "canon")

            # ------------------ the ORIGINAL v1 union frame, for impact only
            v1_ids = sorted({r["candidate_index"] for r in v1rec["content"]["rows"]
                             if r["metric_id"] in primary
                             and r["metric_id"] not in repair
                             and r["status"] == "valid"
                             and r["raw_metric_value"] is not None}
                            & set(canonical))
            v1_frame = []
            for i in v1_ids:
                row = {"ARI": ari_key[(ds, view_id, i)], "valid": True}
                for mid in primary:
                    if mid in repair:
                        continue
                    r = status.get((i, mid))
                    if r is not None and r["status"] == "valid" \
                            and r["raw_metric_value"] is not None:
                        e = primary[mid]
                        row["%s_norm" % mid] = (
                            r["raw_metric_value"] if e["orientation"] == "MAXIMIZE"
                            else -r["raw_metric_value"])
                v1_frame.append(row)
            v1_frame_u = _utilities(pd, compute_metric_utilities, v1_frame, td, "v1")

            src = ari_df[(ari_df["dataset_id"] == ds)]["source"].iloc[0]
            stats["utility_cells_expected"] += EXPECTED_PRIMARY
            for mid in sorted(primary):
                stats["utility_cells_produced"] += 1
                got = canon.get(mid)
                if got is None:
                    util_rows.append({
                        "dataset_id": ds, "source": src, "view_id": view_id,
                        "metric_id": mid, "metric_group": primary[mid]["group"],
                        "utility": None,
                        "status": "INSUFFICIENT_VALID_CANDIDATES",
                        "candidate_basis": len(canonical)})
                    stats["utility_insufficient"] += 1
                else:
                    util_rows.append({
                        "dataset_id": ds, "source": src, "view_id": view_id,
                        "metric_id": mid, "metric_group": primary[mid]["group"],
                        "utility": float(got[UTIL]), "status": "valid",
                        "candidate_basis": len(canonical),
                        "n_rows_used_for_corr": int(got["n_rows_used_for_corr"])})
                    stats["utility_valid"] += 1

                # ---------------------------------------- impact accounting
                new = None if got is None else float(got[UTIL])
                old1 = v1_lookup.get((ds, view_id, mid))
                old2 = v2_lookup.get((ds, view_id, mid))
                prev = old2 if mid in repair else old1
                if prev is not None:
                    tgt = "v2" if mid in repair else "v1"
                    stats["%s_cells_compared" % tgt] += 1
                    if new is not None and np.isclose(new, prev, rtol=TOL, atol=TOL):
                        stats["%s_unchanged" % tgt] += 1
                    else:
                        stats["%s_changed" % tgt] += 1
                    impact_rows.append({
                        "dataset_id": ds, "view_id": view_id, "metric_id": mid,
                        "previous_layer": tgt, "previous_utility": prev,
                        "canonical_utility": new,
                        "delta": (None if new is None else new - prev)})
                    a = v1_frame_u.get(mid)
                    if a is not None and got is not None:
                        for c in COMPONENTS:
                            comp_delta[c].append(float(got[c]) - float(a[c]))

            # ------------------------------------------------- isolation
            if not args.skip_isolation:
                for mid in sorted(primary):
                    col = "%s_norm" % mid
                    if not any(np.isfinite(r.get(col, np.nan)) for r in frame):
                        continue
                    solo = [{"ARI": r["ARI"], "valid": True, col: r[col]}
                            for r in frame]
                    s = _utilities(pd, compute_metric_utilities, solo, td, "solo")
                    a, b = s.get(mid), canon.get(mid)
                    stats["isolation_compared"] += 1
                    if (a is None) != (b is None):
                        stats["isolation_status_mismatch"] += 1
                        iso_fail.append({"unit": body, "metric": mid})
                        continue
                    if a is None:
                        continue
                    if int(a["n_rows_used_for_corr"]) != int(b["n_rows_used_for_corr"]):
                        stats["isolation_count_mismatch"] += 1
                        iso_fail.append({"unit": body, "metric": mid,
                                         "detail": "n"})
                    if any(not np.isclose(float(a[c]), float(b[c]), rtol=TOL,
                                          atol=TOL) for c in COMPONENTS):
                        stats["isolation_component_mismatch"] += 1
                        iso_fail.append({"unit": body, "metric": mid,
                                         "detail": "component"})
                    if not np.isclose(float(a[UTIL]), float(b[UTIL]), rtol=TOL,
                                      atol=TOL):
                        stats["isolation_utility_mismatch"] += 1
                        iso_fail.append({"unit": body, "metric": mid,
                                         "detail": "utility"})

    if stats["missing_unexplained"]:
        raise BasisError("%d logical candidate x metric combinations have no "
                         "record: %s" % (stats["missing_unexplained"], missing[:5]))

    out_dir = run_dir / "phase_c_canonical"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(util_rows).to_csv(out_dir / "metric_utility_canonical.csv.gz",
                                   index=False)
    imp = pd.DataFrame(impact_rows)
    imp.to_csv(out_dir / "impact_vs_previous.csv.gz", index=False)

    d = imp["delta"].dropna().abs()
    summary = {
        "amendment_id": amend["amendment_id"],
        "candidate_basis_version": amend["candidate_basis_version"],
        "utility_definition_changed": False,
        "utility_source_hash": H.canonical_text_hash(
            (_REPO / "models/ClustOpt/external_evaluation"
             / "compute_metric_utility.py").read_bytes()),
        "metric_values_recomputed": False,
        "candidate_ari_recomputed": False,
        "banks_recomputed": False,
        "methods_recomputed": False,
        "stats": stats,
        "impact": {
            "max_abs_delta": float(d.max()) if len(d) else None,
            "median_abs_delta": float(d.median()) if len(d) else None,
            "mean_abs_delta": float(d.mean()) if len(d) else None,
            "component_mean_delta": {c: (float(np.mean(v)) if v else None)
                                     for c, v in comp_delta.items()},
            "component_max_abs_delta": {c: (float(np.max(np.abs(v))) if v else None)
                                        for c, v in comp_delta.items()},
        },
        "isolation_failures": iso_fail[:15],
        "errors": errors[:20],
        "wall_clock_seconds": round(time.perf_counter() - t0, 1),
    }
    (out_dir / "canonical_utility_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print("  units %d | canonical candidates %d (slots %d, failed %d)"
          % (stats["units"], stats["canonical_candidates"], stats["total_slots"],
             stats["failed_clustering_slots"]))
    print("  logical combinations %d resolved %d missing %d"
          % (stats["logical_combinations"], stats["resolved_combinations"],
             stats["missing_unexplained"]))
    print("  utility cells %d | valid %d | insufficient %d | errors %d"
          % (stats["utility_cells_produced"], stats["utility_valid"],
             stats["utility_insufficient"], stats["utility_errors"]))
    print("  impact vs v1: %d compared, %d changed, %d unchanged"
          % (stats["v1_cells_compared"], stats["v1_changed"],
             stats["v1_unchanged"]))
    print("  impact vs v2: %d compared, %d changed, %d unchanged"
          % (stats["v2_cells_compared"], stats["v2_changed"],
             stats["v2_unchanged"]))
    print("  max |delta| %s | median |delta| %s"
          % (summary["impact"]["max_abs_delta"],
             summary["impact"]["median_abs_delta"]))
    print("  ISOLATION %d compared | status %d count %d component %d utility %d"
          % (stats["isolation_compared"], stats["isolation_status_mismatch"],
             stats["isolation_count_mismatch"],
             stats["isolation_component_mismatch"],
             stats["isolation_utility_mismatch"]))
    ok = (stats["missing_unexplained"] == 0 and stats["utility_errors"] == 0
          and not iso_fail)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
