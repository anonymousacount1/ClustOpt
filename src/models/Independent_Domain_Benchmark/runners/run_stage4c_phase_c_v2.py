"""Stage 4C Phase C (v2) -- targeted utility recompute for the repaired six.

Amendment ``stage4c_fixed32_metric_dispatch_repair``.

Candidate ARIs are NOT recomputed: they are read back from the original Phase-C
output, so the repaired metrics are ranked against exactly the same ground-truth
partitioning the original analysis used. Ground truth itself is never reloaded
here.

Utility is computed on a combined frame -- the 61 untouched metrics from the v1
records plus the six repaired metrics from the v2 records -- restricted to
exactly the candidate row set the original Phase C used.

That restriction is the point. The canonical utility scores each metric over a
candidate row set, so the row set is part of the measurement. Two of the
repaired metrics are valid for EVERY candidate, so an unrestricted frame would
admit candidates the original analysis never scored and would move all 61
authoritative utilities (measured: 6822 of 9102 changed). The amendment forbids
recomputing those, so the six are measured on the basis the 61 were measured on.

Given the same rows, the canonical function scores each metric independently of
which other columns are present, so the 61 must come back unchanged. That is
verified here, not assumed: every v1 utility is recomputed alongside the six and
compared against its stored value. Agreement makes composing v1's 61 with v2's 6
equivalent to having run the intended 67-metric analysis once.
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
sys.path.insert(0, str(_REPO / "experiments/external_baselines/Unified_MKR"))
warnings.filterwarnings("ignore")
import logging                                                        # noqa: E402
logging.disable(logging.INFO)

from Independent_Domain_Benchmark.execution import hashing as H        # noqa: E402
from Independent_Domain_Benchmark.execution import store as ST         # noqa: E402

RESULTS = _REPO / "results_analysis" / "independent_domain_benchmark"
CFG = _ROOT / "configs"
VIEW_IDS = ("x_only", "y_only", "xy_2d")
UTILITY_VALUE_COLUMN = "utility_score"


def main(argv=None) -> int:
    import pandas as pd
    from compute_metric_utility import compute_metric_utilities

    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args(argv)

    run_dir = RESULTS / args.run_id
    store = ST.ResultStore(RESULTS, args.run_id)
    registry = json.loads((CFG / "metric_analysis_registry.json")
                          .read_text(encoding="utf-8"))
    amend = json.loads((CFG / "stage4c_metric_dispatch_amendment.json")
                       .read_text(encoding="utf-8"))
    primary = {e["canonical_metric_id"]: e for e in registry["entries"]}
    repair = set(amend["repair_set"])
    untouched = set(primary) - repair

    ari_df = pd.read_csv(run_dir / "phase_c" / "fixed32_candidate_ari.csv.gz")
    v1_util = pd.read_csv(run_dir / "phase_c" / "metric_utility.csv.gz")
    v1_lookup = {(r.dataset_id, r.view_id, r.metric_id): r.utility
                 for r in v1_util[v1_util["status"] == "valid"].itertuples()}

    util_rows: List[Dict[str, Any]] = []
    stats = {"units": 0, "utility_valid": 0, "utility_insufficient": 0,
             "utility_errors": 0, "row_set_differs": 0,
             "untouched_compared": 0, "untouched_identical": 0,
             "untouched_deviating": 0,
             "candidates_outside_v1_row_set": 0}
    deviations: List[Dict[str, Any]] = []
    errors: List[str] = []

    for (ds, view_id), g in ari_df.groupby(["dataset_id", "view_id"]):
        body = "%s__%s" % (ds, view_id)
        v1 = store.load(ST.Layer.EVALUATION, "metrics__" + body)
        v2 = store.load(ST.Layer.EVALUATION, "metrics_v2__" + body)
        if v1 is None or v2 is None:
            errors.append("%s: missing v1=%s v2=%s" % (body, v1 is None, v2 is None))
            continue
        stats["units"] += 1
        source = g["source"].iloc[0]
        ari_by_idx = {int(r.candidate_index): float(r.ari) for r in g.itertuples()}

        by_cand: Dict[int, Dict[str, float]] = {}
        v1_row_ids: set = set()
        for rec, is_v1 in ((v1, True), (v2, False)):
            for r in rec["content"]["rows"]:
                if not r["primary"] or r["status"] != "valid":
                    continue
                mid = r["metric_id"]
                # v1 records still hold the six failed rows; ignore them entirely
                if is_v1 and mid in repair:
                    continue
                if not is_v1 and mid not in repair:
                    continue
                val = r["raw_metric_value"]
                if val is None:
                    continue
                e = primary[mid]
                oriented = val if e["orientation"] == "MAXIMIZE" else -val
                by_cand.setdefault(r["candidate_index"], {})[
                    "%s_norm" % mid] = oriented
                if is_v1:
                    v1_row_ids.add(r["candidate_index"])

        # The canonical utility is computed over a candidate row set, and its
        # per-metric score depends on that set. The two repaired New46 metrics
        # are valid for EVERY candidate, so an unrestricted combined frame would
        # admit candidates the original analysis never scored and would move the
        # 61 authoritative utilities. The amendment forbids recomputing those, so
        # the repaired six are computed on exactly the row basis v1 used. Given
        # the same rows, the canonical function scores each metric independently
        # of which other columns are present, so the 61 are reproduced exactly --
        # which the untouched_* counters below verify rather than assume.
        keep = {i for i in v1_row_ids if i in ari_by_idx}
        dropped = sorted(set(by_cand) & set(ari_by_idx) - keep)
        if dropped:
            stats["row_set_differs"] += 1
            stats["candidates_outside_v1_row_set"] += len(dropped)
        recs = [{"ARI": ari_by_idx[i], "valid": True, **cols}
                for i, cols in sorted(by_cand.items())
                if i in ari_by_idx and i in keep]

        if not recs:
            for mid in repair:
                util_rows.append({
                    "dataset_id": ds, "source": source, "view_id": view_id,
                    "metric_id": mid, "metric_group": primary[mid]["group"],
                    "utility": None, "status": "INSUFFICIENT_VALID_CANDIDATES"})
                stats["utility_insufficient"] += 1
            continue

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "u.csv"
            pd.DataFrame(recs).to_csv(p, index=False)
            try:
                u = compute_metric_utilities(str(p))
                got = {}
                for _, ur in u.iterrows():
                    got[str(ur["metric"])] = float(ur[UTILITY_VALUE_COLUMN])
            except ValueError as exc:
                if "Not enough valid rows" not in str(exc) \
                        and "ARI column not found" not in str(exc):
                    raise
                for mid in repair:
                    util_rows.append({
                        "dataset_id": ds, "source": source, "view_id": view_id,
                        "metric_id": mid, "metric_group": primary[mid]["group"],
                        "utility": None,
                        "status": "INSUFFICIENT_VALID_CANDIDATES",
                        "detail": str(exc)[:120]})
                    stats["utility_insufficient"] += 1
                continue
            except Exception as exc:  # noqa: BLE001
                stats["utility_errors"] += 1
                errors.append("%s: %s: %s" % (body, type(exc).__name__,
                                              str(exc)[:140]))
                continue

        for mid in repair:
            if mid in got:
                util_rows.append({
                    "dataset_id": ds, "source": source, "view_id": view_id,
                    "metric_id": mid, "metric_group": primary[mid]["group"],
                    "utility": got[mid], "status": "valid"})
                stats["utility_valid"] += 1
            else:
                util_rows.append({
                    "dataset_id": ds, "source": source, "view_id": view_id,
                    "metric_id": mid, "metric_group": primary[mid]["group"],
                    "utility": None, "status": "INSUFFICIENT_VALID_CANDIDATES"})
                stats["utility_insufficient"] += 1

        # integrity: the 61 untouched metrics must be unmoved by the composition
        for mid in untouched:
            if mid not in got:
                continue
            old = v1_lookup.get((ds, view_id, mid))
            if old is None:
                continue
            stats["untouched_compared"] += 1
            if np.isclose(got[mid], float(old), rtol=1e-9, atol=1e-12):
                stats["untouched_identical"] += 1
            else:
                stats["untouched_deviating"] += 1
                if len(deviations) < 20:
                    deviations.append({"dataset_id": ds, "view_id": view_id,
                                       "metric_id": mid, "v1": float(old),
                                       "recomputed": got[mid]})

    out_dir = run_dir / "phase_c_v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(util_rows).to_csv(out_dir / "metric_utility_v2.csv.gz", index=False)

    stats["amendment_id"] = amend["amendment_id"]
    stats["repair_set"] = sorted(repair)
    stats["candidate_ari_recomputed"] = False
    stats["ground_truth_reloaded"] = False
    stats["utility_source_hash"] = H.canonical_text_hash(
        (_REPO / "models/ClustOpt/external_evaluation/compute_metric_utility.py")
        .read_bytes())
    stats["untouched_deviation_examples"] = deviations
    stats["errors"] = errors[:20]
    (out_dir / "phase_c_v2_summary.json").write_text(
        json.dumps(stats, indent=2, default=str), encoding="utf-8")

    print("  units %d | repaired-metric utility valid %d / insufficient %d / errors %d"
          % (stats["units"], stats["utility_valid"], stats["utility_insufficient"],
             stats["utility_errors"]))
    print("  untouched 61 metrics: %d compared, %d identical, %d deviating"
          % (stats["untouched_compared"], stats["untouched_identical"],
             stats["untouched_deviating"]))
    print("  candidate row set differs from v1 in %d of %d units"
          % (stats["row_set_differs"], stats["units"]))
    if errors:
        print("  errors: %s" % errors[:3])
    ok = (stats["utility_errors"] == 0 and stats["untouched_deviating"] == 0
          and not errors)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
