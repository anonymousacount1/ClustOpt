"""Stage 2C-4A: freeze the three micro-pilot rows and price the batch.

Selection is deterministic and outcome-blind: the frozen sample is put in its
canonical order (``split_id``, ``rank``, ``view_id`` -- the same key the sample
hash is computed over) and the FIRST eligible row is taken for each of the three
required K categories (Top1, Top10, Dynamic). No ARI, no delta, no reranker
prediction and no runtime measurement participates in the choice, and the three
rows are ordinary members of the frozen sample, so their six searches count
towards the final batch.

``--decide`` is a separate, later invocation: it reads the six completed pilot
searches plus the pilot's target gates and turns *runtime and resources only*
into the batch size (N=12 or N=8) and the worker count. It never reads
``delta_mode``, a RAW/SOFTMAX winner or any ARI.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
from ctypes import wintypes
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from models.Clustering_Repository_Builder.experiments.experiment_execution.output_writer import (  # noqa: E402,E501
    _ext, ensure_dir, read_json, write_csv_atomic, write_json_atomic,
)
from models.ClustOpt_Policy_Predictor.stage2c import stage2c4_common as C  # noqa: E402,E501

PILOT_CATEGORIES: Sequence[str] = ("top1", "top10", "dynamic")
#: Deterministic fallback order if a category has no eligible row at all.
CATEGORY_FALLBACK: Dict[str, Sequence[str]] = {
    "top1": ("top1", "top3", "top5", "top10", "dynamic"),
    "top10": ("top10", "top5", "top3", "top1", "dynamic"),
    "dynamic": ("dynamic", "top10", "top5", "top3", "top1"),
}
#: Batch-size rule, pre-registered: N=12 unless the projection exceeds this.
PRIMARY_HOUR_LIMIT = 6.0
ABORT_HOUR_LIMIT = 8.0
FREE_MB_PER_WORKER = 450.0
FREE_MB_RESERVE = 1200.0
HISTORICAL_SEC_PER_SEARCH = 82.0


def free_mb() -> float:
    class MS(ctypes.Structure):
        _fields_ = [("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    st = MS()
    st.dwLength = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
    return st.ullAvailPhys / 2 ** 20


def canonical_order(rows: pd.DataFrame) -> pd.DataFrame:
    return rows.sort_values(["split_id", "rank", "view_id"],
                            kind="mergesort").reset_index(drop=True)


def select_pilot(rows: pd.DataFrame) -> List[Dict[str, Any]]:
    """First lexical eligible row per K category, without reuse of a row."""
    ordered = canonical_order(rows)
    chosen: List[Dict[str, Any]] = []
    used: set = set()
    for cat in PILOT_CATEGORIES:
        pick = None
        for want in CATEGORY_FALLBACK[cat]:
            cand = ordered[(ordered["k_mode"] == want)
                           & (~ordered["row_id"].isin(used))]
            if len(cand):
                pick = cand.iloc[0]
                break
        if pick is None:
            raise SystemExit("no eligible row for pilot category %s" % cat)
        used.add(pick["row_id"])
        chosen.append({
            "pilot_category": cat,
            "resolved_k_mode": pick["k_mode"],
            "category_substituted": bool(pick["k_mode"] != cat),
            "canonical_position": int(ordered.index[
                ordered["row_id"] == pick["row_id"]][0]),
            **{k: pick[k] for k in ("split_id", "rank", "dataset_id", "view_id",
                                    "family_id", "subfamily_id", "regime_id",
                                    "source", "k_mode", "row_id",
                                    "in_fallback_n8")}})
    return chosen


def cmd_freeze(repo: Path, out: Path, commit: str) -> int:
    rows = C.scientific_rows(C.load_sample(repo))
    pilot = select_pilot(rows)
    P = pd.DataFrame(pilot)
    write_csv_atomic(out / "micro_pilot_rows.csv", P)
    man = {
        "stage": C.STAGE, "sample_hash": C.FROZEN_SAMPLE_HASH,
        "selection_rule": ("first eligible row in the canonical frozen order "
                           "(split_id, rank, view_id) for each of Top1 / Top10 "
                           "/ Dynamic; a row is never reused"),
        "outcome_blind": True,
        "inputs_consulted": ["frozen sample", "frozen OOF-v2 regime per row"],
        "inputs_forbidden": ["ARI", "delta_mode", "RAW/SOFTMAX winner",
                             "reranker prediction", "runtime"],
        "rows": pilot,
        "n_rows": len(pilot), "n_searches": 2 * len(pilot),
        "counts_toward_batch": True,
        "split_1_used": False, "source_commit": commit,
    }
    write_json_atomic(out / "micro_pilot_manifest.json", man)
    print("  micro-pilot rows frozen (outcome-blind):", flush=True)
    for r in pilot:
        print("    %-8s split %02d rank %2d  %s / %s  regime=%s"
              % (r["pilot_category"], r["split_id"], r["rank"],
                 r["dataset_id"], r["view_id"], r["regime_id"]), flush=True)
    return 0


def cmd_decide(repo: Path, out: Path, commit: str, workers_max: int) -> int:
    """Runtime-only batch decision. Reads no outcome column."""
    P = pd.read_csv(_ext(out / "micro_pilot_rows.csv"))
    REC = pd.read_csv(_ext(out / "online_pair_execution_records_pilot.csv"))
    six = REC[REC["row_id"].isin(set(P["row_id"]))]
    if len(six) != 6 or not (six["status"] == "success").all():
        raise SystemExit("expected 6 successful pilot searches, got %d (%s)"
                         % (len(six), six["status"].value_counts().to_dict()))
    # NOTE: only these columns are read. ari / delta are deliberately not.
    rt = pd.to_numeric(six["runtime_sec"], errors="coerce")
    dt = pd.to_numeric(six["delivered_trials"], errors="coerce")
    gates = pd.read_csv(_ext(out / "online_target_gate_checks_pilot.csv"))
    if not bool(gates["all_gates_pass"].all()):
        raise SystemExit("pilot scientific gates failed; batch is blocked")

    tsum = read_json(out / "online_target_summary_pilot.json") or {}
    pilot_overhead_ms = 0.0
    T = pd.read_csv(_ext(out / "raw_softmax_online_targets_pilot.csv.gz"))
    for c in ("raw_feature_build_ms", "softmax_feature_build_ms",
              "raw_reranker_predict_ms", "softmax_reranker_predict_ms",
              "raw_reranker_select_ms", "softmax_reranker_select_ms"):
        pilot_overhead_ms += float(T[c].sum())
    per_search_overhead_ms = pilot_overhead_ms / 6.0

    free = free_mb()
    cpu = os.cpu_count() or 2
    mem_workers = int(max(1, (free - FREE_MB_RESERVE) // FREE_MB_PER_WORKER))
    workers = int(max(1, min(workers_max, cpu, mem_workers)))

    mean_s = float(rt.mean())
    p95_s = float(rt.quantile(0.95))
    # Conservative: price the batch at the pilot p95, not its mean.
    def project(n_searches: int, per_search: float) -> float:
        return n_searches * per_search / max(workers, 1) / 3600.0

    proj12 = project(1080, p95_s)
    proj8 = project(720, p95_s)
    proj12_mean = project(1080, mean_s)
    proj8_mean = project(720, mean_s)
    proj12_hist = project(1080, HISTORICAL_SEC_PER_SEARCH)

    if proj12 <= PRIMARY_HOUR_LIMIT:
        n, why = 12, ("p95-priced N=12 projection %.2f h <= %.1f h"
                      % (proj12, PRIMARY_HOUR_LIMIT))
    elif proj8 <= ABORT_HOUR_LIMIT:
        n, why = 8, ("p95-priced N=12 projection %.2f h exceeds %.1f h; N=8 "
                     "projects %.2f h" % (proj12, PRIMARY_HOUR_LIMIT, proj8))
    else:
        n, why = 0, ("even N=8 projects %.2f h > %.1f h" % (proj8, ABORT_HOUR_LIMIT))

    dec = {
        "stage": C.STAGE, "decision_inputs": ["search runtime", "delivered trials",
                                              "free RAM", "cpu count",
                                              "reranker overhead"],
        "decision_inputs_forbidden_and_unread": [
            "selected ARI", "delta_mode", "RAW/SOFTMAX winner", "effect size"],
        "pilot_searches": 6,
        "search_runtime_sec": {"mean": mean_s, "median": float(rt.median()),
                               "p95": p95_s, "max": float(rt.max()),
                               "min": float(rt.min())},
        "delivered_trials": {"mean": float(dt.mean()), "min": int(dt.min()),
                             "max": int(dt.max()),
                             "n_below_requested": int((dt < C.N_TRIALS).sum())},
        "timeout_incidence": float((dt < C.N_TRIALS).mean()),
        "requested_trials": C.N_TRIALS, "timeout_sec": C.TIMEOUT_SEC,
        "historical_reference_sec_per_search": HISTORICAL_SEC_PER_SEARCH,
        "reranker_overhead_ms_per_search": per_search_overhead_ms,
        "resources": {"cpu_count": cpu, "free_mb": free,
                      "free_mb_reserve": FREE_MB_RESERVE,
                      "free_mb_per_worker_budget": FREE_MB_PER_WORKER,
                      "memory_allowed_workers": mem_workers,
                      "workers_max_allowed": workers_max},
        "workers": workers,
        "projection_hours": {
            "n12_at_p95": proj12, "n8_at_p95": proj8,
            "n12_at_mean": proj12_mean, "n8_at_mean": proj8_mean,
            "n12_at_historical_82s": proj12_hist},
        "primary_rule": "N=12 if the projected batch is <= ~6 h, else N=8",
        "abort_rule": "block if even N=8 projects > ~8 h",
        "n_selected": n, "justification": why,
        "searches_selected": 1080 if n == 12 else (720 if n == 8 else 0),
        "pilot_searches_count_toward_batch": True,
        "source_commit": commit,
    }
    write_json_atomic(out / "micro_pilot_runtime_decision.json", dec)
    print("  pilot runtime: mean %.1fs median %.1fs p95 %.1fs max %.1fs"
          % (mean_s, rt.median(), p95_s, rt.max()), flush=True)
    print("  delivered trials mean %.1f (min %d) | timeout incidence %.2f"
          % (dt.mean(), int(dt.min()), dec["timeout_incidence"]), flush=True)
    print("  free %.0f MB, cpu %d -> workers %d" % (free, cpu, workers), flush=True)
    print("  projections: N=12 %.2f h (p95) / %.2f h (mean); N=8 %.2f h (p95)"
          % (proj12, proj12_mean, proj8), flush=True)
    print("  DECISION: N=%d  (%s)" % (n, why), flush=True)
    return 0 if n else 3


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", default=".")
    p.add_argument("--decide", action="store_true")
    p.add_argument("--workers-max", type=int, default=8)
    args = p.parse_args(argv)
    repo = Path(args.repo_root).resolve()
    out = ensure_dir(repo / C.OUT_REL)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                            capture_output=True, text=True).stdout.strip()
    print("=" * 78)
    print("STAGE 2C-4A  MICRO-PILOT %s"
          % ("RUNTIME DECISION" if args.decide else "ROW FREEZE"))
    print("=" * 78, flush=True)
    return (cmd_decide(repo, out, commit, int(args.workers_max)) if args.decide
            else cmd_freeze(repo, out, commit))


if __name__ == "__main__":
    raise SystemExit(main())
