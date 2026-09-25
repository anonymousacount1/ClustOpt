"""Stage-1C-A scientific validation of the offline 2x3 implementation.

Runs against a SMOKE output tree (never a reportable one). Checks the invariants
that must hold before the full sweep is authorised.

Run:
    .venv_clustopt/Scripts/python.exe scripts/stage1/validate_offline_2x3.py \
        --suffix offline_fixed32_SMOKE_W1
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from scripts.stage1.verify_offline_worker_parity import ARMS, VIEWS, collect  # noqa: E402

RESULTS = []


def rec(name, ok, detail=""):
    RESULTS.append((name, "PASS" if ok else "FAIL", detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", required=True)
    ap.add_argument("--split-id", type=int, default=1)
    args = ap.parse_args()

    from models.metric_utility_mlp.head_groups import (
        METRIC_HEAD_GROUPS, resolve_target_group,
    )
    head14 = set(resolve_target_group("head14"))
    full60 = set(m for ms in METRIC_HEAD_GROUPS.values() for m in ms)
    new46 = full60 - head14

    analyzed = os.path.join(REPO, "results_analysis", "clustering_repository",
                            "analyzed_data")
    data = collect(analyzed, args.suffix, args.split_id)
    print("=" * 78)
    print("STAGE-1C-A SCIENTIFIC VALIDATION (SMOKE, NOT REPORTABLE)")
    print("=" * 78)
    print(f"  records: {len(data)}")
    if not data:
        print("  [FAIL] no records"); return 1

    ok = True
    datasets = {k[0] for k in data}
    ok &= rec("every (dataset, view) has all 6 arms",
              all(len([1 for a in ARMS if (d, v, a) in data]) == 6
                  for d in datasets for v in VIEWS),
              f"{len(datasets)} datasets x 3 views x 6 arms = {len(datasets)*18}")

    # ---- arm semantics ----------------------------------------------------
    bad = []
    for (d, v, a), r in data.items():
        sel, w = r["selected_metrics"], r["selected_weights"]
        inv, strat = r["metric_inventory"], r["utility_strategy"]
        if strat == "uniform":
            n = 14 if inv == "head14" else 60
            if len(sel) != n: bad.append(f"{a}: uniform n={len(sel)}")
            if inv == "head14" and set(sel) != head14: bad.append(f"{a}: HU set")
            if inv == "full60" and set(sel) != full60: bad.append(f"{a}: FU set")
            if any(abs(x - 1.0 / n) > 1e-12 for x in w.values()):
                bad.append(f"{a}: uniform weight != 1/{n}")
        else:
            if len(sel) != 5: bad.append(f"{a}: top-k n={len(sel)}")
        if abs(sum(w.values()) - 1.0) > 1e-9: bad.append(f"{a}: weights sum")
        if any((x is None) or x < 0 for x in w.values()): bad.append(f"{a}: neg weight")
    ok &= rec("arm semantics: sizes, sets, weights sum to 1, non-negative",
              not bad, f"{len(bad)} violations" + (f" {bad[:3]}" if bad else ""))

    # ---- inventory containment -------------------------------------------
    h_bad = [k for k, r in data.items()
             if r["metric_inventory"] == "head14"
             and not set(r["selected_metrics"]) <= head14]
    ok &= rec("Head-14 arms never select a New-46 metric", not h_bad,
              f"{len(h_bad)} violations")
    fo_new = sum(1 for k, r in data.items()
                 if k[2] == "stage1_full60_oracle_top5_raw_fixed50"
                 and set(r["selected_metrics"]) & new46)
    ok &= rec("FO is genuinely unrestricted (does reach New-46)", fo_new > 0,
              f"{fo_new} FO records include a New-46 metric")

    # ---- candidate identity ----------------------------------------------
    fp = defaultdict(set)
    cc = defaultdict(set)
    for (d, v, a), r in data.items():
        fp[(d, v)].add(r["candidate_set_fingerprint"])
        cc[(d, v)].add(r["candidate_count"])
    ok &= rec("all 6 arms saw the SAME candidate set (fingerprint)",
              all(len(s) == 1 for s in fp.values()), f"{len(fp)} (dataset,view) pairs")
    ok &= rec("exactly 32 candidates everywhere",
              all(s == {32} for s in cc.values()), f"observed {sorted(set().union(*cc.values()))}")

    # ---- ceiling is arm-independent --------------------------------------
    ceil = defaultdict(set)
    for (d, v, a), r in data.items():
        ceil[(d, v)].add(round(float(r["best_possible_ari_in_candidate_set"]), 12))
    ok &= rec("candidate ceiling identical across all 6 arms",
              all(len(s) == 1 for s in ceil.values()), f"{len(ceil)} pairs")

    # ---- regret consistency ----------------------------------------------
    rbad = [k for k, r in data.items()
            if abs((r["best_possible_ari_in_candidate_set"] - r["ari"])
                   - r["ari_regret"]) > 1e-12]
    ok &= rec("ari_regret == ceiling - selected_ari", not rbad, f"{len(rbad)} bad")
    neg = [k for k, r in data.items() if r["ari_regret"] < -1e-12]
    ok &= rec("regret is never negative", not neg, f"{len(neg)} negative")

    # ---- ARI never drives selection --------------------------------------
    # Structural evidence: the selected candidate must be an argmax of J. If ARI
    # had influenced selection, some record would select a non-maximal J.
    jbad = [k for k, r in data.items()
            if abs(r["objective_J"] - r["max_objective_J"]) > 1e-12]
    ok &= rec("selected candidate always attains max J (ARI cannot have been used)",
              not jbad, f"{len(jbad)} records select a non-maximal J")
    # Tie diagnostics. NOTE: these are descriptive only. When every tied
    # candidate shares the same ARI the comparison is vacuous -- it can neither
    # support nor refute ARI-based tie-breaking. The real guarantee that ARI
    # never influences selection is structural: offline_candidate_execution
    # computes the argmax over J alone and only reads ARI afterwards.
    tied = [r for r in data.values() if r["n_tied_at_max"] > 1]
    if tied:
        spreads = [r["tie_ari_max"] - r["tie_ari_min"] for r in tied
                   if r["tie_ari_min"] is not None and r["tie_ari_max"] is not None]
        informative = [s for s in spreads if s > 1e-12]
        print(f"       tie diagnostics: {len(tied)}/{len(data)} records tied at max J; "
              f"tie sizes {sorted({r['n_tied_at_max'] for r in tied})}")
        if informative:
            chose_max = sum(1 for r in tied
                            if r["tie_ari_max"] is not None and r["ari"] is not None
                            and abs(r["ari"] - r["tie_ari_max"]) <= 1e-12
                            and (r["tie_ari_max"] - r["tie_ari_min"]) > 1e-12)
            print(f"       {len(informative)} ties have differing ARIs; selected == "
                  f"tie ARI max in {chose_max} of them "
                  f"({chose_max}/{len(informative)} would indicate ARI leakage)")
        else:
            print("       all tied candidates share an identical ARI, so this "
                  "diagnostic is VACUOUS here (max spread 0.0); rely on the "
                  "structural argument above, not on this line")

    # ---- execution mode ---------------------------------------------------
    ok &= rec("execution_mode == offline_fixed32 on every record",
              all(r["execution_mode"] == "offline_fixed32" for r in data.values()))
    ok &= rec("oracle arms marked non-deployable",
              all(r["deployable"] is False for k, r in data.items()
                  if r["utility_strategy"] == "oracle"))
    ok &= rec("HP bound to a Head-14 predictor",
              all("head14" in (r["predictor"] or "") for k, r in data.items()
                  if k[2] == "stage1_head14_mlp_top5_raw_fixed50"))
    ok &= rec("FP bound to the frozen Full-60 predictor",
              all("20260615_231715" in (r["predictor"] or "") for k, r in data.items()
                  if k[2] == "stage1_full60_mlp_top5_raw_fixed50"))

    print()
    print("=" * 78)
    p = sum(1 for _, s, _ in RESULTS if s == "PASS")
    f = sum(1 for _, s, _ in RESULTS if s == "FAIL")
    print(f"  {p}/{len(RESULTS)} PASS, {f} FAIL")
    print(f"  VERDICT: {'IMPLEMENTATION VALIDATED' if ok else 'VALIDATION FAILED'}")
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
