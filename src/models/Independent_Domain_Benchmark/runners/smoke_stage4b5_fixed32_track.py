"""Stage 4B-5 internal smoke — the fixed32 metric-analysis track end to end.

INTERNAL / NON_SCIENTIFIC data only. The frozen external corpus is never touched.

One dataset x three views -> 3 fixed32 banks, 96 candidate slots, all 67 primary
CVIs plus the 2 diagnostics on the SAME partitions, then the offline boundary is
crossed once to compute candidate ARI and the canonical Stage-3B utility.
"""
from __future__ import annotations

import json
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_REPO = _ROOT.parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "models"))
sys.path.insert(0, str(_REPO / "experiments/external_baselines/Unified_MKR"))
warnings.filterwarnings("ignore")

from Independent_Domain_Benchmark.analysis import fixed32_track as F32   # noqa: E402
from Independent_Domain_Benchmark.execution.dataset_view import (        # noqa: E402
    build_dataset_view)

SMOKE = "NON_SCIENTIFIC_SMOKE_TEST"
VIEWS = ("x_only", "y_only", "xy_2d")
UTILITY_SRC = "models/ClustOpt/external_evaluation/compute_metric_utility.py"


def fixture(n=90, seed=20260913):
    """Internal blobs plus their INTERNAL ground truth, kept strictly apart."""
    r = np.random.default_rng(seed)
    X = np.vstack([r.normal([0, 0], .40, (n, 2)), r.normal([5, 0], .40, (n, 2)),
                   r.normal([2.5, 5], .40, (n, 2))])
    y = np.repeat([0, 1, 2], n)
    return X, y


def main() -> int:
    rows = []

    def chk(name, ok, detail=""):
        rows.append({"test": name, "status": "PASS" if ok else "FAIL",
                     "detail": str(detail)[:300]})
        print("  %-44s %s  %s" % (name, rows[-1]["status"], str(detail)[:58]),
              flush=True)

    registry = json.loads(
        (_ROOT / "configs" / "metric_analysis_registry.json").read_text(
            encoding="utf-8"))
    X_full, y_true = fixture()
    views = [build_dataset_view(dataset_id="INVENTED_%s_F32" % SMOKE,
                                source="internal_non_scientific_fixture",
                                view_id=v, X_full=X_full) for v in VIEWS]

    chk("registry_67_primary_2_diagnostics",
        registry["primary_count"] == 67 and registry["diagnostic_count"] == 2,
        "%d primary / %d diagnostics" % (registry["primary_count"],
                                         registry["diagnostic_count"]))
    chk("registry_groups_7_10_46_4",
        registry["group_counts"] == {"Original": 7, "Established": 10,
                                     "New46": 46, "Modern": 4},
        json.dumps(registry["group_counts"]))
    chk("both_historically_dead_new46_retained",
        set(registry["historically_dead_new46_retained"])
        == set(F32.HISTORICALLY_DEAD_NEW46)
        and all(any(e["canonical_metric_id"] == m for e in registry["entries"])
                for m in F32.HISTORICALLY_DEAD_NEW46),
        list(F32.HISTORICALLY_DEAD_NEW46))

    # ---- label-free bank construction, with the ground-truth array unreachable
    banks, gt_calls = {}, {"n": 0}
    import unified_mkr.view_loader as VL
    keep = {n: getattr(VL, n) for n in dir(VL)
            if n.startswith(("load", "build")) and callable(getattr(VL, n))}

    def boom(*a, **k):
        gt_calls["n"] += 1
        raise AssertionError("ground-truth loader invoked during fixed32 generation")

    for n in keep:
        setattr(VL, n, boom)
    try:
        for v in views:
            banks[v.view_id] = F32.build_fixed32_bank(v)
        built, err = True, ""
    except Exception as exc:  # noqa: BLE001
        built, err = False, "%s: %s" % (type(exc).__name__, str(exc)[:160])
    finally:
        for n, f in keep.items():
            setattr(VL, n, f)
    chk("fixed32_banks_built_label_free", built and gt_calls["n"] == 0,
        "%d banks, %d ground-truth loads. %s" % (len(banks), gt_calls["n"], err))
    chk("three_banks_96_slots",
        len(banks) == 3 and sum(b.candidate_count for b in banks.values()) == 96,
        "%d slots" % sum(b.candidate_count for b in banks.values()))
    chk("every_bank_has_32_slots",
        all(b.candidate_count == 32 for b in banks.values()),
        {k: b.candidate_count for k, b in banks.items()})
    chk("failed_slots_preserved_not_resampled",
        all(len(b.candidates) == 32
            and [c.candidate_index for c in b.candidates] == list(range(32))
            for b in banks.values()),
        {k: sum(1 for c in b.candidates if c.status != "valid")
         for k, b in banks.items()})
    chk("bank_carries_no_ari_or_cvi",
        not any(k.lower() in ("ari", "y_true", "cvi", "utility")
                for b in banks.values() for k in vars(b)),
        sorted(vars(banks["xy_2d"]))[:6])
    comp = banks["xy_2d"].provenance["composition"]
    chk("fixed32_composition_recorded_both_sides",
        comp["pre_replacement"].get("constrained") == 4
        and comp["resolved_total"] == 32,
        "pre %s -> post %s" % (json.dumps(comp["pre_replacement"]),
                               json.dumps(comp["post_replacement"])))

    # ---- one common bank shared by every metric
    evals = {}
    for v in views:
        evals[v.view_id] = F32.evaluate_bank_metrics(banks[v.view_id], v, registry)
    per_metric_cands = {}
    for v, rs in evals.items():
        for r in rs:
            per_metric_cands.setdefault((v, r["metric_id"]), set()).add(
                r["candidate_index"])
    same_bank = {v: len({frozenset(s) for (vv, _), s in per_metric_cands.items()
                         if vv == v}) == 1 for v in evals}
    chk("all_metrics_scored_the_same_candidates", all(same_bank.values()),
        json.dumps({k: bool(x) for k, x in same_bank.items()}))
    n_primary = len({r["metric_id"] for r in evals["xy_2d"] if r["primary"]})
    n_diag = len({r["metric_id"] for r in evals["xy_2d"] if not r["primary"]})
    chk("all_67_primary_evaluated", n_primary == 67, "%d primary" % n_primary)
    chk("diagnostics_evaluated_separately", n_diag == 2 and
        all(r["primary"] is False for r in evals["xy_2d"] if not r["primary"]),
        "%d diagnostics, flagged primary=False" % n_diag)
    spaces = {r["metric_id"]: r["evaluation_space"] for r in evals["x_only"]}
    chk("v2_routing_recorded_per_metric",
        sorted(set(spaces.values())) == ["decision", "full"]
        and sum(1 for s in spaces.values() if s == "full") >= 50,
        "%d full-space, %d decision-space"
        % (sum(1 for s in spaces.values() if s == "full"),
           sum(1 for s in spaces.values() if s == "decision")))
    blob = json.dumps(evals["xy_2d"][:200]).lower()
    chk("metric_records_carry_no_ground_truth",
        not any(b in blob for b in ('"ari"', "y_true", "true_k", "utility")),
        "no ARI/y_true/true_k/utility in metric records")

    # ================= OFFLINE BOUNDARY CROSSED HERE =================
    from Independent_Domain_Benchmark.evaluation.offline_boundary import (
        GroundTruth, best_view)
    from sklearn.metrics import adjusted_rand_score
    gt = {v.view_id: GroundTruth(dataset_id=v.dataset_id, view_id=v.view_id,
                                 y_true=y_true, true_k=3,
                                 sample_identity_hash=v.sample_identity_hash)
          for v in views}
    ari_rows = []
    for v in views:
        b = banks[v.view_id]
        for c in b.candidates:
            if c.status != "valid" or c.labels is None:
                continue
            ari_rows.append({"view_id": v.view_id, "candidate_index": c.candidate_index,
                             "ARI": float(adjusted_rand_score(gt[v.view_id].y_true,
                                                              np.asarray(c.labels)))})
    chk("offline_ari_computed_after_completion", len(ari_rows) > 0,
        "%d candidate ARIs over %d views" % (len(ari_rows), len(views)))

    # ---- canonical Stage-3B utility, reused verbatim
    sys.path.insert(0, str(_REPO / "models" / "ClustOpt" / "external_evaluation"))
    from compute_metric_utility import compute_metric_utilities
    import pandas as pd
    util_ok, util_detail, util_rows = False, "", 0
    with tempfile.TemporaryDirectory() as td:
        v = "xy_2d"
        ari_by_idx = {r["candidate_index"]: r["ARI"] for r in ari_rows
                      if r["view_id"] == v}
        by_cand = {}
        for r in evals[v]:
            if not r["primary"] or r["status"] != "valid":
                continue
            # the utility ranks candidates; only the DIRECTION matters, since the
            # NDCG gains come from ARI. Orient once, exactly as Stage 3B did.
            val = r["raw_metric_value"]
            val = val if r["orientation"] == "MAXIMIZE" else -val
            by_cand.setdefault(r["candidate_index"], {})["%s_norm" % r["metric_id"]] = val
        recs = [{"ARI": ari_by_idx[i], "valid": True, **cols}
                for i, cols in sorted(by_cand.items()) if i in ari_by_idx]
        df = pd.DataFrame(recs)
        p = Path(td) / "f32.csv"
        df.to_csv(p, index=False)
        try:
            u = compute_metric_utilities(str(p))
            util_rows = len(u)
            util_ok = util_rows > 0
            util_detail = "%d metric utilities over %d valid candidates" % (
                util_rows, len(df))
        except Exception as exc:  # noqa: BLE001
            util_detail = "%s: %s" % (type(exc).__name__, str(exc)[:160])
    chk("canonical_stage3b_utility_computed", util_ok, util_detail)
    chk("utility_implementation_is_the_frozen_one",
        (_REPO / UTILITY_SRC).is_file()
        and "compute_metric_utilities" in (_REPO / UTILITY_SRC).read_text(
            encoding="utf-8"), UTILITY_SRC)
    chk("bestview_is_method_track_only",
        best_view({"x_only": 0.1, "xy_2d": 0.9}) == "xy_2d",
        "utility stays per dataset x view x metric")

    n_fail = sum(1 for r in rows if r["status"] == "FAIL")
    out = {"smoke": "stage4b5_fixed32_track", "stage": "4B-5", "smoke_tag": SMOKE,
           "banks": len(banks), "candidate_slots": sum(b.candidate_count
                                                       for b in banks.values()),
           "primary_metrics_evaluated": n_primary,
           "diagnostics_evaluated": n_diag,
           "ground_truth_loads_before_offline_phase": gt_calls["n"],
           "candidate_ari_rows": len(ari_rows), "utility_rows": util_rows,
           "valid_slots_per_view": {k: len(b.valid_indices())
                                    for k, b in banks.items()},
           "external_50_used": False,
           "n": len(rows), "n_fail": n_fail, "rows": rows,
           "status": "PASS" if n_fail == 0 else "FAIL"}
    (_ROOT / "validation" / "stage4b5_fixed32_smoke.json").write_text(
        json.dumps(out, indent=2, default=str), encoding="utf-8")
    print("  %d checks | %d fail -> %s" % (len(rows), n_fail, out["status"]))
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
