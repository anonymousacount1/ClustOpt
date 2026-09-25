"""Stage 4B-4 internal smoke — the common runner end to end.

INTERNAL / NON_SCIENTIFIC fixtures only. The frozen external corpus is never
touched and no ARI is computed or inspected anywhere.

Exercises, on real physical layers:

* the view contract (explicit ``X_decision`` / ``X_full``, row alignment),
* physical-group execution and trace sharing,
* atomic COMPLETE persistence,
* resume: a second identical invocation must do **zero** scientific work,
* dependency recovery for an incomplete arm and an incomplete physical trace,
* the corrupted-COMPLETE STOP policy and the stale-provenance STOP policy,
* label isolation with raising stubs on the ground-truth loader,
* runtime fields and the shared-work formula.
"""
from __future__ import annotations

import json
import shutil
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

from Independent_Domain_Benchmark.execution import groups as G          # noqa: E402
from Independent_Domain_Benchmark.execution import store as ST          # noqa: E402
from Independent_Domain_Benchmark.execution.common_runner import (      # noqa: E402
    AutoClustExecutor, BenchmarkRunner, ClustOptExecutor, ML2DACExecutor)
from Independent_Domain_Benchmark.execution.dataset_view import (       # noqa: E402
    ViewContractError, assert_label_free, build_dataset_view)

SMOKE = "NON_SCIENTIFIC_SMOKE_TEST"
BUDGET = 6                      # NON_SCIENTIFIC reduced budget
VIEWS = ("x_only", "y_only", "xy_2d")


def _full(seed=20260912, n=70):
    r = np.random.default_rng(seed)
    return np.vstack([r.normal([0, 0], .35, (n, 2)), r.normal([5, 0], .35, (n, 2)),
                      r.normal([2.5, 5], .35, (n, 2))])


def _views():
    X = _full()
    return [build_dataset_view(dataset_id="INVENTED_%s_RUNNER" % SMOKE,
                               source="internal_non_scientific_fixture",
                               view_id=v, X_full=X) for v in VIEWS]


def _label_free(view) -> bool:
    try:
        assert_label_free(view)
        return True
    except ViewContractError:
        return False


def main() -> int:
    rows = []

    def chk(name, ok, detail=""):
        rows.append({"test": name, "status": "PASS" if ok else "FAIL",
                     "detail": str(detail)[:300]})
        print("  %-44s %s  %s" % (name, rows[-1]["status"], str(detail)[:56]),
              flush=True)

    tmp = Path(tempfile.mkdtemp(prefix="idb_stage4b4_"))
    try:
        views = _views()
        execs = {"AutoClust": AutoClustExecutor(), "ML2DAC": ML2DACExecutor(),
                 "ClustOpt": ClustOptExecutor()}
        families = tuple(execs)
        all_groups = G.physical_groups()
        expected_arms = sum(len(g.method_ids) for g in all_groups.values()
                            if g.family in families)
        expected_groups = sum(1 for g in all_groups.values()
                              if g.family in families)

        chk("view_object_is_label_free", _label_free(views[0]),
            "no ground-truth attribute reachable")
        chk("x_full_is_2d_for_every_view",
            all(v.X_full.shape[1] == 2 for v in views),
            [list(v.X_decision.shape) for v in views])

        store = ST.ResultStore(tmp, "run_alpha")
        runner = BenchmarkRunner(store=store, budget=BUDGET)
        res1 = {v.view_id: runner.run_view(v, execs) for v in views}
        c1 = dict(runner.counters)
        chk("first_run_scored_every_arm",
            all(len(r) == expected_arms for r in res1.values()),
            {k: len(r) for k, r in res1.items()})
        chk("first_run_built_each_group_once",
            c1["physical_built"] == expected_groups * len(views), c1)
        chk("physical_records_complete",
            store.count_complete(ST.Layer.PHYSICAL) == expected_groups * len(views))
        chk("scientific_records_complete",
            store.count_complete(ST.Layer.SCIENTIFIC) == expected_arms * len(views))

        shared = {}
        for v in views:
            for gid, g in all_groups.items():
                if g.family not in families:
                    continue
                shared[(v.view_id, gid)] = len(
                    {res1[v.view_id][m].group_id for m in g.method_ids}) == 1
        chk("one_physical_trace_per_group", all(shared.values()),
            "%d group-view combinations" % len(shared))

        sample = next(iter(res1["xy_2d"].values()))
        rt = sample.runtime
        chk("runtime_fields_present",
            all(k in rt for k in ("preparation_runtime", "physical_trace_runtime",
                                  "arm_scoring_runtime", "method_runtime_equivalent",
                                  "actual_wall_clock_runtime")),
            sorted(k for k in rt if str(k).endswith("runtime")))
        grp = all_groups["autoclust_shared_candidate_slate"]
        phys = res1["xy_2d"][grp.method_ids[0]].runtime["physical_trace_runtime"]
        eqs = [res1["xy_2d"][m].runtime["method_runtime_equivalent"]
               for m in grp.method_ids]
        chk("shared_cost_not_divided", all(e >= phys - 1e-9 for e in eqs),
            "each of %d arms pays the full %.3fs physical cost"
            % (len(eqs), phys))

        runner2 = BenchmarkRunner(store=ST.ResultStore(tmp, "run_alpha"),
                                  budget=BUDGET)
        for v in views:
            runner2.run_view(v, execs)
        c2 = dict(runner2.counters)
        chk("resume_zero_physical_recomputation", c2["physical_built"] == 0, c2)
        chk("resume_zero_scientific_recomputation", c2["arms_scored"] == 0, c2)
        chk("resume_reused_everything",
            c2["arms_reused"] == expected_arms * len(views)
            and c2["physical_reused"] == expected_groups * len(views), c2)

        victim = "IDB_ML2DAC_M2_original_plus_new46"
        akey = "%s__%s__%s" % (views[2].dataset_id, "xy_2d", victim)
        (Path(tmp) / "run_alpha" / "scientific" / (akey + ".json")).unlink()
        r3 = BenchmarkRunner(store=ST.ResultStore(tmp, "run_alpha"), budget=BUDGET)
        r3.run_view(views[2], execs)
        c3 = dict(r3.counters)
        chk("incomplete_arm_recovers_only_itself", c3["arms_scored"] == 1,
            "arms_scored=%d physical_built=%d" % (c3["arms_scored"],
                                                  c3["physical_built"]))

        pkey = "%s__%s__%s" % (views[0].dataset_id, "x_only",
                               "ml2dac_shared_candidate_slate")
        (Path(tmp) / "run_alpha" / "physical" / (pkey + ".json")).unlink()
        r4 = BenchmarkRunner(store=ST.ResultStore(tmp, "run_alpha"), budget=BUDGET)
        r4.run_view(views[0], execs)
        c4 = dict(r4.counters)
        chk("incomplete_trace_recovers_without_touching_valid_arms",
            c4["physical_built"] == 1 and c4["arms_scored"] == 0,
            "physical_built=%d arms_scored=%d" % (c4["physical_built"],
                                                  c4["arms_scored"]))

        akey2 = "%s__%s__%s" % (views[2].dataset_id, "xy_2d",
                                "IDB_AutoClust_A0_original_cvis")
        p = Path(tmp) / "run_alpha" / "scientific" / (akey2 + ".json")
        original = p.read_text(encoding="utf-8")
        rec = json.loads(original)
        rec["content"]["n_evaluations"] = 9999
        p.write_text(json.dumps(rec, indent=2), encoding="utf-8")
        try:
            BenchmarkRunner(store=ST.ResultStore(tmp, "run_alpha"),
                            budget=BUDGET).run_view(views[2], execs)
            corrupt = "ACCEPTED"
        except ST.IntegrityError:
            corrupt = "IntegrityError"
        chk("corrupted_complete_arm_stops", corrupt == "IntegrityError", corrupt)

        p.write_text("{not json", encoding="utf-8")
        try:
            ST.ResultStore(tmp, "run_alpha").load(ST.Layer.SCIENTIFIC, akey2)
            malformed = "ACCEPTED"
        except ST.IntegrityError:
            malformed = "IntegrityError"
        chk("malformed_record_detected", malformed == "IntegrityError", malformed)
        p.write_text(original, encoding="utf-8")

        store5 = ST.ResultStore(tmp, "run_alpha")
        prov = dict(BenchmarkRunner(store=store5, budget=BUDGET)
                    ._base_provenance(views[1]))
        prov["budget"] = 999
        try:
            store5.try_reuse(ST.Layer.PREPARATION,
                             "%s__%s" % (views[1].dataset_id, "y_only"), prov)
            stale = "REUSED"
        except ST.ProvenanceMismatch:
            stale = "ProvenanceMismatch"
        chk("stale_provenance_refused", stale == "ProvenanceMismatch", stale)

        leftovers = list((Path(tmp) / "run_alpha").rglob("*.tmp"))
        chk("no_temporary_files_left", not leftovers, leftovers)

        calls = {"n": 0}
        import unified_mkr.view_loader as VL

        def boom(*a, **k):
            calls["n"] += 1
            raise AssertionError("ground-truth loader invoked during method execution")

        keep = {n: getattr(VL, n) for n in dir(VL)
                if n.startswith(("load", "build")) and callable(getattr(VL, n))}
        for n in keep:
            setattr(VL, n, boom)
        try:
            BenchmarkRunner(store=ST.ResultStore(tmp, "run_labelfree"),
                            budget=BUDGET).run_view(views[2], execs)
            lf_ok, lf_err = True, ""
        except Exception as exc:  # noqa: BLE001
            lf_ok, lf_err = False, "%s: %s" % (type(exc).__name__, str(exc)[:140])
        finally:
            for n, v in keep.items():
                setattr(VL, n, v)
        chk("label_isolation_method_phase", lf_ok and calls["n"] == 0,
            "loader calls=%d %s" % (calls["n"], lf_err))

        blob = json.dumps([r.__dict__ for r in res1["xy_2d"].values()],
                          default=str).lower()
        hits = [b for b in ('"ari"', '"nmi"', '"ami"', '"fmi"', "y_true", "true_k")
                if b in blob]
        chk("no_evaluation_fields_in_arm_records", not hits, hits)

        chk("all_14_methods_executed",
            expected_arms == 14 and all(len(r) == 14 for r in res1.values()),
            "%d arms per view" % expected_arms)
        chk("all_6_physical_groups_executed", expected_groups == 6,
            "%d groups per view" % expected_groups)
        fam = {}
        for m, o in res1["xy_2d"].items():
            fam.setdefault(next(f for f in ("ClustOpt", "AutoClust", "ML2DAC")
                                if f in m), []).append(m)
        chk("family_arm_counts",
            {k: len(v) for k, v in fam.items()} == {"ClustOpt": 6, "AutoClust": 4,
                                                    "ML2DAC": 4},
            {k: len(v) for k, v in fam.items()})
        chk("plan_counts_1_dataset_3_views",
            G.plan_counts(1, 3)["physical_traces"] == 18
            and G.plan_counts(1, 3)["scientific_outputs"] == 42, "18 / 42")
        chk("plan_counts_50_datasets_3_views",
            G.plan_counts(50, 3)["physical_traces"] == 900
            and G.plan_counts(50, 3)["scientific_outputs"] == 2100, "900 / 2100")

        n_fail = sum(1 for r in rows if r["status"] == "FAIL")
        out = {"smoke": "stage4b4_common_runner", "stage": "4B-4",
               "smoke_tag": SMOKE, "budget_is_non_scientific": True,
               "families_exercised": list(families),
               "arms_per_view": expected_arms, "groups_per_view": expected_groups,
               "views": list(VIEWS), "first_run_counters": c1,
               "second_run_counters": c2,
               "thread_controls": runner.thread_controls,
               "external_50_used": False, "labels_used": False,
               "n": len(rows), "n_fail": n_fail, "rows": rows,
               "status": "PASS" if n_fail == 0 else "FAIL"}
        (_ROOT / "validation" / "stage4b4_runner_smoke.json").write_text(
            json.dumps(out, indent=2, default=str), encoding="utf-8")
        print("  %d checks | %d fail -> %s" % (len(rows), n_fail, out["status"]))
        return 0 if n_fail == 0 else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
