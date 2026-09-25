"""Stage-4B-3A1 tests for the AutoClust physical execution layer."""
from __future__ import annotations

import inspect, json, sys, warnings
from pathlib import Path
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_REPO = _ROOT.parents[1]
sys.path.insert(0, str(_REPO)); sys.path.insert(0, str(_ROOT.parent))
warnings.filterwarnings("ignore")

from Independent_Domain_Benchmark.methods import autoclust as AC          # noqa: E402
from Independent_Domain_Benchmark.search import protocol as PROTO         # noqa: E402

SMOKE = "NON_SCIENTIFIC_SMOKE_TEST"
BANNED = ("y", "y_true", "labels_true", "target", "true_k", "ari", "nmi",
          "ami", "fmi", "ground_truth", "class_distribution")


def _X(seed=20260911, n=90):
    r = np.random.default_rng(seed)
    return np.vstack([r.normal([0, 0], .35, (n, 2)), r.normal([4, 0], .35, (n, 2)),
                      r.normal([2, 4], .35, (n, 2))])


def t_api_label_free():
    p = [x for x in inspect.signature(AC.build_autoclust_candidate_trace).parameters]
    bad = [x for x in p if x.lower() in BANNED]
    assert not bad, "label parameter(s) present: %s" % bad
    return {"signature": p, "banned_present": bad}


def t_no_repository_membership(tr):
    # the dataset_id is invented and exists in no master parquet
    assert tr.dataset_id.startswith("INVENTED_"), tr.dataset_id
    return {"dataset_id": tr.dataset_id, "membership_required": False}


def t_seed_identity(tr):
    seeds = {m: PROTO.derive_seed(tr.dataset_id, tr.view_id, m)
             for m in AC.ARM_METHOD_IDS}
    assert len(set(seeds.values())) == 1
    assert tr.resolved_seed == next(iter(seeds.values()))
    assert tr.seed_identity == "autoclust_shared_candidate_slate"
    return {"seeds": seeds, "identity": tr.seed_identity}


def t_trace_id_arm_independent(tr):
    src = inspect.getsource(AC.build_autoclust_candidate_trace)
    blob = src[src.index('ident = json.dumps'):src.index('trace_id = hashlib')]
    assert "method_id" not in blob and "ARM_METHOD_IDS" not in blob
    tr2 = AC.build_autoclust_candidate_trace(_X(), tr.dataset_id, tr.view_id,
                                             budget=tr.budget)
    assert tr2.trace_id == tr.trace_id
    return {"trace_id": tr.trace_id, "reproducible": True,
            "depends_on_method_id": False}


def t_budget(tr):
    assert tr.candidate_count == tr.budget == len(tr.candidates)
    idx = [c.candidate_index for c in tr.candidates]
    assert idx == list(range(tr.budget)), "no resampling / no gaps"
    return {"budget": tr.budget, "candidates": tr.candidate_count,
            "failed_kept": sum(1 for c in tr.candidates if c.status != "valid")}


def t_immutable(tr):
    before = tr.content_hash()
    out = {}
    try:
        tr.dataset_id = "X"; out["metadata"] = "MUTATED"
    except Exception as e:
        out["metadata"] = type(e).__name__
    try:
        tr.candidates.append(None); out["collection"] = "MUTATED"
    except Exception as e:
        out["collection"] = type(e).__name__
    c = tr.candidates[0]
    try:
        c.configuration["n_clusters"] = 99; out["configuration"] = "MUTATED"
    except Exception as e:
        out["configuration"] = type(e).__name__
    lab = next((x.labels for x in tr.candidates if x.labels is not None), None)
    try:
        lab[0] = 12345; out["labels"] = "MUTATED"
    except Exception as e:
        out["labels"] = type(e).__name__
    assert "MUTATED" not in out.values(), out
    assert tr.content_hash() == before
    return out


def t_four_arm_consumption(tr):
    before = tr.content_hash()
    seen = []
    for m in AC.ARM_METHOD_IDS:
        cs = tr.get_candidates_for_scoring()
        seen.append((m, len(cs), sum(tr.availability_mask())))
    assert tr.content_hash() == before
    assert len({s[1] for s in seen}) == 1 and len({s[2] for s in seen}) == 1
    return {"arms": len(seen), "hash_stable": True,
            "availability_identical": True}


def t_no_ground_truth_fields(tr):
    blob = json.dumps({
        "trace": {k: str(v) for k, v in vars(tr).items() if k != "candidates"},
        "cand": [{k: str(v) for k, v in vars(c).items()} for c in tr.candidates[:5]],
    }).lower()
    hits = [b for b in ("\"ari\"", "\"nmi\"", "\"ami\"", "\"fmi\"", "true_k",
                        "y_true") if b in blob]
    assert not hits, hits
    return {"forbidden_found": hits}


def t_sampler_guard():
    """Check CODE, not comments -- the module documents the hazard by name."""
    import re
    src = inspect.getsource(AC.build_autoclust_candidate_trace)
    code = "\n".join(line.split("#")[0] for line in src.splitlines())
    assert "sample_optimizer_configs_constrained" in code, "constrained sampler absent"
    assert "_sample_params" not in code, "native helper reachable in code"
    bare = re.search(r"\bsample_optimizer_configs\b(?!_constrained)", code)
    assert bare is None, "native sampler referenced in code"
    return {"constrained_used": True, "native_reachable": False,
            "checked": "comment-stripped source"}


def main() -> int:
    print("building physical trace (%s)..." % SMOKE, flush=True)
    tr = AC.build_autoclust_candidate_trace(
        _X(), "INVENTED_%s_DS" % SMOKE, "xy_2d", budget=8)
    print("  trace %s | algo %s | seed %d | candidates %d"
          % (tr.trace_id, tr.selected_algorithm, tr.resolved_seed,
             tr.candidate_count), flush=True)
    checks = [("api_label_free", t_api_label_free, ()),
              ("no_repository_membership", t_no_repository_membership, (tr,)),
              ("production_seed_identity", t_seed_identity, (tr,)),
              ("trace_id_arm_independent", t_trace_id_arm_independent, (tr,)),
              ("budget_and_no_resampling", t_budget, (tr,)),
              ("immutability", t_immutable, (tr,)),
              ("four_arm_consumption", t_four_arm_consumption, (tr,)),
              ("no_ground_truth_fields", t_no_ground_truth_fields, (tr,)),
              ("sampler_guard", t_sampler_guard, ())]
    rows, fail = [], 0
    for name, fn, args in checks:
        try:
            rows.append({"test": name, "status": "PASS", "detail": fn(*args)})
        except Exception as e:
            fail += 1
            rows.append({"test": name, "status": "FAIL",
                         "error": "%s: %s" % (type(e).__name__, str(e)[:200])})
        print("  %-28s %s" % (name, rows[-1]["status"]), flush=True)
    out = {"suite": "autoclust_physical_layer", "smoke_tag": SMOKE,
           "trace": {"trace_id": tr.trace_id, "dataset_id": tr.dataset_id,
                     "view_id": tr.view_id, "algorithm": tr.selected_algorithm,
                     "seed": tr.resolved_seed, "candidates": tr.candidate_count,
                     "valid": sum(tr.availability_mask()),
                     "content_hash": tr.content_hash(),
                     "search_space_fingerprint": tr.search_space_fingerprint},
           "external_50_used": False, "labels_used": False,
           "n": len(rows), "n_fail": fail, "rows": rows,
           "status": "PASS" if fail == 0 else "FAIL"}
    (_ROOT / "validation" / "autoclust_physical_layer_tests.json").write_text(
        json.dumps(out, indent=2, default=str), encoding="utf-8")
    print("  %d tests | %d fail -> %s" % (len(rows), fail, out["status"]))
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
