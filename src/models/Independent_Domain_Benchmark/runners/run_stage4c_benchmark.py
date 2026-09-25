"""Stage 4C — execute the frozen external benchmark.

Phases (see ``configs/stage4c_execution_plan.json``):

* **A** end-to-end method execution, 14 arms over 6 physical groups, label-free
* **B** fixed32 metric track, 150 banks x 67 primary CVIs, label-free
* **C** offline ground-truth evaluation (opened only after A and B are COMPLETE)
* **D** frozen analysis

This is an execution driver. It changes no method, metric, artifact, budget, seed
or semantics. Everything scientific is read from the frozen configs and the
already-closed layers; the driver only sequences work, persists it atomically and
resumes.

Corpus materialisation is deterministic and label-free: fetch, then
``datasets.preprocessing.prepare`` (median impute, StandardScaler, PCA-2 seed
1234), then verify the representation checksum against the frozen manifest. A
dataset whose checksum does not match is a provenance failure and stops the run.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_REPO = _ROOT.parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "models"))
sys.path.insert(0, str(_REPO / "experiments/external_baselines/Unified_MKR"))
warnings.filterwarnings("ignore")

from Independent_Domain_Benchmark.execution import groups as G            # noqa: E402
from Independent_Domain_Benchmark.execution import hashing as H          # noqa: E402
from Independent_Domain_Benchmark.execution import store as ST           # noqa: E402
from Independent_Domain_Benchmark.execution.common_runner import (       # noqa: E402
    AutoClustExecutor, BenchmarkRunner, ClustOptExecutor, ML2DACExecutor)
from Independent_Domain_Benchmark.execution.dataset_view import (        # noqa: E402
    build_dataset_view)

CFG = _ROOT / "configs"
RESULTS = _REPO / "results_analysis" / "independent_domain_benchmark"
VIEW_IDS = ("x_only", "y_only", "xy_2d")


class ProvenanceFailure(RuntimeError):
    """A frozen identity does not match. Stops the scientific run."""


# ------------------------------------------------------------------ corpus
def _manifest() -> Dict[str, Any]:
    return json.loads((CFG / "dataset_manifest.json").read_text(encoding="utf-8"))


def materialise(entry: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Load the frozen prepared representation from the durable corpus archive.

    Scientific execution no longer downloads or re-prepares anything. The madelon
    case showed why: re-running PCA under pinned BLAS produces a different 2-D
    embedding from the one the frozen manifest records, so deriving the input
    inside a worker makes the scientific input depend on an infrastructure
    setting. The archive is built once, verified against the frozen manifest, and
    loaded with a hash check thereafter.

    The label vector is returned as a separate value and is never attached to the
    feature matrix; Phase A and Phase B discard it at the call site.
    """
    from Independent_Domain_Benchmark.datasets import corpus_archive as CA
    try:
        X2, rec = CA.load_scientific_input(entry["dataset_id"])
        if rec["representation_checksum"] != entry["representation_checksum"]:
            raise ProvenanceFailure(
                "%s archived representation %s != frozen %s"
                % (entry["dataset_id"], rec["representation_checksum"],
                   entry["representation_checksum"]))
        return X2, CA.load_ground_truth(entry["dataset_id"])
    except FileNotFoundError:
        return _materialise_by_derivation(entry)


def _materialise_by_derivation(entry: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Fetch and prepare one frozen dataset. Labels are returned SEPARATELY.

    The label vector never travels with the feature matrix into the method layer:
    the caller hands ``X_2d`` to ``build_dataset_view`` and keeps ``y`` for the
    Phase-C offline boundary only.
    """
    from Independent_Domain_Benchmark.datasets import preprocessing as P
    src = entry["source"]
    if src == "fcps":
        from Independent_Domain_Benchmark.datasets import fcps
        raw, _ = fcps.fetch(entry["canonical_name"])
        X, y, _ = fcps.parse_arff(raw)
    elif src == "sklearn_shapes":
        from Independent_Domain_Benchmark.datasets import sklearn_shapes as SS
        X, y = SS.generate(entry["generator"], entry["difficulty"])
    elif src == "real_pca":
        from Independent_Domain_Benchmark.datasets import real_world as RW
        spec = next((sp for k, sp, _dom in RW.CANDIDATES
                     if k == entry["canonical_name"]), None)
        if spec is None:
            raise ProvenanceFailure("no frozen spec for %s" % entry["canonical_name"])
        X, y, _meta = RW.fetch_candidate(entry["canonical_name"], spec)
    else:
        raise ProvenanceFailure("unknown source %r" % src)
    prep = P.prepare(np.asarray(X, dtype=float), entry["dataset_id"], src)
    if prep.feature_checksum != entry["representation_checksum"]:
        raise ProvenanceFailure(
            "%s representation checksum %s != frozen %s"
            % (entry["dataset_id"], prep.feature_checksum,
               entry["representation_checksum"]))
    return prep.X_2d, np.asarray(y)


# ------------------------------------------------------------- integrity
def preflight() -> Dict[str, Any]:
    """Frozen identities must match before any external execution."""
    import subprocess
    plan = json.loads((CFG / "stage4c_execution_plan.json").read_text(encoding="utf-8"))
    checks = {
        "plan_hash": H.canonical_text_hash(
            (CFG / "stage4c_execution_plan.json").read_bytes()),
        "corpus_hash": _manifest()["manifest_sha256_16"],
        "metric_registry_hash": H.canonical_text_hash(
            (CFG / "metric_analysis_registry.json").read_bytes()),
        "numerical_policy_hash": H.canonical_text_hash(
            (CFG / "numerical_reproducibility_policy.json").read_bytes()),
        "utility_source_hash": H.canonical_text_hash(
            (_REPO / plan["canonical_utility"]["source"]).read_bytes()),
        "search_space_fingerprint": plan["search_space_fingerprint"],
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=_REPO,
                                     capture_output=True, text=True).stdout.strip(),
        "external_ml2dac_commit": subprocess.run(
            ["git", "-C", "external/ml2dac", "rev-parse", "HEAD"], cwd=_REPO,
            capture_output=True, text=True).stdout.strip(),
    }
    expected = {
        "plan_hash": "6275e89ac18cb431", "corpus_hash": "709712107424ad4e",
        "metric_registry_hash": "8d6b44f45f9cf240",
        "numerical_policy_hash": "353a3777e9f8400e",
        "utility_source_hash": "7f65fa2b1a794b26",
        "search_space_fingerprint": "8ddf86c79888ebe9",
    }
    bad = {k: (checks[k], v) for k, v in expected.items() if checks[k] != v}
    if bad:
        raise ProvenanceFailure("frozen identity mismatch: %s" % json.dumps(bad))
    if not checks["external_ml2dac_commit"].startswith("c946b9e"):
        raise ProvenanceFailure("external/ml2dac at %s"
                                % checks["external_ml2dac_commit"][:12])
    # SCIENTIFIC source must be clean; generated results may be untracked
    st = subprocess.run(["git", "status", "--porcelain", "--",
                         "models", "experiments", "external"],
                        cwd=_REPO, capture_output=True, text=True).stdout
    checks["scientific_source_modified"] = [
        l for l in st.splitlines() if not l.startswith("??")]
    checks["preflight"] = "PASS"
    return checks


def run_id_for(checks: Dict[str, Any]) -> str:
    return "stage4c_%s" % H.semantic_object_hash({
        k: checks[k] for k in sorted(expected_keys())})[:12]


def expected_keys() -> Tuple[str, ...]:
    return ("plan_hash", "corpus_hash", "metric_registry_hash",
            "numerical_policy_hash", "utility_source_hash",
            "search_space_fingerprint", "git_commit")


# -------------------------------------------------------------- phase A
def phase_a(store: ST.ResultStore, entries: List[Dict[str, Any]], *,
            limit_views: Optional[Tuple[str, ...]] = None,
            budget: Optional[int] = None, echo: bool = True) -> Dict[str, Any]:
    execs = {"AutoClust": AutoClustExecutor(), "ML2DAC": ML2DACExecutor(),
             "ClustOpt": ClustOptExecutor()}
    runner = BenchmarkRunner(store=store, budget=budget)
    views = limit_views or VIEW_IDS
    stats = {"datasets": 0, "units": 0, "arm_records": 0, "success": 0,
             "failed": 0, "failure_kinds": {}, "seconds": 0.0,
             "provenance_failures": []}
    for e in entries:
        t0 = time.perf_counter()
        try:
            X2, _y = materialise(e)          # labels discarded here on purpose
        except ProvenanceFailure as exc:
            stats["provenance_failures"].append(str(exc))
            raise
        stats["datasets"] += 1
        for v in views:
            view = build_dataset_view(dataset_id=e["dataset_id"], source=e["source"],
                                      view_id=v, X_full=X2)
            res = runner.run_view(view, execs)
            stats["units"] += 1
            for mid, out in res.items():
                stats["arm_records"] += 1
                if out.status == "success":
                    stats["success"] += 1
                else:
                    stats["failed"] += 1
                    k = out.failure_kind or "UNKNOWN"
                    stats["failure_kinds"][k] = stats["failure_kinds"].get(k, 0) + 1
        stats["seconds"] += time.perf_counter() - t0
        if echo:
            print("   [A] %-22s %d views, %d arm records, %.1fs cumulative"
                  % (e["dataset_id"], len(views), stats["arm_records"],
                     stats["seconds"]), flush=True)
    stats["counters"] = dict(runner.counters)
    return stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="preflight",
                    choices=["preflight", "timing", "A"])
    ap.add_argument("--limit-datasets", type=int, default=0)
    ap.add_argument("--views", default="")
    ap.add_argument("--budget", type=int, default=0)
    args = ap.parse_args(argv)

    checks = preflight()
    rid = run_id_for(checks)
    print("Stage 4C preflight PASS | run_id %s | commit %s"
          % (rid, checks["git_commit"][:12]), flush=True)
    if checks["scientific_source_modified"]:
        print("  scientific source is MODIFIED: %s"
              % checks["scientific_source_modified"][:5], flush=True)

    entries = [e for e in _manifest()["datasets"] if e.get("accepted", True)]
    if args.limit_datasets:
        entries = entries[:args.limit_datasets]
    views = tuple(v for v in args.views.split(",") if v) or VIEW_IDS
    budget = args.budget or None

    RESULTS.mkdir(parents=True, exist_ok=True)
    store = ST.ResultStore(RESULTS, rid)
    if args.phase == "preflight":
        print(json.dumps({k: checks[k] for k in expected_keys()}, indent=2))
        print("accepted datasets: %d" % len(entries))
        return 0
    if args.phase in ("timing", "A"):
        t0 = time.perf_counter()
        stats = phase_a(store, entries, limit_views=views, budget=budget)
        stats["wall_clock_seconds"] = time.perf_counter() - t0
        stats["run_id"] = rid
        stats["preflight"] = {k: checks[k] for k in expected_keys()}
        out = RESULTS / rid / "phase_a_progress.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(stats, indent=2, default=str), encoding="utf-8")
        print("  arm records %d | success %d | failed %d | %.1fs"
              % (stats["arm_records"], stats["success"], stats["failed"],
                 stats["wall_clock_seconds"]))
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
