"""Record the environment of a run that was executed before manifests were written.

``Layer.RUN`` was reserved for "manifest + environment" and nothing ever wrote to
it, so Phase A and most of Phase B finished with no record of the interpreter,
the dependency versions or the thread pinning that produced them. The runner now
writes a manifest at phase start; this fills the gap for the run that already
exists.

It is labelled for what it is. ``captured_at`` is ``post_execution`` and
``captured_at_t0`` is false: this is the environment observed now, on the same
machine and in the same virtual environment that executed the run, NOT a
measurement taken when the run started. The distinction matters, so it is
recorded in the artifact rather than left to a reader's assumption.

Run it with the interpreter that executed the benchmark, or the dependency
versions it records will be someone else's.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_REPO = _ROOT.parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "models"))
sys.path.insert(0, str(_REPO / "experiments/external_baselines/Unified_MKR"))
warnings.filterwarnings("ignore")
import logging                                                        # noqa: E402
logging.disable(logging.INFO)

from Independent_Domain_Benchmark.execution import manifest as MF      # noqa: E402
from Independent_Domain_Benchmark.execution import store as ST         # noqa: E402
from Independent_Domain_Benchmark.execution.common_runner import (     # noqa: E402
    apply_thread_controls)

RESULTS = _REPO / "results_analysis" / "independent_domain_benchmark"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--note", default="")
    args = ap.parse_args(argv)

    run_dir = RESULTS / args.run_id
    if not run_dir.is_dir():
        raise SystemExit("run %s not found" % args.run_id)
    store = ST.ResultStore(RESULTS, args.run_id)

    key = "manifest__post_hoc"
    if store.load(ST.Layer.RUN, key) is not None:
        print("  %s already has %s; refusing to overwrite" % (args.run_id, key))
        return 0

    man = MF.build_manifest(args.run_id, thread_controls=apply_thread_controls(),
                            extra={
        "captured_at": "post_execution",
        "captured_at_t0": False,
        "capture_limitation": (
            "Phase A and most of Phase B executed before any manifest was "
            "written. This records the environment observed after the fact on "
            "the same machine and in the same virtual environment; it is not a "
            "measurement taken at T0 and must not be read as one."),
        "interpreter_executable": sys.executable,
        "note": args.note,
    })
    store.complete(ST.Layer.RUN, key, provenance={
        "run_id": args.run_id, "capture": "post_execution",
        "manifest_version": man["manifest_version"],
        "git_commit": man["git_commit"]}, content=man)

    print("  manifest hash   %s" % man["manifest_hash"])
    print("  interpreter     %s" % sys.executable)
    print("  python          %s" % man["python"])
    print("  dependencies    %s" % json.dumps(man["dependency_versions"]))
    print("  thread env      %s" % json.dumps(man["thread_environment"]))
    print("  git commit      %s (dirty=%s)" % (man["git_commit"][:12],
                                               man["git_dirty"]))
    print("  external/ml2dac %s" % man["external_ml2dac_commit"][:12])
    print("  written to      run/%s" % key)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
