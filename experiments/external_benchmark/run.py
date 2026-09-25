"""External benchmark: public entry point (50 datasets x 3 views = 150 units, 14 arms).

The steps, in order (each one is resumable):

    python experiments/external_benchmark/run.py reconstruct            # rebuild the 50 datasets, verify checksums
    python experiments/external_benchmark/run.py archive                # store them locally for the run (not tracked)
    python experiments/external_benchmark/run.py arms  [-- --workers 4]  # all 14 arms, label-free (prints a RUN_ID)
    python experiments/external_benchmark/run.py banks [-- --workers 4]  # 32-candidate banks + index values
    python experiments/external_benchmark/run.py score        --run-id RUN_ID   # ground-truth ARI of arms and candidates
    python experiments/external_benchmark/run.py repair-indices --run-id RUN_ID  # recompute the index-dispatch repair set
    python experiments/external_benchmark/run.py index-utility --run-id RUN_ID  # index utility on the canonical candidate basis
    python experiments/external_benchmark/run.py core-analysis --run-id RUN_ID  # generalist core and joint pipeline
    python experiments/external_benchmark/run.py tables       --run-id RUN_ID   # method / index / paired / per-source tables
    python experiments/external_benchmark/run.py runtime      --run-id RUN_ID   # runtime decomposition

The ML2DAC-based and AutoClust +established / extended arms need the upstream
ML2DAC code (see baselines/README.md). Options after ``--`` are passed unchanged.
The released results were produced by these steps; compare your tables with
``results/external/``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _dispatch import ROOT, run_module, run_script  # noqa: E402

R = "Independent_Domain_Benchmark.runners."
STEPS = {  # public step -> (implementation module, needs --run-id)
    "archive": (R + "build_stage4_corpus_archive", False),
    "arms": (R + "run_stage4c_parallel", False),
    "banks": (R + "run_stage4c_parallel", False),
    "score": (R + "run_stage4c_phase_c", True),
    "repair-indices": (R + "run_stage4c_metric_repair", True),
    "index-utility": (R + "run_stage4c_canonical_utility", True),
    "core-analysis": (R + "run_stage4c_phase_d_canonical", True),
    "tables": (R + "build_stage4_scientific_review", True),
    "runtime": (R + "build_stage4_runtime_analysis", True),
}


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    passthrough = []
    if "--" in argv:
        i = argv.index("--")
        argv, passthrough = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["reconstruct"] + list(STEPS))
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args(argv)
    if args.step == "reconstruct":
        return run_script(ROOT / "scripts" / "reconstruct_external_datasets.py", passthrough)
    module, needs_run = STEPS[args.step]
    extra = []
    if args.step == "arms":
        extra = ["--phase", "A"]
    elif args.step == "banks":
        extra = ["--phase", "B"]
    if needs_run:
        if not args.run_id:
            ap.error("--run-id is required for step %r (it is printed by the 'arms' step)" % args.step)
        extra += ["--run-id", args.run_id]
    elif args.run_id:
        extra += ["--run-id", args.run_id]
    return run_module(module, extra + passthrough, cwd=ROOT / "src")


if __name__ == "__main__":
    raise SystemExit(main())
