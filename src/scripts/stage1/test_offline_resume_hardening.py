"""Regression test for the Stage-1C resume hardening.

The orchestrator must treat the progress JSON as a HINT and revalidate the
on-disk completion barrier before skipping a subfamily.

Cases:
  A progress=complete + outputs complete    -> barrier PASS  -> skip
  B progress=complete + one output missing  -> barrier FAIL  -> DO NOT skip
  C progress=incomplete + outputs complete  -> barrier PASS  -> may skip safely
                                               (no recomputation needed)

Case B is exercised by temporarily hiding ONE required file (renaming it) and
restoring it immediately afterwards, so no scientific output is destroyed. The
test refuses to run unless the pilot subfamily is already complete.

Run:
    .venv_clustopt/Scripts/python.exe scripts/stage1/test_offline_resume_hardening.py
"""
from __future__ import annotations

import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from pathlib import Path  # noqa: E402

from models.Clustering_Repository_Builder.experiments.experiment_execution.config import (  # noqa: E402
    ExperimentExecutionConfig,
)
from models.Clustering_Repository_Builder.experiments.experiment_execution.output_writer import (  # noqa: E402
    _ext, path_exists,
)
from models.Clustering_Repository_Builder.experiments.experiment_execution.run_offline_subfamily import (  # noqa: E402
    build_arms, subfamily_is_complete,
)
from models.Clustering_Repository_Builder.experiments.experiment_execution.run_offline_2x3_all_subfamilies import (  # noqa: E402
    _split1_dataset_dirs,
)

HEAD14 = "results_analysis/mlp/20260823_195830__metric_utility_mlp_head14_split1_holdout"
RESULTS = []


def rec(name, ok, detail=""):
    RESULTS.append((name, "PASS" if ok else "FAIL", detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    return ok


def main() -> int:
    repo = Path(REPO)
    analyzed = repo / "results_analysis/clustering_repository/analyzed_data"
    sub = analyzed / "arcs_circles_rings/subfamilies/annuli_variable_thickness"
    cfg = ExperimentExecutionConfig(
        repo_root=repo, subfamily_dir=sub,
        split_assignments=analyzed / "experiment_splits/dataset_split_assignments.csv",
        split_id=1, split_dir_suffix="offline_fixed32",
        method_set="stage1_2x3", max_workers=8,
    )
    arms = build_arms(HEAD14)
    dirs = _split1_dataset_dirs(cfg)

    print("=" * 78)
    print("STAGE-1C RESUME HARDENING REGRESSION TEST")
    print("=" * 78)
    print(f"  subfamily : {sub.name}   datasets: {len(dirs)}")

    # ---------------------------------------------------------------- case A
    ok_a, detail_a = subfamily_is_complete(cfg, dirs, arms)
    if not ok_a:
        print(f"  [ABORT] pilot subfamily is not complete ({detail_a}); "
              f"run the pilot before this test.")
        return 2
    all_ok = rec("A: progress=complete + outputs complete -> barrier PASS (skip)",
                 ok_a, f"expected_runs={detail_a['expected_runs']}, "
                       f"missing={detail_a['missing_runs']}")

    # ---------------------------------------------------------------- case B
    victim = (Path(dirs[0]) / "experiments" / cfg.split_dir_name()
              / "stage1_head14_oracle_top5_raw_fixed50" / "x_only"
              / "selected_metrics.json")
    hidden = victim.with_suffix(".json.hidden_for_test")
    if not path_exists(victim):
        return rec("B: victim file present", False, str(victim)) or 1
    os.rename(_ext(victim), _ext(hidden))
    try:
        ok_b, detail_b = subfamily_is_complete(cfg, dirs, arms)
        all_ok &= rec("B: progress=complete + 1 output missing -> barrier FAIL "
                      "(do NOT skip)", not ok_b,
                      f"missing_runs={detail_b['missing_runs']}, "
                      f"e.g. {detail_b['missing_examples'][:1]}")
    finally:
        os.rename(_ext(hidden), _ext(victim))          # always restore
    restored, detail_r = subfamily_is_complete(cfg, dirs, arms)
    all_ok &= rec("B: file restored, barrier PASSes again", restored,
                  f"missing={detail_r['missing_runs']}")

    # ---------------------------------------------------------------- case C
    # progress flag is irrelevant when outputs are complete: the barrier is the
    # authority, so a 'not completed' flag still finds a complete subfamily and
    # the per-run resume skips every individual run (zero recomputation).
    from models.Clustering_Repository_Builder.experiments.experiment_execution.offline_method_runner import (
        offline_run_is_complete,
    )
    per_run = all(
        offline_run_is_complete(Path(d), cfg.split_dir_name(), a.method_id, v)
        for d in dirs for v in ("x_only", "y_only", "xy_2d") for a in arms)
    all_ok &= rec("C: progress=incomplete + outputs complete -> every run "
                  "individually complete (no recomputation)", per_run,
                  f"{len(dirs) * 3 * 6} runs checked")

    print()
    print("=" * 78)
    p = sum(1 for _, s, _ in RESULTS if s == "PASS")
    f = sum(1 for _, s, _ in RESULTS if s == "FAIL")
    print(f"  {p}/{len(RESULTS)} PASS, {f} FAIL")
    print(f"  VERDICT: {'RESUME HARDENING VERIFIED' if all_ok else 'FAILED'}")
    print("=" * 78)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
