"""Verify a dedicated ML2DAC environment: imports + MKR artifacts + (optional) a tiny run.

Run inside the ML2DAC env (from the repo root) with the repo on PYTHONPATH::

    python experiments/external_baselines/ML2DAC/setup/check_ml2dac_install.py \
        --repo-root .

Exit code 0 => environment ready (imports OK + artifacts present). It prints a
clear PASS/FAIL line per check and, if ``--run-tiny`` is passed and everything is
ready, runs one tiny make_blobs dataset end-to-end through the adapter.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path


def _repo_root_default() -> Path:
    # .../experiments/external_baselines/ML2DAC/setup/<this file>
    return Path(__file__).resolve().parents[4]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Check the ML2DAC environment + artifacts.")
    ap.add_argument("--repo-root", type=str, default=str(_repo_root_default()))
    ap.add_argument("--run-tiny", action="store_true",
                    help="If env+artifacts are ready, run one tiny dataset through the adapter.")
    args = ap.parse_args(argv)
    repo_root = Path(args.repo_root).resolve()

    # Make the repo importable so we can reuse the baseline helpers.
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    print(f"Python: {sys.version.splitlines()[0]}")
    print(f"Executable: {sys.executable}")
    print(f"Repo root: {repo_root}")

    ok = True

    # 1. External repo + artifacts (filesystem-only).
    try:
        from experiments.external_baselines.ML2DAC.adapters.artifact_locator import locate_artifacts
        report = locate_artifacts(repo_root)
        present = report["has_pretrained_artifacts"]
        print(f"[{'PASS' if report['ml2dac_src_exists'] else 'FAIL'}] external/ml2dac src present")
        print(f"[{'PASS' if present else 'FAIL'}] MKR artifacts present "
              f"(missing={report['missing_required_artifacts']})")
        ok = ok and report["ml2dac_src_exists"] and present
    except Exception:
        print("[FAIL] artifact locator crashed:")
        traceback.print_exc()
        return 1

    # 2. Imports of the ML2DAC stack.
    from experiments.external_baselines.ML2DAC.utils.path_utils import ensure_ml2dac_on_path
    ensure_ml2dac_on_path(repo_root)
    import_ok = True
    for mod in ["ConfigSpace", "smac", "hdbscan", "pymfe", "sklearn", "numpy", "pandas", "scipy"]:
        try:
            __import__(mod)
            print(f"[PASS] import {mod}")
        except Exception as e:
            print(f"[FAIL] import {mod}: {type(e).__name__}: {e}")
            import_ok = False
    for mod in ["MetaLearning.ApplicationPhase", "Optimizer.OptimizerSMAC",
                "ClusteringCS.ClusteringCS", "ClusterValidityIndices.CVIHandler"]:
        try:
            __import__(mod)
            print(f"[PASS] import {mod}")
        except Exception as e:
            print(f"[FAIL] import {mod}: {type(e).__name__}: {e}")
            import_ok = False
    ok = ok and import_ok

    if import_ok:
        try:
            import sklearn
            print(f"  sklearn={sklearn.__version__}")
        except Exception:
            pass

    if not ok:
        print("\nRESULT: environment NOT ready. See setup_wsl.md.")
        return 1

    print("\nRESULT: environment ready (imports + artifacts).")

    # 3. Optional tiny end-to-end run.
    if args.run_tiny:
        try:
            import numpy as np
            from sklearn.datasets import make_blobs
            from experiments.external_baselines.ML2DAC.adapters.ml2dac_adapter import ML2DACAdapter
            from experiments.external_baselines.ML2DAC.utils.metrics import compute_all_metrics
            X, y = make_blobs(n_samples=300, n_features=2, centers=4, random_state=42, cluster_std=0.6)
            cfg = {"mode": "full_meta_learning", "use_meta_cvi_selection": True,
                   "use_warmstarting": True, "use_algorithm_reduction": True,
                   "time_budget_sec": 60, "n_optimizer_loops": 10, "n_warmstarts": 5,
                   "mf_set_index": 4}
            adapter = ML2DACAdapter(cfg, repo_root)
            res = adapter.fit_predict(X, X, record_type="2d", dataset_id="check_synth")
            if res.get("status") != "success":
                print(f"[FAIL] tiny run: {res.get('error')}")
                return 1
            m = compute_all_metrics(y, np.asarray(res["labels"]), 4)
            print(f"[PASS] tiny run: selected_k={res['selected_k']} cvi={res['selected_cvi']} "
                  f"algo={res['selected_algorithm']} ari={m['ari']:.4f} nmi={m['nmi']:.4f}")
        except Exception:
            print("[FAIL] tiny run crashed:")
            traceback.print_exc()
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
