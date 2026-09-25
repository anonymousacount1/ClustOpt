"""AutoML4Clust smoke test -- tiny end-to-end sanity check.

Steps:
  1. Generate one tiny synthetic dataset (make_blobs).
  2. Verify AutoML4Clust is importable (full diagnostics on failure).
  3. Run AutoML4Clust through the adapter for each record type.
  4. Verify labels are produced; compute ARI / NMI / AMI vs. the synthetic labels.
  5. Save a smoke-test result under
     ``results_analysis/clustering_repository/external_baselines_smoke_tests/AutoML4Clust/``.

The test NEVER hides an error: on any failure it prints the exact error, the
Python version, ``sys.path``, whether ``external/Automl4Clust`` exists, and the
import traceback, then saves a ``failed`` smoke-test result and exits non-zero.

Run::

    python -m experiments.external_baselines.AutoML4Clust.runners.smoke_test_automl4clust
"""
from __future__ import annotations

import argparse
import platform
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from experiments.external_baselines.AutoML4Clust.adapters.automl4clust_adapter import (
    AutoML4ClustAdapter,
)
from experiments.external_baselines.AutoML4Clust.utils.logging_utils import configure_process
from experiments.external_baselines.AutoML4Clust.utils.metrics import compute_all_metrics
from experiments.external_baselines.AutoML4Clust.utils.path_utils import (
    ensure_automl4clust_on_path,
    ensure_dir,
    external_automl4clust_exists,
    external_automl4clust_src,
    read_json,
    write_json_atomic,
)

configure_process()

TAG = "[AutoML4Clust Smoke Test]"


def _repo_root_default() -> Path:
    # .../experiments/external_baselines/AutoML4Clust/runners/<this file>
    return Path(__file__).resolve().parents[4]


def _env_info(repo_root: Path) -> Dict[str, Any]:
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "repo_root": str(repo_root),
        "external_automl4clust_src": str(external_automl4clust_src(repo_root)),
        "external_automl4clust_exists": external_automl4clust_exists(repo_root),
        "sys_path": list(sys.path),
    }


def _print_diagnostics(repo_root: Path, error: Optional[str], tb: Optional[str]) -> None:
    print(f"{TAG} DIAGNOSTICS", flush=True)
    print(f"  python: {sys.version.splitlines()[0]}", flush=True)
    print(f"  executable: {sys.executable}", flush=True)
    print(f"  platform: {platform.platform()}", flush=True)
    print(f"  repo_root: {repo_root}", flush=True)
    print(f"  external/Automl4Clust src: {external_automl4clust_src(repo_root)}", flush=True)
    print(f"  external exists: {external_automl4clust_exists(repo_root)}", flush=True)
    if error:
        print(f"  error: {error}", flush=True)
    print("  sys.path:", flush=True)
    for p in sys.path:
        print(f"    {p}", flush=True)
    if tb:
        print("  traceback:", flush=True)
        print(tb, flush=True)


def _try_import(repo_root: Path) -> Optional[Dict[str, str]]:
    """Attempt to import AutoML4Clust. Returns None on success, else error dict."""
    if not external_automl4clust_exists(repo_root):
        return {
            "error_type": "FileNotFoundError",
            "error_message": f"external/Automl4Clust not found at "
                             f"{external_automl4clust_src(repo_root)}. "
                             f"Run: git submodule update --init --recursive",
            "traceback": "",
        }
    ensure_automl4clust_on_path(repo_root)
    try:
        from Optimizer.Optimizer import SMACOptimizer  # noqa: F401
        from Metrics.MetricHandler import MetricCollection  # noqa: F401
        from Algorithm.ClusteringAlgorithms import run_algorithm  # noqa: F401
        import ConfigSpace  # noqa: F401
        return None
    except Exception as exc:  # noqa: BLE001
        return {
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            ),
        }


def _make_synthetic(seed: int = 42, n: int = 300, k: int = 4):
    from sklearn.datasets import make_blobs
    X, y = make_blobs(n_samples=n, n_features=2, centers=k, random_state=seed,
                      cluster_std=0.60)
    return np.asarray(X, dtype=float), np.asarray(y), k


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="AutoML4Clust smoke test.")
    parser.add_argument("--repo-root", type=str, default=str(_repo_root_default()))
    parser.add_argument("--record-types", nargs="+", default=["2d"],
                        choices=["1d_x", "1d_y", "2d"])
    parser.add_argument("--config", type=str, default=None,
                        help="Optional config path; defaults to smoke_test_config.json.")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()
    out_dir = (
        repo_root / "results_analysis" / "clustering_repository"
        / "external_baselines_smoke_tests" / "AutoML4Clust"
    )
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    print(f"{TAG} Starting", flush=True)

    # Config.
    config_path = args.config or str(
        repo_root / "experiments" / "external_baselines" / "AutoML4Clust"
        / "configs" / "smoke_test_config.json"
    )
    config = read_json(config_path) or {
        "optimizer": "SMAC", "metric": "silhouette", "n_loops": 10,
        "normalize_input": True, "k_range": [2, 8],
    }

    # 1. Import check (explicit, with full diagnostics on failure).
    import_err = _try_import(repo_root)
    if import_err is not None:
        print(f"{TAG} Import FAILED", flush=True)
        _print_diagnostics(repo_root, import_err["error_message"], import_err["traceback"])
        result = {
            "smoke_test": "AutoML4Clust",
            "status": "failed",
            "stage": "import",
            "timestamp": ts,
            "error": import_err,
            "environment": _env_info(repo_root),
            "hint": "AutoML4Clust needs an isolated environment with its legacy "
                    "dependency stack (smac, hpbandster, ConfigSpace, "
                    "scikit-learn==0.21.3, scikit-learn-extra, pyclustering). "
                    "See experiments/external_baselines/AutoML4Clust/setup/.",
        }
        ensure_dir(out_dir)
        saved = out_dir / f"smoke_test_{ts}.json"
        write_json_atomic(saved, result)
        write_json_atomic(out_dir / "latest.json", result)
        print(f"{TAG} Saved results to {saved}", flush=True)
        return 1
    print(f"{TAG} Import OK", flush=True)

    # 2. Synthetic data.
    X, y_true, true_k = _make_synthetic()
    X_2d = X

    # 3. Fit/predict per record type through the adapter.
    adapter = AutoML4ClustAdapter(config, repo_root)
    per_record: Dict[str, Any] = {}
    overall_ok = True
    for record_type in args.record_types:
        if record_type == "1d_x":
            partition = X_2d[:, [0]]
        elif record_type == "1d_y":
            partition = X_2d[:, [1]]
        else:
            partition = X_2d
        adapter_result = adapter.fit_predict(partition, X_2d, record_type=record_type)

        if adapter_result.get("status") != "success":
            overall_ok = False
            print(f"{TAG} Fit/predict FAILED for {record_type}: "
                  f"{adapter_result.get('error')}", flush=True)
            _print_diagnostics(repo_root, adapter_result.get("error_message"),
                               adapter_result.get("traceback"))
            per_record[record_type] = {
                "status": "failed", "error": adapter_result.get("error"),
                "stage": adapter_result.get("stage"),
            }
            continue

        print(f"{TAG} Fit/predict OK ({record_type})", flush=True)
        labels = adapter_result.get("labels")
        if labels is None:
            overall_ok = False
            print(f"{TAG} Labels MISSING for {record_type}", flush=True)
            per_record[record_type] = {"status": "failed", "error": "no labels"}
            continue
        print(f"{TAG} Labels extracted OK ({record_type}, "
              f"selected_k={adapter_result.get('selected_k')})", flush=True)

        metrics = compute_all_metrics(y_true, np.asarray(labels), true_k)
        print(f"{TAG} Metrics computed OK ({record_type}): "
              f"ari={metrics.get('ari'):.4f} nmi={metrics.get('nmi'):.4f} "
              f"ami={metrics.get('ami'):.4f}", flush=True)
        per_record[record_type] = {
            "status": "success",
            "selected_k": adapter_result.get("selected_k"),
            "best_configuration": adapter_result.get("best_configuration"),
            "runtime_sec": adapter_result.get("runtime_sec"),
            "metrics": {k: metrics.get(k) for k in
                        ("ari", "nmi", "ami", "selected_k", "true_k",
                         "exact_k_match", "abs_k_error")},
        }

    result = {
        "smoke_test": "AutoML4Clust",
        "status": "success" if overall_ok else "failed",
        "timestamp": ts,
        "true_k": true_k,
        "n_points": int(X.shape[0]),
        "config_path": str(config_path),
        "records": per_record,
        "environment": _env_info(repo_root),
    }
    ensure_dir(out_dir)
    saved = out_dir / f"smoke_test_{ts}.json"
    write_json_atomic(saved, result)
    write_json_atomic(out_dir / "latest.json", result)
    print(f"{TAG} Saved results to {saved}", flush=True)
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
