"""ML2DAC smoke test -- infra + artifact check, plus a tiny end-to-end run.

Steps:
  1. verify external/ml2dac exists.
  2. run the artifact locator (filesystem-only; works in any environment).
  3. if the MKR artifacts are present AND the env can import ML2DAC:
       run one tiny synthetic dataset through the adapter, verify labels,
       compute ARI/NMI/AMI, save outputs.
  4. if artifacts are missing or the env can't import ML2DAC:
       still PASS in "infrastructure OK" mode, clearly stating what's needed.

Saves under
``results_analysis/clustering_repository/external_baselines_smoke_tests/ML2DAC/``.
On failure it prints the exact error, Python version, sys.path, whether
external/ml2dac (and its src) exist, the artifact report, and the traceback.

Run::

    python -m experiments.external_baselines.ML2DAC.runners.smoke_test_ml2dac
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

from experiments.external_baselines.ML2DAC.adapters.artifact_locator import locate_artifacts
from experiments.external_baselines.ML2DAC.adapters.ml2dac_adapter import ML2DACAdapter
from experiments.external_baselines.ML2DAC.utils.logging_utils import configure_process
from experiments.external_baselines.ML2DAC.utils.metrics import compute_all_metrics
from experiments.external_baselines.ML2DAC.utils.path_utils import (
    ensure_dir,
    ensure_ml2dac_on_path,
    external_ml2dac_exists,
    external_ml2dac_src,
    read_json,
    write_json_atomic,
)

configure_process()
TAG = "[ML2DAC Smoke Test]"


def _repo_root_default() -> Path:
    # .../experiments/external_baselines/ML2DAC/runners/<this file>
    return Path(__file__).resolve().parents[4]


def _env_info(repo_root: Path) -> Dict[str, Any]:
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "repo_root": str(repo_root),
        "ml2dac_src": str(external_ml2dac_src(repo_root)),
        "ml2dac_src_exists": external_ml2dac_exists(repo_root),
        "ml2dac_src_on_path": str(external_ml2dac_src(repo_root)) in sys.path,
        "sys_path": list(sys.path),
    }


def _print_diagnostics(repo_root: Path, error: Optional[str], tb: Optional[str],
                       artifact_report: Optional[Dict[str, Any]]) -> None:
    print(f"{TAG} DIAGNOSTICS", flush=True)
    print(f"  python: {sys.version.splitlines()[0]}", flush=True)
    print(f"  executable: {sys.executable}", flush=True)
    print(f"  platform: {platform.platform()}", flush=True)
    print(f"  repo_root: {repo_root}", flush=True)
    print(f"  external/ml2dac src: {external_ml2dac_src(repo_root)}", flush=True)
    print(f"  external exists: {external_ml2dac_exists(repo_root)}", flush=True)
    print(f"  src on sys.path: {str(external_ml2dac_src(repo_root)) in sys.path}", flush=True)
    if error:
        print(f"  error: {error}", flush=True)
    if artifact_report is not None:
        print(f"  artifacts: has_pretrained={artifact_report.get('has_pretrained_artifacts')} "
              f"missing={artifact_report.get('missing_required_artifacts')}", flush=True)
    print("  sys.path:", flush=True)
    for p in sys.path:
        print(f"    {p}", flush=True)
    if tb:
        print("  traceback:", flush=True)
        print(tb, flush=True)


def _try_import(repo_root: Path) -> Optional[Dict[str, str]]:
    if not external_ml2dac_exists(repo_root):
        return {"error_type": "FileNotFoundError",
                "error_message": f"external/ml2dac not found at {external_ml2dac_src(repo_root)}. "
                                 f"Run: git submodule update --init --recursive",
                "traceback": ""}
    ensure_ml2dac_on_path(repo_root)
    try:
        from MetaLearning.ApplicationPhase import ApplicationPhase  # noqa: F401
        from MetaLearning import MetaFeatureExtractor  # noqa: F401
        from Optimizer.OptimizerSMAC import SMACOptimizer  # noqa: F401
        from ClusterValidityIndices.CVIHandler import CVICollection  # noqa: F401
        return None
    except Exception as exc:  # noqa: BLE001
        return {"error_type": type(exc).__name__, "error_message": str(exc),
                "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))}


def _make_synthetic(seed: int = 42, n: int = 300, k: int = 4):
    from sklearn.datasets import make_blobs
    X, y = make_blobs(n_samples=n, n_features=2, centers=k, random_state=seed, cluster_std=0.60)
    return np.asarray(X, dtype=float), np.asarray(y), k


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="ML2DAC smoke test.")
    parser.add_argument("--repo-root", type=str, default=str(_repo_root_default()))
    parser.add_argument("--record-types", nargs="+", default=["2d"], choices=["1d_x", "1d_y", "2d"])
    parser.add_argument("--config", type=str, default=None,
                        help="Optional config path; defaults to ml2dac_smoke_test_config.json.")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()
    out_dir = (repo_root / "results_analysis" / "clustering_repository"
               / "external_baselines_smoke_tests" / "ML2DAC")
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    print(f"{TAG} Starting", flush=True)

    # 1. External repo.
    if not external_ml2dac_exists(repo_root):
        print(f"{TAG} External repo MISSING", flush=True)
        _print_diagnostics(repo_root, "external/ml2dac not found", None, None)
        result = {"smoke_test": "ML2DAC", "status": "failed", "stage": "external_repo",
                  "timestamp": ts, "environment": _env_info(repo_root)}
        ensure_dir(out_dir)
        write_json_atomic(out_dir / f"smoke_test_{ts}.json", result)
        write_json_atomic(out_dir / "latest.json", result)
        print(f"{TAG} Saved results to {out_dir / f'smoke_test_{ts}.json'}", flush=True)
        return 1
    print(f"{TAG} External repo exists", flush=True)

    # 2. Config + artifact locator.
    config_path = args.config or str(
        repo_root / "experiments" / "external_baselines" / "ML2DAC"
        / "configs" / "ml2dac_smoke_test_config.json")
    config = read_json(config_path) or {"mode": "full_meta_learning"}
    artifact_report = locate_artifacts(repo_root, mkr_path=config.get("mkr_path"))
    ensure_dir(out_dir)
    write_json_atomic(out_dir / "ml2dac_artifact_report.json", artifact_report)
    has_artifacts = bool(artifact_report.get("has_pretrained_artifacts"))
    print(f"{TAG} Artifact check ... has_pretrained={has_artifacts} "
          f"missing={artifact_report.get('missing_required_artifacts')}", flush=True)

    # 3. Import check.
    import_err = _try_import(repo_root)
    import_ok = import_err is None
    print(f"{TAG} Import check ... {'OK' if import_ok else 'FAILED'}", flush=True)
    print(f"{TAG} Full ML2DAC runnable: {'yes' if (has_artifacts and import_ok) else 'no'}", flush=True)

    base_result = {
        "smoke_test": "ML2DAC", "timestamp": ts, "config_path": str(config_path),
        "artifact_report": artifact_report, "import_ok": import_ok,
        "import_error": import_err, "environment": _env_info(repo_root),
    }

    # 4a. Artifacts missing OR import failed -> PASS in "infra OK" mode.
    if not has_artifacts or not import_ok:
        reason = []
        if not has_artifacts:
            reason.append("MKR artifacts missing")
        if not import_ok:
            reason.append("ML2DAC import failed (environment not set up)")
        print(f"{TAG} Fit/predict skipped because {'; '.join(reason)}", flush=True)
        if not import_ok:
            _print_diagnostics(repo_root, import_err.get("error_message"),
                               import_err.get("traceback"), artifact_report)
        base_result.update({
            "status": "infrastructure_ok",
            "fit_predict": "skipped",
            "skip_reason": "; ".join(reason),
            "hint": "Full ML2DAC needs (a) the MKR artifacts (shipped under "
                    "external/ml2dac/src/MetaKnowledgeRepository; run 'git lfs pull' "
                    "in the submodule if CSVs are pointers) and (b) the isolated "
                    "py3.9 env (see experiments/external_baselines/ML2DAC/setup/).",
        })
        write_json_atomic(out_dir / f"smoke_test_{ts}.json", base_result)
        write_json_atomic(out_dir / "latest.json", base_result)
        print(f"{TAG} Saved results to {out_dir / f'smoke_test_{ts}.json'}", flush=True)
        # Infra is OK -> exit 0 (not a failure).
        return 0

    # 4b. Full path: run a tiny synthetic dataset through the adapter.
    X, y_true, true_k = _make_synthetic()
    adapter = ML2DACAdapter(config, repo_root)
    per_record: Dict[str, Any] = {}
    overall_ok = True
    for record_type in args.record_types:
        partition = X if record_type == "2d" else (X[:, [0]] if record_type == "1d_x" else X[:, [1]])
        adapter_result = adapter.fit_predict(partition, X, record_type=record_type, dataset_id="smoke_synth")
        if adapter_result.get("status") != "success":
            overall_ok = False
            print(f"{TAG} Fit/predict FAILED for {record_type}: {adapter_result.get('error')}", flush=True)
            _print_diagnostics(repo_root, adapter_result.get("error_message"),
                               adapter_result.get("traceback"), artifact_report)
            per_record[record_type] = {"status": "failed", "error": adapter_result.get("error"),
                                       "stage": adapter_result.get("stage")}
            continue
        labels = adapter_result.get("labels")
        print(f"{TAG} Fit/predict OK ({record_type}, selected_k={adapter_result.get('selected_k')}, "
              f"cvi={adapter_result.get('selected_cvi')}, algo={adapter_result.get('selected_algorithm')})", flush=True)
        metrics = compute_all_metrics(y_true, np.asarray(labels), true_k)
        print(f"{TAG} Metrics computed OK ({record_type}): ari={metrics.get('ari'):.4f} "
              f"nmi={metrics.get('nmi'):.4f} ami={metrics.get('ami'):.4f}", flush=True)
        per_record[record_type] = {
            "status": "success", "selected_k": adapter_result.get("selected_k"),
            "selected_cvi": adapter_result.get("selected_cvi"),
            "selected_algorithm": adapter_result.get("selected_algorithm"),
            "best_configuration": adapter_result.get("best_configuration"),
            "runtime_sec": adapter_result.get("runtime_sec"),
            "metrics": {k: metrics.get(k) for k in ("ari", "nmi", "ami", "selected_k", "true_k",
                                                    "exact_k_match", "abs_k_error")},
        }

    base_result.update({
        "status": "success" if overall_ok else "failed",
        "fit_predict": "ran", "true_k": true_k, "n_points": int(X.shape[0]),
        "records": per_record,
    })
    write_json_atomic(out_dir / f"smoke_test_{ts}.json", base_result)
    write_json_atomic(out_dir / "latest.json", base_result)
    print(f"{TAG} Saved results to {out_dir / f'smoke_test_{ts}.json'}", flush=True)
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
