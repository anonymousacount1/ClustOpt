"""Run manifest and environment capture.

The environment is recorded, never versioned or pinned from here: the manifest
states what the run actually executed on so a later reader can tell whether two
runs are comparable. The isolated ML2DAC ``.mfenv`` is captured separately from
the main interpreter because the two are deliberately different.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

from . import hashing as H

_ROOT = Path(__file__).resolve().parents[1]
_REPO = _ROOT.parents[1]
MANIFEST_VERSION = "idb_stage4_manifest_v1"

_DEPS = ("numpy", "scipy", "pandas", "sklearn", "joblib", "hdbscan", "cv2",
         "optuna", "torch")


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=_REPO, capture_output=True,
                              text=True, timeout=60).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unavailable"


def _versions() -> Dict[str, str]:
    out: Dict[str, str] = {}
    for name in _DEPS:
        try:
            mod = __import__(name)
            out[name] = str(getattr(mod, "__version__", "unknown"))
        except Exception:  # noqa: BLE001
            out[name] = "absent"
    return out


def mfenv_identity() -> Dict[str, str]:
    """The isolated ML2DAC meta-feature interpreter, captured separately."""
    try:
        from Independent_Domain_Benchmark.methods import ml2dac as ML
        py = ML.resolve_mfenv_python(_REPO)
        out = subprocess.run(
            [str(py), "-c", "import sys,pymfe,numpy,pandas;print(sys.version.split()[0],"
                            "pymfe.__version__,numpy.__version__,pandas.__version__)"],
            capture_output=True, text=True, timeout=180).stdout.split()
        got = dict(zip(("python", "pymfe", "numpy", "pandas"), out))
        got["interpreter"] = str(py)
        got["matches_frozen"] = str(
            {k: got.get(k) for k in ML.FROZEN_MF_ENV} == dict(ML.FROZEN_MF_ENV))
        return got
    except Exception as exc:  # noqa: BLE001
        return {"error": "%s: %s" % (type(exc).__name__, str(exc)[:150])}


def config_hashes() -> Dict[str, Any]:
    cfg = _ROOT / "configs"
    names = ("benchmark_protocol.json", "method_registry.json",
             "metric_registry.json", "dataset_manifest.json",
             "numerical_reproducibility_policy.json",
             "ml2dac_direction_registry.json",
             "ml2dac_cvi_applicability_policy.json",
             "protocol_amendment_ml2dac_scoring_semantics.json")
    return {n: H.file_hashes(cfg / n) for n in names if (cfg / n).is_file()}


def build_manifest(run_id: str, *, thread_controls: Mapping[str, str],
                   extra: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    from Independent_Domain_Benchmark.methods import ml2dac_scoring as MS

    cfg = _ROOT / "configs"
    corpus = json.loads((cfg / "dataset_manifest.json").read_text(
        encoding="utf-8")).get("manifest_sha256_16")
    status = _git("status", "--porcelain")
    man: Dict[str, Any] = {
        "manifest_version": MANIFEST_VERSION, "run_id": run_id,
        "hash_conventions_version": H.HASH_CONVENTIONS_VERSION,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "cpu_count": os.cpu_count(),
        "dependency_versions": _versions(),
        "thread_environment": dict(thread_controls),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(status),
        "git_dirty_paths": [l[3:] for l in status.splitlines()[:40]],
        "submodule_status": _git("submodule", "status"),
        "external_ml2dac_commit": _git("-C", "external/ml2dac", "rev-parse", "HEAD"),
        "corpus_hash": corpus,
        "search_space_fingerprint": "8ddf86c79888ebe9",
        "config_hashes": config_hashes(),
        "scoring_semantics": {"ml2dac": MS.SCORING_SEMANTICS_ID},
        "ml2dac_mfenv": mfenv_identity(),
        "numerical_reproducibility_policy_version": json.loads(
            (cfg / "numerical_reproducibility_policy.json").read_text(
                encoding="utf-8"))["version"],
        "environment_is_recorded_not_versioned": True,
    }
    man.update(dict(extra or {}))
    man["manifest_hash"] = H.semantic_object_hash(man)
    return man
