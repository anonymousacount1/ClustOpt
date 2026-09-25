"""Environment provenance. Runtime claims are only controlled when this matches."""
from __future__ import annotations

import os
import platform
import subprocess
import sys
from typing import Any, Dict

THREAD_ENV = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
              "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}
WORKERS = 8


def apply_thread_policy() -> None:
    """Must run before numpy/sklearn are imported in every worker process."""
    for k, v in THREAD_ENV.items():
        os.environ[k] = v


def capture(repo_root: str) -> Dict[str, Any]:
    def git(*a, cwd=repo_root):
        try:
            return subprocess.run(["git", *a], cwd=cwd, capture_output=True,
                                  text=True).stdout.strip()
        except Exception:
            return ""

    try:
        import psutil
        ram = int(psutil.virtual_memory().total // (1024 ** 2))
        cpus = psutil.cpu_count(logical=True)
    except Exception:
        ram, cpus = None, os.cpu_count()
    mods = {}
    for m in ("numpy", "scipy", "sklearn", "pandas", "hdbscan", "optuna"):
        try:
            mods[m] = __import__(m).__version__
        except Exception:
            mods[m] = None
    return {"machine": platform.node(), "os": platform.platform(),
            "python": sys.version.split()[0], "cpu": platform.processor(),
            "cpu_count": cpus, "ram_mb": ram, "libraries": mods,
            "workers": WORKERS, "thread_env": dict(THREAD_ENV),
            "git_commit": git("rev-parse", "HEAD"),
            "external_submodule_commit": git(
                "rev-parse", "HEAD",
                cwd=os.path.join(repo_root, "external", "ml2dac"))}
