"""Shared helper for the public run wrappers: run an implementation module from src/."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def run_module(module: str, args, cwd: Path = SRC) -> int:
    env = dict(os.environ)
    extra = [str(SRC), str(SRC / "models"), str(SRC / "models" / "ClustOpt"),
             str(SRC / "experiments" / "external_baselines" / "Unified_MKR")]
    env["PYTHONPATH"] = os.pathsep.join(extra + [env.get("PYTHONPATH", "")])
    cmd = [sys.executable, "-m", module] + list(args)
    print("->", " ".join(cmd[2:]), flush=True)
    return subprocess.call(cmd, cwd=str(cwd), env=env)


def run_script(path: Path, args) -> int:
    return subprocess.call([sys.executable, str(path)] + list(args), cwd=str(ROOT))
