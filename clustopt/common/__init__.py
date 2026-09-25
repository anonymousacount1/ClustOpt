"""Shared paths for the CLUSTOPT release.

The scientific implementation lives in ``src/`` under its original module
names (so that imports, recorded hashes and configuration files stay exactly as
they were when the reported results were produced). This package is a thin,
stable public entry point on top of it.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
MODELS = ROOT / "models"
CONFIGS = ROOT / "configs"
DATA = ROOT / "data"
RESULTS = ROOT / "results"

VIEWS = ("x_only", "y_only", "xy_2d")


def ensure_src_on_path() -> None:
    """Make the implementation packages importable (idempotent)."""
    for p in (SRC, SRC / "models", SRC / "models" / "ClustOpt",
              SRC / "experiments" / "external_baselines" / "Unified_MKR"):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)


ensure_src_on_path()
