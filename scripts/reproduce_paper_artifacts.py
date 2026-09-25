"""Rebuild the paper's evidence tables, LaTeX tables and figures from the
released aggregate results (``results/``). No experiment is re-run.

Outputs go to ``analysis/output/paper/{evidence,tables,macros,figures,manifests}``.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "analysis" / "paper"
OUT = ROOT / "analysis" / "output" / "paper"


def run(script: str) -> None:
    print("==>", script)
    subprocess.run([sys.executable, script], cwd=PAPER, check=True)


def main() -> int:
    (OUT / "evidence").mkdir(parents=True, exist_ok=True)
    # the representative-dataset gallery is a frozen selection (it needs the full
    # generated corpus to recompute); every other input is an aggregate table
    shutil.copyfile(ROOT / "results" / "paper_evidence" / "gallery_points.json", OUT / "evidence" / "gallery_points.json")
    for s in ("build_evidence.py", "build_tables.py", "build_figures.py"):
        run(s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
