"""Stage 3A-1: the ML2DAC four-arm metric-inventory decomposition analysis.

Thin runner over :mod:`unified_mkr.decomposition_analysis`, which both baselines
share so their statistical conventions are identical by construction. Analysis
only: A0/A3 are read exactly as the historical runs left them, and no raw result
tree is duplicated into the output directory.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence

_THIS = Path(__file__).resolve()
_PKG_PARENT = _THIS.parents[1]
if str(_PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(_PKG_PARENT))

from unified_mkr import decomposition_analysis as DA  # noqa: E402
from unified_mkr import io_utils  # noqa: E402

OUT_REL = ("results_analysis/clustering_repository/external_baselines/"
           "ML2DAC_metric_decomposition_split1")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="ML2DAC four-arm decomposition")
    ap.add_argument("--repo-root", default=None)
    args = ap.parse_args(argv)
    repo = Path(args.repo_root).resolve() if args.repo_root else io_utils.find_repo_root()
    io_utils.ensure_repo_on_path(repo)
    out = io_utils.ensure_dir(repo / OUT_REL)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                            capture_output=True, text=True).stdout.strip()
    print("=" * 78)
    print("STAGE 3A-1  ML2DAC FOUR-ARM METRIC DECOMPOSITION")
    print("=" * 78, flush=True)
    return DA.run(repo, "ml2dac", out, commit,
                  lambda m: print("[ml2dac-decomp] %s" % m, flush=True))


if __name__ == "__main__":
    raise SystemExit(main())
