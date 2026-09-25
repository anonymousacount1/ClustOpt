r"""Standalone ML2DAC (pymfe) meta-feature pass over an existing subfamily build.

Populates the ``ml2dac_<feature>`` columns in ``metafeatures.parquet`` using the
isolated ``.mfenv`` pymfe interpreter, without touching the ClustOpt env. Run
this after a main build if the ML2DAC meta-features came back ``unavailable``
(or simply rely on the main runner auto-invoking it).

Example
-------
    python experiments/external_baselines/Unified_MKR/runners/run_ml2dac_metafeatures_pass.py \
      --subfamily-out "experiments/external_baselines/Unified_MKR/outputs/arcs_circles_rings/annuli_variable_thickness"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
for _p in [_THIS, *_THIS.parents]:
    if (_p / "models" / "ClustOpt").is_dir():
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
        break

from experiments.external_baselines.Unified_MKR.unified_mkr import io_utils  # noqa: E402
from experiments.external_baselines.Unified_MKR.unified_mkr.logging_utils import log  # noqa: E402
from experiments.external_baselines.Unified_MKR.unified_mkr.pymfe_pass import (  # noqa: E402
    find_mf_python,
    run_pymfe_pass,
)


def main() -> int:
    p = argparse.ArgumentParser(description="ML2DAC pymfe meta-feature pass.")
    p.add_argument("--subfamily-out", required=True, type=str,
                   help="A subfamily output dir (contains records.parquet).")
    p.add_argument("--mf-python", type=str, default=None,
                   help="Path to the pymfe interpreter (default: auto-detect .mfenv).")
    p.add_argument("--repo-root", type=str, default=None)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    repo_root = Path(args.repo_root).resolve() if args.repo_root else io_utils.find_repo_root()
    mf_python = Path(args.mf_python) if args.mf_python else find_mf_python(repo_root)
    if not mf_python or not Path(mf_python).is_file():
        log("[pymfe-pass] ERROR: no pymfe interpreter found. Create one with "
            "setup/create_mf_env, or pass --mf-python.")
        return 1

    log(f"[pymfe-pass] using interpreter: {mf_python}")
    result = run_pymfe_pass(args.subfamily_out, mf_python, overwrite=args.overwrite)
    log(f"[pymfe-pass] result: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
