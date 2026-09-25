"""Link the released models into the implementation workspace (run once).

The frozen experiment configurations refer to trained models by the location
they had when the experiments were run (for example ``model_run_dir`` in
``configs/clustopt/neural_profile_top5/*.json``). Those files are kept
byte-identical, so instead of rewriting them this script creates links from the
recorded locations (under ``src/``) to the public model folders in ``models/``.
Links are directory symlinks, Windows junctions, or copies as a fallback.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

# recorded location (relative to src/)            -> public folder (relative to the release root)
LINKS = {
    "results_analysis/mlp/20260615_231715__metric_utility_mlp_split1_holdout": "models/utility_predictor/neural_full60",
    "results_analysis/mlp/20260823_195830__metric_utility_mlp_head14_split1_holdout": "models/utility_predictor/neural_head14",
    "results_analysis/metric_utility_knn/phase_e1/models": "models/utility_predictor/knn",
    "results_analysis/clustopt_candidate_reranker/stage2b5a_final_models_and_split1_preparation/final_models":
        "models/candidate_reranker",
}


def _link(target: Path, link: Path) -> str:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists() or link.is_symlink():
        return "exists"
    try:
        os.symlink(target, link, target_is_directory=True)
        return "symlink"
    except OSError:
        pass
    if sys.platform == "win32":
        r = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], capture_output=True, text=True)
        if r.returncode == 0:
            return "junction"
    shutil.copytree(target, link)
    return "copy"


def main() -> int:
    for rec, pub in LINKS.items():
        how = _link(ROOT / pub, SRC / rec)
        print("%-9s src/%s -> %s" % (how, rec, pub))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
