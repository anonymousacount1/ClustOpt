"""Stage 3B runner, part 2: ML2DAC selection diagnostics and AutoClust reliance.

Analysis only. Applies two already-persisted models' *stored outputs* (ML2DAC)
and one already-trained model for inference (AutoClust). Nothing is fitted.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metric_atlas import common as C        # noqa: E402
from metric_atlas import ml2dac as ML       # noqa: E402
from metric_atlas import autoclust as AC    # noqa: E402


def log(m: str) -> None:
    print("[3b] %s" % m, flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=None)
    ap.add_argument("--skip-ml2dac", action="store_true")
    ap.add_argument("--skip-autoclust", action="store_true")
    ap.add_argument("--autoclust-max-rows", type=int, default=0,
                    help="0 = use the full held-out Split-1 population")
    args = ap.parse_args(argv)
    repo = Path(args.repo_root).resolve() if args.repo_root else C.REPO
    out = C.out_dir(repo)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                            capture_output=True, text=True).stdout.strip()
    print("=" * 78)
    print("STAGE 3B  METRIC ATTRIBUTION  (part 2: ML2DAC + AutoClust)")
    print("=" * 78, flush=True)

    A = pd.read_csv(C.ext(out / "canonical_metric_atlas.csv"))

    if not args.skip_ml2dac:
        sel = ML.collect_selections(repo, log)
        C.write_csv(sel, out / "ml2dac_selection_records.csv.gz", gz=True)
        star = ML.split1_cvi_star(repo, log)
        C.write_csv(star, out / "ml2dac_split1_cvi_star.csv.gz", gz=True)
        S, Arm, R, F = ML.analyse(sel, star, A, log)
        C.write_csv(S, out / "ml2dac_metric_selection.csv")
        C.write_csv(Arm, out / "ml2dac_cvi_star_diagnostics.csv")
        C.write_csv(R, out / "ml2dac_per_cvi_recall.csv")
        C.write_csv(F, out / "ml2dac_family_metric_selection.csv")

    if not args.skip_autoclust:
        Gp, Ip, Fp, prov = AC.reliance(repo, A, log, max_rows=args.autoclust_max_rows)
        C.write_csv(Gp, out / "autoclust_group_reliance.csv")
        C.write_csv(Ip, out / "autoclust_metric_reliance.csv")
        C.write_csv(Fp, out / "autoclust_family_group_reliance.csv")
        C.write_json(prov, out / "autoclust_reliance_provenance.json")

    C.write_json({"stage": "3B", "part": 2, "created_at": C.now(),
                  "source_commit": commit, "analysis_only": True,
                  "models_trained": 0, "clustering_runs": 0,
                  "ml2dac": "stored selected_cvi + frozen compute_cvi_star on "
                            "stored Split-1 evaluations",
                  "autoclust": "existing A3 ari_predictor.pkl, inference only"},
                 out / "part2_manifest.json")
    log("part 2 complete -> %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
