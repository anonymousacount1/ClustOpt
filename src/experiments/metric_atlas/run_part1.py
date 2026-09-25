"""Stage 3B runner, part 1: atlas, utility, structure, complementarity, ClustOpt.

Analysis only. Reads frozen artefacts, writes tables under the Stage-3B results
root. No clustering, no training, no MKR rebuild.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metric_atlas import common as C          # noqa: E402
from metric_atlas import atlas as AT          # noqa: E402
from metric_atlas import structure as ST      # noqa: E402
from metric_atlas import clustopt as CO       # noqa: E402


def log(m: str) -> None:
    print("[3b] %s" % m, flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=None)
    args = ap.parse_args(argv)
    repo = Path(args.repo_root).resolve() if args.repo_root else C.REPO
    out = C.out_dir(repo)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                            capture_output=True, text=True).stdout.strip()
    print("=" * 78)
    print("STAGE 3B  EXISTING-DOMAIN METRIC ATTRIBUTION ATLAS  (part 1)")
    print("=" * 78, flush=True)

    # ---------------------------------------------------------------- atlas
    A, dead, defn = AT.build_atlas(repo, log)
    C.write_csv(A, out / "canonical_metric_atlas.csv")
    C.write_csv(dead, out / "dead_metric_registry.csv")
    C.write_json(defn, out / "metric_utility_definition.json")

    # -------------------------------------------------------------- utility
    oof = C.load_oof(repo)
    log("OOF table: %d rows, %d datasets, splits %s"
        % (len(oof), oof.dataset_id.nunique(), sorted(oof.heldout_split.unique())[:3] + ["..."]))
    G, V, D = AT.global_utility(repo, oof, A, log)
    C.write_csv(G, out / "metric_global_utility.csv")
    C.write_csv(V, out / "metric_view_level_utility_descriptive.csv")

    S, Ct = AT.group_summary(G, D, log)
    C.write_csv(S, out / "metric_group_utility_summary.csv")
    C.write_csv(Ct, out / "metric_group_contrasts.csv")

    # ------------------------------------------------------------ structure
    keys = ST.family_maps(oof)
    metrics = list(G["metric_id"])
    FAM = ST.enrichment(ST.group_utility(D, metrics, keys, "family_name", log), G,
                        "family_name")
    SUB = ST.enrichment(ST.group_utility(D, metrics, keys, "subfamily_id", log), G,
                        "subfamily_id")
    FAM = FAM.merge(A[["metric_id", "group", "subgroup"]], on="metric_id", how="left")
    SUB = SUB.merge(A[["metric_id", "group", "subgroup"]], on="metric_id", how="left")
    C.write_csv(FAM, out / "metric_family_utility.csv.gz", gz=True)
    C.write_csv(SUB, out / "metric_subfamily_utility.csv.gz", gz=True)

    TOP = G[["metric_id", "group", "subgroup", "top1_freq", "top3_freq",
             "top5_freq", "top10_freq", "mean_rank", "utility_rank_global"]].copy()
    C.write_csv(TOP, out / "metric_topk_coverage.csv")

    # ------------------------------------------------------ complementarity
    Cp, Cov, Go = ST.complementarity(D, A, keys, log)
    C.write_csv(Cp, out / "new46_complementarity.csv")
    C.write_csv(Cov, out / "new46_unique_win_coverage.csv")
    C.write_csv(Go, out / "group_oracle_coverage.csv")

    # ------------------------------------------------------------- ClustOpt
    U, Q, F = CO.usage_and_quality(oof, A, log)
    C.write_csv(U, out / "clustopt_metric_usage.csv")
    C.write_csv(Q, out / "clustopt_metric_prediction_quality.csv")
    C.write_csv(F, out / "clustopt_family_metric_usage.csv")
    C.write_json(CO.stage1_context(repo), out / "stage1_bundle_context.json")

    C.write_json({
        "stage": "3B", "part": 1, "created_at": C.now(), "source_commit": commit,
        "analysis_only": True,
        "population": {
            "utility_table": C.OOF_REL, "splits": "2-16",
            "datasets": int(oof.dataset_id.nunique()), "records": int(len(oof)),
            "split_1_present": False},
        "inferential_unit": "dataset (views averaged)",
        "bootstrap": {"B": C.BOOT_B, "seed": C.BOOT_SEED, "type": "percentile"},
        "multiplicity": "Benjamini-Hochberg within each predeclared test family",
        "clustopt_selection_semantics": {
            "top_k": C.CLUSTOPT_TOP_K, "weighting": C.CLUSTOPT_WEIGHTING,
            "frozen_arm": C.CLUSTOPT_ARM_ID,
            "reconstructed_from": "stored OOF predicted utilities; no model run"},
    }, out / "part1_manifest.json")
    log("part 1 complete -> %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
