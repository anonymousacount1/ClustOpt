"""Stage 2C-4A: freeze the outcome-blind development sample and price the batch.

Selects an ORDERED 12 datasets per development split. Ranks 1-8 are the
pre-registered runtime fallback, so the smaller sample is a deterministic subset
of the larger one and the pilot cannot influence which datasets are retained.

Selection is outcome-blind: it maximises family coverage, then subfamily
diversity, then OOF-v2 ten-regime diversity, and breaks residual ties with seed
42. No RAW/SOFTMAX outcome, no ARI and no Split-1 row-level information is read.

It also derives the exact set of ``(split, regime)`` OOF Candidate-Rerankers the
sample would require, which is what prices the §6 blocker.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from models.Clustering_Repository_Builder.experiments.experiment_execution.output_writer import (  # noqa: E402,E501
    _ext, ensure_dir, read_json, write_csv_atomic, write_json_atomic,
)
from models.ClustOpt_Policy_Predictor.stage2c import regime_inventory as RI  # noqa: E402,E501

OUT_REL = ("results_analysis/clustopt_policy_predictor/"
           "stage2c4_online_weighting_mode_dev")
SCREEN_REL = "results_analysis/clustopt_policy_predictor/stage2c_v2_screening"
SPLIT_CSV = ("results_analysis/clustering_repository/analyzed_data/"
             "experiment_splits/dataset_split_assignments.csv")
N_PRIMARY, N_FALLBACK = 12, 8
SEED = 42
VIEWS = ("x_only", "y_only", "xy_2d")


def oof_v2_regimes(repo: Path) -> pd.DataFrame:
    """Frozen OOF-v2 ten-regime selection per (dataset, view), Splits 2-16.

    Taken from the Stage-2C2A row-level fold files: each row's regime was chosen
    by a v2 fitted WITHOUT that row's split, which is exactly the OOF selection
    the brief requires. The final all-development v2 is deliberately not used.
    """
    frames = []
    for f in sorted((repo / SCREEN_REL).glob("rowlevel_fold_*.csv.gz")):
        frames.append(pd.read_csv(_ext(f), usecols=[
            "dataset_id", "view_id", "outer_split", "v2_selected_regime"]))
    d = pd.concat(frames, ignore_index=True)
    return d.rename(columns={"outer_split": "split_id",
                             "v2_selected_regime": "regime_id"})


def pick_for_split(sub: pd.DataFrame, n: int, rng: np.random.Generator
                   ) -> List[str]:
    """Ordered dataset ids: round-robin over families, then subfamilies.

    Deterministic. Diversity of the OOF-v2 regime set breaks ties between equally
    novel subfamilies; the RNG is consulted last and only for residual ties.
    """
    by_ds = (sub.groupby("dataset_id")
             .agg(family=("family_id", "first"),
                  subfamily=("subfamily_id", "first"),
                  regimes=("regime_id", lambda s: tuple(sorted(set(s)))))
             .reset_index())
    order = np.argsort(rng.random(len(by_ds)))
    by_ds = by_ds.iloc[order].reset_index(drop=True)

    chosen: List[str] = []
    seen_fam: Dict[str, int] = {}
    seen_sub: Dict[str, int] = {}
    seen_reg: Dict[str, int] = {}
    pool = by_ds.to_dict("records")
    while len(chosen) < n and pool:
        def score(r: Dict[str, Any]):
            return (seen_fam.get(r["family"], 0),
                    seen_sub.get(r["subfamily"], 0),
                    min((seen_reg.get(g, 0) for g in r["regimes"]),
                        default=0),
                    r["dataset_id"])
        pool.sort(key=score)
        pick = pool.pop(0)
        chosen.append(pick["dataset_id"])
        seen_fam[pick["family"]] = seen_fam.get(pick["family"], 0) + 1
        seen_sub[pick["subfamily"]] = seen_sub.get(pick["subfamily"], 0) + 1
        for g in pick["regimes"]:
            seen_reg[g] = seen_reg.get(g, 0) + 1
    return chosen


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", default=".")
    args = p.parse_args(argv)
    repo = Path(args.repo_root).resolve()
    out = ensure_dir(repo / OUT_REL)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                            capture_output=True, text=True).stdout.strip()
    t0 = time.time()
    print("=" * 78)
    print("STAGE 2C-4A  OUTCOME-BLIND DEVELOPMENT SAMPLE FREEZE")
    print("=" * 78, flush=True)

    reg = oof_v2_regimes(repo)
    assign = pd.read_csv(_ext(repo / SPLIT_CSV),
                         usecols=["dataset_id", "split_id", "family_id",
                                  "subfamily_id"]).drop_duplicates("dataset_id")
    if (assign[assign["dataset_id"].isin(set(reg["dataset_id"]))]["split_id"]
            == 1).any():
        raise SystemExit("Split 1 present in the development regime table")
    d = reg.merge(assign.drop(columns=["split_id"]), on="dataset_id", how="left")

    rows: List[Dict[str, Any]] = []
    rng = np.random.default_rng(SEED)
    for s in RI.OUTER_SPLITS:
        sub = d[d["split_id"] == s]
        picks = pick_for_split(sub, N_PRIMARY, rng)
        for rank, ds in enumerate(picks, start=1):
            g = sub[sub["dataset_id"] == ds]
            for v in VIEWS:
                gv = g[g["view_id"] == v]
                if gv.empty:
                    raise AssertionError("missing view %s for %s" % (v, ds))
                rows.append({
                    "split_id": s, "rank": rank, "dataset_id": ds,
                    "view_id": v,
                    "family_id": gv["family_id"].iloc[0],
                    "subfamily_id": gv["subfamily_id"].iloc[0],
                    "oof_v2_regime": gv["regime_id"].iloc[0],
                    "in_fallback_n8": rank <= N_FALLBACK})
    SAMPLE = pd.DataFrame(rows)
    write_csv_atomic(out / "stage2c4_dev_sample.csv", SAMPLE)

    payload = json.dumps(
        SAMPLE[["split_id", "rank", "dataset_id", "view_id"]]
        .sort_values(["split_id", "rank", "view_id"])
        .to_dict("records"), separators=(",", ":"))
    shash = hashlib.sha256(payload.encode()).hexdigest()[:16]

    prim = SAMPLE
    fall = SAMPLE[SAMPLE["in_fallback_n8"]]
    need_prim = (prim[["split_id", "oof_v2_regime"]].drop_duplicates()
                 .sort_values(["split_id", "oof_v2_regime"]))
    need_fall = (fall[["split_id", "oof_v2_regime"]].drop_duplicates())
    write_csv_atomic(out / "required_oof_rerankers.csv",
                     need_prim.rename(columns={"oof_v2_regime": "regime_id"}))

    man = {
        "n_primary_per_split": N_PRIMARY, "n_fallback_per_split": N_FALLBACK,
        "fallback_rule": "ranks 1-8 of the same ordered list; the fallback is a "
                         "deterministic subset of the primary sample and the "
                         "pilot cannot change membership",
        "seed": SEED,
        "selection_priorities": ["family coverage", "subfamily diversity",
                                 "OOF-v2 regime diversity",
                                 "seed 42 for residual ties"],
        "regime_source": "Stage-2C2A row-level OOF-v2 selections (v2 fitted "
                         "without the row's own split)",
        "outcome_blind": True, "split_1_used": False,
        "primary": {
            "datasets": int(prim["dataset_id"].nunique()),
            "rows": int(len(prim)), "searches": int(len(prim) * 2),
            "families": int(prim["family_id"].nunique()),
            "subfamilies": int(prim["subfamily_id"].nunique()),
            "regimes": int(prim["oof_v2_regime"].nunique()),
            "required_oof_rerankers": int(len(need_prim))},
        "fallback_n8": {
            "datasets": int(fall["dataset_id"].nunique()),
            "rows": int(len(fall)), "searches": int(len(fall) * 2),
            "families": int(fall["family_id"].nunique()),
            "subfamilies": int(fall["subfamily_id"].nunique()),
            "regimes": int(fall["oof_v2_regime"].nunique()),
            "required_oof_rerankers": int(len(need_fall))},
        "sample_hash": shash, "source_commit": commit,
        "runtime_sec": time.time() - t0}
    write_json_atomic(out / "stage2c4_dev_sample_manifest.json", man)

    print("  primary  : %d datasets, %d rows, %d searches | %d families, "
          "%d subfamilies, %d regimes"
          % (man["primary"]["datasets"], man["primary"]["rows"],
             man["primary"]["searches"], man["primary"]["families"],
             man["primary"]["subfamilies"], man["primary"]["regimes"]),
          flush=True)
    print("  fallback : %d datasets, %d rows, %d searches"
          % (man["fallback_n8"]["datasets"], man["fallback_n8"]["rows"],
             man["fallback_n8"]["searches"]), flush=True)
    print("  sample hash %s" % shash, flush=True)
    print("\n  REQUIRED OOF RERANKERS (split, regime) pairs:", flush=True)
    print("    primary  N=12 : %d" % len(need_prim), flush=True)
    print("    fallback N=8  : %d" % len(need_fall), flush=True)
    print("\n" + prim.groupby("oof_v2_regime").size().rename("rows")
          .to_string(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
