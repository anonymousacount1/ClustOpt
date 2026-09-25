"""Stage 2C-1: assemble the reranker-aware OOF policy target repository.

Consumes the 150 single-exclusion units (reused + newly fitted) and produces the
Stage-2C supervision in the frozen Stage-2A interface: 48,039 dataset/view rows x
20 policy columns, plus the CENTERED transform computed with the unmodified
Stage-2A ``target_transforms.transform``.

Ten regimes, twenty columns. On the fixed-32 slate RAW and SOFTMAX for the same
(source, K) share slate, mask and reranker, and the frozen C2 interface carries
no weighting mode -- so their reranked outcome is identical BY CONSTRUCTION. The
duplication is preserved honestly rather than perturbed.
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
    _ext, ensure_dir, write_csv_atomic, write_json_atomic,
)
from models.ClustOpt_Candidate_Reranker.training.final_models import (  # noqa: E402,E501
    policy_mapping as PM,
)
from models.ClustOpt_Policy_Predictor.stage2c import regime_inventory as RI  # noqa: E402,E501
from models.ClustOpt_Policy_Predictor.training import target_transforms as TT  # noqa: E402,E501

S2A_REL = ("results_analysis/clustopt_policy_predictor/"
           "stage2a3a_view_level_training_data")
GZ = {"index": False, "compression": "gzip"}
KEY = ["dataset_id", "view_id"]


def load_units(repo: Path, inv: pd.DataFrame) -> pd.DataFrame:
    """Long-format (dataset, view, regime) -> reranked OOF ARI + diagnostics."""
    frames: List[pd.DataFrame] = []
    for _, r in inv.iterrows():
        if r["status"] != "EXISTS":
            raise AssertionError("unit %s/%s missing; run build_oof_reranker"
                                 "_targets first" % (r["regime_id"],
                                                     r["split_id"]))
        d = pd.read_csv(_ext(Path(r["unit_dir"]) / "slate_selections.csv.gz"))
        d["regime_id"] = r["regime_id"]
        d["reranker_oof_model_id"] = "%s|fold_%02d|%s" % (
            r["regime_id"], int(r["split_id"]), r["unit_hash"])
        d["source_stage"] = r["source_stage"]
        frames.append(d[["dataset_id", "split_id", "view_id", "regime_id",
                         "selected_candidate_index", "selected_ari",
                         "candidate_oracle_ari", "b_raw_ari", "b_softmax_ari",
                         "reranker_oof_model_id", "source_stage"]])
    return pd.concat(frames, ignore_index=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", default=".")
    args = p.parse_args(argv)
    repo = Path(args.repo_root).resolve()
    out = ensure_dir(repo / RI.STAGE2C_REL)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                            capture_output=True, text=True).stdout.strip()
    t0 = time.time()
    print("=" * 78)
    print("STAGE 2C-1  RERANKER-AWARE OOF TARGET REPOSITORY")
    print("=" * 78, flush=True)

    inv = RI.build(repo)
    n_missing = int((inv["status"] != "EXISTS").sum())
    if n_missing:
        print("  %d units still missing -- run build_oof_reranker_targets"
              % n_missing, flush=True)
        return 1
    long = load_units(repo, inv)
    print("  loaded %d (dataset,view,regime) rows from %d units"
          % (len(long), len(inv)), flush=True)

    mapping = pd.DataFrame(PM.build_mapping())
    order = json.loads((repo / S2A_REL / "policy_target_order.json")
                       .read_text(encoding="utf-8"))
    policies = sorted(order["policies"], key=lambda x: x["index"])
    pol_ids = [p["policy_id"] for p in policies]

    # ---- 10-regime wide matrix --------------------------------------------
    wide10 = long.pivot_table(index=KEY, columns="regime_id",
                              values="selected_ari").reset_index()
    ids = long.drop_duplicates(KEY)[KEY + ["split_id"]]
    wide10 = ids.merge(wide10, on=KEY, how="left")
    regime_cols = [c for c in RI.RERANKER_IDS if c in wide10.columns]
    wide10 = wide10[KEY + ["split_id"] + regime_cols]
    wide10.to_csv(_ext(out / "reranker_aware_targets_10regime.csv.gz"), **GZ)
    print("  10-regime matrix: %d rows x %d regimes"
          % (len(wide10), len(regime_cols)), flush=True)

    # ---- expand to the frozen 20-policy interface --------------------------
    reranker_of = dict(zip(mapping["policy_id"], mapping["reranker_id"]))
    wide20 = wide10[KEY + ["split_id"]].copy()
    for pid in pol_ids:
        wide20["target__ari__%s" % pid] = wide10[reranker_of[pid]].to_numpy()
    wide20.to_csv(_ext(out / "reranker_aware_targets_20policy.csv.gz"), **GZ)

    ari20 = wide20[["target__ari__%s" % p for p in pol_ids]].to_numpy(float)
    cen = TT.transform(ari20, "CENTERED")
    cend = wide20[KEY + ["split_id"]].copy()
    for j, pid in enumerate(pol_ids):
        cend["target__centered__%s" % pid] = cen[:, j]
    cend.to_csv(_ext(out / "centered_reranker_targets.csv.gz"), **GZ)

    # ---- RAW/SOFTMAX equality, by construction and by measurement ----------
    pairs, viol = [], 0
    for rid, g in mapping.groupby("reranker_id"):
        raw = g[g["weighting_mode"] == "RAW"]["policy_id"].iloc[0]
        sof = g[g["weighting_mode"] != "RAW"]["policy_id"].iloc[0]
        a = wide20["target__ari__%s" % raw].to_numpy(float)
        b = wide20["target__ari__%s" % sof].to_numpy(float)
        nmax = float(np.nanmax(np.abs(a - b)))
        viol += int(nmax > 0)
        pairs.append({"reranker_id": rid, "raw_policy": raw,
                      "softmax_policy": sof, "max_abs_diff": nmax,
                      "identical": bool(nmax == 0.0)})
    write_csv_atomic(out / "raw_softmax_equality_check.csv", pd.DataFrame(pairs))
    print("  RAW/SOFTMAX identical for %d/10 regimes (violations %d)"
          % (sum(p["identical"] for p in pairs), viol), flush=True)

    # ---- diagnostics (long format, kept out of the training table) ---------
    diag = long[["dataset_id", "split_id", "view_id", "regime_id",
                 "selected_candidate_index", "selected_ari",
                 "candidate_oracle_ari", "reranker_oof_model_id",
                 "source_stage"]]
    diag.to_csv(_ext(out / "reranker_selection_diagnostics.csv.gz"), **GZ)

    schema_hash = hashlib.sha256(
        json.dumps({"policies": pol_ids, "regimes": list(regime_cols),
                    "columns": list(wide20.columns)},
                   sort_keys=True).encode()).hexdigest()[:16]
    summ = {
        "n_dataset_view_rows": int(len(wide20)),
        "n_datasets": int(wide20["dataset_id"].nunique()),
        "n_views": int(wide20["view_id"].nunique()),
        "n_unique_regimes": len(regime_cols),
        "n_policy_columns": len(pol_ids),
        "splits": sorted(int(s) for s in wide20["split_id"].unique()),
        "split_1_present": bool((wide20["split_id"] == 1).any()),
        "policy_order_hash": order["schema_hash"],
        "policy_semantics_hash": PM.policy_semantics_hash(PM.build_policy_set()),
        "target_schema_hash": schema_hash,
        "centered_formula": "ARI - rowmean(ARI over the 20 policy columns)",
        "centered_impl": "models.ClustOpt_Policy_Predictor.training."
                         "target_transforms.transform(..., 'CENTERED')",
        "raw_softmax_pairs_identical": int(sum(p["identical"] for p in pairs)),
        "raw_softmax_violations": viol,
        "reuse_by_stage": inv["source_stage"].value_counts().to_dict(),
        "source_commit": commit, "runtime_sec": time.time() - t0,
    }
    write_json_atomic(out / "target_repository_summary.json", summ)
    print("  datasets %d | rows %d | split_1_present %s | schema %s"
          % (summ["n_datasets"], summ["n_dataset_view_rows"],
             summ["split_1_present"], schema_hash), flush=True)
    print("\n  done (%.0fs)" % (time.time() - t0), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
