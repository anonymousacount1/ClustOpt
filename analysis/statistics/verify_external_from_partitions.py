"""Recompute the external per-dataset best-view ARI of all 14 arms (including the
policy-selector arms) from the released partitions, without any trained model.

Prerequisite: ``python scripts/reconstruct_external_datasets.py`` (the ground
truth is rebuilt locally from the public sources; it is not shipped).

    python analysis/statistics/verify_external_from_partitions.py
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import clustopt  # noqa: E402,F401


def main() -> int:
    from Independent_Domain_Benchmark.datasets import corpus_archive as CA
    raw = gzip.decompress((ROOT / "results/external/per_dataset/final_partitions_by_unit.jsonl.gz").read_bytes())
    recs = [json.loads(l) for l in raw.decode("utf-8").splitlines()]
    gt, rows = {}, []
    for r in recs:
        if r["dataset_id"] not in gt:
            gt[r["dataset_id"]] = CA.load_ground_truth(r["dataset_id"])
        ok = r["status"] == "success" and r["final_labels"]
        rows.append({"dataset_id": r["dataset_id"], "method_id": r["method_id"], "view_id": r["view_id"],
                     "ari": adjusted_rand_score(gt[r["dataset_id"]], r["final_labels"]) if ok else np.nan})
    bv = pd.DataFrame(rows).groupby(["dataset_id", "method_id"]).ari.max().reset_index()
    ref = pd.read_csv(ROOT / "results/external/per_dataset/method_bestview_per_dataset.csv")
    m = bv.merge(ref, on=["dataset_id", "method_id"])
    m["abs_diff"] = (m.ari - m.bestview_ari).abs()
    summ = m.groupby("method_id").agg(best_view_ari_recomputed=("ari", "mean"),
                                      best_view_ari_released=("bestview_ari", "mean"),
                                      max_abs_diff=("abs_diff", "max"))
    print(summ.round(4).to_string())
    out = ROOT / "analysis/output/statistics"
    out.mkdir(parents=True, exist_ok=True)
    summ.to_csv(out / "external_bestview_recomputed.csv")
    return 0 if m.abs_diff.max() <= 1e-12 else 1


if __name__ == "__main__":
    raise SystemExit(main())
