"""Standalone audit of the stored candidate repository, run before any build.

Proves -- rather than assumes -- the schema facts the rest of Stage 2B-0 relies
on, by sampling real slates across every family and both split extremes:

  * 32 rows, 129 columns, 9 base + 60 raw + 60 normalised CVI columns;
  * CVI order identical to the canonical utility schema;
  * candidate identity fixed per view and constant across datasets;
  * which fields are persisted and which are not (cluster labels are not).

Sampling, not a full sweep: the full-population versions of these assertions run
inside the extractor on all 48,039 slates, so this audit is a fast pre-flight.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from models.Clustering_Repository_Builder.experiments.experiment_execution.output_writer import (  # noqa: E402,E501
    _ext, ensure_dir, write_csv_atomic, write_json_atomic,
)
from models.ClustOpt_Candidate_Reranker.data import candidate_schema as CS  # noqa: E402,E501

OUT_REL = ("results_analysis/clustopt_candidate_reranker/"
           "stage2b0_data_protocol_audit")
SPLIT_CSV_REL = ("results_analysis/clustering_repository/analyzed_data/"
                 "experiment_splits/dataset_split_assignments.csv")


def audit(repo: Path, per_family: int = 2) -> Dict[str, Any]:
    metric_names = CS.load_metric_names(repo)
    assign = pd.read_csv(_ext(repo / SPLIT_CSV_REL))
    assign = assign[assign["split_id"] != 1]
    sample = assign.groupby("family_id", group_keys=False).head(per_family)

    rows: List[Dict[str, Any]] = []
    fingerprints: Dict[str, set] = {}
    columns_seen: Optional[List[str]] = None
    for _, r in sample.iterrows():
        for view in CS.VIEW_IDS:
            path = CS.candidate_csv_path(Path(r["dataset_dir"]), view)
            df = pd.read_csv(_ext(path))
            CS.validate_slate(df, metric_names, path)
            if columns_seen is None:
                columns_seen = list(df.columns)
            algs = df["algorithm"].astype(str).tolist()
            cfgs = df["config_str"].astype(str).tolist()
            fingerprints.setdefault(view, set()).add(
                CS.slate_fingerprint(algs, cfgs))
            valid = df["valid"].map(CS.is_valid_flag)
            ari = pd.to_numeric(df["ARI"], errors="coerce")
            rows.append({
                "dataset_id": r["dataset_id"], "split_id": int(r["split_id"]),
                "family_id": r["family_id"], "view_id": view,
                "n_candidates": len(df), "n_columns": df.shape[1],
                "n_valid": int(valid.sum()),
                "n_ari_nan": int(ari.isna().sum()),
                "invalid_ari_max": float(ari[~valid].max()) if (~valid).any()
                else float("nan"),
                "n_algorithms": df["algorithm"].nunique(),
                "has_labels_column": int(any("label" in c.lower()
                                             and c != "n_labels_unique"
                                             for c in df.columns)),
                "has_runtime_column": int(any("runtime" in c.lower()
                                              for c in df.columns)),
            })
    frame = pd.DataFrame(rows)
    return {
        "frame": frame,
        "summary": {
            "n_slates_sampled": int(len(frame)),
            "n_datasets_sampled": int(frame["dataset_id"].nunique()),
            "all_32_candidates": bool((frame["n_candidates"] == 32).all()),
            "all_129_columns": bool((frame["n_columns"] == 129).all()),
            "n_distinct_fingerprints_per_view":
                {k: len(v) for k, v in fingerprints.items()},
            "view_fingerprints": {k: sorted(v) for k, v in fingerprints.items()},
            "metric_order_hash": CS.metric_order_hash(metric_names),
            "n_metrics": len(metric_names),
            "base_columns": list(CS.BASE_COLUMNS),
            "cluster_labels_persisted": bool(frame["has_labels_column"].any()),
            "per_candidate_runtime_persisted":
                bool(frame["has_runtime_column"].any()),
            "ari_missing_anywhere": int(frame["n_ari_nan"].sum()),
            "max_ari_among_invalid_candidates":
                float(frame["invalid_ari_max"].max()),
            "mean_valid_candidates_per_slate": float(frame["n_valid"].mean()),
        },
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", default=".")
    p.add_argument("--per-family", type=int, default=2)
    args = p.parse_args(argv)
    repo = Path(args.repo_root).resolve()
    out = ensure_dir(repo / OUT_REL)

    print("=" * 78)
    print("STAGE 2B-0  --  STORED CANDIDATE SOURCE AUDIT")
    print("=" * 78, flush=True)
    res = audit(repo, args.per_family)
    write_csv_atomic(out / "candidate_source_audit_sample.csv", res["frame"])
    write_json_atomic(out / "candidate_source_audit.json", res["summary"])
    for k, v in res["summary"].items():
        print(f"  {k:<42} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
