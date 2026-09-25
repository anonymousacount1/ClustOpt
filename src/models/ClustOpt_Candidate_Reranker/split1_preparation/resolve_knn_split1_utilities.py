"""Regenerate the frozen Split-1 KNN utility context (Stage 2B-5B, section 1).

The authoritative artifact ``split1_locked_predictions.parquet`` cannot be read:
no environment on this machine has ``pyarrow`` or ``fastparquet``, and installing
one solely for the final evaluation is not permitted. Per the brief's fallback,
the predictions are regenerated **deterministically from the already-frozen KNN
repository trained on Splits 2-16**, using the production provider verbatim --
``KNNUtilityProvider``, the same class the online policies call.

No Split-1 outcome is involved: this is a label-free nearest-neighbour lookup over
frozen meta-features. Nothing is trained or tuned.
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
    _ext, ensure_dir, path_exists, read_json, write_csv_atomic, write_json_atomic,
)
from models.ClustOpt.cluster_validity_indices.dynamic_metric_selection.knn_predictor_provider import (  # noqa: E402,E501
    get_knn_provider,
)
from models.ClustOpt_Candidate_Reranker.data import candidate_schema as CS  # noqa: E402,E501

OUT_REL = ("results_analysis/clustopt_candidate_reranker/"
           "stage2b5a_final_models_and_split1_preparation")
KNN_MODEL_DIR = "results_analysis/metric_utility_knn/phase_e1/models"
KNN_META = f"{KNN_MODEL_DIR}/final_knn_metadata.json"
UNIFIED_REL = ("results_analysis/clustering_repository/analyzed_data/"
               "unified_training_dataset/unified_training_dataset.csv")
SPLIT_CSV_REL = ("results_analysis/clustering_repository/analyzed_data/"
                 "experiment_splits/dataset_split_assignments.csv")
FEATURE_COLUMNS_REL = ("results_analysis/mlp/20260615_231715__metric_utility_mlp_"
                       "split1_holdout/artifacts/feature_columns.json")
OUT_NAME = "split1_knn_utility_context.csv.gz"
VIEWS = ("x_only", "y_only", "xy_2d")
EXPECTED_ROWS = 3165
GZ = {"index": False, "compression": "gzip"}


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
    print("STAGE 2B-5B  --  RESOLVE SPLIT-1 KNN UTILITY CONTEXT")
    print("=" * 78, flush=True)

    meta = read_json(repo / KNN_META) or {}
    if meta.get("split1_in_training") is not False:
        raise SystemExit("KNN repository has Split 1 in training -- refusing")
    if 1 in list(meta.get("train_splits") or []):
        raise SystemExit("KNN train_splits contains Split 1 -- refusing")
    print(f"  KNN repository: train_splits={meta.get('train_splits')} | "
          f"split1_in_training={meta.get('split1_in_training')} | "
          f"n_metrics={meta.get('n_metrics')}", flush=True)

    provider = get_knn_provider(repo / KNN_MODEL_DIR)
    metric_names = CS.load_metric_names(repo)
    if list(provider.metric_names) != list(metric_names):
        raise SystemExit("KNN provider metric order differs from the canonical "
                         "60-metric schema")
    feat_cols = list(provider.feature_columns)
    print(f"  provider: {len(feat_cols)} features, "
          f"{len(provider.metric_names)} metrics, n_neighbors="
          f"{provider.n_neighbors}", flush=True)

    assign = pd.read_csv(_ext(repo / SPLIT_CSV_REL))
    s1 = assign[assign["split_id"] == 1][["dataset_id"]].drop_duplicates()
    unified = pd.read_csv(_ext(repo / UNIFIED_REL),
                          usecols=["dataset_id", "view_mode"] + feat_cols,
                          low_memory=False)
    sub = unified.merge(s1, on="dataset_id", how="inner")
    sub = sub[sub["view_mode"].isin(VIEWS)]
    if len(sub) != EXPECTED_ROWS:
        raise SystemExit(f"expected {EXPECTED_ROWS} Split-1 dataset/view rows, "
                         f"found {len(sub)}")
    if sub.duplicated(subset=["dataset_id", "view_mode"]).any():
        raise SystemExit("duplicate dataset/view rows in the Split-1 features")
    X = sub[feat_cols].to_numpy(np.float64)
    print(f"  Split-1 feature block: {X.shape}", flush=True)

    preds = np.empty((len(sub), len(metric_names)), dtype=np.float64)
    views = sub["view_mode"].to_numpy()
    for i in range(len(sub)):
        u, _ = provider.predict(X[i], views[i],
                               cache_key=(str(sub["dataset_id"].iloc[i]),
                                          str(views[i])),
                               want_neighbor_debug=False)
        preds[i] = [u[m] for m in metric_names]
        if (i + 1) % 500 == 0:
            print(f"   {i+1}/{len(sub)} predicted", flush=True)

    if not np.isfinite(preds).all():
        raise SystemExit("non-finite KNN utility prediction -- refusing")

    res = pd.DataFrame({"dataset_id": sub["dataset_id"].to_numpy(),
                        "view_id": views})
    for j, m in enumerate(metric_names):
        res[f"knn_utility__{m}"] = preds[:, j]
    res = res.sort_values(["dataset_id", "view_id"]).reset_index(drop=True)
    dest = out / OUT_NAME
    res.to_csv(_ext(dest), **GZ)

    h = hashlib.sha256()
    with open(_ext(dest), "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    manifest = {
        "artifact": OUT_NAME,
        "provenance": "REGENERATED from the frozen KNN repository using the "
                      "production KNNUtilityProvider; the authoritative "
                      "split1_locked_predictions.parquet is unreadable in this "
                      "environment (no pyarrow/fastparquet anywhere on the "
                      "machine) and no dependency was installed",
        "knn_model_dir": KNN_MODEL_DIR,
        "knn_config": meta.get("config"),
        "knn_train_splits": meta.get("train_splits"),
        "split1_in_training": meta.get("split1_in_training"),
        "n_rows": int(len(res)), "expected_rows": EXPECTED_ROWS,
        "n_datasets": int(res["dataset_id"].nunique()),
        "n_views": int(res["view_id"].nunique()),
        "n_utility_columns": len(metric_names),
        "metric_order_hash": CS.metric_order_hash(metric_names),
        "column_order": "canonical 60-metric schema order",
        "duplicates": int(res.duplicated(subset=["dataset_id",
                                                 "view_id"]).sum()),
        "all_finite": True,
        "file_sha256_16": h.hexdigest()[:16],
        "deterministic": "nearest-neighbour lookup over frozen arrays; no "
                         "randomness, no training, no Split-1 outcome used",
        "source_commit": commit,
        "runtime_sec": time.time() - t0,
    }
    write_json_atomic(out / "split1_knn_utility_context_manifest.json", manifest)
    print(f"\n  wrote {len(res)} rows x {len(metric_names)} utilities "
          f"(sha {manifest['file_sha256_16']}) in {time.time()-t0:.0f}s",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
