"""Label-free meta-features (the predictor inputs) for generated datasets.

Datasets must follow the generator layout
``<root>/<family>/subfamilies/<subfamily>/<dataset>/data_data.csv``; features are
written to ``<dataset>/features/features_records.csv`` (one row per view).
"""
from __future__ import annotations

from pathlib import Path

from ..common import ensure_src_on_path

ensure_src_on_path()


def extract(subfamily_root, *, max_datasets=None, overwrite=False, seed=0):
    from models.Clustering_Repository_Builder.features_extraction.config import RunConfig
    from models.Clustering_Repository_Builder.features_extraction.run_extract_subfamily_features import run_subfamily
    cfg = RunConfig(subfamily_root=str(Path(subfamily_root)), overwrite=overwrite,
                    max_datasets=max_datasets, write_parquet=False, random_seed=seed)
    return run_subfamily(cfg)
