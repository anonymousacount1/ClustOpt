"""The clustering search space (algorithms and hyper-parameter ranges)."""
from __future__ import annotations

import json
from pathlib import Path

from ..common import ensure_src_on_path

ensure_src_on_path()


def search_space_from_config(config_path):
    """Build the search space declared in a CLUSTOPT configuration file."""
    from models.ClustOpt.search_space.Clustering_Search_Space import ClusteringSearchSpace
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    return ClusteringSearchSpace.from_dict(cfg["search_space"])


def algorithms(config_path):
    return sorted(json.loads(Path(config_path).read_text(encoding="utf-8"))["search_space"].keys())
