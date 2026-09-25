r"""Path helpers for the AutoClust baseline.

Long-path-safe atomic IO + record-type/view mapping are reused verbatim from the
AutoML4Clust baseline. This module adds the AutoClust-specific locators: the
ML2DAC ``src`` import root and the AutoClust implementation + artifacts living
inside ``external/ml2dac/src/Experiments/RelatedWork/related_work``.

The adapter imports only the ML2DAC *building blocks* (MetaFeatureExtractor,
CVIHandler, ClusteringCS, OptimizerSMAC); it never imports ``AutoClust.py``
itself (that script reads a 139 MB CSV and generates synthetic data at import).
All artifact paths are passed as absolute paths, so no ``chdir`` is needed.
"""
from __future__ import annotations

import sys
from pathlib import Path

from experiments.external_baselines.AutoML4Clust.utils.path_utils import (  # noqa: F401
    RECORD_TYPES,
    RECORD_TYPE_TO_VIEW_ID,
    VIEW_ID_TO_RECORD_TYPE,
    record_type_to_view_id,
    ext,
    ensure_dir,
    path_exists,
    write_json_atomic,
    write_text_atomic,
    write_csv_atomic,
    write_npy_atomic,
    read_json,
    split_dir_name,
)


def external_ml2dac_src(repo_root: str | Path) -> Path:
    """The ML2DAC ``src`` directory (import root for the building blocks)."""
    return Path(repo_root) / "external" / "ml2dac" / "src"


def autoclust_implementation_path(repo_root: str | Path) -> Path:
    """The ML2DAC authors' AutoClust re-implementation file."""
    return external_ml2dac_src(repo_root) / "Experiments" / "RelatedWork" / "AutoClust.py"


def related_work_path(repo_root: str | Path) -> Path:
    """Directory holding AutoClust artifacts (kdtree, metafeatures, MLP, offline CSV)."""
    return external_ml2dac_src(repo_root) / "Experiments" / "RelatedWork" / "related_work"


def autoclust_artifact_paths(repo_root: str | Path) -> dict:
    rw = related_work_path(repo_root)
    return {
        "meanshift_kdtree": rw / "meanshift_kdtree.pkl",
        "meanshift_metafeatures": rw / "meanshift_metafeatures.csv",
        "rw_mlp": rw / "rw_mlp.pkl",
        "offline_opt_csv": rw / "related_work_offline_opt.csv",
        "related_work_dir": rw,
    }


def external_ml2dac_exists(repo_root: str | Path) -> bool:
    return autoclust_implementation_path(repo_root).is_file()


def ensure_ml2dac_on_path(repo_root: str | Path) -> Path:
    """Prepend ``external/ml2dac/src`` to ``sys.path`` (idempotent).

    ML2DAC uses top-level absolute imports (``from MetaLearning... import ...``),
    so its ``src`` dir must be importable directly.
    """
    src = external_ml2dac_src(repo_root)
    s = str(src)
    if s not in sys.path:
        sys.path.insert(0, s)
    return src


__all__ = [
    "RECORD_TYPES", "RECORD_TYPE_TO_VIEW_ID", "VIEW_ID_TO_RECORD_TYPE",
    "record_type_to_view_id", "ext", "ensure_dir", "path_exists",
    "write_json_atomic", "write_text_atomic", "write_csv_atomic",
    "write_npy_atomic", "read_json", "split_dir_name",
    "external_ml2dac_src", "autoclust_implementation_path", "related_work_path",
    "autoclust_artifact_paths", "external_ml2dac_exists", "ensure_ml2dac_on_path",
]
