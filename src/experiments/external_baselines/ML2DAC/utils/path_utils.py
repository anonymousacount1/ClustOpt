r"""Path helpers for the ML2DAC baseline.

The long-path-safe atomic IO + record-type/view mapping are reused verbatim from
the AutoML4Clust baseline (single source of truth). This module adds the
ML2DAC-specific helpers: locating ``external/ml2dac/src`` (the import root) and
the MetaKnowledgeRepository, and wiring ``sys.path``.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Reuse the canonical IO + record-type helpers from the AutoML4Clust baseline.
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
    """Path to the external ML2DAC ``src`` directory (the import root)."""
    return Path(repo_root) / "external" / "ml2dac" / "src"


def external_ml2dac_exists(repo_root: str | Path) -> bool:
    src = external_ml2dac_src(repo_root)
    return (src / "MetaLearning" / "ApplicationPhase.py").is_file()


def default_mkr_path(repo_root: str | Path) -> Path:
    """Path to the shipped MetaKnowledgeRepository inside the submodule."""
    return external_ml2dac_src(repo_root) / "MetaKnowledgeRepository"


def ensure_ml2dac_on_path(repo_root: str | Path) -> Path:
    """Prepend ``external/ml2dac/src`` to ``sys.path`` (idempotent).

    ML2DAC uses top-level absolute imports (``from MetaLearning.ApplicationPhase
    import ...``), so its ``src`` directory must be importable directly. Returns
    the src path for diagnostics.
    """
    src = external_ml2dac_src(repo_root)
    src_str = str(src)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)
    return src


__all__ = [
    "RECORD_TYPES",
    "RECORD_TYPE_TO_VIEW_ID",
    "VIEW_ID_TO_RECORD_TYPE",
    "record_type_to_view_id",
    "ext",
    "ensure_dir",
    "path_exists",
    "write_json_atomic",
    "write_text_atomic",
    "write_csv_atomic",
    "write_npy_atomic",
    "read_json",
    "split_dir_name",
    "external_ml2dac_src",
    "external_ml2dac_exists",
    "default_mkr_path",
    "ensure_ml2dac_on_path",
]
