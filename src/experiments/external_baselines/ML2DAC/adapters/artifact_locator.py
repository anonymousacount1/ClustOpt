"""Locate the ML2DAC MetaKnowledgeRepository (MKR) artifacts.

ML2DAC's application phase needs trained artifacts (evaluated configurations,
meta-feature kdtrees, and a trained CVI classifier). This module checks whether
they are present and produces a normalised report consumed by the adapter,
runners, and smoke test. It does NOT import ML2DAC; it only inspects the
filesystem, so it is safe to run in any environment.

The default meta-feature set is index 4 (``statistical+info-theory+general``) --
the paper-strongest set (see the repository audit). Its trained classifier lives
at ``models/RandomForestClassifier/statistical+info-theory+general/None``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from ..utils.path_utils import (
    default_mkr_path,
    external_ml2dac_exists,
    external_ml2dac_src,
)

# Meta-feature set index 4 = ["statistical","info-theory","general"] -> this string.
DEFAULT_MF_SET_NAME = "statistical+info-theory+general"
DEFAULT_MF_SET_INDEX = 4


def _git_lfs_pointer(path: Path) -> bool:
    """True if a file looks like an unfetched Git-LFS pointer (tiny text stub)."""
    try:
        if path.stat().st_size > 1024:
            return False
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            head = fh.read(64)
        return head.startswith("version https://git-lfs")
    except Exception:
        return False


def locate_artifacts(
    repo_root: str | Path,
    *,
    mkr_path: Optional[str | Path] = None,
    mf_set_name: str = DEFAULT_MF_SET_NAME,
) -> Dict[str, Any]:
    """Inspect the MKR and return a normalised artifact report.

    Parameters
    ----------
    repo_root:
        Repository root (used to find ``external/ml2dac``).
    mkr_path:
        Optional explicit MKR directory. Defaults to the shipped MKR inside the
        submodule.
    mf_set_name:
        Meta-feature-set folder name whose trained classifier is required.
    """
    repo_root = Path(repo_root)
    src = external_ml2dac_src(repo_root)
    mkr = Path(mkr_path) if mkr_path is not None else default_mkr_path(repo_root)

    notes: List[str] = []
    paths: Dict[str, Optional[str]] = {}
    missing: List[str] = []

    repo_ok = external_ml2dac_exists(repo_root)
    if not repo_ok:
        notes.append(
            f"external/ml2dac not found or incomplete (expected {src}). "
            f"Run: git submodule update --init --recursive"
        )

    # Required-at-inference artifacts.
    evaluated_configs = mkr / "evaluated_configs.csv"
    optimal_cvi = mkr / "optimal_cvi.csv"
    kdtree = mkr / "meta_features" / f"{mf_set_name}_kdtree.pkl"
    metafeatures_csv = mkr / "meta_features" / f"{mf_set_name}_metafeatures.csv"
    classifier_default = (
        mkr / "models" / "RandomForestClassifier" / mf_set_name / "None"
    )

    required = {
        "evaluated_configs_csv": evaluated_configs,
        "meta_features_kdtree": kdtree,
        "cvi_classifier_default": classifier_default,
    }
    optional = {
        "optimal_cvi_csv": optimal_cvi,
        "meta_features_csv": metafeatures_csv,
        "models_dir": mkr / "models" / "RandomForestClassifier",
        "meta_features_dir": mkr / "meta_features",
        "mkr_dir": mkr,
    }

    for name, p in {**required, **optional}.items():
        exists = p.exists()
        paths[name] = str(p) if exists else None
        if not exists and name in required:
            missing.append(name)
        elif exists and p.is_file() and _git_lfs_pointer(p):
            # Present but an unfetched LFS pointer -> effectively missing.
            notes.append(f"{name} at {p} is an unfetched Git-LFS pointer "
                         f"(run: git lfs pull in external/ml2dac).")
            paths[name] = None
            if name in required:
                missing.append(f"{name} (lfs-pointer)")

    has_pretrained = repo_ok and not missing
    if has_pretrained:
        try:
            size_mb = evaluated_configs.stat().st_size / (1024 * 1024)
            notes.append(f"evaluated_configs.csv present ({size_mb:.0f} MB).")
        except Exception:
            pass
        notes.append(f"Using meta-feature set '{mf_set_name}' (index "
                     f"{DEFAULT_MF_SET_INDEX}).")

    return {
        "method": "ML2DAC",
        "repo_root": str(repo_root),
        "ml2dac_src": str(src),
        "ml2dac_src_exists": repo_ok,
        "mkr_path": str(mkr),
        "mf_set_name": mf_set_name,
        "has_pretrained_artifacts": bool(has_pretrained),
        "artifact_paths": paths,
        "missing_required_artifacts": missing,
        # Full ML2DAC == the application phase using the shipped MKR.
        "can_run_full_ml2dac": bool(has_pretrained),
        "can_run_application_phase_without_learning_phase": bool(has_pretrained),
        "notes": notes,
    }


__all__ = ["locate_artifacts", "DEFAULT_MF_SET_NAME", "DEFAULT_MF_SET_INDEX"]
