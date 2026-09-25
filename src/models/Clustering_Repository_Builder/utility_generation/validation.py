"""Reusable validation helpers for the utility-generation pipeline.

These run **before** any worker is launched so that obvious configuration
or environment errors fail loudly and immediately.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from .config import UTILITY_MAP_FILES, UtilityGenerationConfig, VIEW_IDS


class ValidationError(RuntimeError):
    """Raised when a pre-run validation step fails."""


def validate_subfamily_dir(subfamily_dir: Path) -> None:
    if not subfamily_dir.exists():
        raise ValidationError(f"Subfamily directory does not exist: {subfamily_dir}")
    if not subfamily_dir.is_dir():
        raise ValidationError(f"Subfamily path is not a directory: {subfamily_dir}")


def validate_repo_root(repo_root: Path) -> None:
    if not repo_root.is_dir():
        raise ValidationError(f"Repo root does not exist: {repo_root}")
    expected = repo_root / "models" / "ClustOpt"
    if not expected.is_dir():
        raise ValidationError(
            f"Repo root does not look like the project root (missing {expected})."
        )


def validate_map_files(cfg: UtilityGenerationConfig) -> None:
    missing: list[Path] = []
    for view_id in VIEW_IDS:
        p = cfg.map_path_for_view(view_id)
        if not p.is_file():
            missing.append(p)
    if missing:
        raise ValidationError(
            "Missing utility map file(s):\n  - " + "\n  - ".join(str(m) for m in missing)
        )


def validate_views(views: tuple[str, ...]) -> None:
    bad = [v for v in views if v not in UTILITY_MAP_FILES]
    if bad:
        raise ValidationError(
            f"Unknown view id(s): {bad}. Allowed: {sorted(UTILITY_MAP_FILES.keys())}"
        )


def validate_metric_mode(mode: str) -> None:
    if mode not in {"fast", "exact"}:
        raise ValidationError(
            f"Unknown metric_mode '{mode}'. Allowed: 'fast', 'exact'."
        )


def validate_clustopt_imports() -> None:
    """Smoke-test the imports the pipeline depends on.

    We import here (not at module top) so that import failures surface as
    actionable ValidationError messages rather than ImportError tracebacks
    deep inside a worker.
    """
    try:
        from models.ClustOpt import ClusteringOptimizationBuilder  # noqa: F401
    except Exception as exc:
        raise ValidationError(f"Failed to import ClusteringOptimizationBuilder: {exc}")
    try:
        from models.ClustOpt.external_evaluation.compute_metric_utility import (  # noqa: F401
            compute_metric_utilities,
        )
    except Exception as exc:
        raise ValidationError(f"Failed to import compute_metric_utilities: {exc}")


def validate_builder_can_load_maps(cfg: UtilityGenerationConfig) -> None:
    """Try constructing a ClustOpt builder for each utility map.

    This catches malformed JSON / schema problems before launching workers.
    """
    from models.ClustOpt import ClusteringOptimizationBuilder

    errors: List[str] = []
    for view_id in VIEW_IDS:
        map_path = cfg.map_path_for_view(view_id)
        try:
            ClusteringOptimizationBuilder(map_path)
        except Exception as exc:
            errors.append(f"{view_id} ({map_path.name}): {type(exc).__name__}: {exc}")
    if errors:
        raise ValidationError(
            "ClustOpt builder failed to load utility map(s):\n  - "
            + "\n  - ".join(errors)
        )


def run_full_validation(cfg: UtilityGenerationConfig) -> None:
    """Run every pre-flight check; raise ``ValidationError`` on the first failure."""
    validate_repo_root(cfg.repo_root)
    validate_subfamily_dir(cfg.subfamily_dir)
    validate_metric_mode(cfg.metric_mode)
    validate_views(cfg.views)
    validate_map_files(cfg)
    validate_clustopt_imports()
    validate_builder_can_load_maps(cfg)
