"""Resolve and validate the 10 method configs × 3 view configs.

Each method maps to a config folder containing ``x_only.json`` / ``y_only.json``
/ ``xy_2d.json``. This module locates them, validates JSON + required keys, and
exposes ``config_path(method, view)``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

from .config import METHODS, VIEW_IDS, MethodSpec

REQUIRED_CONFIG_KEYS: Tuple[str, ...] = ("search_space", "search_algorithm", "cvi")


class ConfigResolutionError(ValueError):
    """Raised when required experiment configs are missing or invalid."""


@dataclass
class ResolvedConfigs:
    config_root: Path
    # (method_name, view_id) -> config path
    paths: Dict[Tuple[str, str], Path] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def config_path(self, method_name: str, view_id: str) -> Path:
        return self.paths[(method_name, view_id)]


def _validate_config_file(path: Path) -> List[str]:
    errors: List[str] = []
    if not path.is_file():
        return [f"missing config: {path}"]
    try:
        with path.open("r", encoding="utf-8") as fh:
            cfg = json.load(fh)
    except Exception as exc:
        return [f"invalid JSON in {path}: {exc}"]
    for key in REQUIRED_CONFIG_KEYS:
        if key not in cfg:
            errors.append(f"{path} missing required key '{key}'")
    return errors


def resolve_configs(
    config_root: Path,
    *,
    methods: Tuple[MethodSpec, ...] = METHODS,
    views: Tuple[str, ...] = VIEW_IDS,
    validate: bool = True,
) -> ResolvedConfigs:
    """Locate and (optionally) validate all method/view configs."""
    resolved = ResolvedConfigs(config_root=Path(config_root))
    for method in methods:
        method_dir = resolved.config_root / method.config_subdir
        for view in views:
            path = method_dir / f"{view}.json"
            resolved.paths[(method.name, view)] = path
            if validate:
                resolved.errors.extend(_validate_config_file(path))
    return resolved


def validate_or_raise(resolved: ResolvedConfigs) -> None:
    if not resolved.ok:
        raise ConfigResolutionError(
            f"{len(resolved.errors)} config error(s):\n  - "
            + "\n  - ".join(resolved.errors)
        )
