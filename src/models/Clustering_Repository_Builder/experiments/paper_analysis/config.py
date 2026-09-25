"""Declarative comparison-suite configuration loading and resolution.

Configs live in ``analysis_configs/*.yaml``. They are loaded verbatim and then
*resolved*: any ``"@representative:<id>"`` placeholder is replaced with the
concrete method name recorded in the **frozen selection manifest**. That
indirection is the mechanism that stops a suite from reselecting a winner in its
own plotting or reporting code -- a config can only name a representative id, and
resolution fails loudly if the manifest does not contain it.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import yaml

from . import paths
from .method_registry import METHOD_BY_NAME

CONFIG_DIR = Path(__file__).resolve().parent / "analysis_configs"

REPRESENTATIVE_PREFIX = "@representative:"

STAGE_CONFIGS: Dict[str, str] = {
    "preliminary-selection": "preliminary_selection.yaml",
    "comparison-01": "comparison_01_metric_utility.yaml",
    "comparison-02": "comparison_02_meta_learning.yaml",
    "comparison-03": "comparison_03_raw_softmax_vbs.yaml",
    "comparison-04": "comparison_04_operational_benchmarks.yaml",
    "comparison-05": "comparison_05_vbs_headroom.yaml",
    "metric-selection": "metric_taxonomy.yaml",
}

# Contract configs that are validated but are not runnable stages.
CONTRACT_CONFIGS: Tuple[str, ...] = ("metric_taxonomy.yaml", "paper_labels.yaml")

# Per-suite subdirectories. Every comparison writes the same eight, so an
# advisor opening any suite finds the same shape.
SUITE_SUBDIRS: Tuple[str, ...] = (
    "config_snapshot", "manifests", "tables", "plots", "reports",
    "formatted_excel", "validation", "package",
)

# Output-root section directories (§23), created up-front so the layout is
# stable whether or not a stage has run.
SECTION_DIRS: Dict[str, str] = {
    "protocol": "00_protocol_and_definitions",
    "preliminary": "01_preliminary_selection",
    "comparison_01": "02_comparison_01_metric_utility",
    "comparison_02": "03_comparison_02_meta_learning",
    "comparison_03": "04_comparison_03_raw_softmax_vbs",
    "comparison_04": "05_comparison_04_operational_benchmarks",
    "comparison_05": "06_comparison_05_vbs_headroom",
    "metric_selection": "07_metric_selection",
    "gap_decomposition": "08_gap_decomposition",
    "runtime_pareto": "09_runtime_and_pareto",
    "statistical_tests": "10_statistical_tests",
    "family_reports": "11_family_reports",
    "subfamily_appendix": "12_subfamily_appendix",
    "raw_reproducibility": "13_raw_and_reproducibility",
    "packaged": "14_packaged_for_paper",
}


class ConfigError(RuntimeError):
    """Raised when a config cannot be loaded or resolved."""


def load_config(name: str) -> Dict[str, Any]:
    """Load one config file by filename or by stage name."""
    filename = STAGE_CONFIGS.get(name, name)
    path = CONFIG_DIR / filename
    text = paths.read_text(path)
    if text is None:
        raise ConfigError(f"config not found: {path}")
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:                              # pragma: no cover
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(data, dict):                             # pragma: no cover
        raise ConfigError(f"{path} must contain a mapping at the top level")
    data["_config_path"] = str(path)
    data["_config_name"] = filename
    return data


def load_all_configs() -> Dict[str, Dict[str, Any]]:
    return {stage: load_config(stage) for stage in STAGE_CONFIGS}


# --------------------------------------------------------------------------- #
# Representative resolution
# --------------------------------------------------------------------------- #
@dataclass
class RepresentativeIndex:
    """Read-only view of the frozen selection manifest."""

    by_id: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_manifest(cls, manifest: pd.DataFrame) -> "RepresentativeIndex":
        by_id: Dict[str, Dict[str, Any]] = {}
        if manifest is None or manifest.empty:
            return cls(by_id)
        globals_only = manifest[manifest.get(
            "selection_scope", pd.Series("global", index=manifest.index)) == "global"]
        for _, r in globals_only.iterrows():
            by_id[str(r["representative_id"])] = r.to_dict()
        return cls(by_id)

    @classmethod
    def from_path(cls, path: Path) -> "RepresentativeIndex":
        if not paths.exists(path):
            raise ConfigError(
                f"frozen selection manifest not found at {path}; run "
                f"--stage preliminary-selection first (comparison suites must "
                f"never reselect their own representatives)")
        return cls.from_manifest(paths.read_csv(path))

    def method(self, representative_id: str) -> str:
        rec = self.by_id.get(representative_id)
        if rec is None:
            raise ConfigError(
                f"representative {representative_id!r} is not in the frozen "
                f"selection manifest; available: {sorted(self.by_id)}")
        return str(rec["selected_method"])

    def record(self, representative_id: str) -> Dict[str, Any]:
        rec = self.by_id.get(representative_id)
        if rec is None:
            raise ConfigError(f"unknown representative {representative_id!r}")
        return rec

    def __contains__(self, representative_id: object) -> bool:
        return str(representative_id) in self.by_id


def resolve_placeholders(node: Any, index: RepresentativeIndex) -> Any:
    """Recursively replace ``@representative:<id>`` with the frozen method name."""
    if isinstance(node, str):
        if node.startswith(REPRESENTATIVE_PREFIX):
            return index.method(node[len(REPRESENTATIVE_PREFIX):])
        return node
    if isinstance(node, list):
        return [resolve_placeholders(v, index) for v in node]
    if isinstance(node, dict):
        return {k: resolve_placeholders(v, index) for k, v in node.items()}
    return node


def resolve_config(config: Dict[str, Any], index: RepresentativeIndex
                   ) -> Dict[str, Any]:
    """Return a copy of ``config`` with every representative placeholder resolved."""
    resolved = resolve_placeholders(copy.deepcopy(config), index)
    resolved["_resolved"] = True
    return resolved


def referenced_representative_ids(node: Any) -> List[str]:
    """Every ``@representative:<id>`` referenced anywhere in a config."""
    found: List[str] = []

    def walk(n: Any) -> None:
        if isinstance(n, str) and n.startswith(REPRESENTATIVE_PREFIX):
            found.append(n[len(REPRESENTATIVE_PREFIX):])
        elif isinstance(n, list):
            for v in n:
                walk(v)
        elif isinstance(n, dict):
            for v in n.values():
                walk(v)

    walk(node)
    return sorted(set(found))


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
REQUIRED_COMPARISON_KEYS: Sequence[str] = (
    "comparison_id", "title", "section", "research_questions",
    "primary_metrics", "required_capabilities", "tables", "plots", "reports",
)

# Configs that are contracts rather than comparison suites: they describe a
# derivation (the metric taxonomy) or a labelling scheme, so the
# comparison-suite key set does not apply to them.
NON_COMPARISON_CONFIGS: Sequence[str] = ("metric_taxonomy.yaml", "paper_labels.yaml")

# Method-name prefixes used by the "did the config name a real method?" heuristic.
# Portfolio ids share some of these prefixes (``knn_vbs``, ``real_utility_vbs``),
# so the ``portfolios`` block is validated against the portfolio registry instead.
_METHOD_NAME_PREFIXES: Sequence[str] = (
    "regressor_", "knn_top", "real_top", "AutoClust", "ML2DAC", "AutoML4Clust",
)


def validate_config(config: Dict[str, Any], *,
                    index: Optional[RepresentativeIndex] = None) -> List[str]:
    """Structural validation of one comparison config."""
    problems: List[str] = []
    name = config.get("_config_name", "<config>")
    if name in NON_COMPARISON_CONFIGS:
        return _validate_contract_config(config, name)
    is_preliminary = str(config.get("comparison_id", "")) == "preliminary"
    required = ("comparison_id", "title", "section") if is_preliminary \
        else REQUIRED_COMPARISON_KEYS
    for key in required:
        if key not in config:
            problems.append(f"{name}: missing required key {key!r}")

    # Every literal method name mentioned must be registered.
    def walk_methods(node: Any, path: str) -> None:
        if isinstance(node, str):
            if node.startswith(REPRESENTATIVE_PREFIX):
                rep = node[len(REPRESENTATIVE_PREFIX):]
                if index is not None and rep not in index:
                    problems.append(f"{name}: {path} references unknown "
                                    f"representative {rep!r}")
            elif node in METHOD_BY_NAME:
                pass
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk_methods(v, f"{path}[{i}]")
        elif isinstance(node, dict):
            for k, v in node.items():
                walk_methods(v, f"{path}.{k}")

    walk_methods(config, "")

    # Method-name blocks: anything that looks like a method must be registered.
    for key in ("candidate_pools", "candidates", "references"):
        block = config.get(key)
        if not isinstance(block, dict):
            continue
        for sub_key, value in block.items():
            items = value if isinstance(value, list) else (
                list(value.values()) if isinstance(value, dict) else [])
            for item in items:
                if not isinstance(item, str) or item.startswith(REPRESENTATIVE_PREFIX):
                    continue
                if item in METHOD_BY_NAME:
                    continue
                if any(item.startswith(p) for p in _METHOD_NAME_PREFIXES):
                    problems.append(f"{name}: {key}.{sub_key} names unregistered "
                                    f"method {item!r}")

    # Portfolio blocks are validated against the portfolio registry, never
    # against the method registry (portfolio ids legitimately share prefixes
    # with method names, e.g. ``knn_vbs``).
    problems.extend(_validate_portfolio_block(config, name))

    if not is_preliminary:
        rq = config.get("research_questions")
        if isinstance(rq, dict):
            for rq_id, body in rq.items():
                if not isinstance(body, dict) or "question" not in body:
                    problems.append(f"{name}: research question {rq_id} has no "
                                    f"'question' text")
        elif rq is not None:
            problems.append(f"{name}: research_questions must be a mapping")
    return problems


def _validate_portfolio_block(config: Dict[str, Any], name: str) -> List[str]:
    """Every portfolio id a config names must exist in the portfolio registry."""
    from .portfolio_registry import PORTFOLIO_BY_ID

    problems: List[str] = []
    block = config.get("portfolios")
    if not isinstance(block, dict):
        return problems
    for sub_key, value in block.items():
        if not isinstance(value, list):
            continue
        for item in value:
            if isinstance(item, str) and item not in PORTFOLIO_BY_ID:
                problems.append(f"{name}: portfolios.{sub_key} names unknown "
                                f"portfolio {item!r}")
    return problems


def _validate_contract_config(config: Dict[str, Any], name: str) -> List[str]:
    """Validation for the taxonomy/label contract configs.

    These declare *rules* rather than a comparison, so they are checked for the
    keys that actually matter: that the taxonomy config states its derivation
    sources and audit thresholds, and that the label config states its scheme.
    """
    problems: List[str] = []
    if name == "metric_taxonomy.yaml":
        for key in ("category_derivation", "subcategory_derivation",
                    "classic_definitions", "audit_requirements",
                    "normalisation_rules"):
            if key not in config:
                problems.append(f"{name}: missing required key {key!r}")
        audit = config.get("audit_requirements") or {}
        if audit.get("max_unknown_metrics", 0) != 0:
            problems.append(f"{name}: max_unknown_metrics must be 0 (the taxonomy "
                            f"must cover every selected metric)")
    elif name == "paper_labels.yaml":
        for key in ("labelling_scheme", "axis_titles", "caption_fragments"):
            if key not in config:
                problems.append(f"{name}: missing required key {key!r}")
    return problems


def validate_all_configs(index: Optional[RepresentativeIndex] = None
                         ) -> List[str]:
    """Validate every stage config plus the standalone contract configs."""
    problems: List[str] = []
    names = list(STAGE_CONFIGS) + [
        c for c in CONTRACT_CONFIGS if c not in STAGE_CONFIGS.values()]
    for name in names:
        try:
            cfg = load_config(name)
        except ConfigError as exc:
            problems.append(str(exc))
            continue
        problems.extend(validate_config(cfg, index=index))
    return problems


# --------------------------------------------------------------------------- #
# Run configuration
# --------------------------------------------------------------------------- #
@dataclass
class RunConfig:
    """Resolved CLI options for one analysis run."""

    repo_root: Path
    analyzed_data_root: Path
    split_assignments: Path
    output_root: Path
    split_id: int = 1
    stage: str = "preliminary-selection"
    collect_workers: Optional[int] = None
    limit_datasets: Optional[int] = None
    methods: Optional[Sequence[str]] = None
    resume: bool = True
    overwrite: bool = False
    dry_run: bool = False
    # Reuse the canonical frames already on disk instead of re-reading the
    # 161k per-run output files. Collection is ~35 min and is pure IO, so a
    # re-run that only changes downstream analysis should not pay it again.
    reuse_frames: bool = False
    # Compute the per-candidate search-selection gap. Reads ~126k trace CSVs,
    # so it is opt-in and cached for all five suites to share.
    trace_gaps: bool = False
    make_plots: bool = True
    make_excel: bool = True
    make_stats: bool = True
    min_plot_importance: str = "appendix"
    bootstrap_samples: int = 2000
    selection_bootstrap_samples: int = 1000
    seed: int = 20260725
    verbose: bool = True

    def section_dir(self, key: str) -> Path:
        return self.output_root / SECTION_DIRS[key]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "repo_root": str(self.repo_root),
            "analyzed_data_root": str(self.analyzed_data_root),
            "split_assignments": str(self.split_assignments),
            "output_root": str(self.output_root),
            "split_id": int(self.split_id),
            "stage": self.stage,
            "collect_workers": self.collect_workers,
            "limit_datasets": self.limit_datasets,
            "methods": list(self.methods) if self.methods else None,
            "resume": bool(self.resume),
            "overwrite": bool(self.overwrite),
            "dry_run": bool(self.dry_run),
            "reuse_frames": bool(self.reuse_frames),
            "trace_gaps": bool(self.trace_gaps),
            "make_plots": bool(self.make_plots),
            "make_excel": bool(self.make_excel),
            "make_stats": bool(self.make_stats),
            "min_plot_importance": self.min_plot_importance,
            "bootstrap_samples": int(self.bootstrap_samples),
            "selection_bootstrap_samples": int(self.selection_bootstrap_samples),
            "seed": int(self.seed),
        }

    def ensure_dirs(self) -> None:
        if self.dry_run:
            return
        for key in SECTION_DIRS:
            paths.mkdirs(self.section_dir(key))
