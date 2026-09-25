"""The stored fixed-32 candidate contract, proven rather than assumed.

Authoritative artifact:  ``<dataset>/utility/<view>/clustopt_results.csv``
32 rows x 129 columns, written by ``Search_Results_Logger`` during repository
generation and consumed unchanged by ``offline_candidate_execution``.

    9   base columns
    60  ``<metric> (w=1.0)``   raw CVI values
    60  ``<metric>_norm``      oriented/normalised CVI values

Only the ``_norm`` block is usable: Stage 0 measured ``-inf`` sentinels in the raw
columns on 100 % of ``valid=False`` rows and 0.42 % of valid rows, while the
normalised block is finite on every valid candidate. The production objective
uses ``_norm`` exclusively, so the package stores ``_norm``.

Cluster labels are NOT persisted. ``Search_Results_Logger._label_stats`` derives
``valid`` / ``n_clusters_wo_noise`` / ``noise_ratio`` / ``n_labels_unique`` from the
labels and then discards them. Any partition descriptor beyond those four would
require re-clustering.
"""
from __future__ import annotations

import ast
import hashlib
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

CANDIDATE_CSV = "clustopt_results.csv"
EXPECTED_CANDIDATES = 32
VIEW_IDS: Tuple[str, ...] = ("x_only", "y_only", "xy_2d")

BASE_COLUMNS: Tuple[str, ...] = (
    "algorithm", "config_str", "aggregate_score", "valid",
    "n_clusters_wo_noise", "noise_ratio", "n_labels_unique", "exception", "ARI",
)
# Label-derived partition descriptors that ARE persisted.
PARTITION_COLUMNS: Tuple[str, ...] = (
    "n_clusters_wo_noise", "noise_ratio", "n_labels_unique", "valid",
)
TARGET_COLUMNS: Tuple[str, ...] = ("ARI",)

METRIC_SCHEMA_REL = ("results_analysis/metric_utility_knn/phase_e1/models/"
                     "utility_metric_schema.json")

# Hyperparameters observed across the fixed grid, per algorithm family. The
# applicability mask is what keeps an inapplicable parameter from being read as
# a meaningful zero.
HYPERPARAMETERS: Tuple[str, ...] = (
    "n_clusters", "n_components", "init", "batch_size", "random_state",
    "linkage", "affinity", "metric", "threshold", "branching_factor",
    "eps", "min_samples", "min_cluster_size", "covariance_type",
    "cluster_selection_method", "target_k",
)


def candidate_csv_path(dataset_dir: Path, view_id: str) -> Path:
    return Path(dataset_dir) / "utility" / view_id / CANDIDATE_CSV


def load_metric_names(repo: Path) -> List[str]:
    """The canonical 60 CVI names, in the frozen utility-schema order."""
    import json
    schema = json.loads((Path(repo) / METRIC_SCHEMA_REL).read_text("utf-8"))
    names = list(schema["metric_names"])
    if len(names) != 60:
        raise ValueError(f"expected 60 canonical metrics, got {len(names)}")
    return names


def metric_order_hash(names: Sequence[str]) -> str:
    return hashlib.sha256("|".join(names).encode()).hexdigest()[:16]


def csv_metric_names(columns: Sequence[str]) -> List[str]:
    """Metric names as they appear in a candidate CSV, in column order."""
    return [c.split(" (w=")[0] for c in columns if "(w=" in c]


def norm_columns(names: Sequence[str]) -> List[str]:
    return [f"{m}_norm" for m in names]


def is_valid_flag(value: Any) -> bool:
    return str(value).strip().lower() in ("true", "1")


# ------------------------------------------------------------- configuration
_ALGO_RE = re.compile(r"^([a-z0-9_]+)(?:-(\d+))?$")


def parse_config(algorithm: str, config_str: str) -> Dict[str, Any]:
    """Algorithm + hyperparameters from the persisted ``config_str`` literal.

    ``config_str`` is a Python dict literal with ``<algo>__<param>`` keys. It is
    parsed with ``ast.literal_eval`` -- never ``eval`` -- and the algorithm
    prefix is stripped so parameters are comparable across families.
    """
    out: Dict[str, Any] = {"algorithm_raw": str(algorithm)}
    m = _ALGO_RE.match(str(algorithm))
    out["algorithm_base"] = m.group(1) if m else str(algorithm)
    out["algorithm_variant_k"] = int(m.group(2)) if (m and m.group(2)) else None
    try:
        cfg = ast.literal_eval(str(config_str))
    except (ValueError, SyntaxError):
        cfg = {}
    params: Dict[str, Any] = {}
    for key, val in (cfg or {}).items():
        if key == "algorithm":
            continue
        params[key.split("__", 1)[1] if "__" in key else key] = val
    out["params"] = params
    return out


def configuration_row(algorithm: str, config_str: str) -> Dict[str, Any]:
    """One flat catalogue row: value + applicability mask per hyperparameter."""
    parsed = parse_config(algorithm, config_str)
    p = parsed["params"]
    row: Dict[str, Any] = {
        "algorithm": parsed["algorithm_raw"],
        "algorithm_base": parsed["algorithm_base"],
        "cfg__constrained_target_k": parsed["algorithm_variant_k"],
        "cfgmask__constrained_target_k": int(parsed["algorithm_variant_k"] is not None),
    }
    for h in HYPERPARAMETERS:
        if h == "target_k":
            continue
        present = h in p
        val = p.get(h)
        row[f"cfgmask__{h}"] = int(present)
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            row[f"cfg__{h}"] = float(val)
            row[f"cfg__{h}_str"] = ""
        else:
            row[f"cfg__{h}"] = np.nan
            row[f"cfg__{h}_str"] = "" if val is None else str(val)
    row["config_str"] = str(config_str)
    return row


def candidate_key(algorithm: str, config_str: str) -> str:
    return f"{algorithm}|{config_str}"


def slate_fingerprint(algorithms: Sequence[str], configs: Sequence[str]) -> str:
    ident = [candidate_key(a, c) for a, c in zip(algorithms, configs)]
    return hashlib.sha256("\n".join(ident).encode("utf-8")).hexdigest()[:16]


def validate_slate(df: pd.DataFrame, metric_names: Sequence[str],
                   path: Optional[Path] = None) -> None:
    """Hard contract for one stored slate."""
    where = f" ({path})" if path else ""
    if len(df) != EXPECTED_CANDIDATES:
        raise ValueError(f"expected {EXPECTED_CANDIDATES} candidates, "
                         f"found {len(df)}{where}")
    missing = [c for c in BASE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing base columns {missing}{where}")
    csv_metrics = csv_metric_names(df.columns)
    if csv_metrics != list(metric_names):
        raise ValueError(f"CVI order deviates from the canonical schema{where}")
    missing_norm = [c for c in norm_columns(metric_names) if c not in df.columns]
    if missing_norm:
        raise ValueError(f"{len(missing_norm)} metrics lack a _norm column{where}")
    ident = df["algorithm"].astype(str) + "|" + df["config_str"].astype(str)
    if ident.duplicated().any():
        raise ValueError(f"duplicate candidate identity within slate{where}")
