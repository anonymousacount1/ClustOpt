"""Unified, de-duplicated internal-metric registry.

The unified metric set is the union of:

1. ML2DAC original CVIs    (CH, DBI, SIL, DBCV, DI, CJI, COP)
2. AutoClust original CVIs (identical 7 internal CVIs in the ML2DAC re-impl)
3. Final ClustOpt utility-map metrics (60)

De-duplication rule: a metric is computed **once** by a single implementation.
The four CVIs shared with ClustOpt (CH/DBI/SIL/DBCV) are computed by ClustOpt's
implementation; only the three ML2DAC-only CVIs (Dunn, Coggins-Jain, COP) are
computed by the original ML2DAC code. Every registry row records which models
use the metric so the later derived MKRs can select the right columns.

``direction`` is the *raw-value* optimisation direction. For the seven ML2DAC /
AutoClust CVIs the exact raw direction is recorded (this is what ML2DAC needs
to rank configurations). For the ClustOpt-only structural metrics the raw
direction is implementation-internal; they carry the sentinel
``"clustopt_utility"`` meaning: the stored value is raw, and ClustOpt's own
``normalize()`` maps it to a higher-is-better utility in ``[0, 1]``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from .configuration import get_cvi_metric_names
from .schemas import METRIC_REGISTRY_COLUMNS, SCHEMA_VERSION  # noqa: F401

# The 7 ML2DAC / AutoClust internal CVIs, expressed in ClustOpt canonical names
# with their raw-value optimisation direction and ML2DAC abbreviation.
ML2DAC_CVIS: Dict[str, Tuple[str, str]] = {
    # canonical_name: (direction, ml2dac_abbrev)
    "calinski_harabasz": ("maximize", "CH"),
    "davies_bouldin": ("minimize", "DBI"),
    "silhouette": ("maximize", "SIL"),
    "dbcv": ("maximize", "DBCV"),
    "dunn_index": ("maximize", "DI"),
    "coggins_jain_index": ("maximize", "CJI"),
    "cop": ("minimize", "COP"),
}

CLUSTOPT_UTILITY_DIRECTION = "clustopt_utility"


def _clustopt_metric_objects():
    from models.ClustOpt.cluster_validity_indices.metrics.metrics_registry import (
        METRIC_REGISTRY,
    )

    return METRIC_REGISTRY


def ml2dac_only_metrics() -> List[str]:
    """The ML2DAC CVIs that are NOT in the ClustOpt registry (computed by ML2DAC)."""
    reg = _clustopt_metric_objects()
    return [name for name in ML2DAC_CVIS if name not in reg]


def unified_metric_order(config_root: str | Path) -> List[str]:
    """Ordered metric_ids for the evaluations table: ClustOpt 60 then ML2DAC-only."""
    clustopt = get_cvi_metric_names(config_root)
    return clustopt + ml2dac_only_metrics()


def build_registry(config_root: str | Path) -> pd.DataFrame:
    """Construct the metric registry DataFrame (one row per unique metric)."""
    reg = _clustopt_metric_objects()
    clustopt_names = get_cvi_metric_names(config_root)
    ml2dac_only = ml2dac_only_metrics()

    rows: List[dict] = []

    for name in clustopt_names:
        metric = reg.get(name)
        is_cvi = name in ML2DAC_CVIS
        direction = ML2DAC_CVIS[name][0] if is_cvi else CLUSTOPT_UTILITY_DIRECTION
        rows.append(
            {
                "metric_id": name,
                "metric_name": name,
                "implementation_source": "clustopt",
                "direction": direction,
                "used_by_ml2dac": is_cvi,
                "used_by_autoclust": is_cvi,
                "used_by_clustopt": True,
                "requires_ground_truth": False,
                "requires_distance_matrix": None,
                "requires_cluster_centers": None,
                "requires_noise_handling": bool(
                    getattr(metric, "needs_original_labels", False)
                ),
            }
        )

    for name in ml2dac_only:
        direction = ML2DAC_CVIS[name][0]
        rows.append(
            {
                "metric_id": name,
                "metric_name": name,
                "implementation_source": "ml2dac",
                "direction": direction,
                "used_by_ml2dac": True,
                "used_by_autoclust": True,
                "used_by_clustopt": False,
                "requires_ground_truth": False,
                "requires_distance_matrix": None,
                "requires_cluster_centers": None,
                "requires_noise_handling": False,
            }
        )

    df = pd.DataFrame(rows, columns=METRIC_REGISTRY_COLUMNS)
    return df
