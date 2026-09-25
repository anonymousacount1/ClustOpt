"""Reuse precomputed ClustOpt metrics from the repository's utility CSVs.

During the original ClustOpt repository build, every dataset view was clustered
with the same utility-map configurations and scored with the full ClustOpt
metric set; the results live at::

    <dataset_dir>/utility/<view>/clustopt_results.csv

Each row holds one configuration's raw metric values (columns ``<metric> (w=1.0)``)
plus ``ARI``, ``n_clusters_wo_noise``, ``noise_ratio`` and ``config_str`` (which is
exactly our ``ConfigSpec.to_search_config()`` dict). Because clustering is
deterministic, those raw values are bit-identical to a fresh recompute (verified
to <=2.3e-10), so for any configuration that already appears in the CSV we can
**skip the expensive metric computation** and reuse the stored values.

Safety rules baked in here:

* Only metrics whose **exact current registry name** has a ``(w=1.0)`` column are
  reused. Metrics renamed / re-implemented since the CSV was written (e.g.
  ``convexity_ratio_image_based``, ``hough_arc_circle_strength`` — both NaN in the
  current registry) have no exact match and are always recomputed, so the reuse
  output equals the full-compute output exactly.
* The caller must pass the freshly re-clustered partition's cluster-count / noise
  so :meth:`PrecomputedView.guard_ok` can confirm the stored row corresponds to
  the labels we actually produced; on any mismatch the caller recomputes.
"""
from __future__ import annotations

import ast
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

RAW_SUFFIX = " (w=1.0)"


def _canon_key(config_dict: dict) -> str:
    return json.dumps(config_dict, sort_keys=True)


@dataclass
class PrecomputedMatch:
    reused: Dict[str, float]
    n_clusters_wo_noise: Optional[int]
    noise_ratio: Optional[float]
    valid: bool


@dataclass
class PrecomputedView:
    available: bool = False
    reusable_metrics: List[str] = field(default_factory=list)
    missing_metrics: List[str] = field(default_factory=list)
    _by_key: Dict[str, PrecomputedMatch] = field(default_factory=dict)

    @classmethod
    def load(
        cls, dataset_dir: str | Path, view_type: str, metric_names: List[str]
    ) -> "PrecomputedView":
        pv = cls(missing_metrics=list(metric_names))
        path = Path(dataset_dir) / "utility" / view_type / "clustopt_results.csv"
        if not path.is_file():
            return pv
        try:
            df = pd.read_csv(path)
        except Exception:
            return pv
        if "config_str" not in df.columns:
            return pv

        raw_cols = {
            c[: -len(RAW_SUFFIX)]: c for c in df.columns if c.endswith(RAW_SUFFIX)
        }
        pv.reusable_metrics = [m for m in metric_names if m in raw_cols]
        pv.missing_metrics = [m for m in metric_names if m not in raw_cols]

        for _, row in df.iterrows():
            try:
                cfg = ast.literal_eval(str(row["config_str"]))
            except (ValueError, SyntaxError):
                continue
            if not isinstance(cfg, dict):
                continue
            reused: Dict[str, float] = {}
            for m in pv.reusable_metrics:
                v = row[raw_cols[m]]
                reused[m] = (
                    float(v) if pd.notna(v) and math.isfinite(float(v)) else float("nan")
                )
            ncw = row.get("n_clusters_wo_noise")
            nr = row.get("noise_ratio")
            pv._by_key[_canon_key(cfg)] = PrecomputedMatch(
                reused=reused,
                n_clusters_wo_noise=int(ncw) if pd.notna(ncw) else None,
                noise_ratio=float(nr) if pd.notna(nr) else None,
                valid=bool(row.get("valid", True)),
            )
        pv.available = len(pv._by_key) > 0
        return pv

    def lookup(self, config_dict: dict) -> Optional[PrecomputedMatch]:
        return self._by_key.get(_canon_key(config_dict))

    @staticmethod
    def guard_ok(
        match: PrecomputedMatch,
        predicted_k: int,
        noise_fraction: float,
        *,
        noise_tol: float = 1e-3,
    ) -> bool:
        """Confirm the stored row matches the freshly re-clustered partition."""
        if match.n_clusters_wo_noise is None:
            return False
        if int(predicted_k) != int(match.n_clusters_wo_noise):
            return False
        if match.noise_ratio is not None:
            if abs(float(noise_fraction) - float(match.noise_ratio)) > noise_tol:
                return False
        return True
