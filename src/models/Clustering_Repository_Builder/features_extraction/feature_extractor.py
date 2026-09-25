"""Per-view feature extraction orchestrator."""
from __future__ import annotations

import time
import uuid
from typing import Optional

import numpy as np

from .blocks.block_a_partition_stats import compute_block_a
from .blocks.block_b_evaluation_stats import compute_block_b
from .blocks.block_g_geometry_structure import compute_block_g
from .blocks.block_l_landmarking import compute_block_l
from .blocks.block_r_relation_features import compute_block_r
from .blocks.block_v_visual_raster import compute_block_v
from .dataset_loader import LoadedDataset
from .feature_schema import (
    AUDIT_COLUMNS,
    BLOCK_COLUMNS,
    blocks_to_record,
    ensure_complete_feature_record,
)
from .view_builder import DatasetView


def _audit_fields(loaded: LoadedDataset, view: DatasetView, status: str) -> dict[str, object]:
    record_id = f"{loaded.dataset_id}__{view.view_mode}"
    return {
        "dataset_id": loaded.dataset_id,
        "record_id": record_id,
        "view_mode": view.view_mode,
        "view_dim": int(view.view_dim),
        "source_dataset_dir": str(loaded.dataset_dir),
        "family_id": loaded.family_id,
        "family_name": loaded.family_name,
        "subfamily_id": loaded.subfamily_id,
        "difficulty": loaded.difficulty,
        "cluster_count_from_generator": loaded.cluster_count_from_generator,
        "dataset_seed": loaded.dataset_seed,
        "family_seed": loaded.family_seed,
        "n_points": int(loaded.n_points),
        "feature_names_original": list(loaded.feature_names),
        "selected_features": list(view.selected_features),
        "has_labels": bool(loaded.has_labels),
        "has_rendering": bool(loaded.has_rendering),
        "has_structural_ground_truth": bool(loaded.has_structural_ground_truth),
        "extraction_status": status,
    }


def _run_block(fn, *args, **kwargs) -> tuple[dict[str, float], float, Optional[str]]:
    t0 = time.perf_counter()
    err: Optional[str] = None
    try:
        result = fn(*args, **kwargs)
    except Exception as exc:  # block-level failure → NaNs
        result = {}
        err = f"{type(exc).__name__}: {exc}"
    return result, time.perf_counter() - t0, err


def extract_features_for_view(
    loaded: LoadedDataset,
    view: DatasetView,
    *,
    skip_landmarking: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> tuple[dict[str, object], dict[str, object], dict[str, float]]:
    """Compute the full 250-feature record (+ audit) for one view.

    Returns a tuple `(record, timing, raw_feature_record)`:

    - ``record``: schema-complete, no empty cells (audit defaults applied,
      non-finite features filled with 0.0). Safe to write to CSV/Parquet.
    - ``timing``: per-block elapsed time + block_errors dict.
    - ``raw_feature_record``: the 250 feature columns BEFORE the fill step,
      so the summary writer can report a pre-fill missing-value snapshot.

    Block-level exceptions are caught and turned into NaNs in ``raw`` (and
    therefore 0.0 in the filled record). Only catastrophic pre-block
    failures will propagate.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    timing: dict[str, object] = {
        "block_A_time_sec": 0.0,
        "block_B_time_sec": 0.0,
        "block_G_time_sec": 0.0,
        "block_V_time_sec": 0.0,
        "block_R_time_sec": 0.0,
        "block_L_time_sec": 0.0,
        "block_errors": {},
    }

    a_vals, a_t, a_err = _run_block(compute_block_a, view.X_view, view.view_dim)
    timing["block_A_time_sec"] = a_t
    if a_err:
        timing["block_errors"]["A"] = a_err

    b_vals, b_t, b_err = _run_block(compute_block_b, view.X_view, view.view_dim)
    timing["block_B_time_sec"] = b_t
    if b_err:
        timing["block_errors"]["B"] = b_err

    g_vals, g_t, g_err = _run_block(compute_block_g, view.X_view, view.view_dim, rng)
    timing["block_G_time_sec"] = g_t
    if g_err:
        timing["block_errors"]["G"] = g_err

    v_vals, v_t, v_err = _run_block(compute_block_v, loaded)
    timing["block_V_time_sec"] = v_t
    if v_err:
        timing["block_errors"]["V"] = v_err

    r_vals, r_t, r_err = _run_block(compute_block_r, view)
    timing["block_R_time_sec"] = r_t
    if r_err:
        timing["block_errors"]["R"] = r_err

    l_vals, l_t, l_err = _run_block(
        compute_block_l, view.X_view, view.view_dim, skip=skip_landmarking, rng=rng
    )
    timing["block_L_time_sec"] = l_t
    if l_err:
        timing["block_errors"]["L"] = l_err

    flat_features = blocks_to_record({
        "A": a_vals,
        "B": b_vals,
        "G": g_vals,
        "V": v_vals,
        "R": r_vals,
        "L": l_vals,
    })

    status = "ok" if not timing["block_errors"] else "partial"
    raw_feature_record = dict(flat_features)  # snapshot BEFORE fill, NaNs preserved
    record = ensure_complete_feature_record({**_audit_fields(loaded, view, status), **flat_features})
    return record, timing, raw_feature_record
