"""Offline worker unit: ONE dataset per worker.

Chosen to match the online :mod:`dataset_worker` unit. Within a worker the
dataset's 3 views x 6 arms are processed sequentially, which gives:

* **no write contention** -- a worker owns every output path under its own
  ``<dataset_dir>/experiments/<split_dir>/`` subtree and no other worker
  touches that dataset;
* **candidate reuse** -- the 32-candidate file for a view is loaded once and
  scored by all six arms, guaranteeing the arms genuinely see the *same*
  candidate set (asserted via the shared fingerprint);
* **predictor reuse** -- the Head-14 / Full-60 predictors are loaded once per
  process and cached by the production predictor cache;
* **exception isolation** -- a failure is captured per (view, arm) and a fatal
  dataset error is surfaced without killing the pool.

A record-level pool was deliberately NOT used here (unlike the online
``run_pool_records``): the offline unit of work is far too small to justify
re-loading the candidate file and the predictors per record.
"""
from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .config import ExperimentExecutionConfig, VIEW_IDS
from .method_runner import RunRecord
from .offline_candidate_execution import (
    ArmSpec, OfflineCandidateError, load_candidate_set,
)
from .offline_method_runner import run_offline_method_view
from .output_writer import read_json


@dataclass
class OfflineDatasetResult:
    dataset_id: str
    dataset_dir: str
    records: List[RunRecord] = field(default_factory=list)
    fatal_error: Optional[str] = None


def _dataset_meta(dataset_dir: Path, cfg: ExperimentExecutionConfig) -> Dict[str, Any]:
    """Dataset metadata, mirroring the ONLINE source of truth.

    ``cluster_count`` and ``difficulty`` live under the ``tags`` block of
    ``data_metadata.json`` -- NOT at the top level. The online
    :mod:`dataset_worker` reads them from ``tags`` (falling back to the loaded
    dataset's ``cluster_count_from_generator``), so the offline path must do the
    same or ``true_K`` -- and therefore ``exact_K`` -- resolve to ``None``.
    """
    meta = read_json(dataset_dir / "data_metadata.json") or {}
    tags = meta.get("tags") or {}
    subfamily_dir = cfg.subfamily_dir
    cluster_count = tags.get("cluster_count")
    if cluster_count is None:
        cluster_count = meta.get("cluster_count",
                                 meta.get("cluster_count_from_generator"))
    return {
        "family": tags.get("family_id") or subfamily_dir.parent.parent.name,
        "subfamily": tags.get("subfamily_id") or subfamily_dir.name,
        "difficulty": tags.get("difficulty", meta.get("difficulty")),
        "cluster_count": None if cluster_count is None else int(cluster_count),
    }


def process_dataset_offline(
    dataset_dir: Path,
    cfg: ExperimentExecutionConfig,
    arms: Sequence[ArmSpec],
    head14: Sequence[str],
    full60: Sequence[str],
) -> OfflineDatasetResult:
    """Score every (view, arm) for one dataset."""
    dataset_dir = Path(dataset_dir)
    result = OfflineDatasetResult(dataset_id=dataset_dir.name,
                                  dataset_dir=str(dataset_dir))
    try:
        meta = _dataset_meta(dataset_dir, cfg)
        for view_id in VIEW_IDS:
            # Load the fixed candidate set ONCE; all six arms score it.
            try:
                candidates = load_candidate_set(dataset_dir, view_id)
            except OfflineCandidateError as exc:
                for arm in arms:
                    rec = RunRecord(
                        dataset_id=dataset_dir.name, dataset_dir=str(dataset_dir),
                        split_id=int(cfg.split_id), family=meta.get("family"),
                        subfamily=meta.get("subfamily"),
                        difficulty=meta.get("difficulty"),
                        cluster_count=meta.get("cluster_count"),
                        view_id=view_id, method_name=arm.method_id,
                        method_group="baseline", status="failed",
                        failure_reason=f"OfflineCandidateError: {exc}",
                    )
                    result.records.append(rec)
                continue

            for arm in arms:
                rec = run_offline_method_view(
                    cfg=cfg, dataset_dir=dataset_dir, dataset_meta=meta,
                    view_id=view_id, arm=arm, candidates=candidates,
                    head14=head14, full60=full60,
                )
                result.records.append(rec)
    except Exception as exc:                                  # dataset-fatal
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        result.fatal_error = f"{type(exc).__name__}: {exc}\n{tb}"
    return result
