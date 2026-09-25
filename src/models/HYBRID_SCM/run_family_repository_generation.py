"""Family-level repository generation runner.

This is the new entry point requested in
``HYBRID_SCM_FAMILY_REPOSITORY_GENERATION_PLAN.md``. It generates one family
of raw 2D datasets per invocation, in memory, by reusing the existing
``generate_dataset`` / ``make_replay_config`` / ``save_artifacts`` pipeline.

Usage::

    python -m models.HYBRID_SCM.run_family_repository_generation \\
        --family-config models/HYBRID_SCM/configs/repository_generation/families/linear_bands_parallel_stripes.json

The runner never modifies any existing flow. ``run_generate.py`` continues
to operate on ``--config``, ``--bundle``, and ``--replay`` exactly as before.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import time
import traceback
from pathlib import Path
from typing import List, Optional

from models.HYBRID_SCM.io.save_utils import save_artifacts
from models.HYBRID_SCM.replay import make_replay_config
from models.HYBRID_SCM.src.generator import generate_dataset

from models.HYBRID_SCM.family_generation.dataset_builder import build_dataset_config
from models.HYBRID_SCM.family_generation.manifest import (
    ManifestRow,
    SubfamilyReport,
    write_family_snapshot,
    write_manifest_csv,
    write_summary_json,
    write_summary_md,
)
from models.HYBRID_SCM.family_generation.planning import (
    DatasetJob,
    FamilyPlan,
    expand_jobs,
    family_job_counts,
    load_family_plan,
)


# =========================================================
# Path helpers
# =========================================================

_REPOSITORY_OUTPUT_ROOT_NAME = "repository_generation"


def _now_timestamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _repo_root() -> Path:
    """Resolve the repository root.

    This file lives at ``<repo_root>/models/HYBRID_SCM/run_family_repository_generation.py``,
    so the repo root is two parents up from this file's directory.
    """
    return Path(__file__).resolve().parents[2]


def _family_run_dir(repo_root: Path, family_id: str, timestamp: str) -> Path:
    """Family run directory under ``<repo_root>/results_analysis/repository_generation/``.

    This mirrors the layout used by ``metric_coverage_validation``: each
    pipeline gets its own dedicated subfolder under the top-level
    ``results_analysis/`` directory at the repository root.
    """
    root = repo_root / "results_analysis" / _REPOSITORY_OUTPUT_ROOT_NAME
    root.mkdir(parents=True, exist_ok=True)
    run_dir = root / f"{timestamp}__{family_id}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


_DIFFICULTY_SHORT = {"easy": "e", "medium": "m", "hard": "h", "very_hard": "vh"}


def _dataset_id(job: DatasetJob) -> str:
    """Spec-compliant descriptive dataset id (stored in manifest + metadata)."""
    return (
        f"{job.subfamily.subfamily_id}__c{int(job.cluster_count)}__{job.difficulty}__"
        f"idx{int(job.global_idx):06d}__s{int(job.dataset_seed)}"
    )


def _dataset_folder_name(job: DatasetJob) -> str:
    """Short on-disk folder name. Windows MAX_PATH (260 chars) makes the
    spec's full descriptive id unsafe when combined with the user's repo
    location and the save_utils auto-prefixed leaf filenames (e.g.
    ``<tag>_structural_ground_truth.json``). The folder name encodes
    cluster_count, abbreviated difficulty, 6-digit global index, and an
    8-hex-char seed, all of which are uniquely identifying given that the
    parent path already includes the subfamily. The full descriptive
    ``dataset_id`` is preserved in the manifest CSV and in ``cfg.tags``
    so searchable lookups remain possible.
    """
    diff_short = _DIFFICULTY_SHORT[job.difficulty]
    seed_hex = f"{int(job.dataset_seed) & 0xFFFFFFFF:08x}"
    return f"c{int(job.cluster_count)}{diff_short}_{int(job.global_idx):06d}_{seed_hex}"


# =========================================================
# Generation step
# =========================================================

def _generate_one(
    plan: FamilyPlan,
    job: DatasetJob,
    family_run_dir: Path,
) -> ManifestRow:
    dataset_id = _dataset_id(job)
    folder_name = _dataset_folder_name(job)
    dataset_dir = family_run_dir / "subfamilies" / job.subfamily.subfamily_id / folder_name
    output_dir_str = str(dataset_dir)

    cfg, n_points_requested = build_dataset_config(plan, job, dataset_id=dataset_id)

    row = ManifestRow(
        dataset_index=int(job.global_idx),
        dataset_id=dataset_id,
        family_id=plan.family_id,
        family_name=plan.family_name,
        subfamily_id=job.subfamily.subfamily_id,
        cluster_count=int(job.cluster_count),
        difficulty=job.difficulty,
        family_seed=int(plan.family_seed),
        dataset_seed=int(job.dataset_seed),
        n_points_requested=int(n_points_requested),
        n_points_generated=None,
        output_dir=output_dir_str,
        status="pending",
    )

    try:
        X, y, meta, feat_names = generate_dataset(cfg)
        replay_cfg = make_replay_config(cfg, meta)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        # Pass a short tag ("data") to keep file prefixes small. The full
        # descriptive name lives on ``cfg.name`` (carried into metadata) and
        # in the manifest's ``dataset_id`` column.
        save_artifacts(
            run_dir=dataset_dir,
            X=X,
            y=y,
            feature_names=feat_names,
            metadata=meta,
            replay_config=replay_cfg,
            tag="data",
        )
        row.n_points_generated = int(X.shape[0])
        row.status = "ok"
        row.error_message = None
    except Exception as exc:
        row.status = "failed"
        row.error_message = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    return row


# =========================================================
# Top-level orchestration
# =========================================================

def run_family(
    family_config_path: Path,
    repo_root: Optional[Path] = None,
    fail_fast: bool = False,
) -> Path:
    plan = load_family_plan(family_config_path)
    jobs = expand_jobs(plan)
    total_planned, per_subfamily_counts = family_job_counts(plan)

    repo_root = (repo_root or _repo_root())
    timestamp = _now_timestamp()
    family_run_dir = _family_run_dir(repo_root, plan.family_id, timestamp)

    write_family_snapshot(plan.raw_config, family_run_dir / "family_config_snapshot.json")

    print(f"Loaded family config: {plan.family_name}")
    print(f"Family id: {plan.family_id}")
    print(f"Family seed: {plan.family_seed}")
    print(f"Total planned: {total_planned} raw datasets")
    print(f"Expected future repository rows after 3 views: {total_planned * 3}")
    print(f"Output root: {family_run_dir}")
    print()

    # Group jobs by subfamily for clean progress printing.
    subfamily_reports: List[SubfamilyReport] = []
    sub_id_to_report: dict[str, SubfamilyReport] = {}
    for sub in plan.subfamilies:
        rep = SubfamilyReport(subfamily_id=sub.subfamily_id, planned=sub.total)
        for cluster_count, cell_total in per_subfamily_counts[sub.subfamily_id].items():
            rep.by_cluster[int(cluster_count)] = {
                "planned": int(cell_total),
                "generated": 0,
                "failed": 0,
            }
        subfamily_reports.append(rep)
        sub_id_to_report[sub.subfamily_id] = rep

    rows: List[ManifestRow] = []
    last_sub_idx: Optional[int] = None
    started = time.time()

    for job in jobs:
        if last_sub_idx is None or job.subfamily_idx != last_sub_idx:
            print(
                f"[Subfamily {job.subfamily_idx}/{len(plan.subfamilies)}] "
                f"{job.subfamily.subfamily_id}"
            )
            print(f"Planned: {job.subfamily.total} datasets")
            for cc in sorted(per_subfamily_counts[job.subfamily.subfamily_id].keys()):
                print(f"  c{cc}: {per_subfamily_counts[job.subfamily.subfamily_id][cc]}")
            last_sub_idx = job.subfamily_idx

        row = _generate_one(plan, job, family_run_dir)
        rows.append(row)
        sub_id_to_report[job.subfamily.subfamily_id].record(
            job.cluster_count, ok=(row.status == "ok")
        )

        n_pts = "?" if row.n_points_generated is None else str(row.n_points_generated)
        print(
            f"[{plan.family_id} | {job.subfamily.subfamily_id}] "
            f"[{job.subfamily_local:03d}/{job.subfamily.total:03d}] "
            f"[{job.global_idx:06d}/{total_planned:06d}] "
            f"clusters={job.cluster_count} difficulty={job.difficulty} "
            f"points={n_pts} seed={job.dataset_seed} status={row.status}"
        )
        if row.status == "ok":
            print(f"saved: {row.output_dir}")
        else:
            print(f"error: {row.error_message}")

        # End-of-subfamily flush log.
        next_idx = job.global_idx
        is_last_in_sub = (
            next_idx == total_planned
            or jobs[next_idx].subfamily_idx != job.subfamily_idx
        )
        if is_last_in_sub:
            rep = sub_id_to_report[job.subfamily.subfamily_id]
            print(f"Finished subfamily {job.subfamily.subfamily_id}")
            print(f"  planned: {rep.planned}")
            print(f"  generated: {rep.generated}")
            print(f"  failed: {rep.failed}")
            print()

        if fail_fast and row.status != "ok":
            print("Fail-fast enabled: stopping after first failure.")
            break

    elapsed = time.time() - started

    write_manifest_csv(rows, family_run_dir / "family_generation_manifest.csv")

    total_generated = sum(1 for r in rows if r.status == "ok")
    total_failed = sum(1 for r in rows if r.status != "ok")
    summary = {
        "family_id": plan.family_id,
        "family_name": plan.family_name,
        "family_seed": plan.family_seed,
        "output_dir": str(family_run_dir),
        "total_planned": int(total_planned),
        "total_generated": int(total_generated),
        "total_failed": int(total_failed),
        "elapsed_seconds": float(elapsed),
        "subfamilies": [
            {
                "subfamily_id": rep.subfamily_id,
                "planned": rep.planned,
                "generated": rep.generated,
                "failed": rep.failed,
                "by_cluster": rep.by_cluster,
            }
            for rep in subfamily_reports
        ],
    }
    write_summary_json(summary, family_run_dir / "family_generation_summary.json")
    write_summary_md(summary, rows, subfamily_reports, family_run_dir / "family_generation_summary.md")

    print(f"Family generation completed: {plan.family_id}")
    print(f"Total planned: {total_planned}")
    print(f"Generated: {total_generated}")
    print(f"Failed: {total_failed}")
    print(f"Expected future repository rows: {total_generated} x 3 = {total_generated * 3}")
    print(f"Output: {family_run_dir}")

    return family_run_dir


# =========================================================
# CLI
# =========================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="HYBRID_SCM family-level repository generation runner.",
    )
    parser.add_argument(
        "--family-config",
        type=str,
        required=True,
        help="Path to a family planning JSON config.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop generation on the first dataset that fails.",
    )
    args = parser.parse_args()

    run_family(Path(args.family_config), fail_fast=bool(args.fail_fast))


if __name__ == "__main__":
    main()
