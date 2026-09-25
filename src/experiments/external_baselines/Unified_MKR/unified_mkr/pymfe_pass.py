"""Environment-isolated ML2DAC (pymfe) meta-feature pass.

ML2DAC meta-features use ``pymfe``, which we deliberately do **not** install in
the ClustOpt environment (it would drag pandas/numpy upgrades and risk breaking
ClustOpt). Instead a dedicated minimal virtualenv (``.mfenv``, created by
``setup/create_mf_env``) holds pymfe, and this module drives it as a subprocess:

1. read the subfamily ``records.parquet`` to find view records still missing
   ML2DAC meta-features,
2. hand the (record_id, data_path, view_type) jobs to ``pymfe_worker.py`` run by
   the ``.mfenv`` interpreter (JSON in / JSON out — no shared deps),
3. merge the returned ``ml2dac_<feature>`` columns + status into the per-dataset
   staging and the aggregated ``metafeatures.parquet``.

The main runner invokes this automatically when an ``.mfenv`` interpreter is
found, so users never switch environments by hand. It is also runnable
standalone (``runners/run_ml2dac_metafeatures_pass.py``).
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from . import io_utils
from .logging_utils import log
from .storage import UnifiedMKRStorage

_WORKER = Path(__file__).resolve().parent / "pymfe_worker.py"


def find_mf_python(repo_root: str | Path) -> Optional[Path]:
    """Locate the dedicated pymfe interpreter (``.mfenv``)."""
    base = Path(repo_root) / "experiments" / "external_baselines" / "Unified_MKR" / ".mfenv"
    candidates = [base / "Scripts" / "python.exe", base / "bin" / "python"]
    for c in candidates:
        if c.is_file():
            return c
    return None


def _records_needing_ml2dac(
    records: pd.DataFrame, metafeatures: pd.DataFrame, overwrite: bool
) -> pd.DataFrame:
    if overwrite or "ml2dac_status" not in metafeatures.columns:
        done = set()
    else:
        ok = metafeatures[metafeatures["ml2dac_status"] == "success"]
        done = set(ok["record_id"].tolist())
    return records[~records["record_id"].isin(done)]


def _run_worker(mf_python: Path, jobs: List[dict]) -> Dict[str, dict]:
    if not jobs:
        return {}
    with tempfile.TemporaryDirectory() as td:
        jobs_path = Path(td) / "jobs.json"
        out_path = Path(td) / "out.json"
        jobs_path.write_text(json.dumps(jobs), encoding="utf-8")
        proc = subprocess.run(
            [str(mf_python), str(_WORKER), str(jobs_path), str(out_path)],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0 or not out_path.is_file():
            raise RuntimeError(
                f"pymfe worker failed (rc={proc.returncode}): {proc.stderr[:500]}"
            )
        return json.loads(out_path.read_text(encoding="utf-8"))


def _apply_results(mf_df: pd.DataFrame, results: Dict[str, dict]) -> pd.DataFrame:
    """Write ml2dac_<feature> columns + status into a metafeatures frame."""
    mf_df = mf_df.copy()
    mf_df = mf_df.set_index("record_id")
    for rid, res in results.items():
        if rid not in mf_df.index:
            continue
        mf_df.loc[rid, "ml2dac_status"] = res["status"]
        mf_df.loc[rid, "ml2dac_runtime_sec"] = res.get("runtime_sec", 0.0)
        mf_df.loc[rid, "ml2dac_error"] = res.get("error", "")
        if res["status"] == "success":
            for name, val in zip(res["names"], res["values"]):
                mf_df.loc[rid, f"ml2dac_{name}"] = val
    return mf_df.reset_index()


def run_pymfe_pass(
    out_dir: str | Path,
    mf_python: str | Path,
    *,
    overwrite: bool = False,
    batch_size: int = 32,
) -> dict:
    """Compute + merge ML2DAC meta-features for one subfamily output dir."""
    storage = UnifiedMKRStorage(out_dir)
    records = io_utils.read_parquet(storage.out_dir / "records.parquet")
    metafeatures = io_utils.read_parquet(storage.out_dir / "metafeatures.parquet")
    if records is None or metafeatures is None or records.empty:
        return {"status": "no_data", "processed": 0}

    todo = _records_needing_ml2dac(records, metafeatures, overwrite)
    log(f"[pymfe-pass] records needing ML2DAC meta-features: {len(todo)}/{len(records)}")
    if todo.empty:
        return {"status": "up_to_date", "processed": 0}

    jobs = [
        {
            "record_id": r.record_id,
            "data_path": r.data_path,
            "view_type": r.view_type,
        }
        for r in todo.itertuples(index=False)
    ]

    all_results: Dict[str, dict] = {}
    for i in range(0, len(jobs), batch_size):
        batch = jobs[i : i + batch_size]
        all_results.update(_run_worker(Path(mf_python), batch))
        log(f"[pymfe-pass] {min(i + batch_size, len(jobs))}/{len(jobs)} records done")

    n_ok = sum(1 for r in all_results.values() if r["status"] == "success")

    # Merge into the aggregated metafeatures table.
    merged = _apply_results(metafeatures, all_results)
    io_utils.write_parquet_atomic(storage.out_dir / "metafeatures.parquet", merged)

    # Also update per-dataset staging so resume stays consistent.
    if "dataset_id" in records.columns:
        rec_to_ds = dict(zip(records["record_id"], records["dataset_id"]))
        by_ds: Dict[str, Dict[str, dict]] = {}
        for rid, res in all_results.items():
            ds = rec_to_ds.get(rid)
            if ds:
                by_ds.setdefault(ds, {})[rid] = res
        for ds, res_map in by_ds.items():
            staged = storage._read_staged(ds, "metafeatures")
            if staged is not None and not staged.empty:
                updated = _apply_results(staged, res_map)
                io_utils.write_parquet_atomic(
                    storage.dataset_stage_dir(ds) / "metafeatures.parquet", updated
                )

    log(f"[pymfe-pass] ML2DAC meta-features populated: {n_ok}/{len(jobs)} succeeded")
    return {"status": "ok", "processed": len(jobs), "succeeded": n_ok}
