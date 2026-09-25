"""Discovery + consistency-checked merge of subfamily Unified-MKR outputs.

Stacked tables (records/metafeatures/evaluations/failures) are concatenated
after reconciling them to a canonical column order. Catalogue tables
(configurations/metric_registry) and ``schema_version.json`` are NOT stacked:
they must be byte-for-byte consistent across every subfamily, so we verify that
and keep a single canonical copy, reporting any divergence.

Nothing here mutates the subfamily outputs. The only transform applied to data
is rewriting ``evaluations.labels_path`` from the subfamily-relative
``labels/<rid>/<cid>.npy`` to the outputs-root-relative
``<family>/<subfamily>/labels/<rid>/<cid>.npy`` so the merged table can locate
labels without copying them.
"""
from __future__ import annotations

import gc
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import pandas as pd

from . import io_utils
from .schemas import FAILURES_COLUMNS

# Files every subfamily output directory must contain.
REQUIRED_PARQUET = (
    "records",
    "metafeatures",
    "configurations",
    "metric_registry",
    "evaluations",
    "failures",
)
REQUIRED_JSON = ("schema_version.json", "build_config.json")

# Tables that are row-stacked across subfamilies.
STACKED_TABLES = ("records", "metafeatures", "evaluations", "failures")
# Tables that must be identical across subfamilies (kept as one canonical copy).
CATALOGUE_TABLES = ("configurations", "metric_registry")


@dataclass
class Subfamily:
    family: str
    subfamily: str
    dir: Path

    @property
    def key(self) -> str:
        return f"{self.family}/{self.subfamily}"


@dataclass
class MergeResult:
    subfamilies: List[Subfamily]
    merged_counts: Dict[str, int] = field(default_factory=dict)
    schema_report: dict = field(default_factory=dict)
    registry_report: dict = field(default_factory=dict)
    configuration_report: dict = field(default_factory=dict)
    canonical_configurations: Optional[pd.DataFrame] = None
    canonical_metric_registry: Optional[pd.DataFrame] = None
    schema_version: Optional[str] = None
    aborted: bool = False
    abort_reason: str = ""


def _log(log: Optional[Callable[[str], None]], msg: str) -> None:
    if log:
        log(msg)


# --- discovery -------------------------------------------------------------

def discover_subfamilies(outputs_root: str | Path) -> List[Subfamily]:
    """Every ``<outputs_root>/<family>/<subfamily>`` holding a records.parquet."""
    root = Path(outputs_root)
    found: List[Subfamily] = []
    for family_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for sub_dir in sorted(p for p in family_dir.iterdir() if p.is_dir()):
            if (sub_dir / "records.parquet").exists():
                found.append(Subfamily(family_dir.name, sub_dir.name, sub_dir))
    return found


def check_required_files(sf: Subfamily) -> List[str]:
    missing: List[str] = []
    for t in REQUIRED_PARQUET:
        if not (sf.dir / f"{t}.parquet").exists():
            missing.append(f"{t}.parquet")
    for j in REQUIRED_JSON:
        if not (sf.dir / j).exists():
            missing.append(j)
    return missing


# --- hashing ---------------------------------------------------------------

def frame_fingerprint(df: pd.DataFrame) -> str:
    """Order-independent SHA256 of a frame's schema + content."""
    d = df.reindex(sorted(df.columns), axis=1)
    try:
        d = d.sort_values(list(d.columns), kind="mergesort").reset_index(drop=True)
    except Exception:
        d = d.reset_index(drop=True)
    h = hashlib.sha256()
    h.update("|".join(f"{c}:{str(d[c].dtype)}" for c in d.columns).encode())
    h.update(pd.util.hash_pandas_object(d, index=False).values.tobytes())
    return h.hexdigest()


# --- schema reconciliation -------------------------------------------------

def _canonical_columns(sf_columns: Dict[str, List[str]]) -> List[str]:
    """Canonical order = first subfamily's column order."""
    first = next(iter(sf_columns.values()))
    return list(first)


def _read(sf: Subfamily, table: str) -> pd.DataFrame:
    df = io_utils.read_parquet(sf.dir / f"{table}.parquet")
    return df if df is not None else pd.DataFrame()


# --- catalogue consistency -------------------------------------------------

def _check_catalogue(
    subfamilies: List[Subfamily], table: str
) -> tuple[dict, Optional[pd.DataFrame]]:
    """Verify a catalogue table is identical across subfamilies."""
    fps: Dict[str, str] = {}
    frames: Dict[str, pd.DataFrame] = {}
    for sf in subfamilies:
        df = _read(sf, table)
        frames[sf.key] = df
        fps[sf.key] = frame_fingerprint(df)

    unique_fps = sorted(set(fps.values()))
    canonical_key = subfamilies[0].key
    canonical = frames[canonical_key]
    identical = len(unique_fps) == 1

    # Group subfamilies by fingerprint so divergence is easy to read.
    groups: Dict[str, List[str]] = {}
    for key, fp in fps.items():
        groups.setdefault(fp, []).append(key)

    report = {
        "table": table,
        "identical_across_subfamilies": identical,
        "n_distinct_fingerprints": len(unique_fps),
        "n_rows_canonical": int(len(canonical)),
        "n_cols_canonical": int(canonical.shape[1]),
        "canonical_subfamily": canonical_key,
        "canonical_fingerprint": fps[canonical_key],
        "fingerprint_groups": {fp: sorted(keys) for fp, keys in groups.items()},
    }

    if not identical:
        # Describe how the divergent groups differ from canonical.
        diffs = []
        canon_cols = set(canonical.columns)
        for fp, keys in groups.items():
            if fp == fps[canonical_key]:
                continue
            other = frames[keys[0]]
            other_cols = set(other.columns)
            diffs.append(
                {
                    "subfamilies": sorted(keys),
                    "missing_columns": sorted(canon_cols - other_cols),
                    "extra_columns": sorted(other_cols - canon_cols),
                    "row_count": int(len(other)),
                }
            )
        report["divergences"] = diffs
    return report, canonical


# --- main entry ------------------------------------------------------------

def run_merge(
    *,
    subfamilies: List[Subfamily],
    write_table: Callable[[str, pd.DataFrame], object],
    outputs_root: Path,
    strict: bool,
    dry_run: bool = False,
    log: Optional[Callable[[str], None]] = None,
) -> MergeResult:
    result = MergeResult(subfamilies=subfamilies)

    # --- 0. schema_version consistency ------------------------------------
    versions = {}
    for sf in subfamilies:
        sv = io_utils.read_json(sf.dir / "schema_version.json") or {}
        versions[sf.key] = sv.get("schema_version")
    distinct_versions = sorted({v for v in versions.values()})
    result.schema_version = distinct_versions[0] if len(distinct_versions) == 1 else None

    # --- 1. stacked-table schema consistency ------------------------------
    schema_report: Dict[str, dict] = {
        "schema_versions": sorted(distinct_versions),
        "schema_version_consistent": len(distinct_versions) == 1,
        "tables": {},
    }
    canonical_cols: Dict[str, List[str]] = {}
    for table in STACKED_TABLES:
        cols_by_sf: Dict[str, List[str]] = {}
        for sf in subfamilies:
            df = _read(sf, table)
            # Empty failures tables carry no columns -> use canonical schema.
            if table == "failures" and df.shape[1] == 0:
                cols_by_sf[sf.key] = list(FAILURES_COLUMNS)
            else:
                cols_by_sf[sf.key] = list(df.columns)
        canon = _canonical_columns(cols_by_sf)
        canon_set = set(canon)
        mismatches = []
        for key, cols in cols_by_sf.items():
            cset = set(cols)
            if cset != canon_set:
                mismatches.append(
                    {
                        "subfamily": key,
                        "missing_columns": sorted(canon_set - cset),
                        "extra_columns": sorted(cset - canon_set),
                    }
                )
        canonical_cols[table] = canon
        schema_report["tables"][table] = {
            "canonical_n_columns": len(canon),
            "consistent": not mismatches,
            "mismatches": mismatches,
        }

    result.schema_report = schema_report

    schema_ok = schema_report["schema_version_consistent"] and all(
        t["consistent"] for t in schema_report["tables"].values()
    )

    # --- 2. catalogue consistency (configurations, metric_registry) -------
    cfg_report, canonical_cfg = _check_catalogue(subfamilies, "configurations")
    reg_report, canonical_reg = _check_catalogue(subfamilies, "metric_registry")
    result.configuration_report = cfg_report
    result.registry_report = reg_report
    result.canonical_configurations = canonical_cfg
    result.canonical_metric_registry = canonical_reg

    catalogue_ok = cfg_report["identical_across_subfamilies"] and reg_report[
        "identical_across_subfamilies"
    ]

    # --- strict gate before any write ------------------------------------
    if strict and not (schema_ok and catalogue_ok):
        result.aborted = True
        reasons = []
        if not schema_report["schema_version_consistent"]:
            reasons.append("schema_version divergence")
        for tbl, info in schema_report["tables"].items():
            if not info["consistent"]:
                reasons.append(f"{tbl} column mismatch")
        if not cfg_report["identical_across_subfamilies"]:
            reasons.append("configurations divergence")
        if not reg_report["identical_across_subfamilies"]:
            reasons.append("metric_registry divergence")
        result.abort_reason = "; ".join(reasons)
        _log(log, f"[merge] STRICT ABORT: {result.abort_reason}")
        return result

    if dry_run:
        _log(log, "[merge] dry-run: skipping writes")
        # Still report the would-be canonical catalogue counts.
        result.merged_counts = {
            "configurations": int(len(canonical_cfg)),
            "metric_registry": int(len(canonical_reg)),
        }
        return result

    # --- 3. write canonical catalogue tables ------------------------------
    write_table("configurations", canonical_cfg)
    write_table("metric_registry", canonical_reg)
    result.merged_counts["configurations"] = int(len(canonical_cfg))
    result.merged_counts["metric_registry"] = int(len(canonical_reg))

    # --- 4. stack + write data tables (one at a time to cap memory) -------
    for table in STACKED_TABLES:
        canon = canonical_cols[table]
        frames: List[pd.DataFrame] = []
        for sf in subfamilies:
            df = _read(sf, table)
            if df.shape[1] == 0:
                # empty (e.g. zero-failure subfamily): contribute no rows
                continue
            # reconcile to canonical column order (missing -> NaN, drop extras)
            df = df.reindex(columns=canon)
            if table == "evaluations":
                df = _rewrite_labels_path(df, sf)
            frames.append(df)
        merged = (
            pd.concat(frames, ignore_index=True)
            if frames
            else pd.DataFrame(columns=canon)
        )
        write_table(table, merged)
        result.merged_counts[table] = int(len(merged))
        _log(log, f"[merge] {table}: {len(merged):,} rows")
        del frames, merged
        gc.collect()

    return result


def _rewrite_labels_path(df: pd.DataFrame, sf: Subfamily) -> pd.DataFrame:
    """Make labels_path relative to the outputs root: <family>/<subfamily>/..."""
    if "labels_path" not in df.columns:
        return df
    prefix = f"{sf.family}/{sf.subfamily}/"

    def _fix(p):
        if isinstance(p, str) and p:
            return prefix + p
        return p

    df = df.copy()
    df["labels_path"] = df["labels_path"].map(_fix)
    return df
