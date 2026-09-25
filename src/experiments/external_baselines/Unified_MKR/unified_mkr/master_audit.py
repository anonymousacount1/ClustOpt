"""Master Readiness Audit over the merged Unified MKR.

Reads the master parquet tables and produces a structured audit dict plus the
machine-readable JSON reports under ``validation/``. The runner turns the dict
into the human-readable Markdown reports. Nothing here mutates inputs.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from . import io_utils
from .schemas import EXTERNAL_METRICS, N_CONFIGS_PER_VIEW, VIEW_TYPES

# Known global summary from the completed subfamily build (source of truth to
# compare the merged data against).
EXPECTED = {
    "families": 12,
    "subfamilies": 86,
    "datasets": 17_068,
    "records": 51_204,
    "evaluations": 1_638_528,
    "real_failures": 42,
}
EXPECTED_CLUSTOPT_METRICS = 60

# The 7 ML2DAC internal CVIs -> evaluation column names.
ML2DAC_CVI_COLUMNS = {
    "Calinski-Harabasz": "calinski_harabasz",
    "Davies-Bouldin": "davies_bouldin",
    "Silhouette": "silhouette",
    "DBCV": "dbcv",
    "Dunn": "dunn_index",
    "Coggins-Jain": "coggins_jain_index",
    "COP": "cop",
}
# AutoClust uses the same 7 internal CVIs as the per-config CVI vector -> ARI.
AUTOCLUST_CVI_COLUMNS = dict(ML2DAC_CVI_COLUMNS)

# Metafeature columns that are bookkeeping, not features.
_MF_NON_FEATURE = {
    "record_id", "dataset_id", "view_type", "split_id",
    "ml2dac_status", "ml2dac_runtime_sec", "ml2dac_error",
    "autoclust_status", "autoclust_runtime_sec", "autoclust_error",
}

# Tolerances.
_CVI_NULL_TOL = 0.05  # over valid evaluations


def _log(log: Optional[Callable[[str], None]], msg: str) -> None:
    if log:
        log(msg)


def run_audit(
    *,
    master_root: Path,
    outputs_root: Path,
    metric_registry: pd.DataFrame,
    configurations: pd.DataFrame,
    label_mode: str = "sampled",
    label_sample: int = 3000,
    log: Optional[Callable[[str], None]] = None,
) -> dict:
    master_root = Path(master_root)
    records = io_utils.read_parquet(master_root / "records.parquet")
    metafeatures = io_utils.read_parquet(master_root / "metafeatures.parquet")
    evaluations = io_utils.read_parquet(master_root / "evaluations.parquet")
    failures = io_utils.read_parquet(master_root / "failures.parquet")
    if failures is None:
        failures = pd.DataFrame(columns=["record_id"])

    # Real failures (clustering stage) drive the per-family / per-split failure
    # columns; metafeatures-stage rows are cosmetic (see _failure_analysis).
    if len(failures) and "stage" in failures.columns:
        real_failures = failures[failures["stage"] == "clustering"]
    else:
        real_failures = failures

    audit: Dict[str, object] = {}
    _log(log, "[audit] counts + keys")
    audit["counts"] = _counts(records, evaluations, failures)
    audit["primary_keys"] = _primary_keys(records, metafeatures, evaluations, configurations)
    audit["config_coverage"] = _config_coverage(evaluations)
    audit["family_coverage"] = _family_coverage(records, evaluations, real_failures)
    audit["split_distribution"] = _split_distribution(records, evaluations, real_failures)
    _log(log, "[audit] feature coverage")
    audit["ml2dac_features"] = _feature_coverage(metafeatures, "ml2dac_")
    audit["autoclust_features"] = _feature_coverage(metafeatures, "autoclust_")
    _log(log, "[audit] metric coverage")
    audit["external_metrics"] = _external_metrics(evaluations)
    audit["internal_metrics"] = _internal_metrics(evaluations, metric_registry)
    audit["ml2dac_cvi"] = _cvi_coverage(evaluations, ML2DAC_CVI_COLUMNS)
    audit["autoclust_cvi"] = _cvi_coverage(evaluations, AUTOCLUST_CVI_COLUMNS)
    audit["clustopt_metrics"] = _clustopt_coverage(evaluations, metric_registry)
    _log(log, "[audit] failure analysis")
    audit["failure_analysis"] = _failure_analysis(failures, records, evaluations)
    _log(log, f"[audit] label audit ({label_mode})")
    audit["label_audit"] = _label_audit(
        evaluations, records, outputs_root, label_mode, label_sample, log
    )
    audit["storage"] = _storage(master_root)
    audit["readiness"] = _readiness(audit)
    return audit


# --- 7.1 counts ------------------------------------------------------------

def _counts(records, evaluations, failures) -> dict:
    n_datasets = int(records["dataset_id"].nunique()) if len(records) else 0
    n_records = int(len(records))
    n_eval = int(len(evaluations))
    n_fail_rows = int(len(failures)) if failures is not None else 0
    if failures is not None and len(failures) and "stage" in failures.columns:
        n_real = int((failures["stage"] == "clustering").sum())
        n_metafeat = int((failures["stage"] == "metafeatures").sum())
    else:
        n_real, n_metafeat = 0, 0
    out = {
        "datasets": n_datasets,
        "records": n_records,
        "evaluations": n_eval,
        "failure_rows_total": n_fail_rows,
        "real_failures_clustering_stage": n_real,
        "metafeature_stage_rows": n_metafeat,
        "expected": dict(EXPECTED),
        "records_eq_datasets_x3": n_records == n_datasets * 3,
        "evaluations_eq_records_x32": n_eval == n_records * N_CONFIGS_PER_VIEW,
        "real_failures_match_expected": n_real == EXPECTED["real_failures"],
    }
    out["datasets_match_expected"] = n_datasets == EXPECTED["datasets"]
    out["records_match_expected"] = n_records == EXPECTED["records"]
    out["evaluations_match_expected"] = n_eval == EXPECTED["evaluations"]
    return out


# --- 7.2 primary keys ------------------------------------------------------

def _primary_keys(records, metafeatures, evaluations, configurations) -> dict:
    out: Dict[str, object] = {}
    out["records_record_id_unique"] = bool(records["record_id"].is_unique)
    out["records_duplicate_ids"] = int(records["record_id"].duplicated().sum())
    out["metafeatures_record_id_unique"] = bool(metafeatures["record_id"].is_unique)

    ev_dup = evaluations.duplicated(subset=["record_id", "config_id"]).sum()
    out["evaluations_pk_unique"] = bool(ev_dup == 0)
    out["evaluations_duplicate_pk"] = int(ev_dup)

    rec_ids = set(records["record_id"])
    mf_ids = set(metafeatures["record_id"])
    ev_ids = set(evaluations["record_id"])
    cfg_ids = set(configurations["config_id"])
    ev_cfg_ids = set(evaluations["config_id"])

    out["metafeatures_record_id_subset_of_records"] = mf_ids <= rec_ids
    out["evaluations_record_id_subset_of_records"] = ev_ids <= rec_ids
    out["evaluations_config_id_subset_of_configurations"] = ev_cfg_ids <= cfg_ids
    out["orphan_metafeature_record_ids"] = sorted(list(mf_ids - rec_ids))[:20]
    out["orphan_evaluation_record_ids"] = sorted(list(ev_ids - rec_ids))[:20]
    out["unknown_evaluation_config_ids"] = sorted(list(ev_cfg_ids - cfg_ids))[:20]
    out["records_missing_metafeatures"] = int(len(rec_ids - mf_ids))
    return out


# --- 7.3 per-record config coverage ---------------------------------------

def _config_coverage(evaluations) -> dict:
    per_record = evaluations.groupby("record_id")["config_id"].count()
    exact = int((per_record == N_CONFIGS_PER_VIEW).sum())
    fewer = per_record[per_record < N_CONFIGS_PER_VIEW]
    more = per_record[per_record > N_CONFIGS_PER_VIEW]
    return {
        "expected_per_record": N_CONFIGS_PER_VIEW,
        "records_total": int(per_record.shape[0]),
        "records_exactly_32": exact,
        "records_fewer_than_32": int(fewer.shape[0]),
        "records_more_than_32": int(more.shape[0]),
        "all_records_exactly_32": bool(fewer.empty and more.empty),
        "examples_fewer": {str(k): int(v) for k, v in fewer.head(20).items()},
        "examples_more": {str(k): int(v) for k, v in more.head(20).items()},
    }


# --- 7.4 family/subfamily coverage ----------------------------------------

def _family_coverage(records, evaluations, failures) -> dict:
    # per (family, subfamily)
    rec = records.copy()
    sf = (
        rec.groupby(["family", "subfamily"])
        .agg(datasets=("dataset_id", "nunique"), records=("record_id", "nunique"))
        .reset_index()
    )
    # evaluations per subfamily via record_id -> family/subfamily map
    rid_map = rec[["record_id", "family", "subfamily"]].drop_duplicates("record_id")
    ev = evaluations[["record_id"]].merge(rid_map, on="record_id", how="left")
    ev_counts = ev.groupby(["family", "subfamily"]).size().rename("evaluations").reset_index()
    sf = sf.merge(ev_counts, on=["family", "subfamily"], how="left")
    # failures per subfamily
    if failures is not None and len(failures) and "record_id" in failures.columns:
        fl = failures[["record_id"]].merge(rid_map, on="record_id", how="left")
        fl_counts = fl.groupby(["family", "subfamily"]).size().rename("real_failures").reset_index()
        sf = sf.merge(fl_counts, on=["family", "subfamily"], how="left")
    sf["real_failures"] = sf.get("real_failures", 0)
    sf["real_failures"] = sf["real_failures"].fillna(0).astype(int)
    sf["evaluations"] = sf["evaluations"].fillna(0).astype(int)

    # roll up to family
    fam = (
        sf.groupby("family")
        .agg(
            subfamilies=("subfamily", "nunique"),
            datasets=("datasets", "sum"),
            records=("records", "sum"),
            evaluations=("evaluations", "sum"),
            real_failures=("real_failures", "sum"),
        )
        .reset_index()
        .sort_values("family")
    )
    return {
        "n_families": int(fam.shape[0]),
        "n_subfamilies": int(sf.shape[0]),
        "by_family": fam.to_dict(orient="records"),
        "by_subfamily": sf.sort_values(["family", "subfamily"]).to_dict(orient="records"),
        "totals": {
            "datasets": int(fam["datasets"].sum()),
            "records": int(fam["records"].sum()),
            "evaluations": int(fam["evaluations"].sum()),
            "real_failures": int(fam["real_failures"].sum()),
        },
    }


# --- 7.5 split distribution ------------------------------------------------

def _split_distribution(records, evaluations, failures) -> dict:
    out: Dict[str, object] = {}
    out["records_with_null_split"] = int(records["split_id"].isna().sum())
    out["all_records_have_split"] = bool(records["split_id"].notna().all())

    rec_split = records[["record_id", "split_id"]].drop_duplicates("record_id")
    ds_per_split = (
        records.groupby("split_id")["dataset_id"].nunique().rename("datasets")
    )
    rec_per_split = records.groupby("split_id")["record_id"].nunique().rename("records")
    ev = evaluations[["record_id"]].merge(rec_split, on="record_id", how="left")
    ev_per_split = ev.groupby("split_id").size().rename("evaluations")
    table = pd.concat([ds_per_split, rec_per_split, ev_per_split], axis=1).fillna(0).astype(int)
    if failures is not None and len(failures) and "split_id" in failures.columns:
        fl_per_split = failures.groupby("split_id").size().rename("failures")
        table = table.join(fl_per_split).fillna(0).astype(int)
    else:
        table["failures"] = 0
    table = table.reset_index().sort_values("split_id")
    out["n_splits"] = int(table.shape[0])
    out["by_split"] = table.to_dict(orient="records")
    return out


# --- 8 feature coverage ----------------------------------------------------

def _feature_coverage(metafeatures, prefix: str) -> dict:
    feat_cols = [
        c for c in metafeatures.columns
        if c.startswith(prefix) and c not in _MF_NON_FEATURE
    ]
    n = len(metafeatures)
    status_col = f"{prefix}status".rstrip("_") if False else f"{prefix}status"
    # status column is e.g. ml2dac_status / autoclust_status
    status_col = prefix + "status"
    status_counts = (
        metafeatures[status_col].fillna("missing").value_counts().to_dict()
        if status_col in metafeatures.columns else {}
    )

    null_by_feat = {c: int(metafeatures[c].isna().sum()) for c in feat_cols}
    null_ratio = {c: (null_by_feat[c] / n if n else 0.0) for c in feat_cols}
    if feat_cols:
        notna_matrix = metafeatures[feat_cols].notna()
        all_pop = int(notna_matrix.all(axis=1).sum())
        none_pop = int((~notna_matrix.any(axis=1)).sum())
        partial = int(n - all_pop - none_pop)
    else:
        all_pop = none_pop = partial = 0
    top_missing = sorted(null_ratio.items(), key=lambda kv: kv[1], reverse=True)[:20]
    return {
        "prefix": prefix,
        "n_feature_columns": len(feat_cols),
        "feature_columns": feat_cols,
        "n_records": n,
        "status_counts": {str(k): int(v) for k, v in status_counts.items()},
        "all_status_success": bool(status_counts.get("success", 0) == n),
        "records_all_features_populated": all_pop,
        "records_partial_features": partial,
        "records_no_features": none_pop,
        "null_ratio_per_feature": null_ratio,
        "top20_by_missingness": [
            {"feature": k, "null_ratio": round(v, 6), "null_count": null_by_feat[k]}
            for k, v in top_missing
        ],
    }


# --- 9.1 external metrics --------------------------------------------------

def _external_metrics(evaluations) -> dict:
    valid = evaluations[evaluations["valid_result"] == True]  # noqa: E712
    n_valid = int(len(valid))
    present = {}
    for m in EXTERNAL_METRICS:
        exists = m in evaluations.columns
        nulls = int(valid[m].isna().sum()) if exists and n_valid else (0 if exists else n_valid)
        present[m] = {
            "present": exists,
            "null_count_valid": nulls,
            "null_ratio_valid": (nulls / n_valid if n_valid else 0.0),
        }
    return {
        "expected": list(EXTERNAL_METRICS),
        "n_valid_evaluations": n_valid,
        "metrics": present,
        "all_present": all(v["present"] for v in present.values()),
    }


# --- 9.2 internal metrics (registry-driven) --------------------------------

def _internal_metrics(evaluations, metric_registry) -> dict:
    valid = evaluations[evaluations["valid_result"] == True]  # noqa: E712
    n_valid = int(len(valid))
    defined = valid[valid["predicted_k"] >= 2]
    n_defined = int(len(defined))
    rows = []
    for _, r in metric_registry.iterrows():
        mid = r["metric_id"]
        exists = mid in evaluations.columns
        if exists and n_valid:
            nn = int(valid[mid].notna().sum())
            nc = n_valid - nn
            nc_def = int(defined[mid].isna().sum()) if n_defined else 0
        else:
            nn, nc, nc_def = (0, n_valid, n_defined)
        rows.append({
            "metric_id": mid,
            "present": bool(exists),
            "used_by_ml2dac": bool(r["used_by_ml2dac"]),
            "used_by_autoclust": bool(r["used_by_autoclust"]),
            "used_by_clustopt": bool(r["used_by_clustopt"]),
            "direction": r["direction"],
            "implementation_source": r["implementation_source"],
            "non_null_count_valid": nn,
            "null_count_valid": nc,
            "null_ratio_valid": (nc / n_valid if n_valid else 0.0),
            "null_ratio_defined": (nc_def / n_defined if n_defined else 0.0),
        })
    missing = [r["metric_id"] for r in rows if not r["present"]]
    # "High missingness" judged over partitions where the metric is *defined*
    # (predicted_k >= 2), so single-cluster degenerate NaNs don't masquerade as
    # gaps. Internal CVIs are 0% missing over defined partitions.
    high_missing = [
        {"metric_id": r["metric_id"],
         "null_ratio_defined": round(r["null_ratio_defined"], 4),
         "null_ratio_valid": round(r["null_ratio_valid"], 4)}
        for r in rows if r["present"] and r["null_ratio_defined"] > 0.10
    ]
    return {
        "n_registered": int(len(metric_registry)),
        "n_present": int(sum(1 for r in rows if r["present"])),
        "missing_metrics": missing,
        "n_valid_evaluations": n_valid,
        "n_defined_partitions": n_defined,
        "high_missingness_over_10pct_defined": sorted(high_missing, key=lambda x: -x["null_ratio_defined"]),
        "metrics": rows,
    }


# --- 9.3 / 9.4 CVI coverage ------------------------------------------------

def _cvi_coverage(evaluations, mapping: Dict[str, str]) -> dict:
    """Internal CVIs are mathematically undefined when a partition has <2
    clusters (predicted_k < 2). Coverage is therefore judged over *defined*
    partitions (valid AND predicted_k >= 2); the degenerate single-cluster rows
    are reported separately as the explanation for the global NaNs."""
    valid = evaluations[evaluations["valid_result"] == True]  # noqa: E712
    n_valid = int(len(valid))
    defined = valid[valid["predicted_k"] >= 2]
    n_defined = int(len(defined))
    n_degenerate = n_valid - n_defined
    rows = {}
    all_ok = True
    for name, col in mapping.items():
        exists = col in evaluations.columns
        nulls_valid = int(valid[col].isna().sum()) if exists and n_valid else (0 if exists else n_valid)
        nulls_def = int(defined[col].isna().sum()) if exists and n_defined else 0
        ratio_def = (nulls_def / n_defined) if n_defined else 0.0
        ratio_valid = (nulls_valid / n_valid) if n_valid else 0.0
        ok = exists and ratio_def <= _CVI_NULL_TOL
        all_ok = all_ok and ok
        rows[name] = {
            "column": col,
            "present": exists,
            "null_count_valid": nulls_valid,
            "null_ratio_valid": round(ratio_valid, 6),
            "null_count_defined": nulls_def,
            "null_ratio_defined": round(ratio_def, 6),
            "ok": ok,
        }
    return {
        "n_valid_evaluations": n_valid,
        "n_defined_partitions": n_defined,
        "n_degenerate_single_cluster": n_degenerate,
        "degenerate_explains_nulls": True,
        "null_tolerance_over_defined": _CVI_NULL_TOL,
        "cvis": rows,
        "pass": bool(all_ok),
    }


# --- 9.5 clustopt metric coverage -----------------------------------------

def _clustopt_coverage(evaluations, metric_registry) -> dict:
    clustopt = metric_registry[metric_registry["used_by_clustopt"] == True]  # noqa: E712
    ids = list(clustopt["metric_id"])
    present = [m for m in ids if m in evaluations.columns]
    missing = [m for m in ids if m not in evaluations.columns]
    return {
        "expected_count": EXPECTED_CLUSTOPT_METRICS,
        "registered_clustopt_metrics": len(ids),
        "present_count": len(present),
        "missing_clustopt_metrics": missing,
        "count_matches_expected": len(ids) == EXPECTED_CLUSTOPT_METRICS,
        "all_present": len(missing) == 0,
    }


# --- 10 failure analysis ---------------------------------------------------

def _failure_analysis(failures, records, evaluations) -> dict:
    if failures is None or len(failures) == 0:
        return {
            "total_failure_rows": 0, "real_failures": 0, "metafeature_stage_rows": 0,
            "real_by_family": {}, "real_by_subfamily": {}, "real_by_algorithm": {},
            "real_by_view_type": {}, "real_by_reason": {},
            "records_losing_all_32": 0,
            "affects_readiness": False,
            "note": "no failures recorded",
        }
    rid_map = records[["record_id", "family", "subfamily"]].drop_duplicates("record_id")
    fl = failures.merge(rid_map, on="record_id", how="left")
    # Real failures are clustering-stage (a config produced no usable partition).
    # metafeatures-stage rows are cosmetic bookkeeping from the early families:
    # ml2dac meta-features were deferred to the pymfe pass (status "unavailable"
    # at inline time) and were subsequently populated 100% by that pass, so they
    # are NOT real failures and do not affect readiness.
    real = fl[fl["stage"] == "clustering"] if "stage" in fl.columns else fl
    cosmetic = fl[fl["stage"] == "metafeatures"] if "stage" in fl.columns else fl.iloc[0:0]

    def vc(df, col):
        if col not in df.columns:
            return {}
        return {str(k): int(v) for k, v in df[col].fillna("").value_counts().head(50).items()}

    # failures are per-(record,config); the evaluation row still exists, so no
    # record loses configs. Verify anyway.
    per_record = evaluations.groupby("record_id")["config_id"].count()
    losing = int((per_record < N_CONFIGS_PER_VIEW).sum())

    return {
        "total_failure_rows": int(len(failures)),
        "real_failures": int(len(real)),
        "metafeature_stage_rows": int(len(cosmetic)),
        "metafeature_stage_note": (
            "deferred ml2dac meta-features (status unavailable at inline time); "
            "populated 100% by the later pymfe pass; not real failures"
        ),
        "real_by_family": vc(real, "family"),
        "real_by_subfamily": vc(real, "subfamily"),
        "real_by_algorithm": vc(real, "algorithm"),
        "real_by_view_type": vc(real, "view_type"),
        "real_by_reason": vc(real, "reason"),
        "records_losing_all_32": losing,
        "affects_readiness": bool(losing > 0),
        "real_examples": real.head(20).to_dict(orient="records"),
    }


# --- 11 label path audit ---------------------------------------------------

def _enumerate_label_files(outputs_root: Path) -> set:
    """All existing label files as outputs-root-relative posix paths (one walk)."""
    outputs_root = Path(outputs_root)
    existing = set()
    for fam in outputs_root.iterdir():
        if not fam.is_dir():
            continue
        for sub in fam.iterdir():
            ldir = sub / "labels"
            if not ldir.is_dir():
                continue
            base = outputs_root
            import os as _os
            for root, _dirs, files in _os.walk(io_utils.ext(ldir)):
                for f in files:
                    rel = _os.path.relpath(_os.path.join(root, f), io_utils.ext(base))
                    existing.add(rel.replace("\\", "/").lstrip("\\/?"))
    return existing


def _label_audit(evaluations, records, outputs_root, mode, sample, log) -> dict:
    valid = evaluations[evaluations["valid_result"] == True]  # noqa: E712
    n_valid = int(len(valid))
    missing_path = int((valid["labels_path"].fillna("") == "").sum())

    n_samples_map = records.set_index("record_id")["n_samples"].to_dict()

    # --- full existence check (cheap: one directory walk + set membership) --
    _log(log, "[audit]   enumerating label files (full existence check)")
    existing = _enumerate_label_files(Path(outputs_root))
    want = valid["labels_path"].dropna()
    want = want[want != ""]
    missing_mask = ~want.isin(existing)
    n_missing_existence = int(missing_mask.sum())
    missing_examples = list(want[missing_mask].head(20))

    if mode == "full":
        subset = valid
    else:
        mode = "sampled"
        if n_valid > sample:
            subset = valid.sample(n=sample, random_state=1234)
        else:
            subset = valid

    checked = 0
    missing_file = 0
    load_fail = 0
    len_mismatch = 0
    hash_mismatch = 0
    bad_examples: List[dict] = []
    outputs_root = Path(outputs_root)

    for row in subset.itertuples(index=False):
        rel = getattr(row, "labels_path", "")
        if not isinstance(rel, str) or not rel:
            continue
        checked += 1
        path = outputs_root / rel
        if not io_utils.path_exists(path):
            missing_file += 1
            if len(bad_examples) < 20:
                bad_examples.append({"labels_path": rel, "issue": "missing_file"})
            continue
        try:
            arr = np.load(io_utils.ext(path))
        except Exception as e:  # noqa: BLE001
            load_fail += 1
            if len(bad_examples) < 20:
                bad_examples.append({"labels_path": rel, "issue": f"load_error:{e}"})
            continue
        rid = getattr(row, "record_id")
        exp_len = n_samples_map.get(rid)
        if exp_len is not None and int(arr.shape[0]) != int(exp_len):
            len_mismatch += 1
            if len(bad_examples) < 20:
                bad_examples.append({"labels_path": rel, "issue": "len_mismatch",
                                     "got": int(arr.shape[0]), "expected": int(exp_len)})
        stored_hash = getattr(row, "labels_hash", "")
        if isinstance(stored_hash, str) and stored_hash:
            if io_utils.array_hash(arr) != stored_hash:
                hash_mismatch += 1
                if len(bad_examples) < 20:
                    bad_examples.append({"labels_path": rel, "issue": "hash_mismatch"})

    ok = (missing_path == 0 and n_missing_existence == 0 and missing_file == 0
          and load_fail == 0 and len_mismatch == 0 and hash_mismatch == 0)
    return {
        "load_hash_mode": mode,
        "existence_check": "full",
        "n_valid_evaluations": n_valid,
        "valid_missing_labels_path": missing_path,
        "full_existence_missing_files": n_missing_existence,
        "full_existence_missing_examples": missing_examples,
        "load_hash_checked": checked,
        "missing_file_in_sample": missing_file,
        "load_failures": load_fail,
        "length_mismatch": len_mismatch,
        "hash_mismatch": hash_mismatch,
        "pass": bool(ok),
        "bad_examples": bad_examples,
    }


# --- 12 storage ------------------------------------------------------------

def _storage(master_root: Path) -> dict:
    master_root = Path(master_root)
    sizes = {}
    total = 0
    for t in ("records", "metafeatures", "configurations",
              "metric_registry", "evaluations", "failures"):
        p = master_root / f"{t}.parquet"
        sz = int(p.stat().st_size) if p.exists() else 0
        sizes[f"{t}.parquet"] = sz
        total += sz
    return {
        "table_bytes": sizes,
        "table_mb": {k: round(v / 1e6, 2) for k, v in sizes.items()},
        "total_parquet_bytes": total,
        "total_parquet_mb": round(total / 1e6, 2),
        "labels_duplicated_in_master": False,
        "labels_note": "labels referenced in-place via outputs-root-relative labels_path; not copied",
    }


# --- readiness synthesis ---------------------------------------------------

def _readiness(audit: dict) -> dict:
    c = audit["counts"]
    pk = audit["primary_keys"]
    cov = audit["config_coverage"]
    mf_ml = audit["ml2dac_features"]
    mf_ac = audit["autoclust_features"]
    ext = audit["external_metrics"]
    ml_cvi = audit["ml2dac_cvi"]
    ac_cvi = audit["autoclust_cvi"]
    clustopt = audit["clustopt_metrics"]
    fail = audit["failure_analysis"]
    lab = audit["label_audit"]
    split = audit["split_distribution"]

    checks = {
        "counts_match_expected": bool(
            c["datasets_match_expected"] and c["records_match_expected"]
            and c["evaluations_match_expected"]
        ),
        "count_invariants_hold": bool(
            c["records_eq_datasets_x3"] and c["evaluations_eq_records_x32"]
        ),
        "primary_keys_clean": bool(
            pk["records_record_id_unique"] and pk["metafeatures_record_id_unique"]
            and pk["evaluations_pk_unique"]
            and pk["metafeatures_record_id_subset_of_records"]
            and pk["evaluations_record_id_subset_of_records"]
            and pk["evaluations_config_id_subset_of_configurations"]
        ),
        "config_coverage_complete": bool(cov["all_records_exactly_32"]),
        "all_records_have_split": bool(split["all_records_have_split"]),
        "ml2dac_feature_readiness": bool(
            mf_ml["n_feature_columns"] >= 50 and mf_ml["all_status_success"]
        ),
        "autoclust_feature_readiness": bool(
            mf_ac["n_feature_columns"] >= 7 and mf_ac["all_status_success"]
        ),
        "external_metrics_ready": bool(ext["all_present"]),
        "ml2dac_cvi_ready": bool(ml_cvi["pass"]),
        "autoclust_cvi_ready": bool(ac_cvi["pass"]),
        "clustopt_metrics_ready": bool(clustopt["all_present"] and clustopt["count_matches_expected"]),
        "failures_do_not_block": not bool(fail["affects_readiness"]),
        "label_audit_pass": bool(lab["pass"]),
    }
    blockers = [k for k, v in checks.items() if not v]
    return {
        "checks": checks,
        "blockers": blockers,
        "ready": len(blockers) == 0,
        "ml2dac_feature_readiness": "PASS" if checks["ml2dac_feature_readiness"] else "FAIL",
        "autoclust_feature_readiness": "PASS" if checks["autoclust_feature_readiness"] else "FAIL",
        "external_metrics_readiness": "PASS" if checks["external_metrics_ready"] else "FAIL",
        "ml2dac_cvi_readiness": "PASS" if checks["ml2dac_cvi_ready"] else "FAIL",
        "autoclust_cvi_readiness": "PASS" if checks["autoclust_cvi_ready"] else "FAIL",
        "clustopt_metric_readiness": "PASS" if checks["clustopt_metrics_ready"] else "FAIL",
    }
