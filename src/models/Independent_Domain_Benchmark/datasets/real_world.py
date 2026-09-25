"""Group C: real labelled datasets, represented through a fixed label-free 2-D PCA.

Every candidate is retrieved by a **pinned OpenML data id** (or a pinned sklearn
loader), so retrieval is reproducible and versioned. Eligibility is decided by the
predeclared rule in ``ELIGIBILITY`` alone; no candidate is ever accepted or
rejected because of how any method performs on it.

The candidate list was assembled from well-known public tabular datasets spanning
several domains, chosen for *diversity and eligibility*, before any method ran.
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ELIGIBILITY = {
    "public": True,
    "reproducibly_obtainable": "pinned OpenML data_id or sklearn bundled loader",
    "numeric_or_convertible": True,
    "labelled": "labels used only for curation and offline scoring",
    "n_classes_min": 2, "n_classes_max": 5,
    "n_rows_min": 200, "n_rows_max": 5000,
    "min_class_size_min": 10,
    "max_missing_fraction": 0.20,
    "not_in_internal_mkr": True,
    "not_selected_on_method_performance": True,
    "forbidden_rejection_reasons": ["poor ClustOpt performance",
                                    "poor metric utility",
                                    "difficult clustering outcome"],
}

#: (key, retrieval spec, domain) -- pinned before any execution.
#: openml ids are stable identifiers; sklearn entries are bundled with the library.
CANDIDATES: Tuple[Tuple[str, Dict[str, Any], str], ...] = (
    ("iris",                 {"sklearn": "load_iris"},        "botany"),
    ("wine",                 {"sklearn": "load_wine"},        "chemistry"),
    ("breast_cancer_wdbc",   {"sklearn": "load_breast_cancer"}, "medical"),
    ("seeds",                {"openml": 1499},                "agriculture"),
    ("banknote",             {"openml": 1462},                "forensics"),
    ("blood_transfusion",    {"openml": 1464},                "medical"),
    ("haberman",             {"openml": 43},                  "medical"),
    ("diabetes_pima",        {"openml": 37},                  "medical"),
    ("ionosphere",           {"openml": 59},                  "physics"),
    ("sonar",                {"openml": 40},                  "signal"),
    ("vehicle",              {"openml": 54},                  "image_shape"),
    ("ecoli",                {"openml": 39},                  "biology"),
    ("glass",                {"openml": 41},                  "materials"),
    ("balance_scale",        {"openml": 11},                  "psychology"),
    ("vertebral_column_3c",  {"openml": 1523},                "medical"),
    ("climate_model_crashes", {"openml": 1467},               "climate"),
    ("qsar_biodeg",          {"openml": 1494},                "chemistry"),
    ("wilt",                 {"openml": 40983},               "remote_sensing"),
    ("phoneme",              {"openml": 1489},                "speech"),
    ("segment",              {"openml": 40984},               "image_shape"),
    ("wall_robot_nav",       {"openml": 1497},                "robotics"),
    ("steel_plates_faults",  {"openml": 1504},                "manufacturing"),
    ("credit_approval",      {"openml": 29},                  "finance"),
    ("german_credit",        {"openml": 31},                  "finance"),
    ("spambase",             {"openml": 44},                  "text_features"),
    ("mammography",          {"openml": 310},                 "medical"),
    ("cardiotocography_3c",  {"openml": 1560},                "medical"),
    ("thyroid_new",          {"openml": 40682},               "medical"),
    ("user_knowledge",       {"openml": 1508},                "education"),
    ("leaf",                 {"openml": 1482},                "botany"),
    ("planning_relax",       {"openml": 1490},                "medical"),
    ("fertility",            {"openml": 1473},                "medical"),
    ("parkinsons",           {"openml": 1488},                "medical"),
    ("musk_clean1",          {"openml": 1116},                "chemistry"),
    ("waveform_5000",        {"openml": 60},                  "signal"),
    # --- second candidate batch -------------------------------------------
    # Added to widen the POOL after the first pass produced fewer eligible real
    # datasets than the frozen target. The eligibility rule in ELIGIBILITY is
    # unchanged, and no method has been executed, so nothing here can be
    # outcome-driven; these are simply further pinned public candidates.
    ("cmc",                  {"openml": 23},                  "demography"),
    ("car_evaluation",       {"openml": 21},                  "consumer"),
    ("titanic",              {"openml": 40945},               "historical"),
    ("churn",                {"openml": 40701},               "telecom"),
    ("pollen",               {"openml": 871},                 "biology"),
    ("kc1_software",         {"openml": 1067},                "software"),
    ("kc2_software",         {"openml": 1063},                "software"),
    ("pc1_software",         {"openml": 1068},                "software"),
    ("pc3_software",         {"openml": 1050},                "software"),
    ("pc4_software",         {"openml": 1049},                "software"),
    ("ozone_level_8hr",      {"openml": 1487},                "environment"),
    ("madelon",              {"openml": 1485},                "synthetic_hard"),
    ("hill_valley",          {"openml": 1479},                "signal"),
    ("ilpd_liver",           {"openml": 1480},                "medical"),
    ("tokyo1",               {"openml": 40705},               "geography"),
    ("analcatdata_authorship", {"openml": 458},               "text_features"),
)


def _numeric(df) -> Tuple[np.ndarray, int, int]:
    """Numeric matrix from a dataframe; categoricals one-hot only if low-cardinality."""
    import pandas as pd
    num = df.select_dtypes(include=[np.number])
    n_dropped = df.shape[1] - num.shape[1]
    if num.shape[1] == 0:
        cats = [c for c in df.columns if df[c].nunique(dropna=True) <= 12]
        if cats:
            num = pd.get_dummies(df[cats], dummy_na=False).astype(float)
            n_dropped = df.shape[1] - len(cats)
    return num.to_numpy(dtype=float), int(num.shape[1]), int(n_dropped)


def fetch_candidate(key: str, spec: Dict[str, Any], timeout: int = 90
                    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Return (X_numeric, y, retrieval_meta). Labels are curation/eval only."""
    meta: Dict[str, Any] = {}
    if "sklearn" in spec:
        from sklearn import datasets as skd
        b = getattr(skd, spec["sklearn"])()
        X, y = np.asarray(b.data, dtype=float), np.asarray(b.target)
        meta.update({"retrieval": "sklearn.datasets.%s" % spec["sklearn"],
                     "version": "bundled with scikit-learn",
                     "licence": "BSD-3-Clause (scikit-learn bundled)",
                     "n_dropped_non_numeric": 0,
                     "original_dim": int(X.shape[1])})
        return X, y, meta
    from sklearn.datasets import fetch_openml
    d = fetch_openml(data_id=spec["openml"], as_frame=True, parser="auto")
    X_df, y_s = d.data, d.target
    X, kept, dropped = _numeric(X_df)
    _, y = np.unique(np.asarray(y_s).astype(str), return_inverse=True)
    meta.update({"retrieval": "sklearn.datasets.fetch_openml(data_id=%d)" % spec["openml"],
                 "openml_data_id": int(spec["openml"]),
                 "openml_name": getattr(d, "details", {}).get("name", key),
                 "openml_version": getattr(d, "details", {}).get("version"),
                 "licence": getattr(d, "details", {}).get("licence", "see OpenML"),
                 "original_dim": int(X_df.shape[1]),
                 "n_numeric_features_used": kept,
                 "n_dropped_non_numeric": dropped})
    return X, y.astype(int), meta


def audit_one(key: str, spec: Dict[str, Any], domain: str) -> Dict[str, Any]:
    rec: Dict[str, Any] = {"dataset_id": "real_%s" % key, "source": "real_pca",
                           "canonical_name": key, "domain": domain,
                           "structural_tags": ["real_pca"],
                           "tag_source": ("no objective geometry label; source tag "
                                          "only, per the no-invented-tags rule")}
    try:
        X, y, meta = fetch_candidate(key, spec)
    except Exception as e:
        rec.update({"accepted": False, "rejection_reason": "unavailable_or_retrieval_failed",
                    "error": "%s: %s" % (type(e).__name__, str(e)[:160])})
        return rec
    rec.update(meta)
    n, d = X.shape
    k = int(len(np.unique(y)))
    sizes = np.bincount(y).tolist()
    miss = float((~np.isfinite(X)).sum()) / max(n * max(d, 1), 1)
    rec.update({"n_rows": int(n), "n_numeric_dim": int(d), "n_classes": k,
                "class_sizes": sizes, "min_class_size": int(min(sizes)) if sizes else 0,
                "missing_fraction": round(miss, 6),
                "sha256_features": hashlib.sha256(
                    np.ascontiguousarray(X, dtype=np.float64).tobytes()).hexdigest()[:16]})
    E = ELIGIBILITY
    if not (E["n_classes_min"] <= k <= E["n_classes_max"]):
        r = "ground_truth_K_outside_2_5"
    elif not (E["n_rows_min"] <= n <= E["n_rows_max"]):
        r = "sample_size_outside_eligibility"
    elif d < 2:
        r = "unusable_feature_schema"
    elif sizes and min(sizes) < E["min_class_size_min"]:
        r = "extreme_tiny_class"
    elif miss > E["max_missing_fraction"]:
        r = "unusable_missingness"
    else:
        r = ""
    rec.update({"accepted": r == "", "rejection_reason": r})
    return rec


def audit_all() -> List[Dict[str, Any]]:
    return [audit_one(k, s, dom) for k, s, dom in CANDIDATES]
