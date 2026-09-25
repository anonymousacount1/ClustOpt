"""Durable Stage-4 corpus archive: acquire once, freeze, never re-derive.

The madelon incident is the reason this exists. Preparing that dataset runs a
PCA on a 2600x500 matrix, and the frozen manifest checksum was computed with
multi-threaded BLAS while Stage-4B-4 pins numeric threads to 1. Single-threaded
BLAS produces a different 2-D embedding, so the same code in the same repository
yields a different scientific input depending on an infrastructure setting. A
corpus that is only reconstructable "in principle" is therefore not reproducible
in practice.

The archive fixes that by storing the exact prepared representation Stage 4 used.
Scientific execution afterwards loads a checksum-verified array; it never calls
PCA, never downloads, and never depends on the numerical environment that
produced the embedding.

Three artifact layers are kept apart on purpose:

``source_raw``
    what was downloaded or generated, before any preparation.
``prepared_pre_pca``
    the post-column-selection numeric matrix, where a reduction happened.
``stage4_X_full_2d``
    the authoritative scientific input, whose hash must equal the frozen
    corpus manifest's ``representation_checksum``.

``ground_truth`` lives in a separate file and is never returned by the loader the
scientific phases use. Publishing labels for reproducibility does not make them
reachable from a ``DatasetView``.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_REPO = _ROOT.parents[1]

ARCHIVE_ROOT = (_REPO / "results_analysis" / "independent_domain_benchmark"
                / "publication" / "corpus")
PREPARATION_ENVIRONMENT = "unpinned_blas"
ARCHIVE_CONTRACT = "stage4_corpus_archive_v1"

#: view construction, recorded so a consumer never has to guess it
VIEW_DEFINITIONS = {
    "x_only": "X_full[:, [0]]",
    "y_only": "X_full[:, [1]]",
    "xy_2d": "X_full",
}

#: redistribution classification. Source licences are recorded from the frozen
#: manifest; where the manifest does not state one we do NOT invent a claim.
REDIST_ALLOWED = "REDISTRIBUTION_ALLOWED"
REDIST_RESTRICTED = "REDISTRIBUTION_RESTRICTED"
REDIST_UNCLEAR = "LICENSE_UNCLEAR"


class ArchiveError(RuntimeError):
    """A frozen corpus identity does not match. Never worked around."""


def _sha_array(a: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(a))
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode())
    h.update(b"|")
    h.update(str(arr.shape).encode())
    h.update(b"|")
    h.update(arr.tobytes(order="C"))
    return h.hexdigest()[:16]


def _sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


def dataset_dir(dataset_id: str) -> Path:
    return ARCHIVE_ROOT / dataset_id


def classify_redistribution(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Classify from recorded metadata only. No legal inference is invented."""
    src = entry["source"]
    lic = str(entry.get("licence") or "").strip()
    if src == "sklearn_shapes":
        return {"raw_redistribution": REDIST_ALLOWED,
                "basis": "generated deterministically from frozen parameters; "
                         "no third-party data is redistributed",
                "prepared_redistribution": REDIST_ALLOWED,
                "manual_review_required": False}
    if src == "fcps" and "MIT" in lic.upper():
        return {"raw_redistribution": REDIST_ALLOWED, "basis": lic,
                "prepared_redistribution": REDIST_ALLOWED,
                "manual_review_required": False}
    if not lic:
        return {"raw_redistribution": REDIST_UNCLEAR,
                "basis": "no licence recorded in the frozen manifest",
                "prepared_redistribution": REDIST_UNCLEAR,
                "manual_review_required": True}
    return {"raw_redistribution": REDIST_RESTRICTED,
            "basis": lic,
            "prepared_redistribution": REDIST_UNCLEAR,
            "manual_review_required": True,
            "note": ("redistribution of a derived 2-D PCA representation is a "
                     "separate question from redistribution of the source and is "
                     "flagged for manual review rather than asserted here")}


def acquire(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Fetch or generate, returning the raw matrix, labels and provenance."""
    src = entry["source"]
    prov: Dict[str, Any] = {"source_group": src,
                            "retrieved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                          time.gmtime())}
    if src == "fcps":
        from . import fcps
        raw, url = fcps.fetch(entry["canonical_name"])
        X, y, names = fcps.parse_arff(raw)
        prov.update({"retrieval": "http", "retrieval_url": url,
                     "raw_bytes_sha256_16": _sha_bytes(raw),
                     "original_filename": "%s.arff" % entry["canonical_name"],
                     "feature_names": list(names), "raw_blob": raw})
    elif src == "sklearn_shapes":
        from . import sklearn_shapes as SS
        X, y = SS.generate(entry["generator"], entry["difficulty"])
        prov.update({"retrieval": "generated",
                     "generator": entry["generator"],
                     "difficulty": entry["difficulty"],
                     "generator_params": entry.get("generator_params"),
                     "seed": entry.get("seed"), "raw_blob": None})
    elif src == "real_pca":
        from . import real_world as RW
        spec = next((sp for k, sp, _d in RW.CANDIDATES
                     if k == entry["canonical_name"]), None)
        if spec is None:
            raise ArchiveError("no frozen spec for %s" % entry["canonical_name"])
        X, y, meta = RW.fetch_candidate(entry["canonical_name"], spec)
        prov.update({"retrieval": meta.get("retrieval"),
                     "openml_data_id": meta.get("openml_data_id"),
                     "openml_name": meta.get("openml_name"),
                     "version": meta.get("version") or entry.get("version"),
                     "original_dim": meta.get("original_dim"),
                     "n_dropped_non_numeric": meta.get("n_dropped_non_numeric"),
                     "raw_blob": None})
    else:
        raise ArchiveError("unknown source %r" % src)
    return {"X_raw": np.asarray(X), "y": np.asarray(y), "provenance": prov}


def prepare_and_archive(entry: Dict[str, Any], *, overwrite: bool = False
                        ) -> Dict[str, Any]:
    """Acquire, prepare, verify against the frozen manifest, and persist.

    Preparation deliberately runs in the environment that reproduces the frozen
    manifest. The result is stored so no consumer ever repeats it.
    """
    from . import preprocessing as P
    d = dataset_dir(entry["dataset_id"])
    rec_path = d / "record.json"
    if rec_path.is_file() and not overwrite:
        return json.loads(rec_path.read_text(encoding="utf-8"))

    got = acquire(entry)
    X_raw, y = got["X_raw"], got["y"]
    prov = got["provenance"]
    raw_blob = prov.pop("raw_blob", None)

    prep = P.prepare(np.asarray(X_raw, dtype=float), entry["dataset_id"],
                     entry["source"])
    if prep.feature_checksum != entry["representation_checksum"]:
        raise ArchiveError(
            "%s prepared checksum %s != frozen %s (preparation environment "
            "differs from the one that froze the manifest)"
            % (entry["dataset_id"], prep.feature_checksum,
               entry["representation_checksum"]))

    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "stage4_X_full_2d.npy", np.ascontiguousarray(prep.X_2d))
    np.save(d / "ground_truth.npy", np.asarray(y))
    art: Dict[str, Any] = {
        "stage4_X_full_2d": {"path": "stage4_X_full_2d.npy",
                             "hash": _sha_array(prep.X_2d),
                             "shape": list(np.shape(prep.X_2d)),
                             "dtype": str(np.asarray(prep.X_2d).dtype)},
        "ground_truth": {"path": "ground_truth.npy", "hash": _sha_array(y),
                         "shape": list(np.shape(y)),
                         "dtype": str(np.asarray(y).dtype),
                         "separate_file": True,
                         "never_returned_by_the_scientific_loader": True},
    }
    if raw_blob is not None:
        (d / "source_raw.arff").write_bytes(raw_blob)
        art["source_raw"] = {"path": "source_raw.arff",
                             "hash": _sha_bytes(raw_blob),
                             "bytes": len(raw_blob)}
    if entry["source"] == "sklearn_shapes":
        np.save(d / "source_raw_X.npy", np.asarray(X_raw))
        art["source_raw"] = {"path": "source_raw_X.npy",
                             "hash": _sha_array(X_raw),
                             "shape": list(np.shape(X_raw)),
                             "note": "deterministically generated pre-preparation X"}
    pca_state = None
    if prep.pca_applied:
        pre_pca = _clean_like_prepare(X_raw)
        np.save(d / "prepared_pre_pca.npy", np.ascontiguousarray(pre_pca))
        art["prepared_pre_pca"] = {
            "path": "prepared_pre_pca.npy", "hash": _sha_array(pre_pca),
            "shape": list(np.shape(pre_pca)),
            "note": ("post-column-selection, post-imputation numeric matrix -- "
                     "exactly what prepare() feeds to StandardScaler and PCA")}
        pca_state = _refit_pca_state(np.asarray(X_raw, dtype=float))
        np.save(d / "pca_components.npy", pca_state.pop("components"))
        np.save(d / "pca_mean.npy", pca_state.pop("mean"))
        art["pca_components"] = {"path": "pca_components.npy"}
        art["pca_mean"] = {"path": "pca_mean.npy"}

    record = {
        "archive_contract": ARCHIVE_CONTRACT,
        "dataset_id": entry["dataset_id"], "source_group": entry["source"],
        "source_identity": {k: prov.get(k) for k in
                            ("retrieval", "retrieval_url", "openml_data_id",
                             "openml_name", "version", "generator", "difficulty",
                             "generator_params", "seed", "original_filename",
                             "raw_bytes_sha256_16")},
        "acquisition_provenance": {"retrieved_at": prov.get("retrieved_at")},
        "preparation": {"protocol_id": prep.report.get("protocol_id"),
                        "environment": PREPARATION_ENVIRONMENT,
                        "pca_applied": bool(prep.pca_applied),
                        "pca_seed": prep.report.get("pca_seed"),
                        "explained_variance_ratio": prep.explained_variance_ratio,
                        "explained_variance_total": prep.explained_variance_total,
                        "dropped_columns": prep.dropped_columns,
                        "n_imputed_cells": prep.n_imputed_cells,
                        "dim_after_cleaning": prep.report.get("dim_after_cleaning"),
                        "labels_used": False},
        "pca_state": pca_state,
        "artifacts": art,
        "n_samples": int(np.shape(prep.X_2d)[0]),
        "prepared_dimensionality": int(np.shape(prep.X_2d)[1]),
        "representation_checksum": prep.feature_checksum,
        "frozen_representation_checksum": entry["representation_checksum"],
        "checksum_matches_frozen_manifest": True,
        "view_definitions": dict(VIEW_DEFINITIONS),
        "redistribution": classify_redistribution(entry),
        "library_versions": _versions(),
    }
    rec_path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    return record


def _clean_like_prepare(X: np.ndarray) -> np.ndarray:
    """Reproduce ``preprocessing.prepare``'s cleaning: drop, then impute.

    The PCA provenance must be fitted on the SAME matrix ``prepare`` used. Fitting
    it on the raw matrix instead fails outright on any dataset with missing
    values, and would silently misreport the components even where it does not.
    """
    X = np.asarray(X, dtype=float)
    keep = np.isfinite(X).any(axis=0) & (np.nanstd(X, axis=0) > 0)
    X = X[:, keep]
    bad = ~np.isfinite(X)
    if bad.any():
        med = np.nanmedian(np.where(np.isfinite(X), X, np.nan), axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        X = X.copy()
        X[bad] = np.take(med, np.where(bad)[1])
    return X


def _refit_pca_state(X: np.ndarray) -> Dict[str, Any]:
    """Fitted PCA provenance, recorded so the reduction is auditable."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    Xs = StandardScaler().fit_transform(_clean_like_prepare(X))
    p = PCA(n_components=2, random_state=1234).fit(Xs)
    return {"components": np.ascontiguousarray(p.components_),
            "mean": np.ascontiguousarray(p.mean_),
            "explained_variance": [float(v) for v in p.explained_variance_],
            "explained_variance_ratio": [float(v) for v in
                                         p.explained_variance_ratio_],
            "singular_values": [float(v) for v in p.singular_values_],
            "n_components": 2, "random_state": 1234,
            "svd_solver": str(p.svd_solver)}


def _versions() -> Dict[str, str]:
    import numpy, scipy, sklearn
    return {"numpy": numpy.__version__, "scipy": scipy.__version__,
            "sklearn": sklearn.__version__}


# ---------------------------------------------------------------- loaders
def load_scientific_input(dataset_id: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """The ONLY loader the scientific phases use. Returns features, never labels."""
    d = dataset_dir(dataset_id)
    rec = json.loads((d / "record.json").read_text(encoding="utf-8"))
    X = np.load(d / rec["artifacts"]["stage4_X_full_2d"]["path"].replace(
        "stage4_X_full_2d.npy", "stage4_X_full_2d.npy"))
    got = _sha_array(X)
    if got != rec["artifacts"]["stage4_X_full_2d"]["hash"]:
        raise ArchiveError("%s X_full hash %s != archived %s"
                           % (dataset_id, got,
                              rec["artifacts"]["stage4_X_full_2d"]["hash"]))
    if got != rec["frozen_representation_checksum"] and \
            rec["representation_checksum"] != rec["frozen_representation_checksum"]:
        raise ArchiveError("%s archived representation is not the frozen one"
                           % dataset_id)
    return np.ascontiguousarray(X), rec


def load_ground_truth(dataset_id: str) -> np.ndarray:
    """Phase C only. Deliberately a separate function from the scientific loader."""
    d = dataset_dir(dataset_id)
    rec = json.loads((d / "record.json").read_text(encoding="utf-8"))
    y = np.load(d / "ground_truth.npy")
    if _sha_array(y) != rec["artifacts"]["ground_truth"]["hash"]:
        raise ArchiveError("%s ground-truth hash mismatch" % dataset_id)
    return y
