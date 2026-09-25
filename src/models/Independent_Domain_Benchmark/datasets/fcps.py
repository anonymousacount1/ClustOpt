"""Group A: the Fundamental Clustering Problems Suite (FCPS).

Retrieved from the `deric/clustering-benchmark` mirror, whose ARFF headers
self-identify as FCPS. Every candidate is audited from the retrieved file itself
-- dimensionality, sample count and ground-truth cluster count are counted, never
recalled -- and eligibility is decided solely by the frozen K in 2..5 protocol.
"""
from __future__ import annotations

import hashlib
import io
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

BASE = ("https://raw.githubusercontent.com/deric/clustering-benchmark/"
        "master/src/main/resources/datasets/artificial/%s.arff")
LICENCE = ("no licence declared by the mirror deric/clustering-benchmark; FCPS "
           "suite by A. Ultsch, used under the citation condition in the ARFF header")
CITATION = ("Ultsch A. (2005), Clustering with SOM: U*C. FCPS suite. "
            "Mirror: github.com/deric/clustering-benchmark")

#: every FCPS member is audited; eligibility is decided from the data, not here
CANDIDATES: Tuple[str, ...] = (
    "atom", "chainlink", "engytime", "golfball", "hepta", "lsun",
    "target", "tetra", "twodiamonds", "wingnut",
)
#: structural tags from the suite's published design intent (pre-outcome)
TAGS: Dict[str, List[str]] = {
    "atom": ["topological", "density_variation", "non_convex", "nested_shell"],
    "chainlink": ["topological", "non_convex", "interlocking_rings"],
    "engytime": ["touching_overlap", "density_variation", "gaussian_mixture"],
    "golfball": ["no_cluster_structure"],
    "hepta": ["isotropic_compact", "well_separated"],
    "lsun": ["mixed", "anisotropic", "linear_band_like", "different_shapes"],
    "target": ["circular_ring", "non_convex", "outliers"],
    "tetra": ["touching_overlap", "isotropic_compact"],
    "twodiamonds": ["touching_overlap", "isotropic_compact"],
    "wingnut": ["density_variation", "linear_band_like"],
}


def fetch(name: str, timeout: int = 30) -> Tuple[bytes, str]:
    """Raw ARFF bytes plus sha256; retrieval only, no interpretation."""
    with urllib.request.urlopen(BASE % name, timeout=timeout) as r:
        raw = r.read()
    return raw, hashlib.sha256(raw).hexdigest()


def parse_arff(raw: bytes) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Minimal ARFF reader: returns (X, y, attribute_names).

    The final attribute is the class column by FCPS/ARFF convention; it is read
    for curation and offline evaluation only and never reaches a method.
    """
    text = raw.decode("utf-8", errors="replace")
    attrs: List[str] = []
    rows: List[List[str]] = []
    in_data = False
    for line in io.StringIO(text):
        s = line.strip()
        if not s or s.startswith("%"):
            continue
        low = s.lower()
        if low.startswith("@attribute"):
            attrs.append(s.split()[1].strip("'\""))
            continue
        if low.startswith("@data"):
            in_data = True
            continue
        if in_data:
            rows.append([t.strip() for t in s.split(",")])
    if not rows:
        raise ValueError("no @data rows")
    arr = np.array(rows, dtype=object)
    ycol = arr[:, -1]
    Xs = arr[:, :-1]
    X = np.array([[float(v) if v not in ("?", "") else np.nan for v in r]
                  for r in Xs], dtype=float)
    _, y = np.unique(ycol.astype(str), return_inverse=True)
    return X, y.astype(int), attrs


def audit_one(name: str) -> Dict[str, Any]:
    """Metadata-only audit of one FCPS candidate. No clustering, no model."""
    rec: Dict[str, Any] = {"dataset_id": "fcps_%s" % name, "source": "fcps",
                           "canonical_name": name,
                           "retrieval_url": BASE % name,
                           "licence": LICENCE, "citation": CITATION,
                           "structural_tags": TAGS.get(name, []),
                           "tag_source": "FCPS published design intent (pre-outcome)"}
    try:
        raw, sha = fetch(name)
        X, y, attrs = parse_arff(raw)
    except Exception as e:  # retrieval or parse problem is a hard audit fact
        rec.update({"accepted": False, "rejection_reason": "retrieval_or_parse_failed",
                    "error": "%s: %s" % (type(e).__name__, e)})
        return rec
    k = int(len(np.unique(y)))
    sizes = np.bincount(y).tolist()
    rec.update({
        "sha256": sha, "n_rows": int(X.shape[0]), "original_dim": int(X.shape[1]),
        "n_classes": k, "class_sizes": sizes,
        "min_class_size": int(min(sizes)), "attributes": attrs,
        "n_missing_cells": int((~np.isfinite(X)).sum()),
    })
    if not (2 <= k <= 5):
        rec.update({"accepted": False,
                    "rejection_reason": "ground_truth_K_outside_2_5"})
    elif X.shape[0] < 100:
        rec.update({"accepted": False, "rejection_reason": "sample_size_below_minimum"})
    else:
        rec.update({"accepted": True, "rejection_reason": ""})
    return rec


def audit_all() -> List[Dict[str, Any]]:
    return [audit_one(n) for n in CANDIDATES]
