"""Group B: 15 frozen sklearn shape datasets = 5 generators x 3 difficulties.

Every parameter here was fixed before any method ran. Difficulty is changed by a
single, documented mechanism per generator so that "harder" always means the same
kind of thing: less separation and/or more noise, never a different task.

The deliberately excluded case is a "no structure" uniform sample: it has no
ground truth, so it cannot enter a paired ARI benchmark.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

N_SAMPLES = 1500
BASE_SEED = 1234

#: generator -> difficulty -> (params, why this is harder)
SPECS: Dict[str, Dict[str, Tuple[Dict[str, Any], str]]] = {
    "circles": {
        "easy":   ({"noise": 0.03, "factor": 0.50},
                   "well-separated concentric rings, low jitter"),
        "medium": ({"noise": 0.07, "factor": 0.60},
                   "rings closer together (factor up) and noisier"),
        "hard":   ({"noise": 0.11, "factor": 0.72},
                   "rings nearly touching with jitter comparable to the gap"),
    },
    "moons": {
        "easy":   ({"noise": 0.04}, "clean interleaving crescents"),
        "medium": ({"noise": 0.09}, "crescent tails begin to overlap"),
        "hard":   ({"noise": 0.15}, "tails substantially mixed"),
    },
    "aniso_blobs": {
        "easy":   ({"cluster_std": 0.50, "shear": 0.30},
                   "mild anisotropy, compact blobs"),
        "medium": ({"cluster_std": 0.85, "shear": 0.60},
                   "stronger shear elongates clusters across each other"),
        "hard":   ({"cluster_std": 1.20, "shear": 0.85},
                   "heavy shear plus wide spread; centroid geometry misleads"),
    },
    "varied_blobs": {
        "easy":   ({"cluster_std": [1.0, 1.0, 1.0]},
                   "equal variances (control for the varied case)"),
        "medium": ({"cluster_std": [1.0, 2.0, 0.5]},
                   "4x variance ratio between the widest and tightest cluster"),
        "hard":   ({"cluster_std": [1.0, 3.0, 0.3]},
                   "10x variance ratio; the wide cluster envelops the tight one"),
    },
    "blobs": {
        "easy":   ({"cluster_std": 0.60, "center_box": (-10.0, 10.0)},
                   "textbook separated isotropic blobs"),
        "medium": ({"cluster_std": 1.40, "center_box": (-8.0, 8.0)},
                   "blobs broadened and centres drawn closer"),
        "hard":   ({"cluster_std": 2.20, "center_box": (-6.0, 6.0)},
                   "blobs overlap materially; boundaries are ambiguous"),
    },
}
DIFFICULTIES = ("easy", "medium", "hard")
#: ground-truth cluster counts, all inside the frozen K in 2..5 protocol
N_CLUSTERS = {"circles": 2, "moons": 2, "aniso_blobs": 3,
              "varied_blobs": 3, "blobs": 4}
#: frozen structural tags, taken from the generator definition (not from results)
TAGS = {"circles": ["circular_ring", "non_convex", "topological"],
        "moons": ["curved", "non_convex", "touching_overlap"],
        "aniso_blobs": ["anisotropic", "elongated"],
        "varied_blobs": ["density_variation"],
        "blobs": ["isotropic_compact"]}


def dataset_id(gen: str, diff: str) -> str:
    return "sk_%s_%s" % (gen, diff)


def seed_for(gen: str, diff: str) -> int:
    """Deterministic per-dataset generator seed; not derived from any outcome."""
    return BASE_SEED + 100 * sorted(SPECS).index(gen) + DIFFICULTIES.index(diff)


def generate(gen: str, diff: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return (X, y). Deterministic given (gen, diff)."""
    from sklearn.datasets import make_blobs, make_circles, make_moons
    p = dict(SPECS[gen][diff][0])
    s = seed_for(gen, diff)
    if gen == "circles":
        return make_circles(n_samples=N_SAMPLES, shuffle=True, random_state=s, **p)
    if gen == "moons":
        return make_moons(n_samples=N_SAMPLES, shuffle=True, random_state=s, **p)
    if gen == "aniso_blobs":
        shear = p.pop("shear")
        X, y = make_blobs(n_samples=N_SAMPLES, centers=N_CLUSTERS[gen],
                          n_features=2, random_state=s, **p)
        return X @ np.array([[1.0, shear], [0.4, 1.0]]), y
    if gen == "varied_blobs":
        return make_blobs(n_samples=N_SAMPLES, centers=N_CLUSTERS[gen],
                          n_features=2, random_state=s, **p)
    if gen == "blobs":
        return make_blobs(n_samples=N_SAMPLES, centers=N_CLUSTERS[gen],
                          n_features=2, random_state=s, **p)
    raise KeyError(gen)


def manifest_entries() -> List[Dict[str, Any]]:
    out = []
    for gen in sorted(SPECS):
        for diff in DIFFICULTIES:
            params, why = SPECS[gen][diff]
            out.append({
                "dataset_id": dataset_id(gen, diff), "source": "sklearn_shapes",
                "generator": gen, "difficulty": diff,
                "generator_params": params, "difficulty_rationale": why,
                "n_samples": N_SAMPLES, "n_clusters": N_CLUSTERS[gen],
                "seed": seed_for(gen, diff), "native_dim": 2,
                "structural_tags": TAGS[gen],
                "tag_source": "generator definition (pre-outcome)",
                "generator_group": gen,
            })
    return out
