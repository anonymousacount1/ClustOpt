"""HDBSCAN wrapper that collapses the clustering into at most ``n_components``
bins by persistence ranking.

The wrapper runs plain HDBSCAN, inspects ``cluster_persistence_`` and merges
the resulting clusters into ``n_components`` "persistence bins":

* **bin 0** — the points belonging to clusters whose persistence is above the
  highest threshold (the "densest" partition),
* **bin j** (1 ≤ j ≤ k-2) — clusters whose persistence falls between the
  (j-1)-th and j-th thresholds (sorted descending),
* **bin k-1** — every other point, including HDBSCAN noise (``-1``).

If HDBSCAN already returns at most ``n_components`` distinct labels (noise
included), the raw labels are returned untouched.

Family conventions
------------------
This wrapper is registered as a suffixed family — the algorithm key in a
config **must** include an explicit ``-N`` suffix:

* ``hdbscan_constrained-2``        — 2-bin variant
* ``hdbscan_constrained-3``        — 3-bin variant
* ``hdbscan_constrained-N``        — generic N-bin variant (any ``N ≥ 2``)

The bare name ``hdbscan_constrained`` is intentionally *not* accepted, so
that the number of persistence bins is unambiguous from the algorithm key
alone.

Config parameters consumed by the wrapper:

* ``n_components`` (optional, ``int``) — overrides ``N`` parsed from the
  name. Useful when callers want to tune ``N`` as a search-space variable.
* ``persistence_thresh_i`` for ``i = 1 .. N-1`` — one float per bin
  boundary; order in the config doesn't matter (the wrapper sorts them
  descending).
* ``persistence_thresh`` (legacy single float) — replicated across every
  boundary so existing 2-bin configs keep working without changes.

Any remaining params are forwarded verbatim to ``hdbscan.HDBSCAN``.
"""
from __future__ import annotations

from typing import Any, Dict, List

import hdbscan
import numpy as np

from models.ClustOpt.clustering_algorithms.abstract_clustering import ClusteringAlgorithm


class HDBSCANConstrainedClustering(ClusteringAlgorithm):
    def __init__(self, name: str, param_config: Dict[str, Any]):
        super().__init__(name, param_config)
        self._default_n_components: int = self._parse_n_from_name(name, fallback=2)
        self.n_components: int = self._default_n_components
        self.thresholds: List[float] = [0.0] * (self.n_components - 1)
        self.hdb_kwargs: Dict[str, Any] = {}

    @staticmethod
    def _parse_n_from_name(name: str, fallback: int) -> int:
        """Extract ``N`` from a ``hdbscan_constrained-N`` registry name."""
        if "-" not in name:
            return fallback
        try:
            value = int(name.rsplit("-", 1)[1])
            return value if value >= 2 else fallback
        except ValueError:
            return fallback

    # --------------------------------------------------------------- instantiate

    def instantiate(self, params: Dict[str, Any]):
        params = dict(params)

        n_components = int(params.pop("n_components", self._default_n_components))
        if n_components < 2:
            n_components = 2
        self.n_components = n_components

        self.thresholds = self._collect_thresholds(params, n_components)

        params.setdefault("min_cluster_size", 5)
        params.setdefault("min_samples", None)
        self.hdb_kwargs = params
        return self

    @staticmethod
    def _collect_thresholds(params: Dict[str, Any], n_components: int) -> List[float]:
        """Pull per-bin thresholds out of ``params`` and return them sorted descending.

        Two declaration styles are accepted:

        * ``persistence_thresh_i`` for ``i = 1 .. n_components-1`` — the
          recommended form for N ≥ 3.
        * ``persistence_thresh`` (a single float) — the legacy form; replicated
          across every bin boundary so the wrapper falls back to the original
          single-threshold semantics when ``n_components == 2``.
        """
        thresholds: List[float] = []
        for i in range(1, n_components):
            key = f"persistence_thresh_{i}"
            if key in params:
                thresholds.append(float(params.pop(key)))

        if not thresholds and "persistence_thresh" in params:
            legacy_value = float(params.pop("persistence_thresh"))
            thresholds = [legacy_value] * (n_components - 1)

        while len(thresholds) < n_components - 1:
            thresholds.append(0.0)

        return sorted(thresholds[: n_components - 1], reverse=True)

    # ------------------------------------------------------------------ fit_predict

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0]
        if n_samples == 0:
            return np.array([], dtype=int)

        hdb = hdbscan.HDBSCAN(**self.hdb_kwargs)
        hdb_labels = hdb.fit_predict(X)

        # If HDBSCAN already produced a small-enough partition, return as-is.
        if np.unique(hdb_labels).size <= self.n_components:
            return hdb_labels

        return self._collapse_by_persistence(hdb, hdb_labels, n_samples)

    def _collapse_by_persistence(
        self,
        hdb: hdbscan.HDBSCAN,
        hdb_labels: np.ndarray,
        n_samples: int,
    ) -> np.ndarray:
        """Reduce ``hdb_labels`` to at most ``n_components`` persistence bins."""
        cluster_persistence = getattr(hdb, "cluster_persistence_", None)
        last_bin = self.n_components - 1

        # Legacy guard: if the top threshold is non-positive the partition is
        # degenerate (nothing can be classified as "dense"), so we collapse the
        # whole space into the last bin — matches the original k=2 behaviour
        # when persistence_thresh was 0.
        if (
            cluster_persistence is None
            or not self.thresholds
            or self.thresholds[0] <= 0.0
        ):
            return np.full(n_samples, last_bin, dtype=int)

        pos_labels = np.unique(hdb_labels[hdb_labels != -1])
        if pos_labels.size == 0 or cluster_persistence.size == 0:
            return np.full(n_samples, last_bin, dtype=int)

        # Per-cluster persistence vector aligned with hdb_labels (defensive
        # truncation in case sklearn / hdbscan returns extra entries).
        cp = np.asarray(cluster_persistence[: len(pos_labels)], dtype=float)
        persistence = np.full(n_samples, -np.inf, dtype=float)
        for lab, p in zip(pos_labels, cp):
            persistence[hdb_labels == lab] = p

        # Walk thresholds from highest to lowest. Each iteration promotes any
        # still-unassigned point whose persistence clears the current
        # threshold into the corresponding bin.
        final = np.full(n_samples, last_bin, dtype=int)
        non_noise = hdb_labels != -1
        for j, t in enumerate(self.thresholds):
            unassigned = final == last_bin
            promote = unassigned & non_noise & (persistence >= t)
            if promote.any():
                final[promote] = j
        return final
