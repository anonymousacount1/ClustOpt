import numpy as np
from sklearn.neighbors import NearestNeighbors
from models.ClustOpt.cluster_validity_indices.metrics.metrics_base import BaseMetric, MetricContext

class NeighborhoodPurityMetric(BaseMetric):
    """
    Local neighborhood purity:
    fraction of k-NN that share the same label as each point.
    Output in [0,1], higher is better.
    """
    name = "neighborhood_purity"
    space = "decision"          # recommended for kNN stability (scale-sensitive)
    higher_is_better = True
    needs_original_labels = False

    def __init__(self, k: int = 10):
        self.k = int(k)

    def evaluate(self, X: np.ndarray, labels: np.ndarray) -> float:
        X = np.asarray(X)
        labels = np.asarray(labels)

        if X.shape[0] < 3:
            return float("-inf")

        uniq = np.unique(labels)
        if uniq.size < 2:
            return float("-inf")

        n = X.shape[0]
        k_eff = min(self.k, n - 1)
        if k_eff <= 0:
            return float("-inf")

        nbrs = NearestNeighbors(n_neighbors=k_eff + 1).fit(X)
        neigh = nbrs.kneighbors(X, return_distance=False)[:, 1:]  # drop self

        # Vectorized purity count
        same = (labels[neigh] == labels[:, None])
        purity = float(np.mean(same))  # already divides by (n*k_eff)

        return purity

    def normalize(self, raw: float, ctx: MetricContext) -> float:
        # already [0,1]
        return raw

class NeighborhoodPurityMultiKMetric(BaseMetric):
    """
    Neighborhood purity averaged over multiple k values.
    For each k: fraction of k-NN that share the same label as each point.
    Output in [0,1], higher is better.
    """
    name = "neighborhood_purity_multi_k"
    space = "decision"
    higher_is_better = True
    needs_original_labels = False

    def __init__(self, ks=(5, 10, 20), agg: str = "mean", weights=None):
        """
        ks: iterable of k values
        agg: "mean" | "median" | "weighted_mean"
        weights: optional list/tuple same length as ks (used only if agg="weighted_mean")
        """
        self.ks = tuple(int(k) for k in ks)
        self.agg = str(agg).lower()
        self.weights = None if weights is None else np.asarray(weights, dtype=float)

        if len(self.ks) == 0:
            raise ValueError("ks must contain at least one k value.")
        if self.agg not in {"mean", "median", "weighted_mean"}:
            raise ValueError("agg must be one of: mean, median, weighted_mean")
        if self.agg == "weighted_mean":
            if self.weights is None or len(self.weights) != len(self.ks):
                raise ValueError("weights must be provided with same length as ks for weighted_mean.")
            s = float(self.weights.sum())
            if s <= 0:
                raise ValueError("weights must sum to a positive value.")
            self.weights = self.weights / s  # normalize

    def evaluate(self, X: np.ndarray, labels: np.ndarray) -> float:
        X = np.asarray(X)
        labels = np.asarray(labels)

        n = X.shape[0]
        if n < 3:
            return float("-inf")

        if np.unique(labels).size < 2:
            return float("-inf")

        # Precompute neighbors once up to max_k
        max_k = min(max(self.ks), n - 1)
        if max_k <= 0:
            return float("-inf")

        nbrs = NearestNeighbors(n_neighbors=max_k + 1).fit(X)
        neigh_all = nbrs.kneighbors(X, return_distance=False)[:, 1:]  # shape (n, max_k)

        per_k_scores = []
        for k in self.ks:
            k_eff = min(k, n - 1)
            if k_eff <= 0:
                continue

            neigh = neigh_all[:, :k_eff]  # first k_eff neighbors
            same = (labels[neigh] == labels[:, None])
            per_k_scores.append(float(np.mean(same)))

        if not per_k_scores:
            return float("-inf")

        vals = np.asarray(per_k_scores, dtype=float)
        if self.agg == "median":
            return float(np.median(vals))
        if self.agg == "weighted_mean":
            # if some k were clipped/skipped, align weights safely
            # simplest: recompute weights for the kept ks in order
            kept = []
            for k in self.ks:
                if min(k, n - 1) > 0:
                    kept.append(k)
            w = np.asarray([self.weights[list(self.ks).index(k)] for k in kept], dtype=float)
            w = w / w.sum()
            return float(np.sum(w * vals))

        # default mean
        return float(np.mean(vals))

    def normalize(self, raw: float, ctx: MetricContext) -> float:
        # already in [0,1]
        return raw
