# metrics/pattern/connectivity_components.py
from __future__ import annotations
import numpy as np
from sklearn.neighbors import NearestNeighbors
from models.ClustOpt.cluster_validity_indices.metrics.metrics_base import BaseMetric, MetricContext

class ConnectivityComponentsMetric(BaseMetric):
    name = "connectivity_components"
    space = "full"
    higher_is_better = True

    def __init__(
        self,
        min_cluster_points: int = 25,
        knn_k: int = 8,
        eps: float = 1e-12,
    ):
        self.min_cluster_points = int(min_cluster_points)
        self.knn_k = int(knn_k)
        self.eps = float(eps)

    def _to_2d(self, X: np.ndarray) -> np.ndarray:
        Xc = X - np.mean(X, axis=0, keepdims=True)
        if Xc.shape[1] == 2:
            return Xc.astype(float)
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        return (U[:, :2] * S[:2]).astype(float)

    def _count_components_knn(self, Z: np.ndarray, k: int) -> int:
        n = Z.shape[0]
        k = int(np.clip(k, 2, max(2, n - 1)))
        nn = NearestNeighbors(n_neighbors=k).fit(Z)
        idx = nn.kneighbors(Z, return_distance=False)

        # build undirected adjacency list
        adj = [set() for _ in range(n)]
        for i in range(n):
            for j in idx[i]:
                if i == j:
                    continue
                adj[i].add(int(j))
                adj[int(j)].add(int(i))

        # BFS components
        seen = np.zeros(n, dtype=bool)
        comps = 0
        for i in range(n):
            if seen[i]:
                continue
            comps += 1
            stack = [i]
            seen[i] = True
            while stack:
                u = stack.pop()
                for v in adj[u]:
                    if not seen[v]:
                        seen[v] = True
                        stack.append(v)
        return comps

    def _cluster_score(self, Xi: np.ndarray) -> float:
        if Xi.shape[0] < self.min_cluster_points or Xi.shape[1] < 2:
            return float("-inf")
        Z = self._to_2d(Xi)
        c = self._count_components_knn(Z, self.knn_k)
        # map components count -> connectivity in [0,1]
        # 1 component -> 1.0, 2 -> 0.5, 3 -> 0.33 ...
        return float(np.clip(1.0 / max(1, c), 0.0, 1.0))

    def evaluate(self, X: np.ndarray, labels: np.ndarray) -> float:
        X = np.asarray(X); labels = np.asarray(labels)
        if X.ndim != 2 or labels.ndim != 1 or X.shape[0] != labels.shape[0] or X.shape[1] < 2:
            return float("-inf")
        uniq = np.unique(labels)
        if uniq.size < 1:
            return float("-inf")

        total = 0.0; total_w = 0.0
        for lab in uniq:
            Xi = X[labels == lab]
            s = self._cluster_score(Xi)
            if not np.isfinite(s) or s == float("-inf"):
                continue
            w = float(Xi.shape[0])
            total += w * s; total_w += w
        return float("-inf") if total_w <= 0 else float(total / total_w)

    def normalize(self, raw: float, ctx: MetricContext) -> float:
        v = float(max(0.0, min(1.0, raw)))
        return float(v ** 0.6)
