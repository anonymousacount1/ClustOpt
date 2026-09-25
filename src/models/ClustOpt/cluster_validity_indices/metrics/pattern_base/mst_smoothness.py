# metrics/pattern/mst_smoothness.py
from __future__ import annotations
import numpy as np

from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

from models.ClustOpt.cluster_validity_indices.metrics.metrics_base import BaseMetric, MetricContext


class MSTSmoothnessMetric(BaseMetric):
    """
    MST Smoothness score in [0,1], higher is smoother (less zigzag, less branching).

    Per cluster:
      1) Project to 2D (SVD/PCA)
      2) Build sparse kNN graph (symmetric)
      3) Compute MST
      4) Smoothness at degree-2 nodes: angle between the two incident edges should be close to pi
         local = 1 - |pi - angle|/pi  in [0,1]
      5) Branching penalty: nodes with degree>2 reduce smoothness (encourage chain-like MST)

    Final per-cluster score:
      score = w_angle * mean(local_smooth) + (1-w_angle) * branch_score
      where branch_score = exp(-mean(max(0,deg-2)) / branch_scale)

    Aggregate: size-weighted mean across clusters.

    Expected noise removed by CVI.
    """
    name = "mst_smoothness"
    space = "full"
    higher_is_better = True
    needs_original_labels = False

    def __init__(
        self,
        min_cluster_points: int = 20,
        knn_k: int = 10,
        w_angle: float = 0.75,
        branch_scale: float = 1.0,
        eps: float = 1e-12,
    ):
        self.min_cluster_points = int(min_cluster_points)
        self.knn_k = int(knn_k)
        self.w_angle = float(w_angle)
        self.branch_scale = float(branch_scale)
        self.eps = float(eps)

    # ---------- helpers ----------
    def _to_2d(self, X: np.ndarray) -> np.ndarray:
        if X.ndim != 2:
            raise ValueError("X must be 2D.")
        n, d = X.shape
        if d < 2:
            raise ValueError("Need at least 2 dims for mst_smoothness.")
        Xc = X - np.mean(X, axis=0, keepdims=True)
        if d == 2:
            return Xc.astype(float)
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        Z = U[:, :2] * S[:2]
        return Z.astype(float)

    def _build_knn_graph(self, Z: np.ndarray) -> csr_matrix:
        n = Z.shape[0]
        k = int(np.clip(self.knn_k, 2, max(2, n - 1)))

        nn = NearestNeighbors(n_neighbors=k, algorithm="auto")
        nn.fit(Z)
        dists, idx = nn.kneighbors(Z, return_distance=True)

        rows = np.repeat(np.arange(n), k)
        cols = idx.reshape(-1)
        data = dists.reshape(-1)

        G = csr_matrix((data, (rows, cols)), shape=(n, n))
        # symmetrize by taking min distance for undirected graph
        GT = G.T
        # where both exist, keep min; where only one exists, keep that
        Gsym = G.minimum(GT) + G.maximum(GT)  # but this doubles when equal
        # fix doubles: use minimum of (G + GT) and (G.maximum(GT))? easiest:
        # build undirected by taking elementwise min of (G + GT) and huge? not needed.
        # We'll re-create clean: keep the smaller nonzero of the two.
        Gsym = G.minimum(GT) + (G - G.minimum(GT)) + (GT - G.minimum(GT))
        # above effectively equals max(G,GT) but without doubling; safe enough.

        # Ensure diagonal zeros
        Gsym.setdiag(0.0)
        Gsym.eliminate_zeros()
        return Gsym

    def _mst_edges(self, G: csr_matrix) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns arrays (u,v,w) for MST edges.
        """
        T = minimum_spanning_tree(G)  # directed sparse; contains n-1 edges if connected
        T = T.tocoo()
        u = T.row.astype(int)
        v = T.col.astype(int)
        w = T.data.astype(float)
        return u, v, w

    def _cluster_score(self, Xi: np.ndarray) -> float:
        n = Xi.shape[0]
        if n < self.min_cluster_points:
            return float("-inf")

        Z = self._to_2d(Xi)

        G = self._build_knn_graph(Z)
        u, v, w = self._mst_edges(G)

        if u.size < max(1, n - 2):
            # graph likely disconnected / too sparse -> fail-safe
            return 0.0

        # undirected adjacency list from MST
        adj: list[list[int]] = [[] for _ in range(n)]
        for a, b in zip(u, v):
            adj[a].append(b)
            adj[b].append(a)

        deg = np.array([len(nei) for nei in adj], dtype=int)

        # --- angle smoothness on degree==2 nodes ---
        nodes2 = np.where(deg == 2)[0]
        if nodes2.size < 3:
            angle_score = 0.0
        else:
            locals_ = []
            for i in nodes2:
                j, k = adj[i][0], adj[i][1]
                a = Z[j] - Z[i]
                b = Z[k] - Z[i]
                na = float(np.linalg.norm(a))
                nb = float(np.linalg.norm(b))
                if na <= self.eps or nb <= self.eps:
                    continue
                cosang = float(np.dot(a, b) / (na * nb))
                cosang = float(np.clip(cosang, -1.0, 1.0))
                ang = float(np.arccos(cosang))  # [0,pi]
                local = 1.0 - abs(np.pi - ang) / np.pi  # 1 when ang~pi
                locals_.append(local)
            angle_score = float(np.mean(locals_)) if locals_ else 0.0
            angle_score = float(np.clip(angle_score, 0.0, 1.0))

        # --- branching penalty ---
        extra = np.maximum(0, deg - 2).astype(float)
        mean_extra = float(np.mean(extra)) if extra.size > 0 else 0.0
        branch_score = float(np.exp(-mean_extra / max(self.branch_scale, self.eps)))  # (0,1]
        branch_score = float(np.clip(branch_score, 0.0, 1.0))

        w = float(np.clip(self.w_angle, 0.0, 1.0))
        score = w * angle_score + (1.0 - w) * branch_score
        return float(np.clip(score, 0.0, 1.0))

    # ---------- contract ----------
    def evaluate(self, X: np.ndarray, labels: np.ndarray) -> float:
        X = np.asarray(X)
        labels = np.asarray(labels)

        if X.ndim != 2 or labels.ndim != 1 or X.shape[0] != labels.shape[0]:
            return float("-inf")
        if X.shape[1] < 2:
            return float("-inf")

        uniq = np.unique(labels)
        if uniq.size < 1:
            return float("-inf")

        total = 0.0
        total_w = 0.0
        for lab in uniq:
            Xi = X[labels == lab]
            s = self._cluster_score(Xi)
            if not np.isfinite(s) or s == float("-inf"):
                continue
            w = float(Xi.shape[0])
            total += w * s
            total_w += w

        if total_w <= 0.0:
            return float("-inf")
        return float(total / total_w)

    def normalize(self, raw: float, ctx: MetricContext) -> float:
        v = float(max(0.0, min(1.0, raw)))
        return float(v ** 0.7)
