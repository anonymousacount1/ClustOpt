import numpy as np
from itertools import combinations
from models.ClustOpt.cluster_validity_indices.metrics.metrics_base import MetricContext, BaseMetric

class SDbwMetric(BaseMetric):
    name = "s_dbw"
    space = "decision"
    higher_is_better = True  # AFTER normalization

    def evaluate(self, X: np.ndarray, labels: np.ndarray) -> float:
        return float(self.s_dbw_index(X, labels))

    def normalize(self, raw: float, ctx: MetricContext) -> float:
        # lower is better -> map to (0,1]
        return 1.0 / (1.0 + max(0.0, raw))

    @staticmethod
    def s_dbw_index(X: np.ndarray, labels: np.ndarray, noise_label: int = -1, eps: float = 1e-12) -> float:
        """
        S_Dbw (lower is better).
        Deterministic, no run-history.
        NOTE: Ideally CVI removes noise; we keep this defensive.
        """
        # Defensive noise removal (OK even if CVI already did it)
        mask = labels != noise_label
        X = X[mask]
        labels = labels[mask]

        unique_labels = np.unique(labels)
        k = len(unique_labels)
        if k < 2:
            return float("-inf")

        clusters = [X[labels == l] for l in unique_labels]
        if any(len(c) < 2 for c in clusters):
            return float("-inf")

        centroids = [c.mean(axis=0) for c in clusters]

        # --- Scattering ---
        sigmas = []
        for c in clusters:
            std_vec = c.std(axis=0, ddof=1)
            sigmas.append(float(np.linalg.norm(std_vec)))

        sigma_clusters = float(np.mean(sigmas))

        std_all = X.std(axis=0, ddof=1)
        sigma_all = float(np.linalg.norm(std_all))

        scattering = sigma_clusters / (sigma_all + eps)

        # --- Density Between ---
        def density(points: np.ndarray, center: np.ndarray, r: float) -> int:
            d2 = np.sum((points - center) ** 2, axis=1)
            return int(np.sum(d2 <= (r * r)))

        dens_ratios = []
        for i, j in combinations(range(k), 2):
            ci, cj = clusters[i], clusters[j]
            mi, mj = centroids[i], centroids[j]

            u_ij = (mi + mj) / 2.0
            r_ij = max(sigmas[i], sigmas[j])

            union_ij = np.vstack([ci, cj])

            dens_ij = density(union_ij, u_ij, r_ij)
            dens_i = density(ci, mi, sigmas[i])
            dens_j = density(cj, mj, sigmas[j])

            denom = max(dens_i, dens_j)
            dens_ratios.append(dens_ij / denom if denom > 0 else 0.0)

        density_between = float(np.mean(dens_ratios)) if dens_ratios else 0.0
        return scattering + density_between
