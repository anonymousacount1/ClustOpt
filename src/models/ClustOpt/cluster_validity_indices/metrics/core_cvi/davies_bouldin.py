# metrics/davies_bouldin.py
import numpy as np
from sklearn.metrics import davies_bouldin_score
from models.ClustOpt.cluster_validity_indices.metrics.metrics_base import MetricContext, BaseMetric

class DaviesBouldinMetric(BaseMetric):
    name = "davies_bouldin"
    space = "decision"
    higher_is_better = True  # AFTER normalization

    def evaluate(self, X: np.ndarray, labels: np.ndarray) -> float:
        return float(davies_bouldin_score(X, labels))

    def normalize(self, raw: float, ctx: MetricContext) -> float:
        # DB in [0, +inf), lower is better -> map to (0,1], higher is better
        return 1.0 / (1.0 + max(0.0, raw))
