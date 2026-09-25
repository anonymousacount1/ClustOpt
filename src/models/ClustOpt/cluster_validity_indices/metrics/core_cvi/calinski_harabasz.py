# metrics/calinski_harabasz.py
import numpy as np
import math
from sklearn.metrics import calinski_harabasz_score
from models.ClustOpt.cluster_validity_indices.metrics.metrics_base import MetricContext, BaseMetric

class CalinskiHarabaszMetric(BaseMetric):
    name = "calinski_harabasz"
    space = "decision"
    higher_is_better = True

    def evaluate(self, X: np.ndarray, labels: np.ndarray) -> float:
        return float(calinski_harabasz_score(X, labels))

    def normalize(self, raw: float, ctx: MetricContext) -> float:
        # Deterministic squashing to [0,1] without run-history
        s = math.log1p(max(0.0, raw))
        c = math.log1p(max(2, ctx.n_samples))
        return s / (s + c)
