import numpy as np
from collections import Counter
from models.ClustOpt.cluster_validity_indices.metrics.metrics_base import BaseMetric, MetricContext

class ClusterSizeBalanceEntropyMetric(BaseMetric):
    """
    Measures how balanced the cluster sizes are using normalized entropy.
    Output in [0,1], where 1 = perfectly balanced sizes, 0 = highly imbalanced.

    NOTE: This metric is label-only (X ignored). Keep it in decision space.
    """
    name = "cluster_size_balance_entropy"
    space = "decision"
    higher_is_better = True
    needs_original_labels = False

    def __init__(self, eps: float = 1e-12):
        self.eps = float(eps)

    def evaluate(self, X: np.ndarray, labels: np.ndarray) -> float:
        labels = np.asarray(labels)

        # If CVI removes noise, labels already exclude -1. If not, metric still works on all labels.
        # But we generally prefer remove_noise=True at CVI layer.

        if labels.size == 0:
            return float("-inf")

        counts = np.array(list(Counter(labels.tolist()).values()), dtype=float)
        if counts.size <= 1:
            # 0 or 1 cluster -> not a meaningful balance metric
            return float("-inf")

        probs = counts / (counts.sum() + self.eps)

        # entropy in nats
        ent = -float(np.sum(probs * np.log(probs + self.eps)))
        max_ent = float(np.log(len(probs)))

        if max_ent <= 0:
            return float("-inf")

        # Balanced score: H / Hmax  in [0,1]
        return ent / max_ent

    def normalize(self, raw: float, ctx: MetricContext) -> float:
        # raw already in [0,1]
        return raw

class ClusterSizeImbalanceEntropyMetric(BaseMetric):
    """
    1 = highly imbalanced cluster sizes, 0 = balanced
    (This is basically your original metric direction.)
    """
    name = "cluster_size_imbalance_entropy"
    space = "decision"
    higher_is_better = True
    needs_original_labels = False

    def __init__(self, eps: float = 1e-12):
        self.eps = float(eps)

    def evaluate(self, X: np.ndarray, labels: np.ndarray) -> float:
        labels = np.asarray(labels)
        if labels.size == 0:
            return float("-inf")

        counts = np.array(list(Counter(labels.tolist()).values()), dtype=float)
        if counts.size < 2:
            return float("-inf")

        p = counts / (counts.sum() + self.eps)
        H = -float(np.sum(p * np.log(p + self.eps)))
        Hmax = float(np.log(len(p)))
        if Hmax <= 0:
            return float("-inf")

        balance = H / Hmax
        return 1.0 - balance  # imbalance in [0,1]

    def normalize(self, raw: float, ctx: MetricContext) -> float:
        return raw
