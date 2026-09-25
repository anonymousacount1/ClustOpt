"""Metric alignment: KNN reuses the MLP metric functions; suite is complete."""
import numpy as np

from models.metric_utility_knn.metrics import compute_full_suite
from models.metric_utility_mlp.metrics import compute_all
from models.metric_utility_knn import config as C


def _fake(n=200, m=60, seed=0):
    rng = np.random.default_rng(seed)
    yt = rng.random(size=(n, m))
    yp = np.clip(yt + rng.normal(scale=0.1, size=(n, m)), 0, 1)
    return yt, yp, [f"metric_{i}" for i in range(m)]


def test_core_metrics_match_mlp_definitions():
    yt, yp, names = _fake()
    suite = compute_full_suite(yt, yp, names)
    core, _ = compute_all(yt, yp, names, topk_list=C.TOPK_LIST,
                          ndcg_list=[3, 5, 10], ndcg_all=True)
    for k, v in core.items():
        assert k in suite
        assert np.isclose(suite[k], v), f"{k} diverged from MLP definition"


def test_best_val_loss_equals_mse():
    yt, yp, names = _fake()
    suite = compute_full_suite(yt, yp, names)
    assert np.isclose(suite["best_val_loss"], suite["mse"])


def test_required_extra_metrics_present():
    yt, yp, names = _fake()
    suite = compute_full_suite(yt, yp, names)
    for k in ["ndcg@1", "cosine_similarity_mean", "pearson_mean", "kendall_std",
              "median_absolute_error", "p95_absolute_error",
              "top5_jaccard", "top5_precision", "top5_recall",
              "dynamic_k_exact_accuracy", "dynamic_k_within_1_accuracy",
              "dynamic_k_mae", "dynamic_k_rmse"]:
        assert k in suite, f"missing metric {k}"
