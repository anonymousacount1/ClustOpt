"""KNN prediction kernel: weighting, exact-match handling, bounds."""
import numpy as np

from models.metric_utility_knn.knn_predictor import predict_from_neighbors


def test_uniform_is_mean_of_neighbours():
    train_u = np.array([[0.2, 0.8], [0.4, 0.6], [1.0, 0.0], [0.0, 1.0]])
    idx = np.array([[0, 1, 2]])
    dist = np.array([[1.0, 2.0, 3.0]])
    pred = predict_from_neighbors(idx, dist, train_u, 3, "uniform")
    assert np.allclose(pred, train_u[[0, 1, 2]].mean(axis=0))


def test_inverse_distance_weights_closer_more():
    train_u = np.array([[1.0, 0.0], [0.0, 1.0]])
    idx = np.array([[0, 1]])
    dist = np.array([[0.1, 10.0]])
    pred = predict_from_neighbors(idx, dist, train_u, 2, "inverse_distance")
    assert pred[0, 0] > pred[0, 1]           # closer neighbour (u=[1,0]) dominates


def test_exact_match_uses_only_exact_neighbours():
    train_u = np.array([[0.5, 0.5], [0.9, 0.1], [0.0, 1.0]])
    idx = np.array([[0, 1, 2]])
    dist = np.array([[0.0, 0.3, 0.7]])       # neighbour 0 is an exact match
    pred = predict_from_neighbors(idx, dist, train_u, 3, "inverse_distance")
    assert np.allclose(pred, train_u[0])


def test_predictions_bounded_0_1():
    rng = np.random.default_rng(3)
    train_u = rng.random(size=(100, 5))
    idx = rng.integers(0, 100, size=(30, 10))
    dist = rng.random(size=(30, 10))
    for w in ("uniform", "inverse_distance"):
        pred = predict_from_neighbors(idx, dist, train_u, 10, w)
        assert pred.shape == (30, 5)
        assert pred.min() >= 0.0 and pred.max() <= 1.0
