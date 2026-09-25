"""Frozen model: save -> reload in a fresh object reproduces predictions exactly."""
import numpy as np
import pytest

from models.metric_utility_knn import config as C
from models.metric_utility_knn.knn_predictor import KnnUtilityPredictor


def _toy_predictor(scaler="standard", distance="euclidean",
                   weighting="uniform", n_neighbors=5):
    rng = np.random.default_rng(0)
    cfg = C.KnnConfig(n_neighbors, scaler, distance, weighting)
    feat = [f"f{i}" for i in range(12)]
    mets = [f"m{i}" for i in range(6)]
    model = KnnUtilityPredictor(cfg, feat, mets)
    for view in C.VIEWS:
        X = rng.normal(size=(400, 12))
        U = rng.random(size=(400, 6))
        rec = np.array([f"{view}_{i}" for i in range(400)])
        model.fit_view(view, X, U, rec)
    return model, cfg, feat, mets, rng


def test_reload_reproduces_predictions(tmp_path):
    model, cfg, feat, mets, rng = _toy_predictor()
    model.save(tmp_path)
    reloaded = KnnUtilityPredictor.load(tmp_path, feat, mets, cfg)
    for view in C.VIEWS:
        q = rng.normal(size=(20, 12))
        p1 = model.predict_utility(q, view)
        p2 = reloaded.predict_utility(q, view)
        assert np.allclose(p1, p2, atol=0, rtol=0)


def test_predict_shape_and_bounds():
    model, cfg, feat, mets, rng = _toy_predictor(weighting="inverse_distance")
    q = rng.normal(size=(15, 12))
    pred = model.predict_utility(q, "x_only")
    assert pred.shape == (15, 6)
    assert pred.min() >= 0.0 and pred.max() <= 1.0


def test_wrong_feature_dim_raises():
    model, cfg, feat, mets, rng = _toy_predictor()
    with pytest.raises(ValueError):
        model.predict_utility(rng.normal(size=(3, 5)), "x_only")
