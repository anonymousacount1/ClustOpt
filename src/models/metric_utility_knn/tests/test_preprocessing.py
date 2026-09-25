"""Preprocessing: train-only fit, imputation, edge cases, reload equality."""
import numpy as np

from models.metric_utility_knn.preprocessing import FeaturePreprocessor


def _data():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 10))
    x[:, 0] = np.nan            # all-NaN feature
    x[5, 1] = np.inf            # infinite value
    x[:, 2] = 3.0               # zero-variance feature
    x[10, 3] = np.nan
    return x


def test_all_nan_feature_imputed_to_zero():
    x = _data()
    pp = FeaturePreprocessor("standard").fit(x)
    assert pp.report_.n_all_nan_features == 1
    out = pp.transform(x)
    assert np.all(np.isfinite(out))
    assert out.shape == x.shape          # dimension preserved


def test_zero_variance_and_inf_handled():
    x = _data()
    pp = FeaturePreprocessor("robust").fit(x)
    assert 2 in pp.report_.zero_variance_feature_idx
    out = pp.transform(x)
    assert np.all(np.isfinite(out))


def test_transform_uses_train_statistics_only():
    x = _data()
    pp = FeaturePreprocessor("standard").fit(x[:150])
    # applying to unseen rows must not raise and stays finite
    out = pp.transform(x[150:])
    assert np.all(np.isfinite(out))


def test_save_load_roundtrip(tmp_path):
    x = _data()
    pp = FeaturePreprocessor("standard").fit(x)
    p = tmp_path / "pp.pkl"
    pp.save(p)
    pp2 = FeaturePreprocessor.load(p)
    assert np.allclose(pp.transform(x), pp2.transform(x), atol=0, rtol=0)
