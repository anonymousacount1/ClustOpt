"""Dynamic Top-K fidelity uses the exact ClustOpt heuristic and stays in [1,10]."""
import numpy as np

from models.metric_utility_knn.metrics import (dynamic_k_metrics,
                                               dynamic_k_confusion)
from models.ClustOpt.dynamic_topk_selection.heuristics.selected_dynamic_topk import (
    select_batch, select)


def test_matches_clustopt_heuristic():
    rng = np.random.default_rng(0)
    y = rng.random(size=(300, 60))
    k_ref, _, _ = select_batch(y)
    # dynamic_k_metrics selects the same K on the "true" side
    _, k_true, _ = dynamic_k_metrics(y, y)
    assert np.array_equal(k_true, k_ref.astype(int))


def test_perfect_prediction_gives_exact_accuracy_one():
    rng = np.random.default_rng(1)
    y = rng.random(size=(200, 60))
    m, kt, kp = dynamic_k_metrics(y, y)
    assert m["dynamic_k_exact_accuracy"] == 1.0
    assert m["dynamic_k_mae"] == 0.0
    assert np.array_equal(kt, kp)


def test_k_bounds_and_confusion_shape():
    rng = np.random.default_rng(2)
    yt = rng.random(size=(200, 60))
    yp = np.clip(yt + rng.normal(scale=0.2, size=yt.shape), 0, 1)
    m, kt, kp = dynamic_k_metrics(yt, yp)
    assert kt.min() >= 1 and kt.max() <= 10
    assert kp.min() >= 1 and kp.max() <= 10
    conf = dynamic_k_confusion(kt, kp)
    assert conf.shape == (10, 10)
    assert conf.to_numpy().sum() == len(kt)


def test_single_record_select_consistent_with_batch():
    y = np.array([[0.9, 0.1, 0.88, 0.2, 0.0]])
    sel = select([f"m{i}" for i in range(5)], y[0])
    k_batch, _, _ = select_batch(y)
    assert sel.selected_k == int(k_batch[0])
