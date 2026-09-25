"""Neighbour search: single max-neighbour query reused for all n_neighbors."""
import numpy as np

from models.metric_utility_knn.neighbor_search import ViewNeighborIndex


def _fit_query(distance):
    rng = np.random.default_rng(1)
    train = rng.normal(size=(500, 8))
    query = rng.normal(size=(20, 8))
    idx = ViewNeighborIndex(distance).fit(train)
    return idx.query(query, max_neighbors=100)


def test_query_returns_sorted_distances():
    for dist in ("euclidean", "cosine"):
        nn = _fit_query(dist)
        assert nn.indices.shape == (20, 100)
        assert nn.distances.shape == (20, 100)
        assert np.all(np.diff(nn.distances, axis=1) >= -1e-9)  # ascending


def test_prefix_reuse_matches_smaller_query():
    """First n columns of the max-neighbour query == an n-neighbour query."""
    rng = np.random.default_rng(2)
    train = rng.normal(size=(300, 6))
    query = rng.normal(size=(10, 6))
    idx = ViewNeighborIndex("euclidean").fit(train)
    big = idx.query(query, max_neighbors=50)
    small = idx.query(query, max_neighbors=10)
    assert np.array_equal(big.indices[:, :10], small.indices)
    assert np.allclose(big.distances[:, :10], small.distances)
