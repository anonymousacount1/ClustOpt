# Clustering Algorithms

The library of clustering algorithms ClustOpt searches over. Each algorithm is a
thin, uniform **wrapper** around a scikit-learn (or HDBSCAN) estimator so the
search engine can treat every algorithm identically: instantiate it from a
parameter dict, fit-predict labels, and ask it for its hyper-parameter ranges.

Parent engine: [README_ClustOpt.md](../README_ClustOpt.md).

---

## Role in the framework

In the project's *objective-engineering* view, the algorithm catalogue is the
**search space `S`** — the set of partitions the optimiser can reach. It is held
fixed across every method (dynamic and baseline) so that observed differences are
attributable to the objective, not the optimiser. The catalogue must therefore be
*expressive enough that the ground-truth-correct partition is reachable for each
structural family* — otherwise no objective, however good, could recover it.

```
search space S  =  ⋃_algorithms { partitions reachable by (algorithm, hyper-params) }
```

---

## Why uniform wrappers

Different clustering algorithms have incompatible APIs and hyper-parameter
semantics (`n_clusters` vs `eps`/`min_samples` vs `min_cluster_size`). The search
engine must not care. Every algorithm subclasses a common `ClusteringAlgorithm`
interface and provides:

| Method | Contract |
|--------|----------|
| `instantiate(params) -> estimator` | build the underlying estimator from a plain dict of hyper-parameters |
| `get_search_space() -> dict`       | declare each hyper-parameter as an `Integer` / `Real` / `Categorical` range |

The search algorithm samples a point from the declared space, calls `instantiate`,
fits, predicts labels, and hands them to the CVI objective. Adding an algorithm is
a single wrapper — no change to the search loop, objective, or context caches.

---

## Available algorithms

Declared and registered in `../search_space/Clustering_Search_Space.py`:

| Config key | Algorithm | Key hyper-parameters |
|------------|-----------|----------------------|
| `kmeans`               | KMeans                        | `n_clusters`, `init`, `max_iter`, `tol`, `algorithm` |
| `minibatch_kmeans`     | MiniBatchKMeans               | `n_clusters`, `init`, `batch_size`, `tol`, `reassignment_ratio` |
| `gmm`                  | Gaussian Mixture Model        | `n_components`, `covariance_type`, `reg_covar`, `init_params` |
| `dbscan`               | DBSCAN                        | `eps`, `min_samples`, `metric` |
| `hdbscan`              | HDBSCAN                       | `min_cluster_size`, `min_samples`, `cluster_selection_method`, `alpha` |
| `hdbscan_constrained-N`| HDBSCAN, collapsed to N bins  | `n_components`, `min_cluster_size`, `min_samples`, `persistence_thresh_i` |
| `spectral`             | SpectralClustering            | `n_clusters`, `affinity`, `gamma`, `n_neighbors`, `assign_labels` |
| `agglomerative`        | AgglomerativeClustering       | `n_clusters`, `linkage`, `metric` |
| `optics`               | OPTICS                        | `min_samples`, `xi`, `min_cluster_size`, `cluster_method` |
| `meanshift`            | MeanShift                     | `bandwidth`, `bin_seeding`, `cluster_all` |
| `birch`                | BIRCH                         | `n_clusters`, `threshold`, `branching_factor` |

### Why this mix — paradigm coverage

The set is deliberately diverse so the search can find the *right paradigm* for
each structural regime, not merely tune one:

| Paradigm | Algorithms | Strong on | Blind to |
|----------|-----------|-----------|----------|
| **Centroid / model-based** | `kmeans`, `minibatch_kmeans`, `gmm` | convex Gaussian blobs | curves, rings, varying density |
| **Density** | `dbscan`, `hdbscan`, `optics`, `meanshift` | arbitrary shapes, noise (arcs, rings, spirals) | well-separated equal-density blobs can be over-merged |
| **Connectivity / hierarchy** | `agglomerative`, `spectral`, `birch` | chains, manifolds, non-Euclidean affinity | cost / parameter sensitivity at scale |

This coverage is a precondition for the whole study: each metric family in the CVI
library has a partner clustering paradigm that *can* produce the structure it
rewards, so a well-chosen objective has a reachable partition to select.

Density and hierarchical methods can emit a **noise label** (`-1`); the CVI layer
filters or accounts for noise (`remove_noise`, `noise_label`) before scoring, and
the experiment k-metrics count only non-noise clusters as `selected_k`.

---

## Hyper-parameter declaration

Each parameter in a config maps to a search dimension:

```json
"kmeans": {
  "n_clusters": {"type": "int",         "low": 2, "high": 10},
  "init":       {"type": "categorical", "choices": ["k-means++", "random"]},
  "max_iter":   {"type": "int",         "low": 100, "high": 300}
}
```

`type` ∈ {`int`, `float`, `categorical`} → `Integer` / `Real` / `Categorical`.
`brute_force` discretises `float` ranges and enumerates the grid; `optuna` samples
them with TPE. The same declaration drives both strategies unchanged.

---

## Limitations

* **Catalogue-bounded.** Any structure no listed algorithm can produce is
  unreachable for every method — a ceiling no objective can exceed.
* **Hyper-parameter ranges matter.** Mis-specified ranges (e.g. an `eps` grid that
  excludes the right scale) silently cap achievable quality.
* **2-D assumptions** in some distance/affinity defaults align with the project's
  planar scope.
