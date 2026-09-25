# Utility-Search Maps (32 configs)

Fixed-size, deterministic ClustOpt search maps used to estimate the
**utility** of CVI / structural metrics — i.e. how well each metric ranks
candidate partitions against ground truth (ARI). These are **not** AutoML
search spaces; they are compact catalogues of 32 diverse partitions per view.

```
configs/utilities_maps/
├── utility_map_x_only_32.json   (1D x-axis,  32 configs, seed 42)
├── utility_map_y_only_32.json   (1D y-axis,  32 configs, seed 73)
├── utility_map_xy_2d_32.json    (2D xy,      32 configs, seed 123)
├── validate_counts.py           (static expand-and-count check, size-aware)
├── benchmark_results/           (deep-instrumented benchmark outputs)
└── README.md
```

## 1. Purpose

Each map produces a fixed set of *candidate partitions* over the same input.
Downstream tooling (`compute_metric_utility.py`,
`select_metric_subset_simple.py`) then asks, per metric:

* Does the metric's ranking of these partitions agree with ARI?
* Does the metric concentrate the best partitions at the top of its order?
* Does it stay consistent across pairs?

Because the map is fixed, *the utility comparison is apples-to-apples across
metrics*. The map size (32) is small enough to run cheaply over the entire
~50 000-record repository, large enough to produce meaningful ranking
signal.

## 2. Why 32 (and not 42 / 64)

Earlier 42- and 64-config maps were prototyped and benchmarked, but proved
too slow for a full-repository sweep. Measured on real data:

| Earlier map     | AMP partition run | PRI-GT 1D run |
|-----------------|-------------------|---------------|
| `*_42.json`     | ~116 s/record     | ~645 s/record |
| `*_64.json`     | ~1.6×–2× slower   | ~1.6×–2× slower |

At 50 004 records, the 42-config map alone would take ~9 days for the
1D-PRI view; the 64 map several days more. The 32 maps cut that cost by
removing Spectral and OPTICS (the two heaviest contributors) and tightening
the per-algorithm grids. Spectral's RBF affinity is `O(n²)`; OPTICS does
several `O(n log n)` passes per config — both bring limited extra signal
relative to their cost.

The 42/64 maps were deleted to keep a single source of truth.

## 3. When to use the 32 maps

* **Default for any full-repository sweep** (cheapest CSV + still 60 metric
  columns per row).
* **Per-record smoke tests / CI**.
* **Comparable utility values across views** (`x_only`, `y_only`, `xy_2d`)
  — all three maps have the same total count (32) and the same algorithm
  budget; only DBSCAN/HDBSCAN ranges and the seed differ.

If you need extra candidate diversity for a single record (e.g. publication
plots), regenerate larger maps locally; the canonical config tree only
ships the 32 size.

## 4. Combinatorial breakdown

Each map sums to **exactly 32 configurations**. All parameter dimensions
are `categorical`, so cardinality is unambiguous and independent of
`max_samples_per_param`.

| Algorithm                | Dim cardinalities                                                                          | Subtotal |
|--------------------------|--------------------------------------------------------------------------------------------|----------|
| `kmeans`                 | n_clusters[4] × init[1] × seed[1]                                                          | **4**    |
| `minibatch_kmeans`       | n_clusters[4] × init[1] × batch_size[1] × seed[1]                                          | **4**    |
| `gmm`                    | n_components[6] × covariance_type[1] × seed[1]                                             | **6**    |
| `agglomerative`          | n_clusters[4] × linkage[1]                                                                 | **4**    |
| `birch`                  | n_clusters[4] × threshold[1] × branching_factor[1]                                         | **4**    |
| `dbscan`                 | eps[4] × min_samples[1]                                                                    | **4**    |
| `hdbscan`                | min_cluster_size[2] × min_samples[1]                                                       | **2**    |
| `hdbscan_constrained-2`  | every param fixed                                                                          | **1**    |
| `hdbscan_constrained-3`  | every param fixed                                                                          | **1**    |
| `hdbscan_constrained-4`  | every param fixed                                                                          | **1**    |
| `hdbscan_constrained-5`  | every param fixed                                                                          | **1**    |
| **TOTAL**                |                                                                                            | **32**   |

The constrained-HDBSCAN family contributes 4 deterministic configs (one per
k = 2,3,4,5), giving the catalogue density-based candidates at every target
k without ever consulting the true cluster count.

## 5. K coverage

Every centroid/partition-style algorithm enumerates `k ∈ {2, 3, 4, 5}`
(GMM goes up to 7 to fill its budget of 6). The constrained-HDBSCAN
quartet adds k = 2..5 by persistence-bin construction. Density-based
methods (`dbscan`, `hdbscan`) let `k` emerge from the data; **the true
cluster count is never used** — the maps remain valid at inference time
when ground truth is unknown.

## 6. Per-map differences

Only the seed and the DBSCAN/HDBSCAN scale ranges differ between views:

| Map                                  | Seed | DBSCAN `eps`              | HDBSCAN `min_cluster_size` | HDBSCAN-constrained `mcs` |
|--------------------------------------|------|---------------------------|----------------------------|----------------------------|
| `utility_map_x_only_32.json`         | 42   | [0.05, 0.10, 0.20, 0.30]  | [10, 50]                   | 25                         |
| `utility_map_y_only_32.json`         | 73   | [0.10, 0.20, 0.30, 0.50]  | [15, 60]                   | 30                         |
| `utility_map_xy_2d_32.json`          | 123  | [0.10, 0.30, 0.50, 0.80]  | [25, 100]                  | 30                         |

The 1D maps use smaller `eps` and `min_cluster_size` because 1D distances
are tighter; the 2D map widens both bands to suit 2-axis Euclidean
geometry.

## 7. Algorithms removed vs the earlier 42/64 maps

| Removed algorithm | Reason                                                                                            |
|-------------------|---------------------------------------------------------------------------------------------------|
| `spectral` (RBF)  | `O(n²)` affinity matrix dominates total cost. Drops repository-wide sweep into multi-week range.  |
| `optics`          | Heavy multi-pass density scan per config; ranking contribution overlaps with DBSCAN / HDBSCAN.    |

The 32 catalogue keeps `dbscan` + `hdbscan` + `hdbscan_constrained-{2,3,4,5}`
for the density-based ranking signal, which is the part actually used by
the metric utility module.

## 8. Reproducibility / seeds

Utility comparisons across metrics must be reproducible. Every stochastic
algorithm pins `random_state` to a single value:

| Map                                  | Seed |
|--------------------------------------|------|
| `utility_map_x_only_32.json`         | 42   |
| `utility_map_y_only_32.json`         | 73   |
| `utility_map_xy_2d_32.json`          | 123  |

The pinned seeds apply to `kmeans`, `minibatch_kmeans`, and `gmm`. The
deterministic-given-hyperparams algorithms (`agglomerative`, `birch`,
`dbscan`, `hdbscan`, `hdbscan_constrained-*`) have no `random_state`.

## 9. CVI section

Every map's CVI block lists **all 60 metrics** from `METRIC_REGISTRY` with
weight 1.0 and standard params (`remove_noise=True`, `noise_label=-1`,
`min_clusters=2`). Downstream utility computation grades one metric at a
time from the per-config CSV columns, so every metric must appear in the
log. The CVI block does not affect the 32-config expansion — only the
number of columns logged per row.

Metrics that cannot evaluate on a particular view (e.g. 2D image-based
metrics on a 1D map) return `-inf` raw → `0.0` normalised through
`BaseMetric.safe_evaluate` / `safe_normalize`; they never crash the
search.

## 10. Count verification

Two independent checks confirm the 32 figure:

1. **Static** (`python validate_counts.py`) — replays the same cardinality
   math that `BruteForceSearch._discretise` performs. The expected total
   per file is parsed from the filename suffix (`..._32.json` → expects
   32), so the same script keeps working for any future size class.
2. **Dynamic** — build each map through `ClusteringOptimizationBuilder`
   and run `search(...)` on tiny synthetic data; the search log will
   contain exactly 32 rows per map.

Latest run:

```
utility_map_x_only_32.json: OK (total=32, expected=32)
utility_map_y_only_32.json: OK (total=32, expected=32)
utility_map_xy_2d_32.json:  OK (total=32, expected=32)
```

## 11. Benchmark instrumentation

A deep-instrumented benchmark lives at the repo root as
`bench_utility_maps.py`. It monkey-patches
`BaseSearchAlgorithm.evaluate_config` and `GenericCVIEvaluator.evaluate`
**without modifying ClustOpt itself** to record:

* total search runtime per (map, scenario)
* per-config wall time, split into fit/eval/normalise/aggregate/log phases
* per-metric wall time per config (60 metrics × 32 configs × 2 scenarios)

Outputs land in `benchmark_results/`:

```
benchmark_results/
├── bench_32_summary.json
├── bench_32_summary.csv
├── bench_32_detailed_timings.csv
├── bench_32_per_metric_timings.csv
└── bench_32_readme_summary.md
```

The instrumentation is turn-key — running the script with no flags
benchmarks the three 32 maps and writes all five artefacts. Default
ClustOpt behaviour is untouched outside the benchmark process.

## 12. Integration

```python
from models.ClustOpt.Clustering_Optimization_Builder import ClusteringOptimizationBuilder

builder = ClusteringOptimizationBuilder(
    "models/ClustOpt/configs/utilities_maps/utility_map_xy_2d_32.json"
)
sa = builder.build_pipeline()
sa.set_ground_truth(y_true, ignore_noise=True)   # required for ARI / utility
sa.search(X_decision, X_full)
sa.export_log_to_csv("results.csv")
# Feed results.csv into compute_metric_utility.compute_metric_utilities(...)
```
