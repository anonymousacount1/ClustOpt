# ClustOpt — Automated Clustering Optimization Engine

ClustOpt is the **clustering engine** of the project: a configurable pipeline
that, given a dataset, *searches* over clustering algorithms and their
hyper-parameters and returns the partition that best maximises an internal
**cluster-validity objective**. It is driven entirely by JSON configs, and its
objective is **pluggable** — which is exactly where this project's research idea
lives.

```
search space  +  search algorithm  +  validity objective  ──►  best (algorithm, params, labels)
```

---

## Research framing: optimization as objective engineering

Most AutoClustering work treats the problem as *optimiser engineering* — better
search over algorithms and hyper-parameters under a fixed validity index. ClustOpt
reframes it as **objective engineering**: the optimiser is held deliberately
simple and interchangeable, and the research effort moves into *the objective the
optimiser maximises*.

```
Given dataset D, search space S, metric library M:
  classical AutoClustering:   π̂ = argmax_{π∈S}  CVI_fixed(π)
  this project (ClustOpt):    π̂ = argmax_{π∈S}  O_D(π),   O_D = Σ wₘ(D)·sₘ(π)

where the weights wₘ(D) are
  • fixed by hand            → cvi.type = "generic"           (baselines)
  • predicted from φ(D)      → cvi.type = "regressor_dynamic" (proposed)
  • read from true u(D)      → cvi.type = "oracle_dynamic"    (ceiling)
```

ClustOpt is used in **two phases**: *offline* it generates the search traces from
which metric utility is measured (see
[utility_generation](../Clustering_Repository_Builder/utility_generation/README_utility_generation.md));
*online* it exploits the learned objective to cluster new, unlabeled data. The same
engine, two CVI modes — only the objective changes.

---

## The core abstraction: `search(X_decision, X_full)`

Every search exposes one signature (`search_algorithm/Base_Search_Algorithm.py`):

```python
result = search.search(X_decision, X_full)
```

| Argument | Meaning | Why separate |
|----------|---------|--------------|
| `X_decision` | the **partition space** — what the clustering algorithm fits on (e.g. just x in `x_only`) | lets us study 1-D projections as the *decision* variable |
| `X_full`     | the **evaluation space** — the full 2-D data used by geometry/image metrics | so shape/image metrics always judge the *true* 2-D structure, even when the partition is 1-D |

**Why this abstraction matters.** Collapsing partition and evaluation into one
space — as every conventional clustering pipeline does — would make 1-D views
meaningless and would prevent the study of *projection effects* on metric choice.
Separating them lets ClustOpt cluster on one axis yet score the result with metrics
that need real 2-D geometry, and it is the mechanism that produces three distinct
`(φ, u)` regimes per dataset throughout the project.

### Why metric caching

A `GlobalMetricContext` is built once per search (pairwise distances, bounds, PCA
projections) and a `PartitionMetricContext` once per candidate's labels (noise
mask, per-cluster indices, cached rasters/skeletons/hulls). Metrics opt into these
caches, so a 50–100-trial search reuses expensive computations instead of
recomputing them per candidate. This is what makes a *combined, multi-metric*
objective — and large-scale utility generation over ~17k datasets — computationally
feasible.

---

## Per-candidate evaluation (three stages)

```
sample (algorithm, params)
   ↓
fit_predict                → candidate labels π
   ↓
evaluate()                 → raw per-metric values {sₘ(π) raw}
   ↓
normalize_scores()         → each metric mapped deterministically to [0,1]
   ↓
aggregate_score()          → weighted mean Σ wₘ·sₘ(π) / Σ wₘ
   ↓
objective value            → returned to the search algorithm
```

**Why normalization is required.** The metrics in `M` are heterogeneous —
unbounded ratios, signed coefficients, "lower-is-better" indices. To combine them
into one objective they must be mapped to a common, comparable `[0,1]` scale with a
consistent "higher = better" direction; otherwise a single large-magnitude metric
would dominate the weighted sum regardless of its actual agreement with structure.
Normalization is deterministic per metric, so the same partition always scores the
same.

---

## What's inside

| Folder / file | Role |
|---------------|------|
| `Clustering_Optimization_Builder.py` | parses a JSON config and wires the pipeline (search space + search algorithm + evaluator) |
| `search_space/`              | declares algorithm hyper-parameter ranges (`Clustering_Search_Space.py`) |
| `clustering_algorithms/`     | sklearn-backed algorithm wrappers — [README_clustering_algorithms.md](clustering_algorithms/README_clustering_algorithms.md) |
| `search_algorithm/`          | the search loop: `brute_force` (exhaustive) and `optuna` (TPE) — `search_registry.py` |
| `cluster_validity_indices/`  | the objective: ~60 metrics + generic/dynamic CVI evaluators — [README_cluster_validity_indices.md](cluster_validity_indices/README_cluster_validity_indices.md) |
| `contexts/`                  | lazy per-search / per-partition caches for metric computation |
| `external_evaluation/`       | ground-truth scoring (ARI, …) + `compute_metric_utility.py` (the utility formula) |
| `profiling/`                 | per-phase timing (instantiate / fit_predict / cvi_evaluate / normalise / aggregate) |
| `configs/`                   | ready-made experiment & utility-map configs |

### Search algorithms

| `search_algorithm.type` | Strategy | Use |
|-------------------------|----------|-----|
| `brute_force` | discretise reals to `max_samples_per_param` and enumerate the full grid | small spaces, exhaustive utility-map generation |
| `optuna`      | TPE sampler, optional `timeout` / `patience` early stop | larger spaces, experiment runs |

That two interchangeable optimisers plug into the same objective is the point: it
keeps the *optimiser* a controlled variable so that improvements are attributable
to the *objective*.

---

## Config shape

```json
{
  "search_space":   { "kmeans": { "n_clusters": {"type":"int","low":2,"high":10}, ... }, "hdbscan": {...} },
  "search_algorithm": { "type": "optuna", "params": { "n_trials": 50, "timeout": 150, "patience": 15 } },
  "cvi": { "type": "generic", "metrics": { "silhouette": 1.0 },
           "params": { "remove_noise": true, "noise_label": -1, "min_clusters": 2 } },
  "runtime": { "metric_mode": "exact" }
}
```

`cvi.type` selects the objective and is the key research switch:

| `cvi.type` | Objective | Used by |
|------------|-----------|---------|
| `generic`           | a fixed, hand-specified metric → weight map | baselines (single CVI, uniform combos) |
| `regressor_dynamic` | metrics chosen **per dataset** from MLP-predicted utilities (top-k + softmax) | the proposed method |
| `oracle_dynamic`    | metrics chosen from the *true* utility vector (upper bound) | ceiling analysis |

### Why fast-mode exists

The `runtime` block tunes metric cost. Image-based metrics rasterise and run
computer-vision operators per cluster, which is expensive on large datasets and
prohibitive across a 17k-dataset utility-generation sweep. `metric_mode: "fast"`
subsamples to `fast_metric_sample_size` (default 3000, fixed seed) so metric
*rankings* are preserved at a fraction of the cost; `"exact"` uses all points when
fidelity matters. Fast-mode is the practical enabler of repository-scale
supervision; the exact/fast trade-off is itself benchmarked under
`configs/utilities_maps/fast_mode_benchmark_results/`.

### `compute_metric_utilities` — the offline supervision formula

`external_evaluation/compute_metric_utility.py` produces the 60 training targets.
Given a search-results table (one row per candidate, with ground-truth `ARI` and
each metric's normalised score), a metric's utility is a clipped, weighted blend of
*how well its ranking of candidates agrees with the ARI ranking* (Spearman,
Kendall, pairwise accuracy, NDCG, Top/Bottom-K). `max(0, ·)` on correlations means
an anti-correlated metric earns **zero**. Full rationale in
[utility_generation](../Clustering_Repository_Builder/utility_generation/README_utility_generation.md).

---

## Main contributions

* **A modular AutoClustering infrastructure** with a *pluggable objective*,
  cleanly separating optimiser, algorithm catalogue, and validity objective.
* **Partition/evaluation-space separation** as a first-class API
  (`search(X_decision, X_full)`), enabling projection-aware clustering and metric
  study.
* **Context-cached, normalisable multi-metric objectives** plus a **fast metric
  mode**, making large-scale, combined-metric search tractable.

## Novelty positioning

Unlike classical AutoClustering systems that optimise a *fixed* validity index,
ClustOpt makes the objective a configurable, per-dataset, learnable construct;
unlike direct algorithm-selection approaches, it intervenes on *what "good" means*,
letting any algorithm in the search space win when it is genuinely correct.

## Configs directory

```
configs/
  experiments/
    baselines/{silhouette_single, calinski_harabasz_single, davies_bouldin_single,
               dbcv_single, uniform_classic_cvi, all_metrics_uniform}/{x_only,y_only,xy_2d}.json
    regressor_metric_selection/regressor_top{1,3,5,10}_softmax_t05/{...}.json
  utilities_maps/   # brute-force "all metrics" runs used to build utility targets + benchmarks
```

## Limitations

* **2-D-oriented metrics.** Many objective metrics assume planar geometry/rendering.
* **Search-space dependence.** Results are bounded by what `S` can express; a
  partition outside the search space can never be found regardless of objective.
* **No uncertainty in the objective.** The aggregated score is a point estimate;
  ClustOpt does not propagate confidence from the (possibly uncertain) weights.

## Extending it

New algorithm → implement the `ClusteringAlgorithm` interface and register it in
the search space. New search strategy → implement `BaseSearchAlgorithm`, register
in `search_registry.py`. New objective → implement `BaseCVI`, register in
`cvi_registry.py`. The builder, logger, and context caches need no changes — a
direct benefit of the objective-engineering design.
