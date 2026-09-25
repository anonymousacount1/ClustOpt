# AutoClustering Experiment Pipeline

Evaluates the trained Metric-Utility MLP inside ClustOpt / AutoClustering by
running dynamic metric-selection methods against fixed-CVI baselines across a
balanced subset of the clustering repository.

> **Role in the pipeline** (stage [4] of the
> [Clustering Repository Builder](../README_Clustering_Repository_Builder.md)).
> This is the **payoff experiment** — the end-to-end test of the project's
> central claim.

---

## Research hypothesis under test

```
H₁ :  using a per-dataset objective O_D(π) = Σ_{m∈top-k(û(D))} wₘ·sₘ(π),
      with û = f(φ(D)) predicted from label-free features,
      yields clusterings of higher external quality Q = ARI
      than any FIXED-CVI objective.

H₀ :  dynamic metric selection does not improve over the best fixed CVI.
```

The experiment instantiates this as a controlled comparison over a balanced,
leakage-free sample of the repository, measuring not only ARI but cluster-count
accuracy and runtime, against several baselines and an oracle ceiling.

### Experimental philosophy

* **Fairness.** Every method — dynamic and baseline — runs inside the *same*
  ClustOpt search space, search algorithm, budget, and runtime mode. Only the
  **objective** differs, so any difference in outcome is attributable to metric
  selection, not to a stronger optimiser or more compute.
* **Reproducibility.** Splits are deterministic and seeded; every run snapshots
  its full config; all writes are atomic. A result can be regenerated exactly.
* **No leakage.** The MLP is applied only to datasets whose `dataset_id` was held
  out of its training, and the three views of a dataset never straddle train/test.
* **Honest use of ground truth.** Labels are touched *only* after clustering, to
  score results — never inside the search objective.

---

## Three independent, separately-runnable components

```
experiments/
  data_splitting/        # 1. split raw datasets into 16 balanced splits (once)
  experiment_execution/  # 2. run ClustOpt experiments for a subfamily/split (resumable)
  result_aggregation/    # 3. aggregate raw results into summaries/plots/reports (rerunnable)
```

## 1. Data splitting

Splits every raw dataset (by `dataset_id`, so the three views stay together) into
16 stratified, balanced splits. Stratified by
`family | subfamily | difficulty | cluster_count` via deterministic round-robin
(seeded), so every split mirrors the repository distribution.

**Why balanced, stratified splits.** Clustering difficulty and the
structure→metric mapping vary enormously across families; an unstratified sample
could over-represent easy convex blobs (where everything works) or one exotic
family (where nothing does), biasing the comparison. Stratification makes each
split a faithful miniature of the repository, so per-method scores are comparable
and aggregate claims generalise across regimes. **Why split by `dataset_id`.** The
three views are highly correlated; letting them straddle splits would leak
information and inflate apparent performance — splitting at the dataset level
prevents this.

```powershell
python -m models.Clustering_Repository_Builder.experiments.data_splitting.run_create_splits `
  --repo-root "<REPO>" `
  --analyzed-data-root "<REPO>\results_analysis\clustering_repository\analyzed_data" `
  --n-splits 16 --seed 42
```

Outputs (under `analyzed_data/experiment_splits/`): `dataset_split_assignments.csv`
/`.json`, `split_summary.csv`/`.md`, and `split_distribution_by_*` /
`split_distribution_full_strata.csv`. Validated: every dataset appears once with
one `split_id` in `1..16`, balanced sizes, no view leakage.

## 2. Experiment execution

For one subfamily and one split id, runs **10 methods × 3 views** per selected
dataset:

* Regressor (dynamic, `cvi.type=regressor_dynamic`): `regressor_top{1,3,5,10}_softmax_t05`
* Baselines (`cvi.type=generic`): `silhouette_single`, `calinski_harabasz_single`,
  `davies_bouldin_single`, `dbcv_single`, `uniform_classic_cvi`, `all_metrics_uniform`

**Why these baselines.** They bracket the space of *fixed* objectives the dynamic
method must beat: four **single classic CVIs** (the conventional default, one per
notion of compactness/separation/density), a **uniform classic combination**
(`uniform_classic_cvi`), and a **uniform combination of all metrics**
(`all_metrics_uniform`). The last is the most important control: it isolates the
value of *selection itself* — if simply averaging every metric did as well, the
learned `f` would add nothing. Beating `all_metrics_uniform` shows that *choosing*
the right metrics, not merely *having* them, is what helps.

Configs come from `models/ClustOpt/configs/experiments/{regressor_metric_selection,baselines}/`.

```powershell
python -m models.Clustering_Repository_Builder.experiments.experiment_execution.run_subfamily_experiments `
  --repo-root "<REPO>" `
  --subfamily-dir "<REPO>\...\subfamilies\<subfamily>" `
  --split-assignments "<REPO>\...\experiment_splits\dataset_split_assignments.csv" `
  --split-id 1 --max-workers 8 --resume
```

Flags: `--resume` / `--overwrite` / `--dry-run` / `--limit-datasets N` /
`--methods ...` / `--views ...` / `--metric-mode fast|exact`.

### Critical: full-2D evaluation space

Partition space and evaluation space are kept separate (reusing the
utility-generation view builder):

| view   | partition (search) input | evaluation (metric) space |
| ------ | ------------------------ | ------------------------- |
| x_only | x column only            | full 2-D `[x, y]`         |
| y_only | y column only            | full 2-D `[x, y]`         |
| xy_2d  | full 2-D `[x, y]`        | full 2-D `[x, y]`         |

ClustOpt's `search(X_decision, X_full)` receives `X_decision` = partition space
and `X_full` = full 2-D, so geometry/image metrics always evaluate the true 2-D
structure even for 1-D partition views.

### Output layout (per dataset)

```
<dataset>/experiments/split_01/
  <method>/<view>/{clustopt_results.csv, best_result.json, external_metrics.json,
                   selected_metrics.json, timing_summary.json, run_log.json,
                   status.json, config_snapshot.json}
  dataset_experiment_summary.json
```

Subfamily/split summaries: `<subfamily>/experiment_execution_summaries/split_XX/`.

All writes are atomic (temp + `os.replace`) and **long-path-safe** (Windows `\\?\`
prefix — the deep method paths exceed `MAX_PATH` and this machine has
`LongPathsEnabled=0`).

### Resume / failure safety

* One worker = one raw dataset (`ProcessPoolExecutor`, no nested pools).
* Resume granularity is `dataset × method × view`: a run is skipped only if its
  `status.json` exists with `status == "success"` (and `--overwrite` is off).
* Any failure writes a `status.json` with the traceback and continues — a single
  failure never crashes the dataset, worker, or subfamily run.

### Fast mode

Runtime block injected into ClustOpt:
`{"metric_mode":"fast","fast_metric_sample_size":3000,"fast_metric_random_state":42}`.
Exact mode remains available via `--metric-mode exact`.

## 3. Result aggregation

Independent of execution; rerun cheaply.

```powershell
python -m models.Clustering_Repository_Builder.experiments.result_aggregation.run_aggregate_results `
  --analyzed-data-root "<REPO>\...\analyzed_data" `
  --split-assignments "<REPO>\...\experiment_splits\dataset_split_assignments.csv" `
  --split-ids 1 --output-dir "<REPO>\results_analysis\autoclustering_experiments"
```

Produces (under `<output-dir>/<timestamp>__<run_name>/`): `raw_results.csv`
(one row per dataset×view×method), `dataset_level_results.csv` (one row per
dataset×method, **oracle best-view** = highest-ARI view), the
`*_method_summary.csv` family, `winrate_summary.csv`, `k_accuracy_summary.csv`,
`k_confusion_matrices/`, `metric_selection_frequency.csv`, `plots/`, `reports/`,
and `aggregation_report.md`.

### Why "oracle best-view" selection is acceptable

`dataset_level_results.csv` collapses the three views per dataset by taking the
**highest-ARI view**. This is applied **identically to every method** (dynamic and
baseline alike), so it cannot favour the proposed method — it is a *common
post-hoc view selector*, not part of any method's objective. It answers a
well-defined question — "with an ideal view chooser, which *objective* wins?" —
while the per-view `raw_results.csv` retains the un-collapsed picture. Because the
oracle is shared, between-method comparisons remain fair.

### Why ARI alone is insufficient — the metrics suite

| Group | Metrics | What it adds beyond ARI |
|-------|---------|--------------------------|
| **External agreement** | ARI, NMI, AMI, V-measure, homogeneity, completeness, Fowlkes–Mallows, purity | multiple, complementary notions of label agreement; (+ predicted/true noise ratio & noise P/R/F1 when noise is present) |
| **K accuracy** | `selected_k` vs `true_k`: `k_correct`, `k_abs_error`, `k_signed_error` → K-accuracy, mean/median K-error, over-/under-/exact-cluster rates, per-method K confusion matrices | whether the method recovers the *right number of clusters* — a failure ARI can partially mask |
| **Win-rate** | paired per-dataset ARI: a method beats a baseline when `ARI(method) > ARI(baseline)+1e-6`; win/tie/loss rate + mean/median ARI gain | head-to-head, per-dataset evidence robust to averaging artefacts |
| **Runtime** | per-phase timing | the dynamic objective must not be prohibitively slower than a fixed CVI to be practical |
| **Selection frequency** | which metrics the regressor picks, per family | interpretability: does `f` select structure-appropriate metrics? |

* **K-accuracy matters** because a high mean ARI can coexist with systematic
  over-/under-segmentation; recovering true-k is a distinct, decision-relevant
  success criterion. `true_k` is read **only** post-clustering, never during search.
* **Win-rate matters** because mean ARI can be dominated by a few easy or hard
  datasets; paired per-dataset comparison shows whether improvements are *broad*.
* **Runtime matters** because dynamic selection adds a prediction step; the
  comparison reports its cost so practicality is part of the verdict.

---

## Main contributions

* A **fair, leakage-controlled benchmark** isolating the effect of *metric
  selection* by holding the optimiser, budget, and search space fixed across all
  methods.
* A **multi-axis evaluation** (external agreement + K-accuracy + paired win-rate +
  runtime + selection interpretability) rather than a single headline number.
* An **`all_metrics_uniform` control** that distinguishes the value of *selecting*
  metrics from merely *having* a large metric library, plus an **oracle ceiling**
  via `oracle_dynamic` for context.

## Novelty positioning

Unlike fixed-CVI clustering evaluations, this benchmark treats the *validity
objective itself* as the experimental variable and measures, per dataset, whether
learning that objective from structure beats every fixed alternative under
identical optimisation.

---

## Limitations

* **Synthetic-only.** All datasets come from `HYBRID_SCM`; real-data evaluation is
  out of scope here.
* **Budget sensitivity.** Outcomes depend on the shared search budget; a different
  budget could change absolute (though not necessarily relative) results.
* **Oracle-view collapse** answers a best-view question; deployment also needs a
  view chooser, which this experiment does not learn.

## Recommended first flow

1. Create splits (once).
2. Dry-run one subfamily/split (`--dry-run`).
3. Small debug (`--limit-datasets 2 --max-workers 2 --resume`).
4. Full subfamily run (`--resume`).
5. Aggregate once results exist.
