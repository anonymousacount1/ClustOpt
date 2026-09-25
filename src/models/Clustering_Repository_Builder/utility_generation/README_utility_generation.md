# Utility Generation — the supervision bridge u(D)

This stage manufactures the **learning targets** of the whole project: for every
dataset it computes a 60-dimensional **metric-utility vector** `u(D)`, where each
coordinate says *how trustworthy a given validity metric is on that dataset*. It
is the bridge between **clustering evaluation** (how metrics behave on candidate
partitions) and **meta-learning** (predicting metric behaviour from appearance).

It reuses the existing ClustOpt search machinery and the existing
`compute_metric_utilities` formula — it does **not** re-implement clustering or
the utility formula.

> **Role in the pipeline** (stage [2] of the
> [Clustering Repository Builder](../README_Clustering_Repository_Builder.md)).
> Inputs are the label-free features
> [φ(D)](../features_extraction/README_features_extraction.md); the targets `u(D)`
> produced here are what
> [metric_utility_mlp](../../metric_utility_mlp/README_metric_utility_mlp.md)
> learns to predict.

---

## Research problem: what "utility" is and why it is needed

A clustering search needs a scalar objective, but ground truth is unavailable at
inference time. So we must learn *which surrogate objective to use*. To learn it,
we need supervision: a measurable quantity that captures **how well a metric
substitutes for ground truth on a given dataset**. That quantity is *utility*.

Formally, fix a dataset `D` with labels `y(D)` (offline only) and run a clustering
search that visits a set of candidate partitions `Π = trace(S, D)`. For each
partition `π ∈ Π` we have the normalised metric score `sₘ(π) ∈ [0,1]` and the
external quality `Q(π) = ARI(π, y(D))`. The utility of metric `m` is the degree to
which **`m`'s ordering of `Π` agrees with ARI's ordering of `Π`**:

```
U(D, m) = Agree( {sₘ(π)}_{π∈Π} , {Q(π)}_{π∈Π} ) ∈ [0,1]
u(D)    = ( U(D,m₁), …, U(D,m₆₀) )
```

A metric is "useful" on `D` precisely when *trusting it to rank candidate
partitions would have led the search toward the ground-truth-best partition.*

### Why raw metric scores are insufficient

One might try to use a metric's raw value `sₘ(π̂)` as supervision. This fails:

* Raw values are **not comparable across metrics or datasets** — different
  metrics live on different scales and respond differently to dataset size,
  density, and noise. Utility is *relative* (agreement with ARI), so it is
  comparable.
* A metric can produce **high absolute scores while ranking partitions wrongly**.
  What matters for steering a search is the *ordering* of candidates, not the
  magnitude of any single score.
* Utility is **search-relevant by construction**: it is measured over exactly the
  partitions the optimiser explores, so a high-utility metric is one that actually
  helps the optimiser, not one that merely looks good in isolation.

### Why external (ground-truth) agreement

ARI against the generator's labels is the only signal that defines "correct"
clustering objectively. Utility therefore anchors metric behaviour to *external*
quality, turning the open-ended question "is this a good metric?" into the
well-posed, supervised question "does this metric rank partitions the way ground
truth does, **on this kind of data**?"

### Why ranking-based, not regression-based, agreement

The search uses a metric only to **compare and rank** candidate partitions and
pick an argmax. So the property that matters is *rank fidelity*, not numerical
closeness to ARI. Utility is built from rank-agreement statistics rather than a
regression error between `sₘ` and `Q`.

### Why several utility signals are combined

No single rank statistic captures everything a search cares about, so utility is a
**bounded, weighted blend** of complementary agreement signals:

| Signal | What it captures | Why it is included |
|--------|------------------|--------------------|
| **Spearman / Kendall** `max(0, ρ)`, `max(0, τ)` | global monotone rank agreement | does the metric order *all* candidates like ARI? |
| **Pairwise accuracy** | local pair-ordering correctness | for two candidates, does the metric pick the better one? — the atomic decision a search makes |
| **Full-list NDCG** | rank quality with position discounting | rewards getting the *top* of the ranking right more than the tail |
| **Top-K overlap & Top-K NDCG** | agreement at the high-quality end | a search mostly exploits the *best* candidates — being right there matters most |
| **Bottom-K overlap & NDCG** | agreement at the low-quality end | a useful metric must also reliably *reject* bad partitions |

* **Top-K agreement matters** because optimisation is dominated by the leading
  candidates; a metric that nails the global trend but mis-orders the very best
  partitions is a poor search objective.
* **NDCG matters** because it is rank-position-aware: errors near the top are
  penalised more than errors deep in the list, mirroring how a search weights its
  decisions.
* **`max(0, ·)` on correlations** encodes a deliberate asymmetry: an
  *anti-correlated* metric earns **zero**, not a negative utility — it is treated
  as *useless*, not *invertible*, because the dynamic objective never inverts
  metrics.
* **Boundedness to `[0,1]`** makes utilities comparable across metrics and
  datasets and gives the MLP a clean, sigmoid-friendly regression/ranking target.

```
candidate partitions Π (from a ClustOpt search)
   ↓ metric m                         ↓ ground truth
metric scores {sₘ(π)}            ARI quality {Q(π)}
   └──────────────┬───────────────────┘
                  ↓  rank-agreement statistics
     Spearman · Kendall · pairwise · NDCG · Top/Bottom-K
                  ↓  bounded weighted blend
            U(D, m) ∈ [0,1]   (one coordinate of u(D))
```

`u(D)` is computed **once per (dataset, view)**: three vectors per dataset,
matching the three feature rows from extraction.

---

## Main contributions

* **Utility as a supervision signal** — a concrete, bounded, rank-based definition
  that converts ground-truth-scored clustering searches into a learnable target.
* **A reusable generator of `(φ, u)` examples at repository scale**, built on the
  *same* ClustOpt search and metric library used online, so the supervision matches
  the deployment objective.
* **Strict offline isolation of ground truth** — ARI is used only to *define*
  utility here and is never exposed to the online path.

## Novelty positioning

Unlike traditional clustering-validity studies that crown a single "best index"
per benchmark, utility generation produces a *graded, per-dataset, rank-based*
score for every metric, explicitly tied to how a search would use it — making
metric quality a *learnable function of dataset structure* rather than a fixed
verdict.

---

## Offline-only

This stage is purely **offline**. It consumes generator labels to compute ARI and
therefore utility; none of this is available, or needed, when ClustOpt later
clusters a new dataset online. The online path predicts `û(D)` from features alone
(see
[dynamic_metric_selection](../../ClustOpt/cluster_validity_indices/dynamic_metric_selection/README_dynamic_metric_selection.md)).

---

## Three views

Three views are generated per dataset, sequentially inside each worker:

| view_id  | Decision space     | Utility map JSON                                     |
|----------|--------------------|------------------------------------------------------|
| `x_only` | 1-D (x axis only)  | `models/ClustOpt/configs/utilities_maps/utility_map_x_only_32.json` |
| `y_only` | 1-D (y axis only)  | `models/ClustOpt/configs/utilities_maps/utility_map_y_only_32.json` |
| `xy_2d`  | 2-D (full xy plane)| `models/ClustOpt/configs/utilities_maps/utility_map_xy_2d_32.json`  |

`X_full` always contains both axes so that image-based / geometry metrics keep
working in 1-D views (partition space ≠ evaluation space).

---

## CLI

```powershell
python -m models.Clustering_Repository_Builder.utility_generation.run_subfamily_utility_generation `
  --repo-root  "." `
  --subfamily-dir "C:\...\analyzed_data\arcs_circles_rings\subfamilies\annuli_variable_thickness" `
  --metric-mode fast `
  --sample-size 3000 `
  --random-state 42 `
  --max-workers 8 `
  --resume
```

Supported flags:

| Flag | Meaning |
|------|---------|
| `--metric-mode {fast,exact}` | ClustOpt metric runtime mode. Default: `fast`. |
| `--sample-size N` | Fast-metric sample size (default 3000). |
| `--random-state N` | Fast-metric random state (default 42). |
| `--max-workers N` | Process-pool size. Default: `min(cpu_count - 1, 8)`. |
| `--resume / --no-resume` | Skip records that are already complete (default: resume on). |
| `--overwrite` | Force recompute existing outputs (implies `--no-resume`). |
| `--dry-run` | Discover datasets, validate maps, write a plan; do not run ClustOpt. |
| `--limit-datasets N` | Process at most N datasets. |
| `--dataset-id SUBSTR` | Process only datasets whose folder name contains `SUBSTR`. |
| `--views x_only y_only xy_2d` | Which views to process. |
| `--progress-every N` | Print verbose progress block every N datasets (default 5). |

## Outputs

Per dataset:

```
<dataset_dir>/utility/
  x_only/
    clustopt_results.csv         # search trace: one row per candidate partition
    metric_utilities.csv         # per-metric utility intermediate
    utility_vector.json          # the 60 utility__<metric> targets for this view
    utility_record_summary.json
    timing_summary.json
    run_log.json
  y_only/ ...
  xy_2d/ ...
  utility_dataset_summary.json
```

Per subfamily (written inside `--subfamily-dir`):

```
utility_generation_summary.csv
utility_generation_summary.json
utility_generation_report.md
utility_generation_failures.csv
utility_generation_checkpoint.json
```

All file writes go through a `*.tmp` → `os.replace(...)` atomic rename so a crash
mid-write never leaves a half-finished file that resume mode would treat as
complete.

## Module layout

```
models/Clustering_Repository_Builder/utility_generation/
  __init__.py
  config.py                          # UtilityGenerationConfig dataclass + map paths
  dataset_discovery.py               # find/validate dataset folders
  record_loader.py                   # load X + y_true + metadata from one folder
  view_builder.py                    # build (X_decision, X_full, y_true) per view
  clustopt_runner.py                 # build + run ClustOpt; collect results.csv
  utility_record_runner.py           # one (dataset, view) end-to-end unit
  utility_output_writer.py           # atomic JSON / CSV writers
  progress_tracker.py                # console progress + counters
  multiprocessing_runner.py          # ProcessPoolExecutor over dataset folders
  summary_report.py                  # subfamily-level CSV/JSON/MD outputs
  validation.py                      # pre-flight validation helpers
  run_subfamily_utility_generation.py  # CLI entry point
```

## Resume vs. overwrite

A view is considered complete when **all** of the following exist:

```
utility/<view_id>/metric_utilities.csv
utility/<view_id>/utility_vector.json
utility/<view_id>/utility_record_summary.json   (with "status": "success")
```

* `--resume` (default): complete views are skipped, anything else runs.
* `--overwrite`: every selected (dataset, view) is recomputed and re-written.

## Multiprocessing model

Dataset-level only, via `concurrent.futures.ProcessPoolExecutor`. Within each
worker the three views run sequentially. No nested multiprocessing, no
metric-level parallelism, no view-level threading.

---

## Limitations

* **Cost.** A full clustering search per (dataset, view) makes this the dominant
  offline expense; `--metric-mode fast` subsamples metric evaluation to contain it.
* **Search-trace dependence.** Utility is defined over the partitions the search
  visits, so it reflects the chosen search space and budget — a metric is credited
  only for structure the search actually explores.
* **ARI-anchored.** Utility inherits ARI's properties (e.g. its handling of noise
  and cluster-count); a different external measure would yield different utilities.

## Not in this stage

* No merge of feature records and utility vectors — that join is
  `training_dataset_builder`.
* No MLP / dataset-splitting / cross-subfamily orchestration.
* No modifications to ClustOpt, to the utility formula, or to feature extraction.
