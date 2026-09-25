# Dynamic Top-K Metric Selection — Pre-Analysis

Statistical **pre-analysis** of ClustOpt utility vectors, to inform the design
of a future *dynamic* Top-K metric-selection heuristic that picks how many
metrics `K` to aggregate from the **shape of each utility vector** rather than a
fixed Top-1 / Top-3 / Top-5 / Top-10.

> This phase does **not** modify ClustOpt optimization behaviour. It is read-only
> analysis + what-if heuristic simulation. It is **not** supervised learning —
> no model is trained to predict `K`.

## What it does

1. **Scan** every utility vector in the clustering repository
   (`results_analysis/clustering_repository/analyzed_data`). The authoritative
   index is `experiment_splits/dataset_split_assignments.csv`
   (17,068 datasets × 3 views = 51,204 records). Each vector lives at
   `<dataset_dir>/utility/<view>/utility_vector.json` (views: `x_only`,
   `y_only`, `xy_2d`), keyed `utility__<metric_name>`.
2. **Per-vector statistics** (`utility_vector_stats.py`): basic descriptives,
   entropy / normalized entropy / participation ratio / perplexity, top-K mass
   ratios, dominance ratios, sorted-curve gaps + elbow, relative & absolute
   threshold counts.
3. **Candidate heuristics** (`heuristic_candidates.py`): relative-threshold,
   cumulative-mass, effective-K, elbow (abs + normalized), hybrid
   conservative (`min`) and moderate (`mean`/`median`). Each records both the
   clipped K (`[1, 10]`) and the raw pre-clip K.
4. **Aggregate / report / plot** (`reporting.py`, `plotting.py`): global,
   family, subfamily, view and per-metric summaries; heuristic K-distribution
   diagnostics flagged against the desired bell-shape; validation reports;
   PNG figures; Markdown reports.

## Files

| file | role |
|---|---|
| `utility_vector_stats.py` | pure-numpy per-vector statistics |
| `heuristic_candidates.py` | candidate dynamic-K heuristics |
| `scan_utility_vectors.py` | parallel, resumable scan over all records |
| `reporting.py` | aggregation tables, validation, Markdown reports |
| `plotting.py` | all figures (Agg backend) |
| `run_pre_analysis.py` | orchestrator / CLI |

## Run

```bash
cd models/ClustOpt/dynamic_topk_selection/pre_analysis
python run_pre_analysis.py                 # full run, auto-timestamped output dir
python run_pre_analysis.py --workers 8
python run_pre_analysis.py --limit 600     # quick smoke test
python run_pre_analysis.py --out-dir <dir> # resume into an existing run
```

Outputs go to
`results_analysis/dynamic_topk_utility_pre_analysis_YYYYMMDD_HHMMSS/`
(record tables as Parquet, summaries as CSV, `reports/`, `plots/`,
`validation/`). The scan is **resumable**: each shard is checkpointed under
`_scan_shards/`, so a re-run with `--out-dir` skips completed shards.

## Notes

- Metric set is consistently **60 metrics** per vector across all families /
  views (the prompt's "~58" estimate; actual is 60). Inconsistencies are
  reported in `validation/scan_report.json`, not silently dropped.
- Utilities are read raw; for mass/entropy/heuristics they are cleaned
  (`NaN→0`, clipped to `[0, 1]`) per spec.
- No final heuristic is chosen here — see the reports for evidence-based
  candidate recommendations to inspect manually.
