> **Internal development notes.** The paper's tables and figures are rebuilt with `python scripts/reproduce_paper_artifacts.py` (see `docs/paper_results_index.md`); `run_build_paper_package.py`, named below, is not part of this release.

# `paper_analysis` — paper-oriented result analysis (v2)

A ground-up redesign of the ClustOpt result-analysis layer, organised around
**five explicit comparison suites** rather than one generic per-method summary.

The previous `result_aggregation` / `result_packaging` packages are **untouched**
and still run; this package writes to a separate output root
(`results_analysis/clustering_repository/paper_analysis_v2`).

---

## The constraint that shapes the whole design

All downstream clustering experiments were executed on **`split_id = 1` only**.
There is no independent split on which a policy could be chosen and then
evaluated. Every ClustOpt single-variant "winner" in this analysis is therefore a
**retrospective** selection, and the architecture makes that impossible to lose:

- the selection engine stamps `selection_type`, `selection_split`,
  `selection_metric`, `is_independently_preselected = False` and
  `paper_claim_level` onto **every** row it produces;
- comparison suites can only name a `@representative:<id>`, which is resolved
  against the **frozen** selection manifest — a suite physically cannot reselect a
  winner inside its own plotting or reporting code;
- statistics involving a representative are labelled
  `post_selection_exploratory` automatically, from the hypothesis's own arms.

Read `00_protocol_and_definitions/terminology.md` in any generated output root
before quoting a number.

---

## Pipeline

```
raw result discovery            collector.py
  -> canonical method registry  method_registry.py     (51 methods, declarative)
  -> canonical run frame        collector.py           (dataset x method x view)
  -> canonical best-view frame  frames.py              (dataset x method, view oracle)
  -> canonical dataset frame    frames.py              (+ cross-method context)
  -> retrospective selection    selection.py           (global + family scope)
  -> representative manifest    preliminary.py         (FROZEN)
  -> representative frames      frames.py
  -> portfolio / VBS engine     vbs.py                 (+ size sensitivity)
  -> gap decomposition          gaps.py                (7 named gaps)
  -> metric taxonomy + usage    metric_registry.py, metric_analysis.py
  -> comparison-suite engine    comparison_engine.py   (declarative configs)
  -> artifact manifest          artifact_manifest.py
  -> comparison packaging       ../paper_packaging/
```

## Modules

| module | responsibility |
|---|---|
| `paths.py` | long-path-safe IO (every run dir is ~270 chars; long paths are disabled on the target machine) |
| `method_registry.py` | one `MethodSpec` per method: taxonomy, oracle level, labels, capabilities, candidate pools, the 17 variant families |
| `portfolio_registry.py` | frozen VBS portfolio membership, `fully_operational` / `oracle_contamination_reason`, subset invariants, deprecated-id aliases |
| `metric_registry.py` | metric taxonomy **derived** from implementation packages and MLP head groups; two separately-named notions of "classic" |
| `collector.py` | run frame; both on-disk schemas; cross-checks the registry against what each run recorded about itself |
| `frames.py` | best-view (view oracle), dataset-level, representative frames; the primary-metric summary block |
| `capability.py` | capability matrix **verified on disk**, preflight, warnings |
| `selection.py` | retrospective selection engine, deterministic tie-break chain, manifests |
| `vbs.py` | dataset-level VBS, headroom, greedy + random-subset size sensitivity, invariants |
| `gaps.py` | search-selection, view, utility-prediction, weighting, metric-count, policy/VBS, external-leader gaps |
| `metric_analysis.py` | metric selection usage with correct cross-policy normalisation; new-metric usage |
| `stats.py` | paired tests, effect sizes, bootstraps, per-family Holm, **selection-aware bootstrap** |
| `stability.py` | per-pool bootstrap that **repeats the selection step inside every resample**; selection frequency, runner-up, entropy |
| `labels.py` | one label manifest across methods, representatives and portfolios; enforces the marker conventions |
| `plots.py` | reusable figure layer; every figure ships PNG + SVG + its plotted data CSV |
| `excel.py` | openpyxl workbooks with Definitions / Labels / Selection Manifest / Capability Warnings sheets |
| `reports.py` | markdown writers; the §34 wording rules live here and only here |
| `artifact_manifest.py` | one row per generated artifact; what the packager consumes |
| `config.py` | declarative config loading + `@representative:` resolution |
| `comparison_engine.py` | executes one comparison config into artifacts |
| `preliminary.py` | the preliminary selection stage |
| `run_analysis.py` | CLI |

## Design rules worth knowing before editing

**Nothing re-derives taxonomy from a name.** Method family, oracle level,
weighting mode and plot label all come from `method_registry`. `is_regressor(name)
-> name.startswith("regressor")` was removed on purpose.

**Runtime meanings never mix.** The primary operational cost is
`{mean,median,std,p90}_total_runtime_all_views_sec` -- you must run all three
views to obtain a Best-View result. `*_selected_best_view_runtime_sec` is a
diagnostic only and is never the headline.

A VBS is **not one executable method**, so it reports
`selected_member_total_all_views_runtime_sec` (a diagnostic: the member the
oracle picked) *and* `portfolio_total_execution_runtime_sec` (the cost of running
every member -- the only quantity that may be called the cost of the VBS). The
validator asserts the second is never below the first, and no VBS point enters an
operational Pareto plot.

End-to-end components are not recorded by any adapter and are never
approximated; the wording used throughout is "total recorded search/runtime
across all three views", never "complete end-to-end runtime".

**A portfolio name never outruns its membership.** The 12-variant MLP/KNN sets
include Dynamic Real-K, which takes the metric count from the real utility
vector, so they are `*_all_variants_vbs`, not "deployable". The 10-variant
subsets that exclude them are `*_strict_deployable_vbs`.
`strict_deployable_clustopt_vbs` (20 members) is the only learned-source ClustOpt
portfolio that may be described as fully operational. The registry validator
fails any id containing "deployable" or "strict" whose membership contradicts it.

**Retrospective selections expose their fragility.** Every selection records its
runner-up, the delta to it, the tie-break reason, and the bootstrap selection
frequency from `stability.py` -- which re-selects inside each resample rather
than bootstrapping an already-chosen winner.

**Missing capabilities are skipped, not null-filled.** The preflight names the
exact method and the exact missing field, the sub-analysis is skipped, the rest of
the comparison continues, and the limitation lands in the report and in
`*_capability_warnings.csv`.

**Selection is deterministic.** The tie-break chain is
`mean^ > median^ > std_v > k_accuracy^ > runtime_v > name^`, later keys consulted
only on a tie within `ari_tie_tolerance`. Practical equivalence is recorded
alongside, never allowed to override the raw ARI winner.

**Explicit column names.** A collapsed frame has `best_view_mean_ari`, never a
bare `mean_ari` that could be mistaken for a mean over all views.

## Usage

```bash
# registries + configs only; reads no experiment output
python -m models.Clustering_Repository_Builder.experiments.paper_analysis.run_analysis \
    --stage validate

# freeze the representatives and portfolios (Split 1)
python -m models.Clustering_Repository_Builder.experiments.paper_analysis.run_analysis \
    --stage preliminary-selection --split-id 1 --collect-workers 12

# one comparison suite (needs the frozen manifest)
python -m models.Clustering_Repository_Builder.experiments.paper_analysis.run_analysis \
    --stage comparison-04 --split-id 1

# all five suites in order
python -m models.Clustering_Repository_Builder.experiments.paper_analysis.run_analysis \
    --stage all-comparisons --split-id 1

# advisor-facing package, driven by the artifact manifest
python -m models.Clustering_Repository_Builder.experiments.paper_packaging.run_build_paper_package \
    --analysis-root results_analysis/clustering_repository/paper_analysis_v2
```

Useful options: `--dry-run`, `--limit-datasets N`, `--no-plots`, `--no-excel`,
`--no-stats`, `--min-plot-importance paper_main`, `--methods A B C`,
`--output-root PATH`.

## Tests

```bash
python -m pytest models/Clustering_Repository_Builder/experiments/paper_analysis/tests -q
```

145 unit tests on synthetic frames (seconds) plus 10 integration tests that read
a handful of real datasets and skip cleanly if the analyzed-data root is absent.
The unit suite covers registry counts, best-view semantics including ties and
total failures, selection determinism, the VBS invariants (one winner per
dataset, never below the best single member, superset dominates subset, runtime
is the member sum), metric-taxonomy coverage, every gap formula and sign, paired
alignment and Holm monotonicity, manifest-driven packaging, and the old
aggregation CLI's continued operation.
