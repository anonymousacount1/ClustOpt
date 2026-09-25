# Independent Domain Benchmark

The final experimental block: an external benchmark of the frozen ClustOpt system
against frozen AutoClust and ML2DAC, on 50 datasets **not** produced by the
internal MKR generator.

**Stage 4A froze everything below. Nothing here has been executed.** No external
search has run, no model has been trained or loaded for inference, and no
authoritative result file exists.

## Why this component sits here

It is a top-level pipeline component parallel to `Clustering_Repository_Builder`,
deliberately **not** nested inside ClustOpt, AutoClust, ML2DAC or the builder: it
consumes all four and belongs to none. Adapters **call** the frozen scientific
implementations; they never re-implement them.

Authoritative results go to `results_analysis/independent_domain_benchmark/`,
never into the synthetic `analyzed_data` tree.

## Corpus (50 datasets, frozen)

| source | n | notes |
|---|---:|---|
| `fcps` | 7 | atom, chainlink, engytime, lsun, tetra, twodiamonds, wingnut |
| `sklearn_shapes` | 15 | 5 generators x {easy, medium, hard} |
| `real_pca` | 28 | pinned OpenML / sklearn loaders, fixed label-free 2-D PCA |

Eligibility is decided by `configs/dataset_manifest.json` alone: public,
reproducibly obtainable, numeric, labelled, **2 ≤ K ≤ 5**, 200–5000 rows. No
dataset was ever accepted or rejected because of how a method performs on it.

## The two rules that matter most

1. **Labels never reach execution.** Imputation, scaling and PCA are fitted on X
   alone; a method receives `ExecutionInput`, which has no label field. Ground
   truth is reachable only through `evaluation/labels.py`, which refuses to hand
   it over until predictions are already persisted.
2. **One common timing boundary.** `ONLINE_RUNTIME = T1 − T0`, where at T0 the
   frozen X for one view is already in memory and T1 is when final labels are
   materialised. Preparation and artifact loading sit outside it;
   `MODEL_INIT_RUNTIME` is measured separately and never mixed into the headline.
   This replaces the historical 150 s ClustOpt-only cap, which was not comparable
   across frameworks.

## Layout

```
configs/     11 frozen JSON contracts (see config_hashes.json)
datasets/    corpus retrieval, frozen preparation, three-view construction
methods/     adapters onto the frozen ClustOpt / AutoClust / ML2DAC systems
search/      budget, K range, matched search space, SHA256 seed derivation
runtime/     the common timer and environment provenance
execution/   runner, strict resume predicate, authoritative storage
evaluation/  label gate and offline scoring
metric_validation/  fixed32 candidate bank and modern comparator CVIs
runners/     audit_stage4a.py, freeze_stage4a_configs.py
```

## Method variants (14)

6 ClustOpt (C0–C5), 4 ML2DAC (M0–M3), 4 AutoClust (A0–A3). Exact IDs, artifact
paths and SHA-256 hashes are in `configs/method_registry.json`. **No method may
be added once external outcomes exist.**

C0/C3 and C1/C4 share a search trace by design — the reranker changes only final
candidate selection — so six outputs require **four** physical traces. A reranker
arm's reported runtime is the shared search cost **plus** its own selection cost,
never the incremental reranker time alone.

## Known blocker carried into Stage 4B

The final Policy Predictor binaries (PP v1, PP v2) were never persisted by any
pipeline script. They are deterministically reconstructible and bit-exactly
verifiable against the stored Split-1 selections. See
`configs/method_registry.json` → `pp_blocker`, and the Stage-4A report §14.

## Reproducing the freeze

```
python models/Independent_Domain_Benchmark/runners/audit_stage4a.py --repo-root .
python models/Independent_Domain_Benchmark/runners/freeze_stage4a_configs.py --repo-root .
python models/Independent_Domain_Benchmark/runners/verify_stage4a_gates.py --repo-root .
```

The audit performs network retrieval and metadata inspection only.
