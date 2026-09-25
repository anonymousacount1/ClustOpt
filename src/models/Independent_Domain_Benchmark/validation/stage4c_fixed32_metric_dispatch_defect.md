# Stage 4C — fixed32 metric-evaluation dispatch defect

Found after Phase D, on `stage4c_v2_323194df26`. The method track (Phase A,
2100/2100 arms) is **not affected**. The defect is confined to the fixed32
metric-analysis track built by `analysis/fixed32_track.py::evaluate_bank_metrics`.

## Symptom

6 of the 67 primary metrics produced **zero** valid values across the entire
corpus — 4776 of 4776 valid candidates, all 150 dataset x view units:

| metric | group | implementation | failure |
|---|---|---|---|
| `dcsi` | Modern | `modern_comparator` | `METRIC_EVALUATION_FAILED:TypeError` |
| `cdbw` | Modern | `modern_comparator` | `METRIC_EVALUATION_FAILED:TypeError` |
| `cvdd` | Modern | `modern_comparator` | `METRIC_EVALUATION_FAILED:TypeError` |
| `cvnn` | Modern | `modern_comparator` | `METRIC_EVALUATION_FAILED:TypeError` |
| `convexity_ratio_image_based` | New46 | `clustopt_registry` | `METRIC_EVALUATION_FAILED:ML2DACScoreError` |
| `hough_arc_circle_strength` | New46 | `clustopt_registry` | `METRIC_EVALUATION_FAILED:ML2DACScoreError` |

Downstream this makes `modern_comparator_analysis.json` carry
`"modern_primary": []`: the modern-comparator comparison, a named Phase D
deliverable, is empty. Phase D covers 61 of 67 primary metrics.

900 of the 948 `INSUFFICIENT_VALID_CANDIDATES` utility cells are these 6
metrics; the remaining 48 are genuine thin-bank cases.

## Root cause

The registry declares three implementation kinds:

| `implementation` | count | correct resolver |
|---|---|---|
| `clustopt_registry` | 60 | `cluster_validity_indices.metrics.metrics_registry.METRIC_REGISTRY[id]` |
| `ml2dac_cvi_collection` | 3 | `ml2dac_scoring.canonical_raw_value` |
| `modern_comparator` | 4 | `metric_validation.external_cvis.<id>` |

`evaluate_bank_metrics` branches on `modern_comparator` only and sends
everything else to the ML2DAC engine. Two independent faults follow.

**1. Modern comparators return a value object, not a float.**
`external_cvis.<id>(X, labels)` returns a `MetricResult` dataclass with fields
`metric_id, value, direction, valid, invalid_reason, runtime_sec, provenance`.
The harness evaluates `float(fn(...))`, which raises `TypeError` on the
dataclass. The metric implementations are correct and were never reached for
their value; only the call site is wrong.

Verified directly on `real_churn` (n=5000), candidate 0:

| metric | runtime | valid | value |
|---|---|---|---|
| `cdbw` | 0.35 s | True | 0.0078926 |
| `cvdd` | 29.51 s | True | 70.5219 |
| `dcsi` | 1.42 s | True | 0.0214304 |

`cvnn` is different, and not a coding fault: it returns
`valid=False, invalid_reason="cvnn_requires_a_comparison_set_for_normalisation"`
with the note *"use cvnn_index over the fixed32 candidate bank"*. CVNN is
normalised across a comparison set of partitions, so it is not defined for a
single candidate in isolation. Evaluating it over the fixed32 bank is a
**semantic decision**, not a bug fix.

**2. Two `clustopt_registry` metrics are not in the ML2DAC 61-class collection.**
Routing all non-modern metrics through the ML2DAC engine works for 61 of the 63,
and raises `CVI_CLASS_UNKNOWN` for `convexity_ratio_image_based` and
`hough_arc_circle_strength`. Both are present in the ClustOpt `METRIC_REGISTRY`
(60 entries, matching the 60 `clustopt_registry` declarations), which the
harness never consults.

## What is NOT affected

* Phase A method track: 2100/2100 arms, all `success`. Method arms score through
  `ml2dac_scoring` / `autoclust_scoring` with their own frozen CVI inventories,
  not through this harness.
* The 61 metrics that did evaluate: their values are unchanged by any fix
  described here, provided the repair is additive (see below).
* Candidate banks and labels: 150/150 banks, 4800/4800 slots kept, every stored
  label reproduces its raw and canonical hash.

## Cost of repair

Only the 6 affected metrics need recomputing; the other 61 are untouched.
Calibrated from measured runtimes (`cvdd` ~ 0.028 s x (n/180)^2):

| item | estimate |
|---|---|
| `cvdd` over 4776 valid candidates | ~3.1 h |
| `cdbw` + `dcsi` | ~0.6 h |
| single worker total | ~3.8 h |
| 5 workers | ~0.8 h |

The two image-based New46 metrics are cheap by comparison.

## Constraint note

No metric implementation needs to change. The repair is confined to the
dispatch in `evaluate_bank_metrics`, plus a decision on CVNN's comparison set.
Because Phase B metric records are frozen COMPLETE artifacts, an additive
amendment writing new record keys and preserving the originals is the
appropriate mechanism, matching the corrected-scoring precedent.
