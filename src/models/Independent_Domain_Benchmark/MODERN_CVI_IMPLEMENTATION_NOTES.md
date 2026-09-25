# Modern CVI Comparators — Implementation Notes

External metric-validation comparators **only**. None of these is ever added to
the ClustOpt, AutoClust or ML2DAC learned inventories.

| metric | reference | licence | our relationship | direction | status |
|---|---|---|---|---|---|
| **CDbw** | PyPI `cdbw` 0.2 | MIT | wrap | maximize | **VALIDATED** |
| **CVNN** | CRAN `fpc::cvnn` | GPL | oracle only, no port | minimize | **VALIDATED** |
| **CVDD** | `hulianyu/CVDD` @ `32cabe0c` | MIT | port with attribution | maximize | **VALIDATED**, with a predeclared applicability rule |
| **DCSI** | `JanaGauss/dcsi` @ `15ba6f7b` | **none** | paper-only reimplementation | maximize | **VALIDATED** at component level |

The primary comparator set is exactly these four and is closed. It is frozen in
`configs/modern_cvi_registry.json`.

## Why the first search missed CVNN and CVDD

Worth stating plainly, because it changed a conclusion.

- **CVNN** — I searched PyPI and GitHub. Its maintained reference lives on
  **CRAN**, which neither query covers. The one Python package I did inspect,
  `validclust`, provides dunn, cop, silhouette, DB and CH — all already native to
  our inventory — and **not** CVNN. I reported "no reference exists" when the
  correct statement was "no reference exists *in the two places I looked*".
- **CVDD** — my GitHub query was `CVDD+cluster+validity`, which returns 0 hits.
  The repository is named simply `CVDD` and its description does not contain that
  phrase. A narrower query produced a false negative, and I treated zero hits as
  evidence of absence.

Both were then located, pinned by commit, and licence-checked.

## Reference runtimes

R and Octave installers require UAC elevation that a non-interactive session
cannot grant (`winget` returned *"You cancelled the installation"* and
`0x800704c7`), and the local MATLAB tree has no `matlab.exe`. Every oracle
therefore runs inside a pinned container, built and stored **outside** any
Git-controlled path. Images, digests and package versions are frozen in
`configs/reference_runtime_manifest.json`.

## Validation results

Fourteen deterministic fixtures (`validation/cvi_fixtures.py`, seed 20260911)
cover K = 1, 2, 3 and 4, spherical and non-convex shapes, unequal densities,
duplicates, a singleton class and touching clusters. Tolerances were fixed before
any comparison ran.

| metric | oracle | cases | max abs diff | result |
|---|---|---|---|---|
| CDbw | PyPI `cdbw` 0.2 called directly | 13 | 0.0 | PASS |
| CVNN | `fpc::cvnn` 2.2.15 | 55 | 4.88e-15 | PASS |
| CVDD | author MATLAB via Octave 9.2.0 | 12 | 2.81e-11 | PASS |
| DCSI | author R, component-wise | 12 | exact per component | PASS |

Two bugs and two definitional problems were found by these oracles, all before
any scientific value was computed.

### CVNN — a real bug the oracle caught

K = 1 candidate partitions were being excluded before the component normalisers
were computed. `fpc` includes them in `max(sep)` and `max(comp)`, so excluding
them shifted every other candidate's value. Fixed; CVNN is set-relative, and the
frozen candidate bank (fixed32) is what makes its values comparable.

### CDbw — the value depends on how clusters are *numbered*

`CDbw` in PyPI `cdbw` 0.2 is deterministic for a fixed label vector but **not
invariant to the integer ordering of the cluster labels**. The same partition,
renumbered, moved the index by up to **0.931** absolute — on `mixed_density_4`,
four orderings give 1.3268, 1.2974, 1.1736 and 0.6017. Compactness, cohesion and
separation all move; the separation term aggregates each cluster's nearest
neighbour over *lower-indexed* clusters only, which is order-dependent by
construction.

Halkidi & Vazirgiannis (2008) define CDbw over an unordered set of clusters, so
this is an implementation artifact — but it means "the CDbw of a partition" is
not a well-defined number without a rule. The wrapper therefore imposes a frozen
canonical ordering that depends only on `(X, partition)`:

> renumber clusters `0..K-1` by ascending lexicographic order of the cluster
> centroid; ties by ascending cluster size, then by the index of the first member.

This selects one member of a set of implementation-dependent values. It does not
modify the algorithm and does not claim to repair it. CDbw values reported from
work that fixes no ordering are not directly comparable to ours.

Found by `validation/test_cdbw_label_invariance.py`; magnitudes in
`validation/cdbw_label_order_sensitivity.json`.

### CVDD — a predeclared applicability rule

If `X` contains two or more exactly identical rows, CVDD is `INVALID` with reason
`CVDD_DUPLICATE_POINT_MST_AMBIGUITY`. The density-involved distance rests on a
minimax path distance from an MST; exact duplicates create zero-weight edges, the
MST becomes non-unique, and the published procedure defines no
implementation-independent tie-break, so no canonical value exists. The rule
invalidates **CVDD only** — never the dataset, view, another CVI, a method run or
candidate generation — and was frozen before external prevalence was inspected.

### DCSI — two routines, one discrepancy

The author repository contains two DCSI routines:

- `calc_DCSI` (two-class) implements Definition 3.2 and produced the published
  measure tables;
- `calc_DCSI_RW` (multiclass, real-world) fills **both** triangles of its
  separation matrix inside the loop and then symmetrises with `+ t()`, which
  doubles every separation entry. The pairwise matrix immediately below it uses
  an upper-triangle loop, so only separation is affected.

Comparison was done **component by component**, because a matching aggregate can
hide two compensating component errors. Core-point sets are identical, per-class
connectedness is identical, separation is exactly `2 ×` ours on every pair, and
every input we call undefined is undefined for the reference too.

`dcsi` is frozen **paper-faithful**. `dcsi_author_compat` reproduces
`calc_DCSI_RW` bit-for-bit and exists so the difference stays auditable; it is a
diagnostic, **not** a fifth CVI, and never enters the registry, a comparator
count, an FDR family or a headline claim. Both live in one implementation behind
a frozen `sep_mode` switch.

Two published values were recomputed end to end from the data shipped beside
them: both reproduce under the **paper** mode to 1.1e-16, including the published
`Sep` and `Conn` components. So the paper-faithful choice agrees with the
definition *and* with the authors' own published numbers.

One earlier claim of mine was wrong and is withdrawn: the two modes are **not**
universally ranking-equivalent. Both pairwise maps are strictly increasing in
`q`, so K = 2 orderings do coincide — but the multiclass value is the mean of a
nonlinear transform, and for K ≥ 3 the orders can reverse. Counterexample, verified
analytically and through the shipped implementation, in
`validation/test_dcsi_variant_ranking.py`. A sensitivity analysis is predeclared
in `configs/dcsi_sensitivity_analysis_plan.json`.

## Licensing rules in force

- **CDbw (MIT)** — wrapped; attribution retained.
- **CVNN (GPL)** — used as an **oracle only**. No `fpc` source is copied or
  ported, so no GPL obligation is incorporated into this repository.
- **CVDD (MIT)** — ported with attribution.
- **DCSI (no licence)** — **no source copying of any kind.** The implementation
  comes from the paper's mathematics; the author repository is consulted only as
  an external numerical oracle in a temporary checkout that is never committed,
  and no author data is committed either.

## Wrapper contract

Comparators legitimately differ on noise, singletons and invalid partitions.
Formulas are **not** altered to force agreement. Every comparator returns either
a finite score **or** an explicit `INVALID` with a metric-specific reason, an
`INVALID` is always metric-local, and the external utility computation uses the
project's existing valid-pair / NaN handling unchanged.
