"""Unified Meta-Knowledge Repository (MKR) builder.

Subfamily-level construction of one *unified raw* repository from which the
following model-specific in-domain knowledge bases can later be derived
(NOT implemented in this phase):

* ML2DAC original / ML2DAC + ClustOpt metrics
* AutoClust original / AutoClust + ClustOpt metrics

See ``UNIFIED_MKR_DESIGN.md`` for the authoritative specification.

Design principles realised here:

* Reuse ClustOpt for search-space expansion, clustering, and the 60 final
  utility-map metrics (``models/ClustOpt``).
* Reuse the original ML2DAC source for the ML2DAC-only CVIs (Dunn / Coggins-Jain
  / COP) and for ML2DAC + AutoClust meta-features (``external/ml2dac/src``).
* Reuse the ClustOpt dataset loader / view builder and the project's
  long-path-safe atomic IO helpers.
* Compute every overlapping metric exactly once (metric de-duplication).
"""

SCHEMA_VERSION = "1.0.0"
