"""Paper-oriented result analysis (v2).

A ground-up redesign of the ClustOpt result-analysis layer around **five
explicit comparison suites** rather than one generic per-method summary. See
``README_paper_analysis.md`` for the architecture and
``00_protocol_and_definitions/terminology.md`` in every generated output root
for the terminology contract.

The old :mod:`..result_aggregation` package is left untouched and still runs;
this package writes to a *separate* output root (``paper_analysis_v2``).
"""
