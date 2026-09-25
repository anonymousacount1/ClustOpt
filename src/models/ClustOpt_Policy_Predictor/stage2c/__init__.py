"""Stage 2C: reranker-aware policy targets.

Lives inside the existing Policy Predictor package on purpose: Stage 2C changes
only the SUPERVISION of the Stage-2A predictor, never its architecture, so it
extends that package rather than forking a second one. Stage-2A modules are
imported unmodified and remain independently reproducible.
"""
