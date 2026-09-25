"""Utility helpers for the AutoML4Clust baseline (paths, IO, metrics, logging,
safe execution, dataset loading). All helpers are self-contained so the baseline
stays isolated from the rest of the repository where practical, while reusing the
canonical ClustOpt dataset loader / metric formulas for exact comparability.
"""
