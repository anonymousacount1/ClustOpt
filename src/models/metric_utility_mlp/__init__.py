"""
metric_utility_mlp
==================

Isolated package implementing a supervised MLP pipeline that predicts the
*utility vector* of clustering metrics from extracted dataset meta-features.

Input : 250 dataset meta-features
Output: 60 metric-utility scores in [0, 1]

The package is fully self-contained and must not depend on (or be imported by)
ClustOpt / HYBRID_SCM / the repository builder. See README.md for usage.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "1.0.0"
