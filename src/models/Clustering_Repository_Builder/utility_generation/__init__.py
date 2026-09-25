"""Clustering Repository — utility-vector generation pipeline.

Multiprocessing pipeline that takes a subfamily directory (full of generated
dataset folders) and produces metric utility vectors for the three standard
views (``x_only``, ``y_only``, ``xy_2d``).

This package is intentionally isolated from feature extraction and from
ClustOpt internals: it only *calls* existing ClustOpt entry points and the
existing ``compute_metric_utilities`` function.
"""
from __future__ import annotations

from .config import UtilityGenerationConfig, VIEW_IDS

__all__ = ["UtilityGenerationConfig", "VIEW_IDS"]
