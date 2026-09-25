"""Adapter layer for the ML2DAC baseline.

This is the ONLY place in the repository that imports and calls the external
ML2DAC project. Everything else talks to :class:`ML2DACAdapter` and the
normalised result dict it returns. The external source under ``external/ml2dac``
is never modified.
"""
from .ml2dac_adapter import ML2DACAdapter
from .artifact_locator import locate_artifacts

__all__ = ["ML2DACAdapter", "locate_artifacts"]
