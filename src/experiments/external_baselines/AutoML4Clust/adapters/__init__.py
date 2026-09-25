"""Adapter layer for the AutoML4Clust baseline.

This is the ONLY place in the repository that imports and calls the external
AutoML4Clust project. Everything else talks to :class:`AutoML4ClustAdapter`
and the normalised result dict it returns. The external source under
``external/Automl4Clust`` is never modified.
"""
from .automl4clust_adapter import AutoML4ClustAdapter

__all__ = ["AutoML4ClustAdapter"]
