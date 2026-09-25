"""AutoML4Clust external baseline.

Wraps the external AutoML4Clust project (``external/Automl4Clust``) as a
black-box AutoClustering baseline. The external source is never modified; all
integration happens through the :mod:`adapters` layer.

Fairness contract (see README_AutoML4Clust_baseline.md): AutoML4Clust never
receives true labels, true K, ClustOpt metric utilities, or family metadata.
Ground truth is used only for *external* evaluation after labels are predicted.
"""

__all__ = ["adapters", "runners", "utils"]
