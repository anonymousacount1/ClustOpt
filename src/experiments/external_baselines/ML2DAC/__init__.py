"""ML2DAC external baseline.

Wraps the external ML2DAC project (``external/ml2dac``) -- a *meta-learning*
AutoClustering approach that (a) predicts a CVI, (b) selects warmstart
configurations from similar prior datasets, and (c) reduces the algorithm
search space. The external source is never modified; integration happens only
through the :mod:`adapters` layer.

Fairness contract (see README_ML2DAC_baseline.md): ML2DAC never receives true
labels, true K, ClustOpt metric utilities, or family metadata. Ground truth is
used only for *external* evaluation after labels are predicted.

This package mirrors the AutoML4Clust baseline structure and reuses its utils
(metrics / IO / dataset loader) so results are directly comparable.
"""

__all__ = ["adapters", "runners", "utils"]
