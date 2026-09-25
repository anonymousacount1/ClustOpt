"""External AutoClustering baselines.

Each baseline is an isolated, model-specific sub-package that wraps an external
project through adapters (never modifying the external source). The baselines
reuse the ClustOpt dataset/split/metric conventions so results are directly
comparable.

Planned sub-packages:
    AutoML4Clust/   -- this milestone
    ML2DAC/         -- future
    cSmartML/       -- future
"""
