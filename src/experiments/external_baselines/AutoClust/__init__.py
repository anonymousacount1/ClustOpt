"""AutoClust external AutoClustering baseline.

Wraps the **ML2DAC authors' re-implementation** of AutoClust (Poulakis et al.,
ICDM 2020), which lives inside the existing ML2DAC submodule at
``external/ml2dac/src/Experiments/RelatedWork/AutoClust.py``. AutoClust has no
official public implementation from its original authors; the ML2DAC authors
re-implemented it for their related-work comparison, and that is the version used
here.

This is a *separate* baseline from ML2DAC. Only the adapter imports the external
code (the ML2DAC building blocks); ``external/ml2dac`` is never modified.
"""
