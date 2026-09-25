"""Runners for the AutoML4Clust baseline.

* :mod:`smoke_test_automl4clust` -- tiny end-to-end sanity check.
* :mod:`run_automl4clust_dataset` -- one dataset, all requested record types
  (also exposes ``process_single_dataset`` reused by the subfamily runner).
* :mod:`run_automl4clust_subfamily_split` -- the final multiprocessing runner
  over a whole subfamily/split.
"""
