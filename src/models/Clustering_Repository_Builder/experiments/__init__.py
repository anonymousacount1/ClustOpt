"""AutoClustering experiment pipeline.

Three independent components:

* :mod:`data_splitting`        - split raw datasets into balanced experiment splits.
* :mod:`experiment_execution`  - run ClustOpt experiments for a subfamily/split.
* :mod:`result_aggregation`    - aggregate raw results into research summaries.

Each component has its own CLI (``run_*.py``) and is runnable independently so
splitting is done once, execution is resumable, and aggregation can be rerun
without repeating expensive Optuna searches.
"""
