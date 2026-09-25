"""Tests for the paper-oriented analysis.

Run from the repo root::

    python -m pytest models/Clustering_Repository_Builder/experiments/paper_analysis/tests -q

The unit tests use synthetic mini-frames and never touch the experiment output
tree, so they run in seconds. The integration test is marked ``slow`` and reads a
handful of real datasets.
"""
