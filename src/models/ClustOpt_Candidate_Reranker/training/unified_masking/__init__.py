"""Stage 2B-3: one masking-aware reranker across all 10 deployable regimes.

Each outer-training slate carries exactly ONE regime, so the unified model trains
on the same number of candidate rows as a dedicated specialist. That isolates
shared cross-regime learning from a 10x larger training set.
"""
