"""Nested (pair-exclusion) cross-fitting infrastructure for Policy Predictor
model selection inside Splits 2-16.

Stage 2A-3A produced FINAL-training data: a row from split t uses utilities from a
model trained on S minus {t}. That is correct for the final predictor, but invalid for
model selection: when outer split s is held out, a training row from split t carries
utilities from a model that trained on s.

This package removes that indirect leakage by building 105 unordered pair-exclusion
utility sources -- one per {a,b} -- trained on S minus {a,b}.

No Policy Predictor is trained here.
"""
