"""Stage 2B-5A: the ten final dedicated contextual rerankers.

One model per deployable observability regime, trained on ALL Splits 2-16 with
single-exclusion OOF utility context. These are the artifacts the Split-1 online
evaluation will load; nothing here touches Split 1.
"""
