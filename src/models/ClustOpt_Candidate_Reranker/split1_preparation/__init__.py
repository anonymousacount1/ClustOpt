"""Stage 2B-5A: structural preparation of the Split-1 online-trace evaluation.

Everything here is preparation. No reranker prediction is made on Split 1, no
candidate is selected, and no ARI value is read -- the trace reader excludes the
ARI column by ``usecols``, so the guarantee is structural rather than a promise.
"""
