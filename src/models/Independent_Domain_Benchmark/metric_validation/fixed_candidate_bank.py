"""Stage-4B implementation target. Stage 4A freezes the contract only.

Honour the frozen protocol in ``configs/``:
``benchmark_protocol.json``, ``runtime_protocol.json``, ``result_schemas.json``,
``resume_semantics.json``, ``metric_registry.json``.

Reuses the frozen offline_fixed32 slate to produce 32 candidate partitions
per (dataset, view), independently of any method's search trajectory.
"""
