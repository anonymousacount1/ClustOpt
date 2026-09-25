"""Stage-4B implementation target. Stage 4A freezes the contract only.

Honour the frozen protocol in ``configs/``:
``benchmark_protocol.json``, ``runtime_protocol.json``, ``result_schemas.json``,
``resume_semantics.json``, ``metric_registry.json``.

Executes one (dataset, view, method) unit under the frozen protocol and
persists a result.json that contains NO labels and NO offline score.
"""
