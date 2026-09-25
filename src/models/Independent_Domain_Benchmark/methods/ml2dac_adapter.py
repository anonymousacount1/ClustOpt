"""Adapter for the frozen ML2DAC implementation.

STAGE 4A: interface only. This adapter must CALL the existing frozen ML2DAC
components rather than duplicate them, and must not retrain, recalibrate or
domain-adapt anything.

Implementation is a Stage-4B task. See ``configs/method_registry.json`` for the
exact artifacts, hashes, training-split provenance and entrypoints it must bind
to, and ``configs/runtime_protocol.json`` for what falls inside ONLINE_RUNTIME.
"""
from __future__ import annotations

from .base import MethodAdapter, MethodResult  # noqa: F401
