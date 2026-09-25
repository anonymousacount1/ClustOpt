"""Safe-execution helpers: never let one degenerate run crash the pipeline.

The whole baseline follows the ClustOpt robustness philosophy -- a failure in
any single (dataset, record) unit is captured into a structured ``failed``
result (with traceback) and never propagates. These helpers standardise that
capture.

A best-effort thread-based timeout is provided. NOTE: the underlying
AutoML4Clust optimizers (SMAC / hpbandster) cannot be interrupted cleanly, so a
timeout leaves the worker thread running in the background (daemonised) and
simply reports a timeout result. The natural, deterministic budget is the
optimizer iteration count (``n_loops``); the timeout is only a safety net and
is disabled by default. See README for the budget discussion.
"""
from __future__ import annotations

import threading
import time
import traceback
from typing import Any, Callable, Dict, Optional, Tuple


def build_failure(
    error: BaseException,
    *,
    runtime_sec: float = 0.0,
    stage: Optional[str] = None,
) -> Dict[str, Any]:
    """Standard structured failure payload (matches the task's schema)."""
    tb = "".join(traceback.format_exception(type(error), error, error.__traceback__))
    payload: Dict[str, Any] = {
        "status": "failed",
        "error_type": type(error).__name__,
        "error_message": str(error),
        "traceback": tb,
        "runtime_sec": float(runtime_sec),
    }
    if stage is not None:
        payload["stage"] = stage
    return payload


class TimeoutExceeded(RuntimeError):
    """Raised (logically) when a guarded call exceeds its wall-clock budget."""


def safe_call(
    fn: Callable[..., Any],
    *args,
    timeout_sec: Optional[float] = None,
    **kwargs,
) -> Tuple[Optional[Any], Optional[Dict[str, Any]], float]:
    """Run ``fn`` and capture failures as a structured payload.

    Returns ``(result, failure_payload, runtime_sec)``. Exactly one of
    ``result`` / ``failure_payload`` is non-None on a normal/exception path; on
    timeout, ``failure_payload`` is set.

    ``timeout_sec`` (best-effort, thread-based) is only applied when a positive
    value is passed. When None/0 the call runs synchronously with no timeout.
    """
    t0 = time.perf_counter()

    if not timeout_sec or timeout_sec <= 0:
        try:
            result = fn(*args, **kwargs)
            return result, None, time.perf_counter() - t0
        except BaseException as exc:  # noqa: BLE001 - we re-package every failure
            return None, build_failure(exc, runtime_sec=time.perf_counter() - t0), \
                time.perf_counter() - t0

    box: Dict[str, Any] = {}

    def _target() -> None:
        try:
            box["result"] = fn(*args, **kwargs)
        except BaseException as exc:  # noqa: BLE001
            box["error"] = exc

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join(timeout=float(timeout_sec))
    runtime = time.perf_counter() - t0

    if thread.is_alive():
        err = TimeoutExceeded(
            f"AutoML4Clust call exceeded the {timeout_sec:.0f}s wall-clock budget "
            f"(thread left running in background)."
        )
        return None, build_failure(err, runtime_sec=runtime, stage="timeout"), runtime

    if "error" in box:
        return None, build_failure(box["error"], runtime_sec=runtime), runtime
    return box.get("result"), None, runtime


__all__ = ["build_failure", "safe_call", "TimeoutExceeded"]
