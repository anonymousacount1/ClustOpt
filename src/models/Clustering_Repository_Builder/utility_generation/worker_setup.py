"""Process-level setup helpers shared by the main runner and every worker.

* Reconfigures ``sys.stdout`` / ``sys.stderr`` to UTF-8 with ``errors='replace'``
  so a stray emoji in third-party code can't crash a process under cp1252 /
  cp1255 / other narrow Windows code pages.
* Filters out the well-known third-party warnings that bury real progress
  output (BIRCH ConvergenceWarning, hdbscan ``invalid value in divide``,
  scipy ``ConstantInputWarning``, sklearn ``force_all_finite`` deprecation).
"""
from __future__ import annotations

import os
import sys
import warnings


_DONE_FLAG = "_clustering_repo_utility_gen_setup_done"


def _reconfigure_stdio() -> None:
    """Make stdout/stderr UTF-8 tolerant. Safe to call multiple times."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            # Best-effort: if reconfigure isn't supported (older Python, weird
            # stream wrappers), leave it as-is rather than crash setup.
            pass


def _install_warning_filters() -> None:
    """Silence the known-noisy warnings the pipeline can't act on."""
    # sklearn BIRCH "Number of subclusters found (1) by BIRCH is less than (N)"
    # — fires for tiny / degenerate slices in 1-D views; the configuration is
    # still recorded with valid=False and downstream code handles it correctly.
    warnings.filterwarnings(
        "ignore",
        message=r".*Number of subclusters found.*by BIRCH.*",
        category=UserWarning,
    )
    try:
        from sklearn.exceptions import ConvergenceWarning
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
    except Exception:
        pass

    # hdbscan validity: "invalid value encountered in divide" — fires when a
    # cluster collapses; the resulting NaN is treated as invalid downstream.
    warnings.filterwarnings(
        "ignore",
        message=r".*invalid value encountered in divide.*",
        category=RuntimeWarning,
    )

    # scipy ConstantInputWarning from spearmanr/kendalltau when a metric
    # column is constant across all configs.
    try:
        from scipy.stats import ConstantInputWarning  # type: ignore[attr-defined]
        warnings.filterwarnings("ignore", category=ConstantInputWarning)
    except Exception:
        warnings.filterwarnings(
            "ignore",
            message=r".*input array is constant.*",
            category=RuntimeWarning,
        )

    # sklearn 1.6 deprecation surfaced by hdbscan<=0.8.40
    warnings.filterwarnings(
        "ignore",
        message=r".*force_all_finite.*was renamed to.*ensure_all_finite.*",
        category=FutureWarning,
    )


# Thread-limit variables consulted by the numerical stacks at import time.
_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
)


def limit_compute_threads(n_threads: int = 1) -> None:
    """Cap numerical-library threading for THIS process.

    Rationale (Stage-1E): the online sweep runs 8 worker processes on an
    8-core machine. With uncapped BLAS/OpenMP each worker spawns up to 8
    threads, so ~64 threads contend for 8 cores. Measured effect was severe:
    recorded per-run compute time averaged 92.6 s while actual elapsed time per
    worker was ~960 s, i.e. roughly 10x CPU starvation, projecting a 26-day
    sweep. One compute thread per worker restores true 8-way process
    parallelism.

    This changes SPEED ONLY. It cannot change results: the Optuna sampler is
    seeded, clustering ``random_state`` values are pinned, and the protocol has
    no timeout or patience, so nothing in the pipeline is time-dependent.

    Must be called BEFORE numpy / scipy / sklearn / torch are imported for the
    environment variables to take effect (they are read when those libraries
    initialise their pools). ``configure_process()`` is invoked at module import
    in every runner and at the top of every worker entry point, which on Windows
    ``spawn`` puts it ahead of the heavy imports. As a belt-and-braces measure we
    additionally clamp any pools that are already loaded via ``threadpoolctl``
    and set ``torch`` intra/inter-op threads when torch is already imported.
    """
    n = str(max(1, int(n_threads)))
    for var in _THREAD_ENV_VARS:
        os.environ[var] = n

    # Clamp pools that may already be initialised in this process.
    try:
        import threadpoolctl
        threadpoolctl.threadpool_limits(limits=int(n))
    except Exception:
        pass

    # torch keeps its own pools. Only touch it if it is ALREADY imported --
    # importing torch here would slow every process start and risk re-entrant
    # initialisation under Windows spawn.
    if "torch" in sys.modules:
        try:
            torch = sys.modules["torch"]
            torch.set_num_threads(int(n))
            try:
                torch.set_num_interop_threads(int(n))
            except Exception:
                # Raises if inter-op parallelism was already started; harmless.
                pass
        except Exception:
            pass


def thread_configuration() -> dict:
    """Introspect the effective thread configuration (for run provenance)."""
    info = {v: os.environ.get(v) for v in _THREAD_ENV_VARS}
    info["cpu_count"] = os.cpu_count()
    try:
        import threadpoolctl
        info["threadpools"] = [
            {"user_api": p.get("user_api"), "internal_api": p.get("internal_api"),
             "num_threads": p.get("num_threads"),
             "prefix": p.get("prefix")}
            for p in threadpoolctl.threadpool_info()
        ]
    except Exception as exc:
        info["threadpools"] = f"<unavailable: {exc}>"
    if "torch" in sys.modules:
        try:
            info["torch_num_threads"] = sys.modules["torch"].get_num_threads()
        except Exception:
            info["torch_num_threads"] = None
    return info


def configure_process(thread_limit: int = 1) -> None:
    """Top-level entry point for both main and worker processes.

    Idempotent: a flag on :mod:`sys` prevents re-applying the same filters
    when the function is called twice (e.g. once at import, once inside the
    worker function).

    ``thread_limit`` caps numerical-library threading (default 1) -- see
    :func:`limit_compute_threads`. The thread cap is applied on EVERY call, even
    when the idempotency flag is already set, because a child process may import
    a numerical library after its first ``configure_process()`` call.
    """
    limit_compute_threads(thread_limit)
    if getattr(sys, _DONE_FLAG, False):
        return
    _reconfigure_stdio()
    _install_warning_filters()
    setattr(sys, _DONE_FLAG, True)
