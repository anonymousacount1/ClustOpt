"""Stage 2C-2B: decide the worker count from a REAL concurrent probe.

Fits one representative pair-exclusion reranker serially, then two of them
simultaneously, sampling memory throughout. Two workers are accepted only if the
free-RAM floor stays above a hard reserve and the per-fit slowdown is not
pathological. This is an engineering decision; it touches no scientific choice.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import subprocess
import sys
import threading
import time
from ctypes import wintypes
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from models.Clustering_Repository_Builder.experiments.experiment_execution.output_writer import (  # noqa: E402,E501
    ensure_dir, write_json_atomic,
)
from models.ClustOpt_Candidate_Reranker.training import model_registry as MR  # noqa: E402,E501
from models.ClustOpt_Policy_Predictor.stage2c.run_pair_exclusion_rerankers import (  # noqa: E402,E501
    OUT_REL,
)

FREE_MB_RESERVE = 1200.0        # hard floor: never eat the last GB
SLOWDOWN_LIMIT = 2.6            # 2 workers must beat serial in throughput


def free_mb() -> float:
    class MS(ctypes.Structure):
        _fields_ = [("dwLength", wintypes.DWORD),
                    ("dwMemoryLoad", wintypes.DWORD),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    st = MS()
    st.dwLength = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
    return st.ullAvailPhys / 2**20


def fit_once(n_rows: int, n_feat: int, seed: int, out: List[float]) -> None:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_rows, n_feat), dtype=np.float32)
    y = rng.standard_normal(n_rows).astype(np.float32)
    t0 = time.perf_counter()
    MR.build_model().fit(X, y)
    out.append(time.perf_counter() - t0)
    del X, y


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", default=".")
    p.add_argument("--rows", type=int, default=1_000_000)
    args = p.parse_args(argv)
    repo = Path(args.repo_root).resolve()
    out = ensure_dir(repo / OUT_REL)
    print("=" * 78)
    print("STAGE 2C-2B  CONCURRENT WORKER PREFLIGHT")
    print("=" * 78, flush=True)

    n_feat = 347
    floor = {"v": free_mb()}
    stop = threading.Event()

    def sampler() -> None:
        while not stop.is_set():
            floor["v"] = min(floor["v"], free_mb())
            time.sleep(0.25)

    start_free = free_mb()
    print("  free at start: %.0f MB | probe matrix %.2f GB"
          % (start_free, args.rows * n_feat * 4 / 2**30), flush=True)

    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    serial: List[float] = []
    fit_once(args.rows, n_feat, 0, serial)
    serial_floor = floor["v"]
    print("  serial fit %.1fs | free floor %.0f MB" % (serial[0], serial_floor),
          flush=True)

    floor["v"] = free_mb()
    conc: List[float] = []
    ths = [threading.Thread(target=fit_once, args=(args.rows, n_feat, i + 1, conc))
           for i in range(2)]
    t0 = time.perf_counter()
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    conc_wall = time.perf_counter() - t0
    conc_floor = floor["v"]
    stop.set()

    slow = (max(conc) / serial[0]) if serial[0] else float("inf")
    throughput_gain = (2 * serial[0]) / conc_wall if conc_wall else 0.0
    safe = bool(conc_floor > FREE_MB_RESERVE and slow < SLOWDOWN_LIMIT
                and throughput_gain > 1.0)
    workers = 2 if safe else 1
    res: Dict[str, Any] = {
        "probe_rows": args.rows, "n_features": n_feat,
        "free_mb_at_start": start_free,
        "serial_fit_sec": serial[0], "serial_free_floor_mb": serial_floor,
        "concurrent_fit_sec": conc, "concurrent_wall_sec": conc_wall,
        "concurrent_free_floor_mb": conc_floor,
        "per_fit_slowdown": slow, "throughput_gain_vs_serial": throughput_gain,
        "free_mb_reserve": FREE_MB_RESERVE,
        "slowdown_limit": SLOWDOWN_LIMIT,
        "two_workers_safe": safe, "workers": workers,
        "swap_observable": False,
        "note": "engineering decision only; no scientific parameter depends on "
                "the worker count",
    }
    write_json_atomic(out / "resource_preflight.json", res)
    print("  concurrent: fits %s | wall %.1fs | free floor %.0f MB"
          % ([round(c, 1) for c in conc], conc_wall, conc_floor), flush=True)
    print("  slowdown %.2fx | throughput gain %.2fx -> workers = %d"
          % (slow, throughput_gain, workers), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
