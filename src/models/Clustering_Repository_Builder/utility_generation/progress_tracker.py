"""Console progress reporter and aggregate counters for utility generation.

The tracker is invoked from the main process only. Workers return
:class:`DatasetWorkerResult` objects and the tracker accumulates them, mirroring
the verbose-every-N + one-liner-otherwise pattern used by the feature
extraction pipeline.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

from .utility_record_runner import RecordResult


def _fmt_elapsed(seconds: float) -> str:
    if not seconds or seconds < 0:
        return "00:00:00"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


@dataclass
class DatasetWorkerResult:
    """Aggregate returned from one dataset worker (one dataset, all views)."""

    dataset_id: str
    dataset_dir: str
    records: list[RecordResult] = field(default_factory=list)
    fatal_error: Optional[str] = None
    total_runtime_sec: float = 0.0


class ProgressTracker:
    def __init__(self, total_datasets: int, total_views: int, *, progress_every: int = 5) -> None:
        self.total_datasets = max(int(total_datasets), 0)
        self.total_views = max(int(total_views), 1)
        self.expected_records = self.total_datasets * self.total_views
        self.progress_every = max(int(progress_every), 1)

        self.start_time: float = 0.0
        self.datasets_done = 0
        self.datasets_ok = 0
        self.datasets_partial = 0
        self.datasets_failed = 0
        self.records_success = 0
        self.records_failed = 0
        self.records_skipped = 0
        self.fatal_datasets = 0
        self.dataset_times: list[float] = []

        self.completed_records: list[RecordResult] = []

    # -------------------------------------------------------------------- start
    def start(self, *, subfamily_dir: Path, metadata: dict) -> None:
        self.start_time = time.perf_counter()
        print("=" * 70, flush=True)
        print("Clustering Repository Utility Generation", flush=True)
        print("=" * 70, flush=True)
        print(flush=True)
        print(f"Subfamily directory:", flush=True)
        print(f"  {subfamily_dir}", flush=True)
        print(flush=True)
        for key in ("family_id", "subfamily_id"):
            v = metadata.get(key)
            if v:
                print(f"  {key}: {v}", flush=True)
        print(flush=True)
        print(f"Dataset folders found: {metadata.get('datasets_found', self.total_datasets)}", flush=True)
        print(f"Views per dataset: {self.total_views}", flush=True)
        print(f"Expected utility records: {self.expected_records}", flush=True)
        print(f"Metric mode: {metadata.get('metric_mode')}", flush=True)
        print(f"Sample size: {metadata.get('sample_size')}", flush=True)
        print(f"Random state: {metadata.get('random_state')}", flush=True)
        print(f"Max workers: {metadata.get('max_workers')}", flush=True)
        print(f"Resume: {metadata.get('resume')}", flush=True)
        print(f"Overwrite: {metadata.get('overwrite')}", flush=True)
        if metadata.get("dry_run"):
            print("Dry run: True (no ClustOpt will be executed)", flush=True)
        print(f"Progress every: {self.progress_every} datasets", flush=True)
        print(flush=True)

    # --------------------------------------------------------- on dataset done
    def on_dataset_done(self, idx: int, result: DatasetWorkerResult) -> None:
        self.datasets_done += 1
        self.dataset_times.append(float(result.total_runtime_sec))

        if result.fatal_error:
            self.fatal_datasets += 1
            self.datasets_failed += 1
            pct = (idx / max(self.total_datasets, 1)) * 100.0
            print(
                f"[{idx:04d}/{self.total_datasets:04d} | {pct:5.1f}%] FATAL "
                f"dataset={result.dataset_id} ({result.total_runtime_sec:.1f}s)",
                flush=True,
            )
            print(f"  error: {result.fatal_error.splitlines()[0]}", flush=True)
            print(f"  action: recorded; continuing with other datasets", flush=True)
            self._maybe_periodic_summary(idx)
            return

        view_lines: list[str] = []
        n_ok = n_skipped = n_failed = 0
        for rec in result.records:
            self.completed_records.append(rec)
            if rec.status == "success":
                self.records_success += 1
                n_ok += 1
            elif rec.status == "skipped":
                self.records_skipped += 1
                n_skipped += 1
            else:
                self.records_failed += 1
                n_failed += 1
            view_lines.append(self._format_view_line(rec))

        # Roll-up dataset status (matches the format we write inside each worker).
        if n_failed == 0 and n_ok + n_skipped == len(result.records):
            self.datasets_ok += 1
        elif n_ok == 0 and n_skipped == 0:
            self.datasets_failed += 1
        else:
            self.datasets_partial += 1

        pct = (idx / max(self.total_datasets, 1)) * 100.0
        avg = self._avg_dataset_time()
        eta = self._eta(idx)
        verbose = (idx == 1) or (idx % self.progress_every == 0) or (idx == self.total_datasets)

        if verbose:
            print(
                f"[{idx:04d}/{self.total_datasets:04d} | {pct:5.1f}%] dataset={result.dataset_id} "
                f"({result.total_runtime_sec:.1f}s)",
                flush=True,
            )
            for line in view_lines:
                print(line, flush=True)
            print(
                f"  totals: ok={self.records_success} failed={self.records_failed} "
                f"skipped={self.records_skipped} / expected={self.expected_records}",
                flush=True,
            )
            print(
                f"  elapsed: {_fmt_elapsed(time.perf_counter() - self.start_time)} "
                f"| avg dataset: {avg:.1f}s | ETA: {_fmt_elapsed(eta)}",
                flush=True,
            )
        else:
            # Concise one-liner with a per-dataset roll-up similar to features.
            print(
                f"[{idx:04d}/{self.total_datasets:04d} | {pct:5.1f}%] "
                f"{result.dataset_id} ok={n_ok} skipped={n_skipped} failed={n_failed} "
                f"time={result.total_runtime_sec:.1f}s "
                f"running ok/fail/skip={self.records_success}/{self.records_failed}/{self.records_skipped} "
                f"ETA={_fmt_elapsed(eta)}",
                flush=True,
            )

        self._maybe_periodic_summary(idx)

    # ---------------------------------------------------------------- finalize
    def finish(self) -> dict[str, object]:
        elapsed = time.perf_counter() - self.start_time
        return {
            "datasets_found": int(self.total_datasets),
            "datasets_processed": int(self.datasets_done),
            "datasets_ok": int(self.datasets_ok),
            "datasets_partial": int(self.datasets_partial),
            "datasets_failed": int(self.datasets_failed),
            "datasets_fatal": int(self.fatal_datasets),
            "records_expected": int(self.expected_records),
            "records_success": int(self.records_success),
            "records_failed": int(self.records_failed),
            "records_skipped": int(self.records_skipped),
            "records_success_rate": (
                float(self.records_success) / float(self.expected_records)
                if self.expected_records else 0.0
            ),
            "runtime_total_sec": float(elapsed),
            "runtime_total_str": _fmt_elapsed(elapsed),
            "avg_dataset_time_sec": float(self._avg_dataset_time()),
        }

    def print_final_summary(self, subfamily_id: str, output_files: Iterable[str]) -> None:
        elapsed = time.perf_counter() - self.start_time
        rate = (
            (self.records_success / self.expected_records * 100.0)
            if self.expected_records else 0.0
        )
        print(flush=True)
        print("=" * 70, flush=True)
        print("Utility generation completed", flush=True)
        print("=" * 70, flush=True)
        print(flush=True)
        print(f"Subfamily: {subfamily_id}", flush=True)
        print(f"Datasets: total={self.total_datasets} processed={self.datasets_done}", flush=True)
        print(
            f"  ok={self.datasets_ok} partial={self.datasets_partial} "
            f"failed={self.datasets_failed} fatal={self.fatal_datasets}",
            flush=True,
        )
        print(
            f"Records: ok={self.records_success} failed={self.records_failed} "
            f"skipped={self.records_skipped} / expected={self.expected_records} "
            f"(success rate={rate:.1f}%)",
            flush=True,
        )
        print(f"Total runtime: {_fmt_elapsed(elapsed)}", flush=True)
        print(f"Avg per dataset: {self._avg_dataset_time():.1f}s", flush=True)
        print(flush=True)
        print("Saved subfamily-level files:", flush=True)
        for f in output_files:
            print(f"  {f}", flush=True)

    # ------------------------------------------------------------------ helpers
    def _format_view_line(self, rec: RecordResult) -> str:
        if rec.status == "success":
            return (
                f"    {rec.view_id:<7} -> success  "
                f"({rec.runtime_sec:.1f}s, n_metrics={rec.n_metrics}, "
                f"top1={rec.top1_metric}={rec.top1_utility:.3f})"
            )
        if rec.status == "skipped":
            return f"    {rec.view_id:<7} -> skipped  (already complete)"
        # failed
        first = (rec.error_message or "").splitlines()[0] if rec.error_message else ""
        return f"    {rec.view_id:<7} -> FAILED   [{first[:160]}]"

    def _maybe_periodic_summary(self, idx: int) -> None:
        if idx == self.total_datasets:
            return
        if idx % self.progress_every != 0:
            return
        elapsed = time.perf_counter() - self.start_time
        avg = self._avg_dataset_time()
        eta = self._eta(idx)
        rate = (
            (self.records_success / self.expected_records * 100.0)
            if self.expected_records else 0.0
        )
        print(flush=True)
        print("Progress summary:", flush=True)
        print(f"  datasets done: {self.datasets_done} / {self.total_datasets}", flush=True)
        print(
            f"  dataset status: ok={self.datasets_ok} partial={self.datasets_partial} "
            f"failed={self.datasets_failed} fatal={self.fatal_datasets}",
            flush=True,
        )
        print(
            f"  records: ok={self.records_success} failed={self.records_failed} "
            f"skipped={self.records_skipped} / expected={self.expected_records} "
            f"(success rate={rate:.1f}%)",
            flush=True,
        )
        print(f"  elapsed: {_fmt_elapsed(elapsed)}", flush=True)
        print(f"  avg per dataset: {avg:.1f}s", flush=True)
        print(f"  ETA: {_fmt_elapsed(eta)}", flush=True)
        print(flush=True)

    def _avg_dataset_time(self) -> float:
        if not self.dataset_times:
            return 0.0
        return float(sum(self.dataset_times) / len(self.dataset_times))

    def _eta(self, processed: int) -> float:
        avg = self._avg_dataset_time()
        remaining = max(self.total_datasets - processed, 0)
        return avg * remaining

    @property
    def record_results(self) -> List[RecordResult]:
        return list(self.completed_records)

    @property
    def failed_records(self) -> List[RecordResult]:
        return [r for r in self.completed_records if r.status == "failed"]
