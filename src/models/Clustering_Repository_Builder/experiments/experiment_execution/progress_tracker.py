"""Console progress reporter for experiment execution.

Counts at the run level (dataset × method × view) and prints periodic
high-level progress with elapsed time and ETA. Invoked from the main process
only; workers return :class:`DatasetWorkerResult` objects.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Iterable, List

from .dataset_worker import DatasetWorkerResult
from .method_runner import RunRecord


def _fmt_elapsed(seconds: float) -> str:
    if not seconds or seconds < 0:
        return "00:00:00"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class ExperimentProgressTracker:
    def __init__(
        self,
        *,
        total_datasets: int,
        runs_per_dataset: int,
        progress_every: int = 1,
    ) -> None:
        self.total_datasets = max(int(total_datasets), 0)
        self.runs_per_dataset = max(int(runs_per_dataset), 1)
        self.expected_runs = self.total_datasets * self.runs_per_dataset
        self.progress_every = max(int(progress_every), 1)

        self.start_time = 0.0
        self.datasets_done = 0
        self.runs_success = 0
        self.runs_failed = 0
        self.runs_skipped = 0
        self.fatal_datasets = 0
        self.dataset_times: List[float] = []
        self.completed_records: List[RunRecord] = []

    def start(self, *, metadata: Dict) -> None:
        self.start_time = time.perf_counter()
        print("=" * 70, flush=True)
        print("AutoClustering Experiment Execution", flush=True)
        print("=" * 70, flush=True)
        print(f"Subfamily dir:  {metadata.get('subfamily_dir')}", flush=True)
        print(f"Split id:       {metadata.get('split_id')}", flush=True)
        print(f"Datasets in subfamily: {metadata.get('datasets_discovered')}", flush=True)
        print(f"Datasets selected for split: {self.total_datasets}", flush=True)
        print(f"Methods ({len(metadata.get('methods', []))}): "
              f"{', '.join(metadata.get('methods', []))}", flush=True)
        print(f"Views ({len(metadata.get('views', []))}): "
              f"{', '.join(metadata.get('views', []))}", flush=True)
        print(f"Runs per dataset: {self.runs_per_dataset}", flush=True)
        print(f"Expected records: {self.total_datasets} datasets × 3 views = "
              f"{self.total_datasets * 3}", flush=True)
        print(f"Expected runs:    {self.expected_runs}", flush=True)
        print(f"Max workers: {metadata.get('max_workers')}", flush=True)
        print(f"Resume: {metadata.get('resume')} | Overwrite: {metadata.get('overwrite')} "
              f"| Dry-run: {metadata.get('dry_run')}", flush=True)
        print(f"Metric mode: {metadata.get('metric_mode')} "
              f"(sample={metadata.get('sample_size')})", flush=True)
        print(flush=True)

    def on_dataset_done(self, idx: int, result: DatasetWorkerResult) -> None:
        self.datasets_done += 1
        self.dataset_times.append(float(result.total_runtime_sec))

        if result.fatal_error:
            self.fatal_datasets += 1
            print(f"[dataset {idx}/{self.total_datasets}] FATAL "
                  f"{result.dataset_id}: {result.fatal_error.splitlines()[0]}", flush=True)
            print("  action: recorded; continuing.", flush=True)
            return

        n_ok = n_failed = n_skipped = 0
        for rec in result.records:
            self.completed_records.append(rec)
            if rec.status == "success":
                self.runs_success += 1; n_ok += 1
            elif rec.status == "skipped":
                self.runs_skipped += 1; n_skipped += 1
            else:
                self.runs_failed += 1; n_failed += 1

        runs_done = self.runs_success + self.runs_failed + self.runs_skipped
        eta = self._eta(self.datasets_done)
        verbose = (idx == 1) or (idx % self.progress_every == 0) or (idx == self.total_datasets)
        if verbose:
            print(
                f"[dataset {idx:>4}/{self.total_datasets}] {result.dataset_id} "
                f"({result.total_runtime_sec:.1f}s)  "
                f"completed={n_ok} skipped={n_skipped} failed={n_failed}",
                flush=True,
            )
            print(
                f"  Global: datasets_done={self.datasets_done}/{self.total_datasets} | "
                f"runs_done={runs_done}/{self.expected_runs} | "
                f"ok={self.runs_success} failed={self.runs_failed} skipped={self.runs_skipped}",
                flush=True,
            )
            print(
                f"  Elapsed={_fmt_elapsed(time.perf_counter() - self.start_time)} | "
                f"avg_dataset={self._avg():.1f}s | ETA={_fmt_elapsed(eta)}",
                flush=True,
            )

    def finish(self) -> Dict[str, object]:
        elapsed = time.perf_counter() - self.start_time
        return {
            "datasets_selected": int(self.total_datasets),
            "datasets_processed": int(self.datasets_done),
            "datasets_fatal": int(self.fatal_datasets),
            "expected_runs": int(self.expected_runs),
            "runs_success": int(self.runs_success),
            "runs_failed": int(self.runs_failed),
            "runs_skipped": int(self.runs_skipped),
            "runs_success_rate": (
                self.runs_success / self.expected_runs if self.expected_runs else 0.0
            ),
            "runtime_total_sec": float(elapsed),
            "runtime_total_str": _fmt_elapsed(elapsed),
            "avg_dataset_time_sec": float(self._avg()),
        }

    def print_final_summary(self, output_files: Iterable[str]) -> None:
        elapsed = time.perf_counter() - self.start_time
        print(flush=True)
        print("=" * 70, flush=True)
        print("Experiment execution completed", flush=True)
        print("=" * 70, flush=True)
        print(f"Datasets: selected={self.total_datasets} processed={self.datasets_done} "
              f"fatal={self.fatal_datasets}", flush=True)
        print(f"Runs: ok={self.runs_success} failed={self.runs_failed} "
              f"skipped={self.runs_skipped} / expected={self.expected_runs}", flush=True)
        print(f"Total runtime: {_fmt_elapsed(elapsed)} | avg/dataset={self._avg():.1f}s", flush=True)
        print("Saved execution summary files:", flush=True)
        for f in output_files:
            print(f"  {f}", flush=True)

    @property
    def record_results(self) -> List[RunRecord]:
        return list(self.completed_records)

    @property
    def failed_records(self) -> List[RunRecord]:
        return [r for r in self.completed_records if r.status == "failed"]

    def _avg(self) -> float:
        return float(sum(self.dataset_times) / len(self.dataset_times)) if self.dataset_times else 0.0

    def _eta(self, processed: int) -> float:
        return self._avg() * max(self.total_datasets - processed, 0)
