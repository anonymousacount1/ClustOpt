"""Progress reporting / timing tracker for the feature extraction runner.

Centralizes:
- start-of-run summary
- per-dataset status (verbose every `progress_every`, one-liner otherwise)
- failure status
- periodic aggregate summary
- final summary
- aggregate dataset and block timing
"""
from __future__ import annotations

import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


def _fmt_elapsed(seconds: float) -> str:
    if not seconds or seconds < 0:
        return "00:00:00"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


@dataclass
class DatasetTiming:
    load_time_sec: float = 0.0
    view_build_time_sec: float = 0.0
    block_A_time_sec: float = 0.0
    block_B_time_sec: float = 0.0
    block_G_time_sec: float = 0.0
    block_V_time_sec: float = 0.0
    block_R_time_sec: float = 0.0
    block_L_time_sec: float = 0.0
    save_time_sec: float = 0.0
    total_time_sec: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass
class _FailureRecord:
    dataset_dir: str
    dataset_id: str
    error_stage: str
    error_message: str
    traceback_short: str


class ProgressTracker:
    """Console + summary aggregator. Single-threaded for now."""

    def __init__(self, total_datasets: int, progress_every: int = 10) -> None:
        self.total_datasets = max(int(total_datasets), 0)
        self.progress_every = max(int(progress_every), 1)
        self.start_time: float = 0.0
        self.dataset_times: list[float] = []
        self.records_created = 0
        self.processed_ok = 0
        self.failed = 0
        self.skipped = 0
        self.failures: list[_FailureRecord] = []
        self.block_totals: dict[str, float] = {
            "block_A_time_sec": 0.0,
            "block_B_time_sec": 0.0,
            "block_G_time_sec": 0.0,
            "block_V_time_sec": 0.0,
            "block_R_time_sec": 0.0,
            "block_L_time_sec": 0.0,
            "load_time_sec": 0.0,
            "view_build_time_sec": 0.0,
            "save_time_sec": 0.0,
        }
        self._current_t0: Optional[float] = None
        self._current_dataset_id: Optional[str] = None
        self._verbose_flag: bool = True

    # ------------------------------------------------------------------ start
    def start(self, subfamily_root: Path, metadata: dict) -> None:
        self.start_time = time.perf_counter()
        print("=" * 60)
        print("Clustering Repository Feature Extraction")
        print("=" * 60)
        print()
        print(f"Subfamily root:")
        print(f"  {subfamily_root}")
        print()
        print(f"Detected:")
        for key in ("family_id", "subfamily_id"):
            if metadata.get(key) is not None:
                print(f"  {key}: {metadata[key]}")
        print()
        print(f"Dataset folders found: {metadata.get('datasets_found', self.total_datasets)}")
        print(f"Expected feature records: {metadata.get('expected_records', self.total_datasets * 3)}")
        print(f"Views per dataset: 3")
        print(f"Feature columns per record: {metadata.get('feature_count', 250)}")
        print(f"Overwrite: {metadata.get('overwrite', False)}")
        print(f"Skip landmarking: {metadata.get('skip_landmarking', False)}")
        print(f"Progress every: {self.progress_every}")
        print()
        print(f"Output files:")
        for fname in metadata.get("output_files", []):
            print(f"  {fname}")
        print()

    # ------------------------------------------------------------- per dataset
    def start_dataset(self, index: int, dataset_id: str) -> None:
        self._current_t0 = time.perf_counter()
        self._current_dataset_id = dataset_id
        self._verbose_flag = (index == 1) or (index % self.progress_every == 0) or (index == self.total_datasets)
        if self._verbose_flag:
            pct = (index / max(self.total_datasets, 1)) * 100.0
            print(f"[{index:03d}/{self.total_datasets:03d} | {pct:5.1f}%] dataset={dataset_id}")

    def stage(self, label: str, ok: bool = True, detail: str = "", elapsed: Optional[float] = None) -> None:
        if not self._verbose_flag:
            return
        status = "ok" if ok else "FAIL"
        suffix = f" ({elapsed:.2f}s)" if elapsed is not None else ""
        extra = f": {detail}" if detail else ""
        print(f"  {label}... {status}{extra}{suffix}")

    # ------------------------------------------------------------------ finish
    def finish_dataset_ok(self, index: int, dataset_id: str, records: int, timing: dict[str, float]) -> None:
        elapsed = (timing.get("total_time_sec") or (time.perf_counter() - (self._current_t0 or time.perf_counter())))
        self.dataset_times.append(float(elapsed))
        self.processed_ok += 1
        self.records_created += int(records)
        for key in self.block_totals.keys():
            v = float(timing.get(key, 0.0) or 0.0)
            self.block_totals[key] += v

        avg = self._avg()
        throughput = self._throughput()
        eta = self._eta(index)
        if self._verbose_flag:
            print(f"  records: {records}")
            print(f"  elapsed dataset: {elapsed:.2f}s")
            print(f"  avg: {avg:.2f}s/dataset | throughput: {throughput:.1f} datasets/min | ETA: {_fmt_elapsed(eta)}")
        else:
            pct = (index / max(self.total_datasets, 1)) * 100.0
            print(
                f"[{index:03d}/{self.total_datasets:03d} | {pct:5.1f}%] ok dataset={dataset_id} "
                f"records={records} time={elapsed:.2f}s avg={avg:.2f}s ETA={_fmt_elapsed(eta)}"
            )

        if index % self.progress_every == 0 and index != self.total_datasets:
            self.print_periodic_summary()

    def finish_dataset_failed(
        self,
        index: int,
        dataset_id: str,
        stage: str,
        error: BaseException,
        dataset_dir: Optional[str] = None,
    ) -> None:
        self.failed += 1
        pct = (index / max(self.total_datasets, 1)) * 100.0
        err_message = f"{type(error).__name__}: {error}"
        tb_short = "".join(traceback.format_exception_only(type(error), error)).strip()
        self.failures.append(
            _FailureRecord(
                dataset_dir=str(dataset_dir or ""),
                dataset_id=str(dataset_id),
                error_stage=stage,
                error_message=err_message,
                traceback_short=tb_short,
            )
        )
        print(f"[{index:03d}/{self.total_datasets:03d} | {pct:5.1f}%] failed dataset={dataset_id}")
        print(f"  stage: {stage}")
        print(f"  error: {err_message}")
        print(f"  action: recorded in failed_datasets.csv and continuing")

    def skip_dataset(self, index: int, dataset_id: str, reason: str = "") -> None:
        self.skipped += 1
        if not reason:
            reason = "already extracted"
        print(f"[{index:03d}/{self.total_datasets:03d}] skipped dataset={dataset_id} ({reason})")

    # ----------------------------------------------------------- periodic agg
    def print_periodic_summary(self) -> None:
        elapsed = time.perf_counter() - self.start_time
        avg = self._avg()
        throughput = self._throughput()
        processed = self.processed_ok + self.failed + self.skipped
        eta = self._eta(processed)
        print()
        print("Progress summary:")
        print(f"  processed: {processed} / {self.total_datasets}")
        print(f"  ok: {self.processed_ok}")
        print(f"  failed: {self.failed}")
        print(f"  skipped: {self.skipped}")
        print(f"  records created: {self.records_created}")
        print(f"  elapsed: {_fmt_elapsed(elapsed)}")
        print(f"  avg time: {avg:.2f}s/dataset")
        print(f"  throughput: {throughput:.1f} datasets/min")
        print(f"  ETA: {_fmt_elapsed(eta)}")
        print()

    # ---------------------------------------------------------------- finals
    def finish(self) -> dict[str, object]:
        elapsed = time.perf_counter() - self.start_time
        avg = self._avg()
        throughput = self._throughput()
        return {
            "datasets_found": int(self.total_datasets),
            "processed_successfully": int(self.processed_ok),
            "failed": int(self.failed),
            "skipped": int(self.skipped),
            "records_created": int(self.records_created),
            "runtime_total_sec": float(elapsed),
            "runtime_total_str": _fmt_elapsed(elapsed),
            "avg_time_per_dataset_sec": float(avg),
            "throughput_datasets_per_min": float(throughput),
            "block_timing_totals_sec": dict(self.block_totals),
            "block_timing_means_sec": {
                k: (v / self.processed_ok) if self.processed_ok else 0.0
                for k, v in self.block_totals.items()
            },
            "failures": [asdict(f) for f in self.failures],
        }

    def print_final_summary(self, subfamily_id: str, output_files: list[str], expected_records: int, parquet_status: str) -> None:
        elapsed = time.perf_counter() - self.start_time
        print()
        print("=" * 60)
        print("Feature extraction completed")
        print("=" * 60)
        print()
        print(f"Subfamily: {subfamily_id}")
        print(f"Datasets found: {self.total_datasets}")
        print(f"Processed successfully: {self.processed_ok}")
        print(f"Failed: {self.failed}")
        print(f"Skipped: {self.skipped}")
        print()
        print(f"Feature records created: {self.records_created}")
        print(f"Expected records from successful datasets: {expected_records}")
        print(f"Feature columns per record: 250")
        print()
        print(f"Runtime:")
        print(f"  total elapsed: {_fmt_elapsed(elapsed)}")
        print(f"  avg time per dataset: {self._avg():.2f}s")
        print(f"  throughput: {self._throughput():.1f} datasets/min")
        print()
        print(f"Saved:")
        for fname in output_files:
            print(f"  {fname}")
        print(f"  parquet_status: {parquet_status}")

    # ---------------------------------------------------------------- helpers
    def _avg(self) -> float:
        if not self.dataset_times:
            return 0.0
        return float(sum(self.dataset_times) / len(self.dataset_times))

    def _throughput(self) -> float:
        elapsed = time.perf_counter() - self.start_time
        if elapsed <= 0 or self.processed_ok == 0:
            return 0.0
        return float((self.processed_ok / elapsed) * 60.0)

    def _eta(self, processed: int) -> float:
        avg = self._avg()
        remaining = max(self.total_datasets - processed, 0)
        return avg * remaining
