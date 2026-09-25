"""Numerical helpers — safe division, robust stats, entropy/gini."""
from __future__ import annotations

from typing import Iterable

import numpy as np

from ..config import EPS


def safe_div(num: float, den: float) -> float:
    if den is None:
        return np.nan
    try:
        if not np.isfinite(num) or not np.isfinite(den):
            return np.nan
        if abs(den) < EPS:
            return np.nan
        return float(num) / float(den)
    except Exception:
        return np.nan


def finite_values(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x).ravel()
    return arr[np.isfinite(arr)]


def safe_skew(x: np.ndarray) -> float:
    a = finite_values(x)
    if a.size < 3:
        return np.nan
    mean = a.mean()
    std = a.std(ddof=0)
    if std < EPS:
        return 0.0
    return float(np.mean(((a - mean) / std) ** 3))


def safe_kurtosis(x: np.ndarray) -> float:
    """Excess kurtosis."""
    a = finite_values(x)
    if a.size < 4:
        return np.nan
    mean = a.mean()
    std = a.std(ddof=0)
    if std < EPS:
        return 0.0
    return float(np.mean(((a - mean) / std) ** 4) - 3.0)


def normalized_entropy(counts: np.ndarray) -> float:
    """Shannon entropy of `counts` divided by log(num_bins). Returns NaN for
    empty input, 0 when only a single non-empty bin exists.
    """
    arr = np.asarray(counts, dtype=float).ravel()
    if arr.size == 0:
        return np.nan
    total = arr.sum()
    if total <= 0:
        return np.nan
    p = arr / total
    nz = p[p > 0]
    if nz.size <= 1:
        return 0.0
    h = -np.sum(nz * np.log(nz))
    return float(h / np.log(arr.size))


def gini(values: np.ndarray) -> float:
    """Gini coefficient of non-negative values. NaN if undefined."""
    a = finite_values(values)
    if a.size == 0:
        return np.nan
    a = np.abs(a)
    if a.sum() <= EPS:
        return 0.0
    a_sorted = np.sort(a)
    n = a_sorted.size
    cum = np.cumsum(a_sorted)
    return float((2.0 * np.sum((np.arange(1, n + 1)) * a_sorted) - (n + 1) * cum[-1]) / (n * cum[-1]))


def percentiles(x: np.ndarray, qs: Iterable[float]) -> dict[float, float]:
    a = finite_values(x)
    if a.size == 0:
        return {q: np.nan for q in qs}
    pct = np.percentile(a, list(qs))
    return {q: float(p) for q, p in zip(qs, pct)}


def mad_from_median(x: np.ndarray) -> float:
    a = finite_values(x)
    if a.size == 0:
        return np.nan
    med = np.median(a)
    return float(np.median(np.abs(a - med)))


def cv(values: np.ndarray) -> float:
    a = finite_values(values)
    if a.size == 0:
        return np.nan
    m = a.mean()
    if abs(m) < EPS:
        return np.nan
    return float(a.std(ddof=0) / m)


def safe_zscores(x: np.ndarray) -> np.ndarray:
    a = finite_values(x)
    if a.size == 0:
        return np.empty(0)
    m = a.mean()
    s = a.std(ddof=0)
    if s < EPS:
        return np.zeros_like(a)
    return (a - m) / s


def histogram_density(values: np.ndarray, bins: int) -> np.ndarray:
    a = finite_values(values)
    if a.size == 0:
        return np.zeros(bins)
    counts, _ = np.histogram(a, bins=bins)
    return counts.astype(float)


def safe_corrcoef(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size != y.size or x.size < 2:
        return np.nan
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return np.nan
    xs = x[mask]
    ys = y[mask]
    sx = xs.std(ddof=0)
    sy = ys.std(ddof=0)
    if sx < EPS or sy < EPS:
        return np.nan
    return float(np.mean((xs - xs.mean()) * (ys - ys.mean())) / (sx * sy))


def joint_histogram(x: np.ndarray, y: np.ndarray, bins: int) -> np.ndarray:
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return np.zeros((bins, bins))
    h, _, _ = np.histogram2d(x[mask], y[mask], bins=bins)
    return h.astype(float)


def shannon_entropy_from_probs(p: np.ndarray, base: float | None = None) -> float:
    nz = p[p > 0]
    if nz.size == 0:
        return 0.0
    h = -float(np.sum(nz * np.log(nz)))
    if base is not None:
        h /= np.log(base)
    return h


def mutual_information_hist(x: np.ndarray, y: np.ndarray, bins: int) -> float:
    h = joint_histogram(x, y, bins)
    total = h.sum()
    if total <= 0:
        return np.nan
    pxy = h / total
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    denom = px @ py
    mask = (pxy > 0) & (denom > 0)
    if not np.any(mask):
        return 0.0
    return float(np.sum(pxy[mask] * np.log(pxy[mask] / denom[mask])))


def safe_acf_peak(values: np.ndarray, max_lag: int | None = None) -> float:
    """Strongest absolute autocorrelation at lag > 0. NaN if not computable."""
    a = finite_values(values)
    if a.size < 4:
        return np.nan
    a = a - a.mean()
    var = float(np.dot(a, a))
    if var < EPS:
        return np.nan
    n = a.size
    if max_lag is None:
        max_lag = min(n - 1, 64)
    max_lag = max(1, int(max_lag))
    peak = 0.0
    for lag in range(1, max_lag + 1):
        num = float(np.dot(a[:-lag], a[lag:]))
        peak = max(peak, abs(num / var))
    return peak


def safe_fft_peak_ratio(values: np.ndarray) -> float:
    a = finite_values(values)
    if a.size < 4:
        return np.nan
    a = a - a.mean()
    spectrum = np.abs(np.fft.rfft(a))
    if spectrum.size < 2:
        return np.nan
    spectrum = spectrum[1:]
    if spectrum.size == 0:
        return np.nan
    peak = float(spectrum.max())
    total = float(spectrum.sum())
    if total < EPS:
        return np.nan
    return peak / total


def linear_r2(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan
    xs = x[mask]
    ys = y[mask]
    if xs.std(ddof=0) < EPS or ys.std(ddof=0) < EPS:
        return np.nan
    coeffs = np.polyfit(xs, ys, 1)
    preds = np.polyval(coeffs, xs)
    ss_res = float(np.sum((ys - preds) ** 2))
    ss_tot = float(np.sum((ys - ys.mean()) ** 2))
    return float(1.0 - ss_res / ss_tot) if ss_tot > EPS else np.nan


def poly_r2(x: np.ndarray, y: np.ndarray, degree: int) -> float:
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < degree + 2:
        return np.nan
    xs = x[mask]
    ys = y[mask]
    if xs.std(ddof=0) < EPS or ys.std(ddof=0) < EPS:
        return np.nan
    try:
        coeffs = np.polyfit(xs, ys, degree)
    except np.linalg.LinAlgError:
        return np.nan
    preds = np.polyval(coeffs, xs)
    ss_res = float(np.sum((ys - preds) ** 2))
    ss_tot = float(np.sum((ys - ys.mean()) ** 2))
    return float(1.0 - ss_res / ss_tot) if ss_tot > EPS else np.nan


def coerce_float(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return np.nan
    return out if np.isfinite(out) else np.nan
