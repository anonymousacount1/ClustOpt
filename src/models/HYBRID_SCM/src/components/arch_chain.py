from __future__ import annotations

import numpy as np

from ..priors import sample_param_dict
from .base import Component, register
from ._helpers import (
    _check_dims,
    _fill_irrelevant_dims,
    _generate_intervals_from_spec,
    _resolve_refs,
)


class ArchChain(Component):
    name = "arch_chain"

    def sample_params(self, rng, n_features, param_spec, parent_params):
        spec = _resolve_refs(param_spec, parent_params)
        s = sample_param_dict(rng, spec)
        i, j = _check_dims(s.get("dims", [0, 1]), n_features, "arch_chain")

        legacy_segments_spec = s.get("segments", None)
        if isinstance(legacy_segments_spec, (list, tuple)) and len(legacy_segments_spec) > 0 and all(isinstance(seg, dict) for seg in legacy_segments_spec):
            segments = []
            for seg in legacy_segments_spec:
                center = float(seg.get("center", seg.get("x0")))
                halfwidth = float(seg["halfwidth"])
                segments.append({
                    "center": center,
                    "halfwidth": halfwidth,
                    "height": float(seg["height"]),
                    "sharpness": float(seg.get("sharpness", 2.8)),
                    "mode": str(seg.get("mode", s.get("mode", "up_down"))).lower(),
                })
            x_low = float(min(seg["center"] - seg["halfwidth"] for seg in segments))
            x_high = float(max(seg["center"] + seg["halfwidth"] for seg in segments))
            return {
                "dims": [i, j],
                "legacy_segments": True,
                "segments": segments,
                "y_base": float(s.get("y_base", s.get("base", -66.0))),
                "trend_slope": float(s.get("trend_slope", 0.0)),
                "thickness": float(s.get("thickness", 0.25)),
                "x_low": x_low,
                "x_high": x_high,
                "irrelevant": s.get("irrelevant", {"mode": "gaussian_mid", "sigma": float(s.get("other_sigma", 0.25))}),
            }

        x_low = float(s.get("x_low", 3.6e6))
        x_high = float(s.get("x_high", 4.7e6))
        intervals = s.get("intervals", None)
        if intervals is None and s.get("interval_generation", None) is not None:
            intervals = _generate_intervals_from_spec(rng, s["interval_generation"], x_low=x_low, x_high=x_high)
        mode_raw = str(s.get("mode", "up_down")).lower()
        mode_aliases = {
            "nested": "up_down",
            "arch": "up_down",
            "arches": "up_down",
            "monotone": "monotone_up",
            "rise": "monotone_up",
            "rising": "monotone_up",
        }
        mode = mode_aliases.get(mode_raw, mode_raw)
        if mode not in ("up_down", "monotone_up"):
            raise ValueError(
                "arch_chain.mode must be one of: 'up_down', 'monotone_up' "
                "(aliases supported: 'nested', 'arch', 'arches', 'monotone', 'rise', 'rising')."
            )
        return {
            "dims": [i, j],
            "legacy_segments": False,
            "x_low": x_low,
            "x_high": x_high,
            "intervals": intervals,
            "y_base": float(s.get("y_base", s.get("base", -66.0))),
            "mode": mode,
            "amp_min": float(s.get("amp_min", 6.0)),
            "amp_max": float(s.get("amp_max", 34.0)),
            "amp_schedule": str(s.get("amp_schedule", "uniform")).lower(),
            "grow_power": float(s.get("grow_power", 1.6)),
            "grow_floor": float(s.get("grow_floor", 0.2)),
            "hump_power": float(s.get("hump_power", 1.0)),
            "sharpness_min": float(s.get("sharpness_min", 2.0)),
            "sharpness_max": float(s.get("sharpness_max", 4.5)),
            "thickness": float(s.get("thickness", 0.25)),
            "thickness_growth": float(s.get("thickness_growth", 0.0)),
            "thickness_schedule": str(s.get("thickness_schedule", "linear")).lower(),
            "flat_tail_frac": float(s.get("flat_tail_frac", 0.0)),
            "cap_to_bounds": bool(s.get("cap_to_bounds", True)),
            "bounds_margin": float(s.get("bounds_margin", 0.35)),
            "irrelevant": s.get("irrelevant", {"mode": "gaussian_mid", "sigma": float(s.get("other_sigma", 0.25))}),
        }

    def _amp_by_schedule(self, k, n, p):
        amin, amax = float(p["amp_min"]), float(p["amp_max"])
        sched = str(p.get("amp_schedule", "uniform")).lower()
        if n <= 1 or sched == "uniform":
            return None
        t = k / (n - 1)
        if sched == "increase":
            gp = max(1e-6, float(p.get("grow_power", 1.6)))
            gf = float(p.get("grow_floor", 0.2))
            frac = gf + (1.0 - gf) * (t ** gp)
            return amin + frac * (amax - amin)
        if sched == "hump":
            hp = max(1e-6, float(p.get("hump_power", 1.0)))
            h = max(0.0, 1.0 - abs(2.0 * t - 1.0)) ** hp
            return amin + h * (amax - amin)
        return None

    def sample_points(self, rng, n_points, n_features, bounds_low, bounds_high, params_used, parent_params=None):
        p = params_used
        i, j = _check_dims(p["dims"], n_features, "arch_chain")

        if bool(p.get("legacy_segments", False)):
            segments = p["segments"]
            widths = np.array([2.0 * seg["halfwidth"] for seg in segments], dtype=float)
            probs = widths / widths.sum()
            counts = rng.multinomial(n_points, probs)
            X = np.zeros((n_points, n_features), dtype=float)
            _fill_irrelevant_dims(rng, X, bounds_low, bounds_high, (i, j), p.get("irrelevant", None))
            out = 0
            for seg, m in zip(segments, counts):
                if m <= 0:
                    continue
                xa = seg["center"] - seg["halfwidth"]
                xb = seg["center"] + seg["halfwidth"]
                x = rng.uniform(xa, xb, size=m)
                t = (x - xa) / max(1e-12, xb - xa)
                mode = seg.get("mode", "up_down")
                sharp = float(seg.get("sharpness", 2.8))
                height = float(seg["height"])
                y_base = float(p.get("y_base", -66.0)) + float(p.get("trend_slope", 0.0)) * (seg["center"] - segments[0]["center"])
                if mode == "monotone_up":
                    y = y_base + height * (t ** sharp)
                else:
                    core = np.clip(1.0 - (2.0 * t - 1.0) ** 2, 0.0, 1.0)
                    y = y_base + height * (core ** sharp)
                y = y + rng.normal(0.0, float(p.get("thickness", 0.25)), size=m)
                X[out:out + m, i] = x
                X[out:out + m, j] = y
                out += m
            np.clip(X, bounds_low, bounds_high, out=X)
            return X

        intervals = p.get("intervals", None)
        if intervals is None or len(intervals) == 0:
            intervals = [[float(p["x_low"]), float(p["x_high"])]]
        lens = np.array([max(1e-12, float(b) - float(a)) for a, b in intervals], dtype=float)
        probs = lens / lens.sum()
        pts_per = rng.multinomial(int(n_points), probs)
        X = np.zeros((n_points, n_features), dtype=float)
        _fill_irrelevant_dims(rng, X, bounds_low, bounds_high, (i, j), p.get("irrelevant", None))
        y_base = float(p["y_base"])
        out = 0
        for k, ((xa, xb), m) in enumerate(zip(intervals, pts_per)):
            if m <= 0:
                continue
            x = rng.uniform(float(xa), float(xb), size=m)
            t = (x - float(xa)) / max(1e-12, float(xb) - float(xa))
            amp = self._amp_by_schedule(k, len(intervals), p)
            if amp is None:
                amp = rng.uniform(float(p["amp_min"]), float(p["amp_max"]))
            if bool(p.get("cap_to_bounds", True)):
                max_amp_allowed = float(bounds_high[j]) - float(p.get("bounds_margin", 0.35)) - y_base
                if max_amp_allowed > 0:
                    amp = min(amp, max_amp_allowed)
            sharp = rng.uniform(float(p["sharpness_min"]), float(p["sharpness_max"]))
            mode = str(p.get("mode", "up_down")).lower()
            if mode == "monotone_up":
                y = y_base + amp * (t ** sharp)
                ft = float(p.get("flat_tail_frac", 0.0))
                if ft > 0:
                    t0 = 1.0 - min(max(ft, 0.0), 0.95)
                    y0 = y_base + amp * (t0 ** sharp)
                    y = np.where(t >= t0, y0, y)
            else:
                core = np.clip(1.0 - (2.0 * t - 1.0) ** 2, 0.0, 1.0)
                y = y_base + amp * (core ** sharp)
            frac = k / max(1, len(intervals) - 1)
            thick = float(p["thickness"])
            if str(p.get("thickness_schedule", "linear")).lower() == "linear":
                thick = thick * (1.0 + float(p.get("thickness_growth", 0.0)) * frac)
            y = y + rng.normal(0.0, thick, size=m)
            X[out:out + m, i] = x
            X[out:out + m, j] = y
            out += m
        np.clip(X, bounds_low, bounds_high, out=X)
        return X


register(ArchChain())
