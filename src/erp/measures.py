"""ERP measurement utilities.

These are simple, deterministic signal measurements (peak/mean/FAL, peak-to-peak).
They are used after component windows are defined via the collapsed localizer.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def _window_mask(times_ms: np.ndarray, window_ms: Tuple[float, float]) -> np.ndarray:
    start, end = window_ms
    return (times_ms >= start) & (times_ms <= end)


def mean_amplitude(signal: np.ndarray, times_ms: np.ndarray, window_ms: Tuple[float, float]) -> float:
    mask = _window_mask(times_ms, window_ms)
    if not np.any(mask):
        raise ValueError(
            f"Window [{window_ms[0]}, {window_ms[1]}] ms has no samples "
            f"within provided times [{times_ms[0]:.1f}, {times_ms[-1]:.1f}] ms"
        )
    return float(np.mean(signal[mask]))


def peak_latency(
    signal: np.ndarray,
    times_ms: np.ndarray,
    window_ms: Tuple[float, float],
    polarity: str = "auto",
) -> float:
    mask = _window_mask(times_ms, window_ms)
    if not np.any(mask):
        raise ValueError(
            f"Window [{window_ms[0]}, {window_ms[1]}] ms has no samples "
            f"within provided times [{times_ms[0]:.1f}, {times_ms[-1]:.1f}] ms"
        )

    sig = signal[mask]
    tms = times_ms[mask]

    pol = str(polarity).lower()
    if pol == "auto":
        pol = "negative" if float(np.mean(sig)) < 0 else "positive"

    idx = int(np.argmin(sig) if pol == "negative" else np.argmax(sig))
    return float(tms[idx])


def peak_amplitude(
    signal: np.ndarray,
    times_ms: np.ndarray,
    window_ms: Tuple[float, float],
    polarity: str = "auto",
) -> float:
    mask = _window_mask(times_ms, window_ms)
    if not np.any(mask):
        raise ValueError(
            f"Window [{window_ms[0]}, {window_ms[1]}] ms has no samples "
            f"within provided times [{times_ms[0]:.1f}, {times_ms[-1]:.1f}] ms"
        )

    sig = signal[mask]
    pol = str(polarity).lower()
    if pol == "auto":
        pol = "negative" if float(np.mean(sig)) < 0 else "positive"

    return float(np.min(sig) if pol == "negative" else np.max(sig))


def fractional_area_latency(
    signal: np.ndarray,
    times_ms: np.ndarray,
    window_ms: Tuple[float, float],
    fraction: float = 0.5,
    polarity: str = "auto",
) -> float:
    if not 0.0 < float(fraction) < 1.0:
        raise ValueError(f"fraction must be between 0 and 1 (exclusive), got {fraction}")

    mask = _window_mask(times_ms, window_ms)
    if not np.any(mask):
        raise ValueError(
            f"Window [{window_ms[0]}, {window_ms[1]}] ms has no samples "
            f"within provided times [{times_ms[0]:.1f}, {times_ms[-1]:.1f}] ms"
        )

    signal_w = signal[mask]
    times_w = times_ms[mask]

    pol = str(polarity).lower()
    if pol == "auto":
        pol = "negative" if float(np.mean(signal_w)) < 0 else "positive"

    working = (-signal_w) if pol == "negative" else signal_w

    dt = np.diff(times_w)  # ms
    avg = (working[:-1] + working[1:]) / 2.0
    incremental = dt * avg
    cumulative = np.cumsum(incremental)
    cumulative = np.concatenate([[0.0], cumulative])
    total_area = float(cumulative[-1])

    if total_area <= 0:
        raise ValueError(
            f"Total area is non-positive ({total_area:.6f}). "
            f"Check window/polarity. Mean signal: {float(np.mean(signal_w)):.3f}"
        )

    target = float(fraction) * total_area
    idx = int(np.searchsorted(cumulative, target))
    if idx <= 0:
        return float(times_w[0])
    if idx >= len(cumulative):
        return float(times_w[-1])

    area_before = float(cumulative[idx - 1])
    area_after = float(cumulative[idx])
    time_before = float(times_w[idx - 1])
    time_after = float(times_w[idx])
    if area_after == area_before:
        return time_after
    frac_between = (target - area_before) / (area_after - area_before)
    return float(time_before + frac_between * (time_after - time_before))


def compute_peak_to_peak_metrics(
    signal: np.ndarray,
    times_ms: np.ndarray,
    p1_window_ms: Tuple[float, float],
    n1_window_ms: Tuple[float, float],
    p_polarity: str = "positive",
    n_polarity: str = "negative",
) -> dict:
    p1_mask = _window_mask(times_ms, p1_window_ms)
    n1_mask = _window_mask(times_ms, n1_window_ms)
    if not np.any(p1_mask):
        raise ValueError(f"P1 window [{p1_window_ms[0]}, {p1_window_ms[1]}] ms has no samples")
    if not np.any(n1_mask):
        raise ValueError(f"N1 window [{n1_window_ms[0]}, {n1_window_ms[1]}] ms has no samples")

    p1_lat = peak_latency(signal, times_ms, p1_window_ms, polarity=p_polarity)
    n1_lat = peak_latency(signal, times_ms, n1_window_ms, polarity=n_polarity)
    p1_amp = peak_amplitude(signal, times_ms, p1_window_ms, polarity=p_polarity)
    n1_amp = peak_amplitude(signal, times_ms, n1_window_ms, polarity=n_polarity)

    return {
        "p1_amp": float(p1_amp),
        "n1_amp": float(n1_amp),
        "p2p_amp": float(p1_amp - n1_amp),
        "p1_lat_ms": float(p1_lat),
        "n1_lat_ms": float(n1_lat),
    }

