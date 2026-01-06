"""Global Field Power (GFP) utilities and FWHM windowing.

This module is used by collapsed localizers to define unbiased component windows.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np


def _compute_gfp_from_data(data: np.ndarray) -> np.ndarray:
    if data.ndim != 2:
        raise ValueError(f"Data must be 2D (n_channels, n_times), got shape {data.shape}")
    if data.shape[0] < 2:
        raise ValueError(f"Need at least 2 channels for GFP, got {data.shape[0]}")
    return np.std(data, axis=0, ddof=1)


def compute_gfp(evoked) -> Tuple[np.ndarray, np.ndarray]:
    """Compute GFP from an MNE Evoked and return (times_ms, gfp_uv)."""
    data = evoked.get_data()  # (n_channels, n_times) in Volts
    times_ms = (evoked.times * 1000.0).astype(float)
    gfp_uv = _compute_gfp_from_data(data) * 1e6
    return times_ms, gfp_uv.astype(float)


def find_gfp_peak(
    gfp: np.ndarray,
    times_ms: np.ndarray,
    search_range_ms: Tuple[int, int],
    min_gfp_ratio: float = 0.1,
) -> Tuple[int, float]:
    start_ms, end_ms = search_range_ms
    mask = (times_ms >= start_ms) & (times_ms <= end_ms)
    if not np.any(mask):
        raise ValueError(
            f"No samples found in search range [{start_ms}, {end_ms}] ms. "
            f"Data spans [{times_ms[0]:.1f}, {times_ms[-1]:.1f}] ms"
        )

    gfp_w = gfp[mask]
    times_w = times_ms[mask]
    peak_idx = int(np.argmax(gfp_w))
    peak_amp = float(gfp_w[peak_idx])
    peak_lat_ms = int(np.round(times_w[peak_idx]))

    global_max = float(np.max(gfp))
    if global_max > 0 and peak_amp < float(min_gfp_ratio) * global_max:
        raise ValueError(
            f"No meaningful GFP peak found in search range [{start_ms}, {end_ms}] ms. "
            f"Peak GFP ({peak_amp:.3f}) is less than {min_gfp_ratio*100}% of global maximum ({global_max:.3f})."
        )

    return peak_lat_ms, peak_amp


def compute_fwhm_window(
    signal: np.ndarray,
    times_ms: np.ndarray,
    peak_idx: int,
    min_width_samples: int = 5,
) -> Tuple[float, float]:
    """Compute FWHM window around a (positive) peak using half-maximum crossings."""
    if peak_idx < min_width_samples or peak_idx >= len(signal) - min_width_samples:
        peak_time = float(times_ms[peak_idx])
        epoch_start = float(times_ms[0])
        epoch_end = float(times_ms[-1])
        sample_duration = float(np.mean(np.diff(times_ms)))
        required_margin = float(min_width_samples) * sample_duration
        raise ValueError(
            f"Cannot compute FWHM: peak at {peak_time:.1f} ms is too close to epoch edge. "
            f"Epoch spans [{epoch_start:.1f}, {epoch_end:.1f}] ms. "
            f"Need ~{required_margin:.1f} ms margin on each side."
        )

    peak_val = float(signal[peak_idx])
    if peak_val <= 0:
        raise ValueError("FWHM requires a positive peak value (pass a positized signal).")
    half = 0.5 * peak_val

    # Left crossing
    li = peak_idx
    while li > 0 and float(signal[li]) >= half:
        li -= 1
    if li == peak_idx:
        raise ValueError("Could not find left half-maximum crossing.")
    # Crossing is between li and li+1
    l0, l1 = li, li + 1

    # Right crossing
    ri = peak_idx
    while ri < len(signal) - 1 and float(signal[ri]) >= half:
        ri += 1
    if ri == peak_idx:
        raise ValueError("Could not find right half-maximum crossing.")
    r0, r1 = ri - 1, ri

    def _interp_time(i0: int, i1: int) -> float:
        y0 = float(signal[i0])
        y1 = float(signal[i1])
        x0 = float(times_ms[i0])
        x1 = float(times_ms[i1])
        if y1 == y0:
            return x1
        frac = (half - y0) / (y1 - y0)
        return float(x0 + frac * (x1 - x0))

    start_ms = _interp_time(l0, l1)
    end_ms = _interp_time(r0, r1)
    if end_ms <= start_ms:
        raise ValueError(f"Invalid FWHM window: end ({end_ms:.3f}) <= start ({start_ms:.3f})")
    return start_ms, end_ms


def gfp_peak_and_window(
    data: np.ndarray,
    times_ms: np.ndarray,
    search_range_ms: Tuple[int, int],
    component_name: str = "",
) -> Dict[str, Any]:
    if data.ndim != 2:
        raise ValueError(f"Data must be 2D (n_channels, n_times), got shape {data.shape}")
    if data.shape[1] != len(times_ms):
        raise ValueError(
            f"Time dimension mismatch: data has {data.shape[1]} samples, times_ms has {len(times_ms)} points"
        )

    gfp_uv = _compute_gfp_from_data(data) * 1e6
    peak_lat_ms, peak_amp = find_gfp_peak(gfp_uv, times_ms, search_range_ms)
    peak_idx = int(np.argmin(np.abs(times_ms - peak_lat_ms)))
    win_start, win_end = compute_fwhm_window(gfp_uv, times_ms, peak_idx)

    return {
        "gfp": gfp_uv.astype(float),
        "peak_latency_ms": int(peak_lat_ms),
        "peak_amplitude": float(peak_amp),
        "window_start_ms": float(win_start),
        "window_end_ms": float(win_end),
        "fwhm_ms": float(win_end - win_start),
        "component_name": str(component_name),
        "search_range_ms": tuple(search_range_ms),
    }

