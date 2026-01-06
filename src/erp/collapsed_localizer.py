"""Collapsed localizers for ERP component window selection.

Implements:
- GFP collapsed localizer (reference-free)
- ROI collapsed localizer (polarity-aware)
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

import numpy as np

from .gfp_measures import compute_fwhm_window as _compute_fwhm_window_signal
from .gfp_measures import gfp_peak_and_window


def compute_fwhm_window(
    *, times_ms: np.ndarray, signal: np.ndarray, peak_idx: int
) -> tuple[float, float, float]:
    """Return (start_ms, end_ms, fwhm_ms) for a positized 1D signal peak."""
    start_ms, end_ms = _compute_fwhm_window_signal(signal, times_ms, peak_idx)
    return float(start_ms), float(end_ms), float(end_ms - start_ms)


def compute_collapsed_localizer_gfp(
    *,
    evokeds_by_set: Dict[str, List],
    component_name: str,
    search_range_ms: tuple[int, int],
) -> Dict[str, Any]:
    import mne

    all_evokeds: List = []
    for evoked_list in evokeds_by_set.values():
        all_evokeds.extend(evoked_list)
    if not all_evokeds:
        raise ValueError(f"No evoked data available for collapsed localizer ({component_name})")

    grand_avg = mne.combine_evoked(all_evokeds, weights="equal")
    data = grand_avg.get_data()
    times_ms = (grand_avg.times * 1000.0).astype(float)

    result = gfp_peak_and_window(
        data=data, times_ms=times_ms, search_range_ms=search_range_ms, component_name=component_name
    )
    result["times_ms"] = times_ms
    result["n_subjects"] = len(all_evokeds)
    result["n_conditions"] = len(evokeds_by_set)
    result["method"] = "gfp"
    return result


def compute_collapsed_localizer_roi(
    *,
    evokeds_by_set: Dict[str, List],
    roi_channels: Iterable[str],
    component_name: str,
    search_range_ms: tuple[int, int],
    polarity: str = "positive",
) -> Dict[str, Any]:
    import mne

    all_evokeds: List = []
    for evoked_list in evokeds_by_set.values():
        all_evokeds.extend(evoked_list)
    if not all_evokeds:
        raise ValueError(
            f"No evoked data available for collapsed localizer (ROI) ({component_name})"
        )

    grand_avg = mne.combine_evoked(all_evokeds, weights="equal")
    times_ms = (grand_avg.times * 1000.0).astype(float)

    try:
        roi_ev = grand_avg.copy().pick_channels(list(roi_channels), ordered=False)
    except Exception as e:
        raise ValueError(f"ROI localizer failed to pick channels: {e}")
    if roi_ev.data.size == 0:
        raise ValueError("ROI localizer has no data after picking channels")

    roi_curve = roi_ev.data.mean(axis=0) * 1e6  # µV

    start_ms, end_ms = search_range_ms
    mask = (times_ms >= start_ms) & (times_ms <= end_ms)
    if not np.any(mask):
        raise ValueError(f"No samples in ROI localizer search range [{start_ms}, {end_ms}] ms")

    pol = str(polarity).lower()
    positized = roi_curve if pol == "positive" else -roi_curve
    window_indices = np.where(mask)[0]
    peak_idx_in_window = int(np.argmax(positized[mask]))
    peak_idx = int(window_indices[peak_idx_in_window])

    win_start, win_end, fwhm_ms = compute_fwhm_window(times_ms=times_ms, signal=positized, peak_idx=peak_idx)
    peak_latency_ms = int(np.round(times_ms[peak_idx]))
    peak_amplitude = float(roi_curve[peak_idx])  # signed µV

    return {
        "trace": roi_curve,
        "trace_label": "ROI mean",
        "trace_units": "µV",
        "times_ms": times_ms,
        "peak_latency_ms": peak_latency_ms,
        "peak_amplitude": peak_amplitude,
        "window_start_ms": float(win_start),
        "window_end_ms": float(win_end),
        "fwhm_ms": float(fwhm_ms),
        "component_name": component_name,
        "n_subjects": len(all_evokeds),
        "n_conditions": len(evokeds_by_set),
        "search_range_ms": tuple(search_range_ms),
        "polarity": pol,
        "method": "roi",
    }

