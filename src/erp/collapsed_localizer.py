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


def _collapse_per_subject_then_average(evokeds_by_set: Dict[str, List]):
    """
    Scientific default: collapse within each subject across conditions (equal weight),
    then average across subjects (equal weight).

    This avoids overweighting subjects who contribute more evokeds (e.g., due to missingness).

    Requires per-evoked subject id. We use Evoked.comment as a lightweight subject-id tag
    (set in scripts/run_analysis.py). If subject ids cannot be inferred, we fall back to
    the legacy behavior (combine all evokeds equally).
    """
    import mne
    from collections import defaultdict

    all_evokeds: list = []
    by_subject: dict[str, list] = defaultdict(list)
    has_any_sid = False
    has_missing_sid = False

    for evoked_list in evokeds_by_set.values():
        for evk in evoked_list:
            all_evokeds.append(evk)
            sid = str(getattr(evk, "comment", "") or "").strip()
            if sid:
                has_any_sid = True
                by_subject[sid].append(evk)
            else:
                has_missing_sid = True

    if not all_evokeds:
        return [], 0, 0

    # Only use per-subject collapsing if we have usable subject ids for all evokeds
    # and at least 2 distinct subjects (otherwise it's equivalent).
    if not has_any_sid or has_missing_sid or len(by_subject) < 2:
        grand = mne.combine_evoked(all_evokeds, weights="equal")
        return [grand], 1, len(all_evokeds)

    # Collapse within subject, then average subjects
    subj_collapsed = [mne.combine_evoked(v, weights="equal") for v in by_subject.values() if v]
    if not subj_collapsed:
        grand = mne.combine_evoked(all_evokeds, weights="equal")
        return [grand], 1, len(all_evokeds)

    grand = mne.combine_evoked(subj_collapsed, weights="equal")
    return [grand], len(subj_collapsed), len(all_evokeds)


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

    combined, n_subjects, n_evokeds = _collapse_per_subject_then_average(evokeds_by_set)
    if not combined:
        raise ValueError(f"No evoked data available for collapsed localizer ({component_name})")

    grand_avg = combined[0]
    data = grand_avg.get_data()
    times_ms = (grand_avg.times * 1000.0).astype(float)

    result = gfp_peak_and_window(
        data=data, times_ms=times_ms, search_range_ms=search_range_ms, component_name=component_name
    )
    result["times_ms"] = times_ms
    result["n_subjects"] = int(n_subjects)
    result["n_conditions"] = len(evokeds_by_set)
    result["n_evokeds"] = int(n_evokeds)
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

    combined, n_subjects, n_evokeds = _collapse_per_subject_then_average(evokeds_by_set)
    if not combined:
        raise ValueError(
            f"No evoked data available for collapsed localizer (ROI) ({component_name})"
        )

    grand_avg = combined[0]
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
        "n_subjects": int(n_subjects),
        "n_conditions": len(evokeds_by_set),
        "n_evokeds": int(n_evokeds),
        "search_range_ms": tuple(search_range_ms),
        "polarity": pol,
        "method": "roi",
    }

