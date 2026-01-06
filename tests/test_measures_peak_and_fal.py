from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _import_measures():
    import sys

    sys.path.insert(0, str(_repo_root() / "src"))
    from erp.measures import (
        fractional_area_latency,
        peak_amplitude,
        peak_latency,
    )

    return fractional_area_latency, peak_amplitude, peak_latency


def test_peak_latency_and_amplitude_polarity_aware():
    _, peak_amplitude, peak_latency = _import_measures()

    times_ms = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
    signal = np.array([0.0, 1.0, 0.0, -2.0, 0.0])
    window = (0.0, 40.0)

    assert peak_amplitude(signal, times_ms, window, polarity="positive") == 1.0
    assert peak_latency(signal, times_ms, window, polarity="positive") == 10.0

    assert peak_amplitude(signal, times_ms, window, polarity="negative") == -2.0
    assert peak_latency(signal, times_ms, window, polarity="negative") == 30.0


def test_fractional_area_latency_monotonic_case():
    fractional_area_latency, _, _ = _import_measures()

    times_ms = np.array([0.0, 1.0, 2.0, 3.0])
    signal = np.array([1.0, 1.0, 1.0, 1.0])

    lat = fractional_area_latency(signal, times_ms, window_ms=(0.0, 3.0), fraction=0.5, polarity="positive")
    assert lat == pytest.approx(1.5, abs=1e-6)

    lat_neg = fractional_area_latency(-signal, times_ms, window_ms=(0.0, 3.0), fraction=0.5, polarity="negative")
    assert lat_neg == pytest.approx(1.5, abs=1e-6)

