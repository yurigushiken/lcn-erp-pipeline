from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_compute_fwhm_window_synthetic_gaussian():
    import sys

    sys.path.insert(0, str(_repo_root() / "src"))
    from erp.collapsed_localizer import compute_fwhm_window

    times_ms = np.linspace(0.0, 500.0, 501)
    mu = 200.0
    sigma = 20.0
    signal = np.exp(-0.5 * ((times_ms - mu) / sigma) ** 2)

    peak_idx = int(np.argmax(signal))
    start_ms, end_ms, fwhm_ms = compute_fwhm_window(times_ms=times_ms, signal=signal, peak_idx=peak_idx)

    assert start_ms < mu < end_ms
    assert end_ms > start_ms
    assert fwhm_ms == pytest.approx(end_ms - start_ms, abs=1e-9)

