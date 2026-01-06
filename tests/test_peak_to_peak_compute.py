from __future__ import annotations

from pathlib import Path

import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_compute_peak_to_peak_metrics_synthetic_signal():
    # Synthetic timebase (ms)
    times_ms = np.linspace(0.0, 500.0, 501)

    # Build synthetic ROI signal: +P1 at ~90ms, -N1 at ~170ms (µV scale)
    def gauss(x, mu, sigma, amp):
        return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    signal = (
        gauss(times_ms, 90.0, 12.0, 4.0)  # +4 µV peak near 90 ms (P1)
        - gauss(times_ms, 170.0, 14.0, 6.0)  # -6 µV peak near 170 ms (N1)
        + 0.1 * np.random.RandomState(0).normal(size=times_ms.size)  # small noise
    )

    p1_window = (60.0, 120.0)
    n1_window = (125.0, 200.0)

    import sys

    sys.path.insert(0, str(_repo_root() / "src"))
    from erp.measures import compute_peak_to_peak_metrics

    result = compute_peak_to_peak_metrics(
        signal=signal,
        times_ms=times_ms,
        p1_window_ms=p1_window,
        n1_window_ms=n1_window,
        p_polarity="positive",
        n_polarity="negative",
    )

    assert set(result.keys()) == {"p1_amp", "n1_amp", "p2p_amp", "p1_lat_ms", "n1_lat_ms"}
    assert result["p1_amp"] > 0.5  # clearly positive
    assert result["n1_amp"] < -0.5  # clearly negative
    # p2p = P1_amp - N1_amp (subtracting a negative increases the value)
    assert np.isclose(result["p2p_amp"], result["p1_amp"] - result["n1_amp"], rtol=0.05, atol=0.1)
    # Latencies within expected windows
    assert p1_window[0] <= result["p1_lat_ms"] <= p1_window[1]
    assert n1_window[0] <= result["n1_lat_ms"] <= n1_window[1]

