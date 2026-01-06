from __future__ import annotations

from pathlib import Path

import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_compute_gfp_non_negative_and_length_matches_time():
    import sys

    sys.path.insert(0, str(_repo_root() / "src"))
    from erp.gfp_measures import compute_gfp

    import mne

    sfreq = 100.0
    times = np.arange(0, 1.0, 1.0 / sfreq)
    data = np.random.RandomState(0).normal(size=(4, times.size)) * 1e-6  # volts
    info = mne.create_info(ch_names=["E1", "E2", "E3", "E4"], sfreq=sfreq, ch_types="eeg")
    evoked = mne.EvokedArray(data, info, tmin=0.0, verbose=False)

    times_ms, gfp_uv = compute_gfp(evoked)
    assert len(times_ms) == len(gfp_uv) == times.size
    assert np.all(gfp_uv >= 0)

