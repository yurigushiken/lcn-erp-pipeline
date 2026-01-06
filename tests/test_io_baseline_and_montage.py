from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _import_io():
    import sys

    sys.path.insert(0, str(_repo_root() / "src"))
    from erp.io import apply_montage_sfp, validate_baseline_window

    return apply_montage_sfp, validate_baseline_window


def _make_synthetic_epochs(tmin_s: float = -0.1, tmax_s: float = 0.4, sfreq: float = 100.0):
    import mne

    n_times = int(round((tmax_s - tmin_s) * sfreq)) + 1
    info = mne.create_info(ch_names=["E1", "E2"], sfreq=sfreq, ch_types=["eeg", "eeg"])
    data = np.zeros((1, 2, n_times), dtype=float)  # (n_epochs, n_ch, n_times)
    return mne.EpochsArray(data, info=info, tmin=tmin_s, verbose=False)


def test_validate_baseline_window_raises_when_outside():
    _, validate_baseline_window = _import_io()
    epochs = _make_synthetic_epochs(tmin_s=-0.1, tmax_s=0.4)

    with pytest.raises(ValueError) as e:
        validate_baseline_window(epochs, (-200, 0))  # starts before tmin
    assert "Baseline" in str(e.value)


def test_apply_montage_sfp_raises_if_missing_file(tmp_path: Path):
    apply_montage_sfp, _ = _import_io()
    epochs = _make_synthetic_epochs()
    missing = tmp_path / "missing.sfp"

    with pytest.raises(FileNotFoundError):
        apply_montage_sfp(epochs, str(missing))

