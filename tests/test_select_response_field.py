from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _import_select():
    import sys

    sys.path.insert(0, str(_repo_root() / "src"))
    from erp.select import filter_by_response, select_epochs, filter_epochs_by_response

    return filter_by_response, select_epochs, filter_epochs_by_response


def test_filter_by_response_all_noop():
    filter_by_response, _, _ = _import_select()

    df = pd.DataFrame({"ACC": [1, 0, 1]})
    out = filter_by_response(df, "ALL", response_field="ACC")
    assert len(out) == 3


def test_filter_by_response_acc1_acc0_uses_configured_field():
    filter_by_response, _, _ = _import_select()

    df = pd.DataFrame({"ACC": [1, 0, 1]})
    acc1 = filter_by_response(df, "ACC1", response_field="ACC")
    acc0 = filter_by_response(df, "ACC0", response_field="ACC")
    assert acc1["ACC"].tolist() == [1, 1]
    assert acc0["ACC"].tolist() == [0]


def test_filter_by_response_raises_if_field_missing():
    filter_by_response, _, _ = _import_select()

    df = pd.DataFrame({"Other": [1, 0, 1]})
    with pytest.raises(ValueError) as e:
        filter_by_response(df, "ACC1", response_field="ACC")
    assert "Required metadata column missing" in str(e.value)


def test_select_epochs_by_condition_field():
    _, select_epochs, _ = _import_select()
    import mne

    info = mne.create_info(ch_names=["E1"], sfreq=100.0, ch_types=["eeg"])
    data = np.zeros((3, 1, 10), dtype=float)
    epochs = mne.EpochsArray(data, info=info, tmin=-0.1, verbose=False)
    epochs.metadata = pd.DataFrame({"CondX": ["12", "23", "12"]})

    sub = select_epochs(epochs, condition_codes=["12"], metadata_filters=None, condition_field="CondX")
    assert len(sub) == 2


def test_filter_epochs_by_response_uses_epochs_mask():
    _, _, filter_epochs_by_response = _import_select()
    import mne

    info = mne.create_info(ch_names=["E1"], sfreq=100.0, ch_types=["eeg"])
    data = np.zeros((3, 1, 10), dtype=float)
    epochs = mne.EpochsArray(data, info=info, tmin=-0.1, verbose=False)
    epochs.metadata = pd.DataFrame({"ACC": [1, 0, 1]})

    out = filter_epochs_by_response(epochs, response="ACC1", response_field="ACC")
    assert len(out) == 2

