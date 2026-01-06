from __future__ import annotations

from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

import glob
import os

import pandas as pd


def discover_epoch_files(root: str, pattern: str) -> List[str]:
    search_glob = os.path.join(root, pattern)
    return sorted(glob.glob(search_glob, recursive=True))


def read_epochs(file_path: str):
    """Read an MNE Epochs object from a FIF file (import on demand)."""
    import mne

    return mne.read_epochs(file_path, preload=False, verbose=False)


def _validate_montage_path(montage_sfp: str) -> None:
    if not os.path.isfile(montage_sfp):
        raise FileNotFoundError(f"Montage file not found: {montage_sfp}")


def apply_montage_sfp(epochs, montage_sfp: str) -> None:
    """Apply a custom montage from an .sfp file; raise actionable error on mismatched labels."""
    import mne

    _validate_montage_path(montage_sfp)
    montage = mne.channels.read_custom_montage(montage_sfp)

    # Validate channel names (case-insensitive)
    epoch_chs = [ch.upper() for ch in epochs.ch_names]
    montage_chs = [ch.upper() for ch in getattr(montage, "ch_names", [])]
    unmatched = [ch for ch in epoch_chs if ch not in montage_chs]
    if unmatched:
        raise ValueError(
            "Montage application failed: unmatched channel labels found. "
            f"Unmatched (sample): {unmatched[:10]} (total={len(unmatched)}); "
            f"montage={montage_sfp}"
        )

    epochs.set_montage(montage, match_case=False, on_missing="warn", verbose=False)


def validate_baseline_window(epochs, baseline_ms: Tuple[int, int]) -> None:
    start_ms, end_ms = baseline_ms
    tmin_ms = int(round(epochs.tmin * 1000.0))
    tmax_ms = int(round(epochs.tmax * 1000.0))
    if start_ms < tmin_ms or end_ms > tmax_ms:
        raise ValueError(
            f"Baseline [{start_ms}, {end_ms}] ms outside epoch range [{tmin_ms}, {tmax_ms}] ms"
        )


def extract_subject_id(epochs, file_path: str) -> str:
    """Best-effort subject id extraction.

    Priority:
    1) epochs.metadata['SubjectID'] if present
    2) filename pattern: sub-<ID>
    3) filename stem
    """
    import re

    md = getattr(epochs, "metadata", None)
    if md is not None and "SubjectID" in md.columns:
        vals = md["SubjectID"].astype(str).unique()
        if len(vals) > 0:
            return str(vals[0])

    m = re.search(r"sub-([A-Za-z0-9]+)", os.path.basename(file_path))
    if m:
        return m.group(1)
    return os.path.splitext(os.path.basename(file_path))[0]

