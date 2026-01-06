from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional

import pandas as pd


def filter_by_response(
    metadata: pd.DataFrame, response: str, response_field: str = "Target.ACC"
) -> pd.DataFrame:
    """Filter rows by response mode.

    Response modes:
      - ALL: return metadata unchanged
      - ACC1: keep rows where response_field == 1
      - ACC0: keep rows where response_field == 0
    """
    mode = str(response).upper()
    if mode == "ALL":
        return metadata

    if response_field not in metadata.columns:
        raise ValueError(f"Required metadata column missing: {response_field!r}")

    if mode == "ACC1":
        return metadata.loc[metadata[response_field] == 1]
    if mode == "ACC0":
        return metadata.loc[metadata[response_field] == 0]

    raise ValueError(f"Unknown response mode: {response}. Valid options: ALL, ACC1, ACC0")


def build_condition_mask(
    metadata: pd.DataFrame, condition_codes: List[str], condition_field: str = "Condition"
) -> pd.Series:
    if condition_field not in metadata.columns:
        raise ValueError(f"Required metadata column missing: {condition_field!r}")
    if not condition_codes:
        return pd.Series(True, index=metadata.index)
    return metadata[condition_field].astype(str).isin([str(c) for c in condition_codes])


def build_metadata_filter_mask(
    metadata: pd.DataFrame, filters: Mapping[str, Iterable[str]], set_name: str
) -> pd.Series:
    mask: Optional[pd.Series] = None
    for column, raw_values in filters.items():
        if column not in metadata.columns:
            raise ValueError(
                f"Condition set '{set_name}' references missing metadata column: '{column}'"
            )
        values = [str(v) for v in raw_values]
        col_mask = metadata[column].astype(str).isin(values)
        mask = col_mask if mask is None else (mask & col_mask)
    return mask if mask is not None else pd.Series(True, index=metadata.index)


def filter_epochs_by_response(epochs, response: str, response_field: str = "Target.ACC"):
    if str(response).upper() == "ALL":
        return epochs

    md = getattr(epochs, "metadata", None)
    if md is None:
        raise ValueError("Epochs metadata is required for response filtering")

    mode = str(response).upper()
    if response_field not in md.columns:
        raise ValueError(f"Required metadata column missing: {response_field!r}")

    if mode == "ACC1":
        mask = (md[response_field] == 1).to_numpy()
        return epochs[mask]
    if mode == "ACC0":
        mask = (md[response_field] == 0).to_numpy()
        return epochs[mask]

    raise ValueError(f"Unknown response mode: {response}. Valid options: ALL, ACC1, ACC0")


def select_epochs(
    epochs,
    *,
    condition_codes: List[str],
    metadata_filters: Optional[Mapping[str, Iterable[str]]],
    condition_field: str = "Condition",
    set_name: str = "condition_set",
):
    md = getattr(epochs, "metadata", None)
    if md is None:
        raise ValueError("Epochs metadata is required for condition selection")

    cond_mask = build_condition_mask(
        md, condition_codes=[str(c) for c in condition_codes], condition_field=condition_field
    )
    mf = metadata_filters or {}
    meta_mask = build_metadata_filter_mask(md, mf, set_name=set_name)
    combined = (cond_mask & meta_mask).astype(bool)
    return epochs[combined]


__all__ = [
    "filter_by_response",
    "build_condition_mask",
    "build_metadata_filter_mask",
    "filter_epochs_by_response",
    "select_epochs",
]

