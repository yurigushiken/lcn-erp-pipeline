from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _import_stats():
    import sys

    sys.path.insert(0, str(_repo_root() / "src"))
    from erp.statistics import ERPStatistics

    return ERPStatistics


def test_filter_data_and_descriptives():
    ERPStatistics = _import_stats()

    df = pd.DataFrame(
        {
            "subject_id": ["01", "01", "02", "02"],
            "component": ["N1", "N1", "N1", "N1"],
            "condition": ["A", "B", "A", "B"],
            "snr": [1.0, 1.0, 2.0, 2.0],
            "mean_amplitude_roi": [1.0, 2.0, 1.5, 2.5],
        }
    )

    stats = ERPStatistics(df)
    filtered = stats.filter_data(component="N1", min_snr=1.5, dropna=True)
    assert set(filtered["subject_id"]) == {"02"}

    desc = stats.descriptives(component="N1", dv="mean_amplitude_roi", groupby="condition")
    assert set(desc.columns) >= {"condition", "count", "mean", "std", "sem", "min", "max"}


def test_run_lmm_returns_named_effects():
    ERPStatistics = _import_stats()

    rng = np.random.RandomState(0)
    subjects = [f"{i:02d}" for i in range(1, 11)]
    rows = []
    for s in subjects:
        for cond in ["A", "B"]:
            snr = 1.0 + rng.rand()
            y = 5.0 + (1.0 if cond == "B" else 0.0) + 0.5 * snr + rng.normal(scale=0.2)
            rows.append({"subject_id": s, "component": "N1", "condition": cond, "snr": snr, "dv": y})
    df = pd.DataFrame(rows)

    stats = ERPStatistics(df.rename(columns={"dv": "mean_amplitude_roi"}))
    res = stats.run_lmm(
        data=stats.filter_data(component="N1", dropna=True),
        dv="mean_amplitude_roi",
        fixed="condition + snr",
        random="subject_id",
        method="lbfgs",
    )

    names = [row.get("name") for row in res.get("summary", [])]
    assert "Intercept" in names
    assert any("condition" in str(n).lower() for n in names)
    assert any("snr" == str(n).lower() for n in names)

