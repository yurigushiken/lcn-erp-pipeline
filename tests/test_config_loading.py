from __future__ import annotations

from pathlib import Path

import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _import_loaders():
    import sys

    repo_root = _repo_root()
    sys.path.insert(0, str(repo_root / "src"))
    from erp.config import load_analysis_config, load_components_config, load_electrodes_config

    return load_analysis_config, load_components_config, load_electrodes_config


def test_load_analysis_config_requires_keys(tmp_path: Path):
    load_analysis_config, _, _ = _import_loaders()

    p = tmp_path / "bad.yaml"
    p.write_text("dataset: {}\n", encoding="utf-8")

    with pytest.raises(ValueError) as e:
        load_analysis_config(str(p))
    assert "Missing required key" in str(e.value)


def test_load_analysis_config_optional_measurement(tmp_path: Path):
    load_analysis_config, _, _ = _import_loaders()

    p = tmp_path / "ok.yaml"
    p.write_text(
        "\n".join(
            [
                "dataset: {root: data, pattern: '*.fif', montage_sfp: 'assets/net/AdultAverageNet128_v1.sfp'}",
                "selection: {response: ALL, response_field: Target.ACC, min_epochs_per_set: 1, condition_sets: []}",
                "components: [N1]",
                "preprocessing: {baseline_ms: [-100, 0]}",
                "roi: {min_channels: 1}",
                "plots: {formats: [png], dpi: 100}",
                "outputs: {plots_dir: 'docs/assets/plots/x', tables_dir: 'docs/assets/tables/x'}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    cfg = load_analysis_config(str(p))
    assert isinstance(cfg.measurement, dict)
    assert cfg.measurement == {}


def test_load_components_config_shape():
    _, load_components_config, _ = _import_loaders()
    comps = load_components_config(str(_repo_root()))
    assert "N1" in comps
    assert isinstance(comps["N1"], dict)
    assert "window_ms" in comps["N1"]


def test_load_electrodes_config_shape():
    _, _, load_electrodes_config = _import_loaders()
    rois = load_electrodes_config(str(_repo_root()))
    assert "N1_L" in rois
    assert isinstance(rois["N1_L"], list)
    assert rois["N1_L"][0].startswith("E")

