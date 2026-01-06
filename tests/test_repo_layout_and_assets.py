from pathlib import Path


def test_montage_file_exists():
    repo_root = Path(__file__).resolve().parents[1]
    montage = repo_root / "assets" / "net" / "AdultAverageNet128_v1.sfp"
    assert montage.is_file(), f"Missing montage file: {montage}"


def test_example_analysis_config_exists():
    repo_root = Path(__file__).resolve().parents[1]
    cfg = repo_root / "configs" / "analyses" / "example.yaml"
    assert cfg.is_file(), f"Missing example analysis config: {cfg}"

