from __future__ import annotations

from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_run_analysis_no_data_creates_page(tmp_path: Path):
    import sys

    sys.path.insert(0, str(_repo_root()))

    # Create an empty data dir
    empty_data = tmp_path / "empty_data"
    empty_data.mkdir(parents=True, exist_ok=True)

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "dataset:",
                f"  root: {str(empty_data).replace('\\\\', '/')}",
                "  pattern: '*.fif'",
                "  montage_sfp: assets/net/AdultAverageNet128_v1.sfp",
                "selection:",
                "  response: ALL",
                "  response_field: Target.ACC",
                "  min_epochs_per_set: 1",
                "  condition_sets: []",
                "components: [N1]",
                "preprocessing: {baseline_ms: [-100, 0]}",
                "roi: {min_channels: 1}",
                "plots: {formats: [png], dpi: 80}",
                "outputs:",
                f"  plots_dir: {str((tmp_path / 'docs' / 'assets' / 'plots' / 'no_data')).replace('\\\\', '/')}",
                f"  tables_dir: {str((tmp_path / 'docs' / 'assets' / 'tables' / 'no_data')).replace('\\\\', '/')}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    from scripts.run_analysis import main

    code = main(["--config", str(cfg_path)])
    assert code == 1

