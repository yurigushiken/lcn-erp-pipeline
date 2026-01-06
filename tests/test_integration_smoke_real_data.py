from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.integration
def test_smoke_run_analysis_real_data(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    data_root = repo_root / "data" / "lab_data_original_with_primes"
    if not data_root.exists():
        pytest.skip("Real data directory not present")

    # Write a temp config that outputs into tmp_path (no repo pollution)
    cfg_path = tmp_path / "analysis.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "dataset:",
                f"  root: {str(data_root).replace('\\\\', '/')}",
                "  pattern: sub-*_preprocessed-epo.fif",
                f"  montage_sfp: {str(repo_root / 'assets' / 'net' / 'AdultAverageNet128_v1.sfp').replace('\\\\', '/')}",
                "selection:",
                "  response: ALL",
                "  response_field: Target.ACC",
                "  min_epochs_per_set: 1",
                "  condition_sets:",
                "    - name: 1 to 2",
                "      conditions: ['12']",
                "components: [N1]",
                "preprocessing: {baseline_ms: [-100, 0]}",
                "roi: {min_channels: 1}",
                "plots: {formats: [png], dpi: 80}",
                "outputs:",
                f"  plots_dir: {str((tmp_path / 'docs' / 'assets' / 'plots' / 'smoke')).replace('\\\\', '/')}",
                f"  tables_dir: {str((tmp_path / 'docs' / 'assets' / 'tables' / 'smoke')).replace('\\\\', '/')}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    import sys

    sys.path.insert(0, str(repo_root))
    from scripts.run_analysis import main

    code = main(["--config", str(cfg_path), "--max_subjects", "1"])
    assert code == 0

    # Core outputs
    plots_dir = tmp_path / "docs" / "assets" / "plots" / "smoke"
    tables_dir = tmp_path / "docs" / "assets" / "tables" / "smoke"
    assert (plots_dir / "smoke-collapsed_localizer.png").exists()
    assert (plots_dir / "smoke-N1.png").exists()
    assert (plots_dir / "smoke-N1-no_topo.png").exists()
    assert (tables_dir / "subject_measurements.csv").exists()
    assert (tables_dir / "collapsed_localizer_results.json").exists()

    # Publication variants should be created, but not referenced anywhere automatically.
