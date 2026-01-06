from __future__ import annotations

from pathlib import Path

import json
import pandas as pd


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _import_report_generator():
    import sys

    sys.path.insert(0, str(_repo_root() / "src"))
    from erp.summary_report import StatisticalReportGenerator

    return StatisticalReportGenerator


def test_report_sections_ordered_lmm_first(tmp_path: Path):
    StatisticalReportGenerator = _import_report_generator()

    stats_dir = tmp_path / "docs" / "assets" / "stats" / "demo"
    stats_dir.mkdir(parents=True, exist_ok=True)

    # Minimal summary file
    (stats_dir / "statistical_summary.json").write_text(
        json.dumps(
            {
                "analysis_id": "demo",
                "components_tested": ["N1"],
                "dependent_variables": ["mean_amplitude_roi"],
                "analysis_settings": {"fixed": "condition + snr"},
            }
        ),
        encoding="utf-8",
    )

    # Minimal ANOVA + LMM + pairwise artifacts
    pd.DataFrame({"Source": ["condition"], "ddof2": [8], "p-unc": [0.04]}).to_csv(
        stats_dir / "anova_N1_mean_amplitude_roi.csv", index=False
    )
    (stats_dir / "lmm_N1_mean_amplitude_roi.json").write_text(
        json.dumps(
            {
                "summary": [
                    {"name": "Intercept", "P>|z|": 0.001},
                    {"name": "condition", "P>|z|": 0.008},
                    {"name": "snr", "P>|z|": 0.003},
                ]
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame({"A": ["cond1"], "B": ["cond2"], "p-corr": [0.22]}).to_csv(
        stats_dir / "pairwise_N1_mean_amplitude_roi.csv", index=False
    )

    report_gen = StatisticalReportGenerator(stats_dir=stats_dir, analysis_id="demo")
    out = stats_dir / "STATISTICAL_REPORT.md"
    report_gen.generate_report(out)

    text = out.read_text(encoding="utf-8")
    lmm_pos = text.find("**Linear Mixed-Effects Model")
    anova_pos = text.find("**Repeated-Measures ANOVA")
    pairwise_pos = text.find("**Pairwise Comparisons")
    assert lmm_pos != -1 and anova_pos != -1 and pairwise_pos != -1
    assert lmm_pos < anova_pos < pairwise_pos
    assert "snr" in text.lower()


def test_anova_effective_n_reported(tmp_path: Path):
    StatisticalReportGenerator = _import_report_generator()

    stats_dir = tmp_path / "docs" / "assets" / "stats" / "demo"
    stats_dir.mkdir(parents=True, exist_ok=True)

    (stats_dir / "statistical_summary.json").write_text(
        json.dumps({"analysis_id": "demo", "components_tested": ["N1"], "dependent_variables": ["mean_amplitude_roi"]}),
        encoding="utf-8",
    )
    # ddof2=8 => n=9 complete cases for one-factor rmANOVA
    pd.DataFrame({"Source": ["condition"], "ddof2": [8], "p-unc": [0.04]}).to_csv(
        stats_dir / "anova_N1_mean_amplitude_roi.csv", index=False
    )
    (stats_dir / "lmm_N1_mean_amplitude_roi.json").write_text(
        json.dumps({"summary": [{"name": "Intercept", "P>|z|": 0.5}]}), encoding="utf-8"
    )
    pd.DataFrame({"A": ["cond1"], "B": ["cond2"], "p-corr": [0.22]}).to_csv(
        stats_dir / "pairwise_N1_mean_amplitude_roi.csv", index=False
    )

    report_gen = StatisticalReportGenerator(stats_dir=stats_dir, analysis_id="demo")
    out = stats_dir / "STATISTICAL_REPORT.md"
    report_gen.generate_report(out)

    text = out.read_text(encoding="utf-8")
    assert "n=9" in text.replace(" ", "")

