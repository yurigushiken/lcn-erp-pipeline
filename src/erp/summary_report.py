from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


class StatisticalReportGenerator:
    """Generate a concise, LMM-first statistical report from saved outputs.

    This reads output files produced by scripts/run_statistics.py and produces
    a markdown report in a stable, testable order:
      1) LMM (Primary)
      2) ANOVA (Supplementary)
      3) Pairwise (Supplementary)
    """

    def __init__(self, *, stats_dir: Path, analysis_id: str):
        self.stats_dir = Path(stats_dir)
        self.analysis_id = str(analysis_id)
        self.summary: Dict[str, Any] = {}

        summary_path = self.stats_dir / "statistical_summary.json"
        if summary_path.exists():
            with open(summary_path, "r", encoding="utf-8") as f:
                self.summary = json.load(f) or {}

    def generate_report(self, output_path: Path) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        components = self.summary.get("components_tested") or []
        dvs = self.summary.get("dependent_variables") or []
        if not components:
            # Best-effort fallback: infer from files
            components = self._infer_components()
        if not dvs:
            dvs = self._infer_dvs()

        lines = [f"# Statistical Analysis Report: {self.analysis_id}", ""]
        lines.append("## Inferential Statistics (LMM-first)")
        lines.append("")

        for component in components:
            for dv in dvs:
                lines.append(f"### {component} — {dv}")
                lines.append("")
                lines.extend(self._section_lmm(component, dv))
                lines.extend(self._section_anova(component, dv))
                lines.extend(self._section_pairwise(component, dv))

        output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    def _section_lmm(self, component: str, dv: str) -> list[str]:
        path = self.stats_dir / f"lmm_{component}_{dv}.json"
        if not path.exists():
            return ["**Linear Mixed-Effects Model (Primary Analysis)**", "", "_No LMM output found._", ""]

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f) or {}
        summary = data.get("summary") or []

        lines = ["**Linear Mixed-Effects Model (Primary Analysis)**", ""]
        if not summary:
            lines.append("_LMM summary was empty._")
            lines.append("")
            return lines

        for row in summary:
            name = str(row.get("name", ""))
            p = row.get("P>|z|")
            try:
                p_str = f"{float(p):.3f}"
            except Exception:
                p_str = str(p)
            lines.append(f"- {name}: p={p_str}")
        lines.append("")
        return lines

    def _section_anova(self, component: str, dv: str) -> list[str]:
        path = self.stats_dir / f"anova_{component}_{dv}.csv"
        if not path.exists():
            return ["**Repeated-Measures ANOVA (Supplementary Analysis)**", "", "_No ANOVA output found._", ""]

        df = pd.read_csv(path)
        lines = ["**Repeated-Measures ANOVA (Supplementary Analysis)**", ""]

        # Effective N warning (listwise deletion) — use ddof2 + 1 for one-way rmANOVA
        n_eff = _effective_n_from_anova(df)
        if n_eff is not None:
            lines.append(f"_Note: ANOVA uses listwise deletion; effective n={n_eff} complete cases._")
        else:
            lines.append("_Note: ANOVA uses listwise deletion; effective n could not be inferred._")
        lines.append("")
        return lines

    def _section_pairwise(self, component: str, dv: str) -> list[str]:
        path = self.stats_dir / f"pairwise_{component}_{dv}.csv"
        if not path.exists():
            return ["**Pairwise Comparisons (Supplementary Analysis)**", "", "_No pairwise output found._", ""]

        df = pd.read_csv(path)
        lines = ["**Pairwise Comparisons (Supplementary Analysis)**", ""]
        if len(df) == 0:
            lines.append("_No pairwise rows._")
            lines.append("")
            return lines

        # Print first few comparisons
        for _, row in df.head(5).iterrows():
            a = row.get("A")
            b = row.get("B")
            # Prefer family-corrected p if present, then per-file corrected p, then uncorrected.
            p = row.get("p-corr-family", row.get("p-corr", row.get("p-unc", "n/a")))
            lines.append(f"- {a} vs {b}: p={p}")
        lines.append("")
        return lines

    def _infer_components(self) -> list[str]:
        comps = set()
        for p in self.stats_dir.glob("lmm_*.json"):
            # lmm_{component}_{dv}.json
            stem = p.stem
            parts = stem.split("_")
            if len(parts) >= 3:
                comps.add(parts[1])
        return sorted(comps)

    def _infer_dvs(self) -> list[str]:
        dvs = set()
        for p in self.stats_dir.glob("lmm_*.json"):
            parts = p.stem.split("_")
            if len(parts) >= 3:
                dvs.add("_".join(parts[2:]))
        return sorted(dvs)


def _effective_n_from_anova(anova_df: pd.DataFrame) -> Optional[int]:
    if "Source" not in anova_df.columns:
        return None
    # Prefer row for within factor 'condition' if present
    row = anova_df[anova_df["Source"].astype(str).str.lower() == "condition"]
    if len(row) == 0:
        row = anova_df.iloc[:1]
    if "ddof2" not in anova_df.columns:
        return None
    try:
        ddof2 = float(row["ddof2"].values[0])
    except Exception:
        return None
    if pd.isna(ddof2):
        return None
    return int(ddof2) + 1


__all__ = ["StatisticalReportGenerator"]

