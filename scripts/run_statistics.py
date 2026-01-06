#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from erp.statistics import ERPStatistics, compute_cohens_d_ci, save_json
from erp.stats_plots import plot_boxplot, plot_effect_sizes, plot_violin
from erp.summary_report import StatisticalReportGenerator


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _hash_file(path: Path) -> str:
    try:
        return hashlib.sha1(path.read_bytes()).hexdigest()[:7]
    except Exception:
        return "unknown"


def _resolve_path(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (REPO_ROOT / path)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run statistics on subject_measurements.csv")
    parser.add_argument("--config", required=True, help="Path to statistics YAML")
    args = parser.parse_args(argv)

    cfg_path = Path(args.config)
    cfg = _load_yaml(cfg_path)
    cfg["_config_path"] = str(cfg_path)

    input_csv = _resolve_path(str(cfg.get("input_csv", "")))
    output_dir = _resolve_path(str(cfg.get("output_dir", "")))
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = ERPStatistics(input_csv)
    filters = cfg.get("filters", {}) or {}
    min_snr = filters.get("min_snr", None)
    dropna = bool(filters.get("dropna", True))

    components = cfg.get("components", []) or []
    dvs = cfg.get("dependent_variables", []) or []

    tests_cfg = cfg.get("tests", {}) or {}
    plots_cfg = cfg.get("plots", {}) or {}

    cfg_hash = _hash_file(cfg_path)
    analysis_id = output_dir.name
    build_stamp = f"{analysis_id} · cfg:{cfg_hash} · {datetime.now().date().isoformat()}"

    summary: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "analysis_id": analysis_id,
        "components_tested": components,
        "dependent_variables": dvs,
        "tests_run": {k: bool(v.get("enabled", False)) for k, v in tests_cfg.items() if isinstance(v, dict)},
        "analysis_settings": {
            "fixed": tests_cfg.get("lmm", {}).get("fixed", ""),
            "random": tests_cfg.get("lmm", {}).get("random", ""),
            "filters": {"min_snr": min_snr, "dropna": dropna},
        },
    }

    # Run tests
    for component in components:
        comp_data = stats.filter_data(component=component, min_snr=min_snr, dropna=dropna)
        for dv in dvs:
            if dv not in comp_data.columns:
                continue

            # LMM
            if tests_cfg.get("lmm", {}).get("enabled", False):
                fixed = str(tests_cfg["lmm"].get("fixed", "condition"))
                random = str(tests_cfg["lmm"].get("random", "subject_id"))
                method = str(tests_cfg["lmm"].get("method", "lbfgs"))
                lmm_res = stats.run_lmm(data=comp_data, dv=dv, fixed=fixed, random=random, method=method)
                save_json(output_dir / f"lmm_{component}_{dv}.json", lmm_res)

            # ANOVA
            if tests_cfg.get("anova", {}).get("enabled", False):
                within = str(tests_cfg["anova"].get("within", "condition"))
                subject = str(tests_cfg["anova"].get("subject", "subject_id"))
                anova = stats.run_anova(data=comp_data, dv=dv, within=within, subject=subject)
                anova.to_csv(output_dir / f"anova_{component}_{dv}.csv", index=False)

            # Pairwise
            if tests_cfg.get("pairwise", {}).get("enabled", False):
                within = str(tests_cfg["pairwise"].get("within", "condition"))
                subject = str(tests_cfg["pairwise"].get("subject", "subject_id"))
                correction = str(tests_cfg["pairwise"].get("correction", "fdr_bh"))
                effsize = str(tests_cfg["pairwise"].get("effsize", "cohen"))
                pair = stats.run_pairwise(
                    data=comp_data,
                    dv=dv,
                    within=within,
                    subject=subject,
                    correction=correction,
                    effsize=effsize,
                )

                # Add Cohen's d CI columns when possible (paired within-subjects)
                try:
                    # Pivot to subject x condition to align paired samples
                    wide = comp_data.pivot_table(index=subject, columns=within, values=dv, aggfunc="mean")
                    ci_lows = []
                    ci_highs = []
                    ds = []
                    for _, r in pair.iterrows():
                        a = str(r.get("A"))
                        b = str(r.get("B"))
                        if a in wide.columns and b in wide.columns:
                            sub = wide[[a, b]].dropna()
                            x = sub[a].to_numpy()
                            y = sub[b].to_numpy()
                            d_val, lo, hi = compute_cohens_d_ci(x=x, y=y, paired=True)
                            ds.append(d_val)
                            ci_lows.append(lo)
                            ci_highs.append(hi)
                        else:
                            ds.append(float("nan"))
                            ci_lows.append(float("nan"))
                            ci_highs.append(float("nan"))
                    # Preserve pingouin's existing cohen column but add CIs
                    pair["ci_lower"] = ci_lows
                    pair["ci_upper"] = ci_highs
                except Exception:
                    pass
                pair.to_csv(output_dir / f"pairwise_{component}_{dv}.csv", index=False)

            # LMM pairwise
            if tests_cfg.get("lmm_pairwise", {}).get("enabled", False):
                fixed = str(tests_cfg["lmm_pairwise"].get("fixed", tests_cfg.get("lmm", {}).get("fixed", "condition")))
                random = str(tests_cfg["lmm_pairwise"].get("random", tests_cfg.get("lmm", {}).get("random", "subject_id")))
                correction = str(tests_cfg["lmm_pairwise"].get("correction", "hs"))
                lmm_pair = stats.run_lmm_pairwise(data=comp_data, dv=dv, fixed=fixed, random=random, correction=correction)
                lmm_pair.to_csv(output_dir / f"lmm_pairwise_{component}_{dv}.csv", index=False)

            # Descriptives
            if cfg.get("descriptives", {}).get("enabled", False):
                groupby = str(cfg.get("descriptives", {}).get("groupby", "condition"))
                desc = stats.descriptives(component=component, dv=dv, groupby=groupby, min_snr=min_snr, dropna=dropna)
                desc.to_csv(output_dir / f"descriptives_{component}_{dv}.csv", index=False)

            # Plots
            if plots_cfg.get("enabled", False):
                plots_dir = output_dir / str(plots_cfg.get("plots_subdir", "plots"))
                plots_dir.mkdir(parents=True, exist_ok=True)

                title = f"{component} {dv.replace('_', ' ').title()}"
                dpi = int(plots_cfg.get("dpi", 300))
                figsize = tuple(plots_cfg.get("figsize", [8, 6]))

                plot_boxplot(
                    data=comp_data,
                    dv=dv,
                    groupby="condition",
                    title=title,
                    output_path=plots_dir / f"boxplot_{component}_{dv}.png",
                    show_points=bool(plots_cfg.get("boxplot", {}).get("show_points", True)),
                    show_mean=bool(plots_cfg.get("boxplot", {}).get("show_mean", True)),
                    dpi=dpi,
                    figsize=figsize,  # type: ignore[arg-type]
                    build_stamp=build_stamp,
                )
                plot_violin(
                    data=comp_data,
                    dv=dv,
                    groupby="condition",
                    title=title,
                    output_path=plots_dir / f"violin_{component}_{dv}.png",
                    dpi=dpi,
                    figsize=figsize,  # type: ignore[arg-type]
                    build_stamp=build_stamp,
                )

                # Effect sizes: if pairwise exists and has CI columns, plot them
                pair_path = output_dir / f"pairwise_{component}_{dv}.csv"
                if pair_path.exists():
                    pair_df = pd.read_csv(pair_path)
                    if len(pair_df) > 0 and "cohen" in pair_df.columns:
                        # Optional CI columns may not exist; plot_effect_sizes handles that.
                        plot_effect_sizes(
                            effects=pair_df,
                            title=f"{component} Effect sizes (Cohen's d)",
                            output_path=plots_dir / f"effect_sizes_{component}_{dv}.png",
                            effect_col="cohen",
                            ci_low_col="ci_lower",
                            ci_high_col="ci_upper",
                            show_ci=bool(plots_cfg.get("effect_size", {}).get("show_ci", True)),
                            threshold_lines=list(plots_cfg.get("effect_size", {}).get("threshold_lines", [0.2, 0.5, 0.8])),
                            threshold_labels=list(plots_cfg.get("effect_size", {}).get("threshold_labels", ["Small", "Medium", "Large"])),
                            dpi=dpi,
                            figsize=figsize,  # type: ignore[arg-type]
                            build_stamp=build_stamp,
                        )

    # Save summary + report
    save_json(output_dir / str(cfg.get("output", {}).get("summary_filename", "statistical_summary.json")), summary)

    report_gen = StatisticalReportGenerator(stats_dir=output_dir, analysis_id=analysis_id)
    report_gen.generate_report(output_dir / "STATISTICAL_REPORT.md")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

