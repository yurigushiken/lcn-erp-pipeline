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


def _method_map(name: str) -> str | None:
    m = str(name or "").lower()
    # statsmodels.multitest names + common aliases used in configs
    mapping = {
        "hs": "holm-sidak",
        "holm-sidak": "holm-sidak",
        "holm": "holm",
        "bonferroni": "bonferroni",
        "sidak": "sidak",
        "fdr_bh": "fdr_bh",
        "none": None,
    }
    return mapping.get(m, "holm-sidak")


def _apply_family_correction(*, output_dir: Path, correction_family: str, method: str) -> None:
    """
    Second-stage multiple-comparisons correction across a broader family than the per-file correction.

    This adds a `p-corr-family` column to:
    - pairwise_{component}_{dv}.csv  (from pingouin, uses p-unc as input)
    - lmm_pairwise_{component}_{dv}.csv (uses p-unc as input)

    Family definitions:
    - component: correct within each component across all DVs and all pairs
    - dv:        correct within each DV across all components and all pairs
    - all:       correct across everything in the output_dir
    - none/empty: do nothing
    """
    from statsmodels.stats.multitest import multipletests

    fam = str(correction_family or "").lower().strip()
    if fam in {"", "none"}:
        return

    m = _method_map(method)
    if m is None:
        return

    def _load(kind: str) -> list[tuple[str, str, Path, pd.DataFrame]]:
        out = []
        for p in sorted(output_dir.glob(f"{kind}_*.csv")):
            # expected: {kind}_{component}_{dv}.csv
            parts = p.stem.split("_")
            if len(parts) < 3:
                continue
            component = parts[1]
            dv = "_".join(parts[2:])
            try:
                df = pd.read_csv(p)
            except Exception:
                continue
            if "p-unc" not in df.columns:
                continue
            out.append((component, dv, p, df))
        return out

    # Pairwise and LMM-pairwise share the same correction logic.
    for kind in ("pairwise", "lmm_pairwise"):
        items = _load(kind)
        if not items:
            continue

        # Build a single long table of p-values with a stable row id.
        records = []
        for component, dv, pth, df in items:
            for idx, p_unc in enumerate(df["p-unc"].astype(float).fillna(1.0).to_list()):
                if fam == "component":
                    key = component
                elif fam == "dv":
                    key = dv
                elif fam == "all":
                    key = "all"
                else:
                    # Unknown family choice -> do nothing
                    return
                records.append({"key": key, "path": str(pth), "row": idx, "p": float(p_unc)})

        if not records:
            continue

        rec_df = pd.DataFrame.from_records(records)
        rec_df["p_corr_family"] = float("nan")

        for key, grp in rec_df.groupby("key", dropna=False):
            pvals = grp["p"].astype(float).values
            try:
                _, p_corr, _, _ = multipletests(pvals, method=m)
            except Exception:
                p_corr = np.full_like(pvals, fill_value=np.nan, dtype=float)
            rec_df.loc[grp.index, "p_corr_family"] = p_corr

        # Write back into each file
        by_path = rec_df.groupby("path")
        for component, dv, pth, df in items:
            p_str = str(pth)
            if p_str not in by_path.groups:
                continue
            g = rec_df.loc[by_path.groups[p_str]]
            # preserve original order
            g = g.sort_values("row")
            if len(g) != len(df):
                continue
            df["p-corr-family"] = g["p_corr_family"].astype(float).values
            df.to_csv(pth, index=False)


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
    correction_family = str(cfg.get("correction_family", "none"))

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

    # Optional second-stage correction across a broader family (pairwise + lmm_pairwise).
    # This is applied after all files are written so it can span components/DVs.
    family_method = str(tests_cfg.get("pairwise", {}).get("correction", "fdr_bh"))
    _apply_family_correction(output_dir=output_dir, correction_family=correction_family, method=family_method)

    # Save summary + report
    save_json(output_dir / str(cfg.get("output", {}).get("summary_filename", "statistical_summary.json")), summary)

    report_gen = StatisticalReportGenerator(stats_dir=output_dir, analysis_id=analysis_id)
    report_gen.generate_report(output_dir / "STATISTICAL_REPORT.md")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

