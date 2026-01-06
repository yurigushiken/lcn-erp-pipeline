from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib

# Avoid backend switching if pyplot already loaded (tests)
import sys

if "matplotlib.pyplot" not in sys.modules:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt


def _ylabel_for(dv: str) -> str:
    dv_l = dv.lower()
    if "latency" in dv_l:
        return "Latency (ms)"
    if "amplitude" in dv_l:
        return "Amplitude (µV)"
    return dv.replace("_", " ").title()


def plot_boxplot(
    *,
    data: pd.DataFrame,
    dv: str,
    groupby: str,
    title: str,
    output_path: Union[str, Path],
    show_points: bool = True,
    show_mean: bool = True,
    dpi: int = 300,
    figsize: Tuple[float, float] = (8, 6),
    build_stamp: Optional[str] = None,
) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    df = data[[groupby, dv]].dropna()
    groups = sorted(df[groupby].astype(str).unique())
    values = [df[df[groupby].astype(str) == g][dv].astype(float).values for g in groups]

    fig, ax = plt.subplots(figsize=figsize)
    ax.boxplot(values, labels=groups, showfliers=False)
    ax.set_title(title)
    ax.set_xlabel(groupby)
    ax.set_ylabel(_ylabel_for(dv))

    if show_points:
        for i, vals in enumerate(values, start=1):
            x = np.full_like(vals, i, dtype=float)
            jitter = (np.random.RandomState(0).rand(len(vals)) - 0.5) * 0.15
            ax.scatter(x + jitter, vals, s=12, alpha=0.6, color="#333")

    if show_mean:
        means = [float(np.mean(v)) if len(v) else float("nan") for v in values]
        ax.scatter(range(1, len(means) + 1), means, marker="D", color="#d62728", s=40, zorder=3)

    if build_stamp:
        fig.text(0.995, 0.002, build_stamp, ha="right", va="bottom", fontsize=6, color="#666")

    fig.tight_layout()
    fig.savefig(out, dpi=int(dpi))
    plt.close(fig)


def plot_violin(
    *,
    data: pd.DataFrame,
    dv: str,
    groupby: str,
    title: str,
    output_path: Union[str, Path],
    dpi: int = 300,
    figsize: Tuple[float, float] = (8, 6),
    build_stamp: Optional[str] = None,
) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    df = data[[groupby, dv]].dropna()
    groups = sorted(df[groupby].astype(str).unique())
    values = [df[df[groupby].astype(str) == g][dv].astype(float).values for g in groups]

    fig, ax = plt.subplots(figsize=figsize)
    parts = ax.violinplot(values, showmeans=False, showmedians=True, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_facecolor("#4c78a8")
        pc.set_edgecolor("#333")
        pc.set_alpha(0.6)
    ax.set_xticks(range(1, len(groups) + 1))
    ax.set_xticklabels(groups)
    ax.set_title(title)
    ax.set_xlabel(groupby)
    ax.set_ylabel(_ylabel_for(dv))
    if build_stamp:
        fig.text(0.995, 0.002, build_stamp, ha="right", va="bottom", fontsize=6, color="#666")
    fig.tight_layout()
    fig.savefig(out, dpi=int(dpi))
    plt.close(fig)


def plot_effect_sizes(
    *,
    effects: pd.DataFrame,
    title: str,
    output_path: Union[str, Path],
    effect_col: str = "cohen",
    ci_low_col: str = "ci_lower",
    ci_high_col: str = "ci_upper",
    label_col_a: str = "A",
    label_col_b: str = "B",
    dpi: int = 300,
    figsize: Tuple[float, float] = (8, 6),
    build_stamp: Optional[str] = None,
    show_ci: bool = True,
    threshold_lines: Optional[list[float]] = None,
    threshold_labels: Optional[list[str]] = None,
) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    df = effects.copy()
    if len(df) == 0:
        return

    labels = [f"{a} vs {b}" for a, b in zip(df[label_col_a], df[label_col_b])]
    y = np.arange(len(labels))
    eff = df[effect_col].astype(float).values
    lo = df[ci_low_col].astype(float).values if ci_low_col in df.columns else np.full_like(eff, np.nan)
    hi = df[ci_high_col].astype(float).values if ci_high_col in df.columns else np.full_like(eff, np.nan)

    fig, ax = plt.subplots(figsize=figsize)
    ax.axvline(0, color="#999", linewidth=1)

    # Threshold lines (Cohen's d: small/medium/large). Draw both + and -.
    th = threshold_lines or [0.2, 0.5, 0.8]
    for v in th:
        try:
            vv = float(v)
        except Exception:
            continue
        ax.axvline(vv, color="#bbb", linewidth=1, linestyle=":")
        ax.axvline(-vv, color="#bbb", linewidth=1, linestyle=":")

    ax.scatter(eff, y, color="#000")
    if show_ci and np.all(np.isfinite(lo)) and np.all(np.isfinite(hi)):
        ax.hlines(y, lo, hi, color="#000", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_title(title)
    ax.set_xlabel("Effect size (Cohen's d)")
    ax.invert_yaxis()

    # Optional threshold labels near top of axis
    if threshold_labels and len(threshold_labels) == len(th):
        try:
            y_top = -0.75
            for vv, lbl in zip(th, threshold_labels):
                ax.text(float(vv), y_top, str(lbl), ha="center", va="bottom", fontsize=8, color="#666")
        except Exception:
            pass

    if build_stamp:
        fig.text(0.995, 0.002, build_stamp, ha="right", va="bottom", fontsize=6, color="#666")
    fig.tight_layout()
    fig.savefig(out, dpi=int(dpi))
    plt.close(fig)

