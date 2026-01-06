from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Tuple

import sys
import matplotlib

# Headless-safe backend (CI / servers). Avoid switching backends if pyplot is already loaded
# (e.g., in tests), which triggers a Matplotlib deprecation warning.
if "matplotlib.pyplot" not in sys.modules:
    matplotlib.use("Agg")


def _sanitize_filename(name: str) -> str:
    # Remove control chars and replace invalid Windows characters
    name = re.sub(r"[\r\n\t]", " ", name)
    name = re.sub(r"[<>:\\/\|\?\*]", "_", name)
    return name.strip()


def save_figure(fig, path_like, dpi: int = 300, bbox_inches: Optional[str] = None) -> None:
    """Filesystem-safe figure saver (Windows-safe; normalizes separators and retries on errno=22)."""
    out_path = Path(path_like)
    safe_name = _sanitize_filename(out_path.name)
    out_path = out_path.with_name(safe_name)

    try:
        out_path = out_path.expanduser().resolve(strict=False)
    except Exception:
        out_path = Path(os.path.abspath(str(out_path)))

    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _attempt(p: Path) -> None:
        kw = {"dpi": int(dpi)}
        if bbox_inches is not None:
            kw["bbox_inches"] = bbox_inches
        fig.savefig(str(p), **kw)

    try:
        _attempt(out_path)
    except OSError as e:
        if getattr(e, "errno", None) == 22:
            # Retry with a simplified absolute path string
            alt = Path(os.path.abspath(str(out_path)))
            _attempt(alt)
        else:
            raise


def make_erp_figure(
    *,
    curves_by_label: Dict[str, Iterable[float]],
    times_ms,
    title: str,
    subtitle: Optional[str] = None,
    sem_by_label: Optional[Mapping[str, Iterable[float]]] = None,
    show_sem: bool = False,
    latencies_by_label: Optional[Mapping[str, float]] = None,
    peak_amplitudes_by_label: Optional[Mapping[str, float]] = None,
    latency_annotation_label: str = "Peak",
    colors: Optional[Mapping[str, str]] = None,
    linestyles: Optional[Mapping[str, str]] = None,
    xlim_ms: Optional[Tuple[float, float]] = None,
    ylimit_uv: Optional[float] = None,
    epochs_by_label: Optional[Mapping[str, int]] = None,
) -> "matplotlib.figure.Figure":
    import matplotlib.pyplot as plt
    import numpy as np

    # NOTE: Avoid constrained_layout here because we place a suptitle + a subtitle
    # via fig.text; constrained_layout does not reliably reserve space for those,
    # which can cause the title/subtitle to overlap the legend/plot.
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    for label, y in curves_by_label.items():
        y = np.asarray(list(y), dtype=float)
        kw = {}
        if colors and label in colors:
            kw["color"] = colors[label]
        if linestyles and label in linestyles:
            kw["linestyle"] = linestyles[label]
        legend_label = str(label)
        try:
            lat = None
            amp = None
            if latencies_by_label and label in latencies_by_label:
                lat = float(latencies_by_label[label])
            if peak_amplitudes_by_label and label in peak_amplitudes_by_label:
                amp = float(peak_amplitudes_by_label[label])
            if lat is not None and amp is not None and np.isfinite(lat) and np.isfinite(amp):
                legend_label = f"{legend_label} ({lat:.0f}ms, {amp:.1f} µV)"
        except Exception:
            pass
        try:
            if epochs_by_label and label in epochs_by_label:
                legend_label = f"{legend_label} [{int(epochs_by_label[label])} epochs]"
        except Exception:
            pass

        ax.plot(times_ms, y, label=legend_label, **kw)
        if show_sem and sem_by_label and label in sem_by_label:
            sem = np.asarray(list(sem_by_label[label]), dtype=float)
            ax.fill_between(times_ms, y - sem, y + sem, alpha=0.18, linewidth=0, **({k: v for k, v in kw.items() if k == "color"}))

    # Peak markers: vertical latency + horizontal amplitude lines
    if latencies_by_label:
        for label, lat in latencies_by_label.items():
            try:
                color = (colors.get(label) if colors else None) or "#444"
                ls = (linestyles.get(label) if linestyles else None) or "-"
                ax.axvline(float(lat), color=color, alpha=0.7, linestyle=ls, linewidth=0.8)
            except Exception:
                continue
    if peak_amplitudes_by_label:
        for label, amp in peak_amplitudes_by_label.items():
            try:
                color = (colors.get(label) if colors else None) or "#666"
                ls = (linestyles.get(label) if linestyles else None) or "-"
                ax.axhline(float(amp), color=color, alpha=0.6, linestyle=ls, linewidth=0.8)
            except Exception:
                continue

    ax.axvline(0, color="#999", linewidth=1, alpha=0.6)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude (µV)")
    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold", y=0.97)
    if subtitle:
        enhanced = subtitle
        if latencies_by_label:
            enhanced = f"{enhanced} | Vertical lines = {latency_annotation_label} Latency"
        fig.text(0.5, 0.90, enhanced, ha="center", fontsize=9)
    if xlim_ms is not None:
        ax.set_xlim(xlim_ms)
    if ylimit_uv is not None:
        ax.set_ylim((-float(ylimit_uv), float(ylimit_uv)))
    ax.grid(True, which="major", linestyle=":", linewidth=0.5, alpha=0.25)
    ax.legend(loc="upper left", fontsize=8, frameon=True, framealpha=0.9)

    # Reserve top space for title/subtitle so they never overlap the axes/legend.
    try:
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.84))
    except Exception:
        pass
    return fig


def make_collapsed_localizer_figure(
    *,
    localizer_results: Dict[str, Dict],
    title: str,
    subtitle: str,
    xlim_ms: Optional[Tuple[float, float]] = None,
) -> "matplotlib.figure.Figure":
    import matplotlib.pyplot as plt
    import numpy as np

    n_components = len(localizer_results)
    if n_components == 0:
        raise ValueError("No localizer results to plot")

    fig, axes = plt.subplots(
        n_components,
        1,
        figsize=(8.5, 2.8 * n_components),
        constrained_layout=True,
        squeeze=False,
    )
    axes = axes.flatten()

    for idx, (comp, res) in enumerate(sorted(localizer_results.items())):
        res = res or {}
        ax = axes[idx]

        times_ms = res.get("times_ms")
        if times_ms is None:
            continue

        if "gfp" in res:
            trace = np.asarray(res["gfp"], dtype=float)
            trace_label = "GFP (all channels)"
            y_label = "GFP (µV)"
        else:
            trace = np.asarray(res.get("trace", []), dtype=float)
            trace_label = res.get("trace_label", "ROI mean")
            y_label = "µV"

        peak_lat = float(res.get("peak_latency_ms", np.nan))
        ws = float(res.get("window_start_ms", np.nan))
        we = float(res.get("window_end_ms", np.nan))
        fwhm = float(res.get("fwhm_ms", np.nan))
        peak_amp = float(res.get("peak_amplitude", np.nan))
        sr = res.get("search_range_ms")

        ax.plot(times_ms, trace, label=trace_label, color="#2c7bb6", linewidth=2.0, zorder=3)
        ax.axvspan(ws, we, alpha=0.2, color="#d7191c", label=f"FWHM window ({fwhm:.1f} ms)", zorder=1)
        ax.axvline(0, color="#666", linewidth=1, alpha=0.5, linestyle="-", label="Stimulus onset", zorder=2)
        ax.axvline(peak_lat, color="#d7191c", linewidth=2, linestyle="--", label=f"Peak: {peak_lat:.0f} ms", zorder=4)

        if isinstance(sr, (tuple, list)) and len(sr) == 2:
            ax.axvline(float(sr[0]), color="#999", linewidth=1, alpha=0.4, linestyle=":", zorder=2)
            ax.axvline(float(sr[1]), color="#999", linewidth=1, alpha=0.4, linestyle=":", label=f"Search range: [{sr[0]}, {sr[1]}] ms", zorder=2)

        ax.set_xlabel("Time (ms)")
        ax.set_ylabel(y_label)
        ax.set_title(f"{comp} Component", fontsize=11, loc="left", fontweight="bold")
        if xlim_ms is not None:
            ax.set_xlim(xlim_ms)
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.legend(loc="best", fontsize=8, framealpha=0.9)

        metrics_text = f"Peak: {peak_amp:.2f} µV\nFWHM: {fwhm:.1f} ms\nWindow: [{ws:.1f}, {we:.1f}] ms"
        ax.text(
            0.98,
            0.97,
            metrics_text,
            transform=ax.transAxes,
            fontsize=7,
            va="top",
            ha="right",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#2c7bb6", alpha=0.8),
        )

    fig.suptitle(title, fontsize=13, fontweight="bold")
    if subtitle:
        fig.text(0.5, 0.02, subtitle, ha="center", fontsize=9, style="italic")
    return fig


def make_peak_to_peak_figure(
    *,
    curves_by_label: Dict[str, Iterable[float]],
    times_ms,
    p2p_by_label: Mapping[str, float],
    p1_lat_by_label: Mapping[str, float],
    n1_lat_by_label: Mapping[str, float],
    title: str,
    subtitle: Optional[str] = None,
    colors: Optional[Mapping[str, str]] = None,
    linestyles: Optional[Mapping[str, str]] = None,
    xlim_ms: Optional[Tuple[float, float]] = None,
    ylimit_uv: Optional[float] = None,
    hline_color: str = "#000000",
    hline_style: str = ":",
    p1_amp_by_label: Optional[Mapping[str, float]] = None,
    n1_amp_by_label: Optional[Mapping[str, float]] = None,
    epochs_by_label: Optional[Mapping[str, int]] = None,
) -> "matplotlib.figure.Figure":
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    # Reserve space for big title + subtitle
    try:
        fig.subplots_adjust(top=0.78)
    except Exception:
        pass

    # Plot waveforms
    for label, y in curves_by_label.items():
        y = np.asarray(list(y), dtype=float)
        kw = {}
        if colors and label in colors:
            kw["color"] = colors[label]
        if linestyles and label in linestyles:
            kw["linestyle"] = linestyles[label]

        # Legend label with epochs + delta
        legend_label = str(label)
        try:
            if epochs_by_label and label in epochs_by_label:
                legend_label = f"{legend_label} ({int(epochs_by_label[label])} epochs)"
        except Exception:
            pass
        try:
            if label in p2p_by_label:
                legend_label = f"{legend_label}   Δ = {float(p2p_by_label[label]):.1f} µV"
        except Exception:
            pass
        ax.plot(times_ms, y, label=legend_label, **kw)

    ax.axvline(0, color="#999", linewidth=1, alpha=0.6)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude (µV)")

    if xlim_ms is not None:
        ax.set_xlim(xlim_ms)
    if ylimit_uv is not None:
        ax.set_ylim((-float(ylimit_uv), float(ylimit_uv)))

    # Horizontal lines at P1/N1 peak amplitudes (dotted)
    if p1_amp_by_label:
        for label, amp in p1_amp_by_label.items():
            try:
                color = colors[label] if colors and label in colors else hline_color
                ax.axhline(float(amp), color=color, linestyle=hline_style, linewidth=0.9, alpha=0.9)
            except Exception:
                continue
    if n1_amp_by_label:
        for label, amp in n1_amp_by_label.items():
            try:
                color = colors[label] if colors and label in colors else hline_color
                ax.axhline(float(amp), color=color, linestyle=hline_style, linewidth=0.9, alpha=0.9)
            except Exception:
                continue

    ax.grid(True, which="major", linestyle=":", linewidth=0.5, alpha=0.25)
    ax.legend(loc="upper right", fontsize=9, frameon=True, framealpha=0.9)

    if title:
        fig.suptitle(title, fontsize=18, fontweight="bold", y=0.96)
    if subtitle:
        fig.text(0.5, 0.89, subtitle, ha="center", fontsize=12)
    return fig


def make_component_figure(
    *,
    curves_by_label: Dict[str, Iterable[float]],
    sem_by_label: Optional[Mapping[str, Iterable[float]]],
    times_ms,
    topomap_by_label: Mapping[str, "np.ndarray"],
    info,
    title: str,
    subtitle: Optional[str] = None,
    latencies_by_label: Optional[Mapping[str, float]] = None,
    peak_amplitudes_by_label: Optional[Mapping[str, float]] = None,
    half_window_ms: Optional[float] = None,
    highlight_channels: Optional[Iterable[str]] = None,
    exclude_non_scalp: bool = True,
    non_scalp_labels: Optional[Iterable[str]] = None,
    latency_annotation_label: str = "Peak",
    colors: Optional[Mapping[str, str]] = None,
    linestyles: Optional[Mapping[str, str]] = None,
    xlim_ms: Optional[Tuple[float, float]] = None,
    ylimit_uv: Optional[float] = None,
    epochs_by_label: Optional[Mapping[str, int]] = None,
    show_sem: bool = False,
    include_topomaps: bool = True,
) -> "matplotlib.figure.Figure":
    import numpy as np
    import matplotlib.pyplot as plt
    import mne

    if not include_topomaps:
        return make_erp_figure(
            curves_by_label=curves_by_label,
            sem_by_label=sem_by_label,
            show_sem=show_sem,
            latencies_by_label=latencies_by_label,
            peak_amplitudes_by_label=peak_amplitudes_by_label,
            latency_annotation_label=latency_annotation_label,
            times_ms=times_ms,
            title=title,
            subtitle=subtitle,
            colors=colors,
            linestyles=linestyles,
            xlim_ms=xlim_ms,
            ylimit_uv=ylimit_uv,
            epochs_by_label=epochs_by_label,
        )

    labels = list(sorted(curves_by_label.keys()))
    n = max(1, len(labels))

    fig = plt.figure(figsize=(7.8, 7.2))
    # Dynamic layout: when there are few conditions, allocate more vertical room for the
    # waveform row and increase spacing between waveform and topomaps (reference-style).
    n_cols = n
    is_sparse = n_cols <= 2
    height_ratios = [2.9, 1.6] if is_sparse else [2.5, 1.9]
    hspace = 0.40 if is_sparse else 0.18
    width_ratios = [1.0] * n_cols + [0.10]

    # Reserve space for title/subtitle at top, and optional ROI caption at bottom.
    gs = fig.add_gridspec(
        2,
        n + 1,
        height_ratios=height_ratios,
        width_ratios=width_ratios,
        hspace=hspace,
        wspace=0.15,
        left=0.08,
        right=0.95,
        top=0.86,
        bottom=0.06,
    )

    ax_overlay = fig.add_subplot(gs[0, :])
    for label in labels:
        y = np.asarray(list(curves_by_label[label]), dtype=float)
        kw = {}
        if colors and label in colors:
            kw["color"] = colors[label]
        if linestyles and label in linestyles:
            kw["linestyle"] = linestyles[label]
        legend_label = str(label)
        try:
            lat = None
            amp = None
            if latencies_by_label and label in latencies_by_label:
                lat = float(latencies_by_label[label])
            if peak_amplitudes_by_label and label in peak_amplitudes_by_label:
                amp = float(peak_amplitudes_by_label[label])
            if lat is not None and amp is not None and np.isfinite(lat) and np.isfinite(amp):
                legend_label = f"{legend_label} ({lat:.0f}ms, {amp:.1f} µV)"
        except Exception:
            pass
        try:
            if epochs_by_label and label in epochs_by_label:
                legend_label = f"{legend_label} [{int(epochs_by_label[label])} epochs]"
        except Exception:
            pass
        ax_overlay.plot(times_ms, y, label=legend_label, **kw)
        if show_sem and sem_by_label and label in sem_by_label:
            sem = np.asarray(list(sem_by_label[label]), dtype=float)
            ax_overlay.fill_between(times_ms, y - sem, y + sem, alpha=0.18, linewidth=0, **({k: v for k, v in kw.items() if k == "color"}))

    # Peak markers: vertical latency + horizontal amplitude lines
    if latencies_by_label:
        for label, lat in latencies_by_label.items():
            try:
                color = (colors.get(label) if colors else None) or "#444"
                ls = (linestyles.get(label) if linestyles else None) or "-"
                ax_overlay.axvline(float(lat), color=color, alpha=0.7, linestyle=ls, linewidth=0.8)
            except Exception:
                continue
    if peak_amplitudes_by_label:
        for label, amp in peak_amplitudes_by_label.items():
            try:
                color = (colors.get(label) if colors else None) or "#666"
                ls = (linestyles.get(label) if linestyles else None) or "-"
                ax_overlay.axhline(float(amp), color=color, alpha=0.6, linestyle=ls, linewidth=0.8)
            except Exception:
                continue

    ax_overlay.axvline(0, color="#999", linewidth=1, alpha=0.6)
    ax_overlay.set_xlabel("Time (ms)")
    ax_overlay.set_ylabel("Amplitude (µV)")
    if xlim_ms is not None:
        ax_overlay.set_xlim(xlim_ms)
    if ylimit_uv is not None:
        ax_overlay.set_ylim((-float(ylimit_uv), float(ylimit_uv)))
    ax_overlay.grid(True, which="major", linestyle=":", linewidth=0.5, alpha=0.25)
    ax_overlay.legend(loc="upper left", fontsize=8, frameon=True, framealpha=0.9)

    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold", y=0.97)
    if subtitle:
        enhanced = subtitle
        if latencies_by_label:
            enhanced = f"{enhanced} | Vertical lines = {latency_annotation_label} Latency"
        fig.text(0.5, 0.90, enhanced, ha="center", fontsize=9)

    # Topomaps (fixed ±5 µV like the reference)
    vecs = [np.asarray(topomap_by_label[l], dtype=float) for l in labels if l in topomap_by_label]
    vlim = (-5.0, 5.0)

    highlight_set = set(str(ch) for ch in (highlight_channels or []))

    # Optional: exclude non-scalp electrodes from topomaps (EGI HydroCel)
    picks = None
    if exclude_non_scalp:
        try:
            ch_names = [ch.get("ch_name") for ch in info.get("chs", [])]
            default_non_scalp = {
                "E1", "E8", "E14", "E17", "E21", "E25", "E32", "E38", "E43", "E44",
                "E48", "E49", "E113", "E114", "E119", "E120", "E121", "E125", "E126", "E127", "E128"
            }
            exclude_set = set(non_scalp_labels or []) or default_non_scalp
            picks = [idx for idx, nm in enumerate(ch_names) if nm not in exclude_set]
            if not picks:
                picks = None
        except Exception:
            picks = None

    im = None
    # Adaptive topo title styling (more conditions → slightly smaller font and tighter padding).
    title_size = 8 if n_cols <= 4 else 7
    title_pad = 1.0 if is_sparse else 0.5
    for i, label in enumerate(labels):
        ax = fig.add_subplot(gs[1, i])
        title_color = (colors.get(label) if colors else None) or "black"
        title_text = str(label)
        try:
            if latencies_by_label and label in latencies_by_label and half_window_ms is not None:
                display_latency = float(latencies_by_label[label])
                half_win = float(half_window_ms)
                annot = str(latency_annotation_label or "FAL")
                title_text = (
                    f"{label}\n{annot} {display_latency:.1f} ms\n(±{half_win:.0f} ms)"
                )
        except Exception:
            title_text = str(label)
        ax.set_title(title_text, fontsize=title_size, color=title_color, pad=title_pad)
        vec = topomap_by_label.get(label)
        if vec is None:
            ax.axis("off")
            continue
        # Apply non-scalp filtering
        sub_vec = vec
        sub_info = info
        if picks is not None:
            try:
                sub_vec = vec[picks]
                sub_info = mne.pick_info(info, picks, copy=True)
            except Exception:
                sub_vec = vec
                sub_info = info

        mask = None
        mask_params = None
        try:
            if highlight_set:
                sub_ch_names = [ch.get("ch_name") for ch in sub_info.get("chs", [])]
                mask = np.asarray([nm in highlight_set for nm in sub_ch_names], dtype=bool)
                if mask.any():
                    mask_params = dict(
                        marker="o",
                        markerfacecolor=(1.0, 1.0, 0.0, 0.7),
                        markeredgecolor="k",
                        linewidth=0,
                        markersize=4,
                    )
                else:
                    mask = None
        except Exception:
            mask = None
            mask_params = None
        im, _ = mne.viz.plot_topomap(
            np.asarray(sub_vec, dtype=float),
            sub_info,
            axes=ax,
            show=False,
            contours=6,
            vlim=vlim,
            cmap="RdBu_r",
            mask=mask,
            mask_params=mask_params,
        )

    # Colorbar
    if im is not None:
        cax = fig.add_subplot(gs[1, -1])
        cb = fig.colorbar(im, cax=cax, label="Amplitude (µV)")
        try:
            cb.set_ticks([-5, -2.5, 0, 2.5, 5])
        except Exception:
            pass

    # ROI electrode caption (if provided)
    if highlight_set:
        roi_text = ", ".join(sorted(highlight_set))
        fig.text(0.5, 0.01, f"Yellow sensors = ERP ROI electrodes: {roi_text}", ha="center", va="bottom", fontsize=7, color="#444")

    return fig

