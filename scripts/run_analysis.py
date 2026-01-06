#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


import json
import time
import datetime as dt

import numpy as np

from erp.config import load_analysis_config, load_components_config, load_electrodes_config
from erp.io import (
    apply_montage_sfp,
    discover_epoch_files,
    extract_subject_id,
    read_epochs,
    validate_baseline_window,
)
from erp.select import filter_epochs_by_response, select_epochs
from erp.collapsed_localizer import compute_collapsed_localizer_gfp, compute_collapsed_localizer_roi
from erp.measures import fractional_area_latency, peak_amplitude, peak_latency
from erp.plots import (
    make_collapsed_localizer_figure,
    make_component_figure,
    make_erp_figure,
    make_peak_to_peak_figure,
    save_figure,
)
def _analysis_id_from_outputs(plots_dir: str, tables_dir: str) -> str:
    """
    Derive a stable analysis_id without relying on markdown report pages.
    Convention: plots_dir and tables_dir should end in the same folder name (analysis_id).
    """
    plots_name = Path(plots_dir).name
    tables_name = Path(tables_dir).name
    return plots_name or tables_name


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run ERP analysis (collapsed localizer + ROI/GFP)")
    parser.add_argument("--config", required=True, help="Path to analysis YAML")
    parser.add_argument("--max_subjects", type=int, default=0, help="Limit subjects processed (0 = no limit)")
    no_topo_group = parser.add_mutually_exclusive_group()
    no_topo_group.add_argument(
        "--save-no-topo",
        dest="save_no_topo",
        action="store_true",
        default=True,
        help="Save overlay-only figures without topographies (*-no_topo.png). Enabled by default.",
    )
    no_topo_group.add_argument(
        "--no-save-no-topo",
        dest="save_no_topo",
        action="store_false",
        help="Disable overlay-only figures without topographies (*-no_topo.png).",
    )
    args = parser.parse_args(argv)

    cfg = load_analysis_config(args.config)

    plots_dir = cfg.outputs.get("plots_dir")
    tables_dir = cfg.outputs.get("tables_dir")
    if not plots_dir or not tables_dir:
        raise SystemExit("Config outputs must include plots_dir and tables_dir")

    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    analysis_id = _analysis_id_from_outputs(str(plots_dir), str(tables_dir))

    components_cfg = load_components_config(str(REPO_ROOT))
    electrodes_cfg = load_electrodes_config(str(REPO_ROOT))

    data_root = cfg.dataset.get("root", "data")
    pattern = cfg.dataset.get("pattern", "**/*.fif")
    # Allow absolute dataset roots, otherwise treat as repo-relative
    root_path = Path(data_root)
    if not root_path.is_absolute():
        root_path = REPO_ROOT / root_path

    fif_files = discover_epoch_files(str(root_path), pattern)
    if args.max_subjects and args.max_subjects > 0:
        fif_files = fif_files[: int(args.max_subjects)]

    if len(fif_files) == 0:
        return 1

    t_start = time.perf_counter()

    baseline = tuple(cfg.preprocessing.get("baseline_ms", [-100, 0]))
    response = str(cfg.selection.get("response", "ALL")).upper()
    response_field = str(cfg.selection.get("response_field", "Target.ACC"))
    sets = cfg.selection.get("condition_sets", []) or []
    min_epochs = int(cfg.selection.get("min_epochs_per_set", 1))
    show_sem = bool(cfg.plots.get("show_sem", False))
    try:
        ylimit_uv = float(cfg.plots.get("ylimit_uv", 6))
    except Exception:
        ylimit_uv = 6.0

    meas_cfg = getattr(cfg, "measurement", {}) or {}
    latency_mode = str(meas_cfg.get("latency", "peak")).lower()
    amplitude_mode = str(meas_cfg.get("amplitude", "peak")).lower()

    montage_sfp = str(cfg.dataset.get("montage_sfp", ""))
    montage_path = Path(montage_sfp)
    if not montage_path.is_absolute():
        montage_path = REPO_ROOT / montage_path

    # Collect evokeds per condition set
    set_name_to_evokeds: dict[str, list] = {s["name"]: [] for s in sets}
    set_name_to_total_epochs: dict[str, int] = {s["name"]: 0 for s in sets}
    qc_rows: list[dict] = []
    subject_evokeds: list[dict] = []

    for fif in fif_files:
        epochs = read_epochs(fif)
        apply_montage_sfp(epochs, str(montage_path))

        validate_baseline_window(epochs, baseline)
        epochs.apply_baseline((baseline[0] / 1000.0, baseline[1] / 1000.0))

        subj_id = extract_subject_id(epochs, fif)

        for item in sets:
            name = str(item["name"])
            codes = [str(c) for c in (item.get("conditions") or [])]
            metadata_filters = item.get("metadata_filters") or {}
            set_response = str(item.get("response") or response).upper()

            try:
                ep_r = filter_epochs_by_response(epochs, response=set_response, response_field=response_field)
                sub = select_epochs(
                    ep_r,
                    condition_codes=codes,
                    metadata_filters=metadata_filters,
                    condition_field=str(item.get("condition_field") or "Condition"),
                    set_name=name,
                )
            except Exception as e:
                qc_rows.append(
                    {
                        "subject": subj_id,
                        "set": name,
                        "included": False,
                        "epoch_count": 0,
                        "exclusion_reason": str(e),
                    }
                )
                continue

            epoch_count = int(len(sub))
            if epoch_count < min_epochs:
                qc_rows.append(
                    {
                        "subject": subj_id,
                        "set": name,
                        "included": False,
                        "epoch_count": epoch_count,
                        "exclusion_reason": f"insufficient_epochs(<{min_epochs})",
                    }
                )
                continue

            evk = sub.average()
            # Tag evoked with subject id so collapsed-localizer can collapse per-subject first.
            # (MNE uses evoked.comment as a free-form string label.)
            try:
                evk.comment = str(subj_id)
            except Exception:
                pass
            set_name_to_evokeds[name].append(evk)
            set_name_to_total_epochs[name] += epoch_count
            subject_evokeds.append({"subject_id": subj_id, "condition": name, "evoked": evk, "n_epochs": epoch_count})
            qc_rows.append(
                {
                    "subject": subj_id,
                    "set": name,
                    "included": True,
                    "epoch_count": epoch_count,
                    "exclusion_reason": "",
                }
            )

    total_evokeds = sum(len(v) for v in set_name_to_evokeds.values())
    if total_evokeds == 0:
        return 1

    # Collapsed localizer per component
    collapsed_results: dict[str, dict] = {}
    failed_components: list[str] = []
    for comp in cfg.components:
        comp_cfg = components_cfg.get(comp, {}) if isinstance(components_cfg, dict) else {}
        sr = comp_cfg.get("window_ms")
        if not sr or len(sr) != 2:
            failed_components.append(comp)
            continue
        loc_cfg = comp_cfg.get("localizer", {}) if isinstance(comp_cfg, dict) else {}
        method = str(loc_cfg.get("method", "roi")).lower()
        if method == "gfp":
            res = compute_collapsed_localizer_gfp(
                evokeds_by_set=set_name_to_evokeds, component_name=comp, search_range_ms=tuple(sr)
            )
        else:
            roi_names = loc_cfg.get("roi_names") or comp_cfg.get("rois") or []
            roi_channels: list[str] = []
            for rn in roi_names:
                if rn in electrodes_cfg:
                    roi_channels.extend(electrodes_cfg[rn])
            if not roi_channels:
                failed_components.append(comp)
                continue
            polarity = str(loc_cfg.get("polarity", ("negative" if comp.upper().startswith("N") else "positive")))
            res = compute_collapsed_localizer_roi(
                evokeds_by_set=set_name_to_evokeds,
                roi_channels=roi_channels,
                component_name=comp,
                search_range_ms=tuple(sr),
                polarity=polarity,
            )
        collapsed_results[comp] = res

    # Collapsed localizer figure
    first_list = next((v for v in set_name_to_evokeds.values() if v), None)
    if first_list is None:
        return 1
    first_evoked = first_list[0]
    epoch_end_ms = float(first_evoked.times[-1] * 1000.0)
    xlim_ms = (float(baseline[0]), epoch_end_ms)
    cl_fig = make_collapsed_localizer_figure(
        localizer_results=collapsed_results,
        title=f"{analysis_id}: Collapsed Localizer",
        subtitle=f"baseline {baseline} ms; collapsed across all conditions; FWHM windows",
        xlim_ms=xlim_ms,
    )
    cl_out = Path(plots_dir) / f"{analysis_id}-collapsed_localizer.png"
    save_figure(cl_fig, str(cl_out), dpi=int(cfg.plots.get("dpi", 300)))
    import matplotlib.pyplot as plt

    plt.close(cl_fig)

    # Track saved figures for run_metrics (count only).
    saved_figs: list[str] = [str(Path(cl_out).resolve())]

    # Subject-level measurements
    subject_rows: list[dict] = []
    baseline_start, baseline_end = float(baseline[0]), float(baseline[1])

    for rec in subject_evokeds:
        subj_id = rec["subject_id"]
        cond = rec["condition"]
        evk = rec["evoked"]
        times_ms = (evk.times * 1000.0).astype(float)
        baseline_mask = (times_ms >= baseline_start) & (times_ms <= baseline_end)

        for comp in cfg.components:
            res = collapsed_results.get(comp)
            if not res:
                continue
            comp_cfg = components_cfg.get(comp, {}) if isinstance(components_cfg, dict) else {}
            roi_names = comp_cfg.get("rois") or []
            roi_channels: list[str] = []
            for rn in roi_names:
                if rn in electrodes_cfg:
                    roi_channels.extend(electrodes_cfg[rn])
            if not roi_channels:
                continue

            ws = float(res["window_start_ms"])
            we = float(res["window_end_ms"])
            window_mask = (times_ms >= ws) & (times_ms <= we)
            if not np.any(window_mask):
                continue

            polarity = str((comp_cfg.get("localizer") or {}).get("polarity") or ("negative" if comp.upper().startswith("N") else "positive"))

            roi_ev = evk.copy().pick_channels(roi_channels, ordered=False)
            roi_curve = (roi_ev.data.mean(axis=0) * 1e6).astype(float)

            baseline_noise = float(np.std(roi_curve[baseline_mask])) if np.any(baseline_mask) else float("nan")
            mean_amp = float(np.mean(roi_curve[window_mask]))
            try:
                peak_amp = float(peak_amplitude(roi_curve, times_ms, (ws, we), polarity=polarity))
            except Exception:
                peak_amp = float("nan")
            try:
                pk_lat = float(peak_latency(roi_curve, times_ms, (ws, we), polarity=polarity))
            except Exception:
                pk_lat = float("nan")
            try:
                fal = float(fractional_area_latency(roi_curve, times_ms, (ws, we), fraction=0.5, polarity=polarity))
            except Exception:
                fal = float("nan")

            snr = abs(mean_amp) / baseline_noise if baseline_noise and baseline_noise > 0 else float("nan")

            subject_rows.append(
                {
                    "subject_id": subj_id,
                    "component": comp,
                    "condition": cond,
                    "latency_frac_area_ms": fal,
                    "mean_amplitude_roi": mean_amp,
                    "peak_latency_ms": pk_lat,
                    "peak_amplitude_roi": peak_amp,
                    "n_epochs": int(rec["n_epochs"]),
                    "baseline_noise_uv": baseline_noise,
                    "snr": snr,
                    "collapsed_peak_latency_ms": float(res.get("peak_latency_ms", float("nan"))),
                    "window_start_ms": ws,
                    "window_end_ms": we,
                    "fwhm_ms": float(res.get("fwhm_ms", float("nan"))),
                }
            )

    import csv

    subj_csv = Path(tables_dir) / "subject_measurements.csv"
    if subject_rows:
        with open(subj_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(subject_rows[0].keys()))
            writer.writeheader()
            writer.writerows(subject_rows)

    # Component ERP overlay plots + condition_measurements.csv
    condition_rows: list[dict] = []
    for comp in cfg.components:
        res = collapsed_results.get(comp)
        comp_cfg = components_cfg.get(comp, {}) if isinstance(components_cfg, dict) else {}
        roi_names = comp_cfg.get("rois") or []
        roi_channels: list[str] = []
        for rn in roi_names:
            if rn in electrodes_cfg:
                roi_channels.extend(electrodes_cfg[rn])
        if not roi_channels:
            continue

        ws = float(res["window_start_ms"]) if res else float("nan")
        we = float(res["window_end_ms"]) if res else float("nan")
        polarity = str((comp_cfg.get("localizer") or {}).get("polarity") or ("negative" if comp.upper().startswith("N") else "positive"))

        curves_by_label: dict[str, np.ndarray] = {}
        sem_by_label: dict[str, np.ndarray] = {}
        topomap_by_label: dict[str, np.ndarray] = {}
        latencies_by_label: dict[str, float] = {}
        peak_amps_by_label: dict[str, float] = {}
        times_ms = None
        gav_info = None
        half_window_ms = float(res["fwhm_ms"]) / 2.0 if res and "fwhm_ms" in res else None
        for set_name, evoked_list in set_name_to_evokeds.items():
            if len(evoked_list) == 0:
                continue
            # ROI curves per subject for SEM
            subj_curves = []
            for evk in evoked_list:
                if times_ms is None:
                    times_ms = (evk.times * 1000.0).astype(float)
                try:
                    roi_ev = evk.copy().pick_channels(roi_channels, ordered=False)
                    subj_curves.append((roi_ev.data.mean(axis=0) * 1e6).astype(float))
                except Exception:
                    continue
            if not subj_curves:
                continue
            stacked = np.stack(subj_curves, axis=0)
            mean_curve = stacked.mean(axis=0)
            sem_curve = stacked.std(axis=0, ddof=1) / np.sqrt(stacked.shape[0]) if stacked.shape[0] > 1 else np.zeros_like(mean_curve)
            curves_by_label[set_name] = mean_curve
            sem_by_label[set_name] = sem_curve

            # Topomap vector from grand average within collapsed-localizer window
            if res and np.isfinite(ws) and np.isfinite(we):
                try:
                    import mne

                    gav = mne.grand_average(evoked_list)
                    if gav_info is None:
                        gav_info = gav.info
                    evk_win = gav.copy().crop(tmin=ws / 1000.0, tmax=we / 1000.0)
                    topomap_by_label[set_name] = (evk_win.data.mean(axis=1) * 1e6).astype(float)
                except Exception:
                    pass

            if res and times_ms is not None:
                window_mask = (times_ms >= ws) & (times_ms <= we)
                if np.any(window_mask):
                    mean_amp = float(np.mean(mean_curve[window_mask]))
                    try:
                        pk_amp = float(peak_amplitude(mean_curve, times_ms, (ws, we), polarity=polarity))
                    except Exception:
                        pk_amp = float("nan")
                    try:
                        fal = float(fractional_area_latency(mean_curve, times_ms, (ws, we), fraction=0.5, polarity=polarity))
                    except Exception:
                        fal = float("nan")
                    try:
                        pk_lat = float(peak_latency(mean_curve, times_ms, (ws, we), polarity=polarity))
                    except Exception:
                        pk_lat = float("nan")

                    # Selected markers for overlay (Peak vs FAL, Peak vs Mean)
                    try:
                        latencies_by_label[set_name] = float(pk_lat if latency_mode == "peak" else fal)
                    except Exception:
                        pass
                    try:
                        peak_amps_by_label[set_name] = float(pk_amp if amplitude_mode == "peak" else mean_amp)
                    except Exception:
                        pass

                    condition_rows.append(
                        {
                            "analysis_id": analysis_id,
                            "component": comp,
                            "condition": set_name,
                            "n_subjects": int(len(subj_curves)),
                            "collapsed_peak_latency_ms": float(res.get("peak_latency_ms", float("nan"))),
                            "window_start_ms": ws,
                            "window_end_ms": we,
                            "fwhm_ms": float(res.get("fwhm_ms", float("nan"))),
                            "mean_amplitude_roi": mean_amp,
                            "peak_amplitude_roi": pk_amp,
                            "latency_frac_area_ms": fal,
                            "latency_peak_ms": pk_lat,
                            "roi_channels": ";".join(roi_channels),
                        }
                    )

        if curves_by_label and times_ms is not None:
            subtitle = (
                f"baseline {baseline} ms; FWHM window [{ws:.1f}, {we:.1f}] ms"
                if res
                else f"baseline {baseline} ms; window unavailable"
            )
            # Colors/linestyles come from YAML condition_sets
            condition_cfg_map = {str(s.get("name")): s for s in sets}
            colors_map = {
                name: str(condition_cfg_map[name].get("color"))
                for name in curves_by_label.keys()
                if name in condition_cfg_map and condition_cfg_map[name].get("color")
            }
            linestyles_map = {
                name: str(condition_cfg_map[name].get("linestyle"))
                for name in curves_by_label.keys()
                if name in condition_cfg_map and condition_cfg_map[name].get("linestyle")
            }
            if res and gav_info is not None and topomap_by_label:
                fig = make_component_figure(
                    curves_by_label=curves_by_label,
                    sem_by_label=sem_by_label,
                    times_ms=times_ms,
                    topomap_by_label=topomap_by_label,
                    info=gav_info,
                    title=f"{analysis_id}: {comp} ({response})",
                    subtitle=subtitle,
                    xlim_ms=xlim_ms,
                    ylimit_uv=ylimit_uv,
                    epochs_by_label=set_name_to_total_epochs,
                    colors=colors_map,
                    linestyles=linestyles_map,
                    show_sem=show_sem,
                    latencies_by_label=latencies_by_label,
                    peak_amplitudes_by_label=peak_amps_by_label,
                    half_window_ms=half_window_ms,
                    highlight_channels=roi_channels,
                    exclude_non_scalp=bool(cfg.plots.get("exclude_non_scalp", True)),
                    non_scalp_labels=cfg.plots.get("non_scalp_labels") or None,
                    latency_annotation_label=("Peak" if latency_mode == "peak" else "FAL"),
                    include_topomaps=True,
                )
            else:
                fig = make_erp_figure(
                    curves_by_label=curves_by_label,
                    sem_by_label=sem_by_label,
                    times_ms=times_ms,
                    title=f"{analysis_id}: {comp} ({response})",
                    subtitle=subtitle,
                    xlim_ms=xlim_ms,
                    ylimit_uv=ylimit_uv,
                    epochs_by_label=set_name_to_total_epochs,
                    colors=colors_map,
                    linestyles=linestyles_map,
                    show_sem=show_sem,
                    latencies_by_label=latencies_by_label,
                    peak_amplitudes_by_label=peak_amps_by_label,
                    latency_annotation_label=("Peak" if latency_mode == "peak" else "FAL"),
                )
            out_path = Path(plots_dir) / f"{analysis_id}-{comp}.png"
            save_figure(fig, str(out_path), dpi=int(cfg.plots.get("dpi", 300)), bbox_inches="tight")
            plt.close(fig)
            saved_figs.append(str(Path(out_path).resolve()))

            # Optional publication variant: overlay-only (no topomaps)
            if args.save_no_topo and res and gav_info is not None and topomap_by_label:
                fig2 = make_component_figure(
                    curves_by_label=curves_by_label,
                    sem_by_label=sem_by_label,
                    times_ms=times_ms,
                    topomap_by_label=topomap_by_label,
                    info=gav_info,
                    title=f"{analysis_id}: {comp} ({response})",
                    subtitle=subtitle,
                    xlim_ms=xlim_ms,
                    ylimit_uv=ylimit_uv,
                    epochs_by_label=set_name_to_total_epochs,
                    colors=colors_map,
                    linestyles=linestyles_map,
                    show_sem=show_sem,
                    latencies_by_label=latencies_by_label,
                    peak_amplitudes_by_label=peak_amps_by_label,
                    half_window_ms=half_window_ms,
                    highlight_channels=roi_channels,
                    latency_annotation_label=("Peak" if latency_mode == "peak" else "FAL"),
                    include_topomaps=False,
                )
                out_path2 = Path(plots_dir) / f"{analysis_id}-{comp}-no_topo.png"
                save_figure(fig2, str(out_path2), dpi=int(cfg.plots.get("dpi", 300)), bbox_inches="tight")
                plt.close(fig2)

    # Peak-to-peak (bar)
    p2p_cfg = (cfg.measurement or {}).get("peak_to_peak", {}) if hasattr(cfg, "measurement") else {}
    p2p_enabled = bool(p2p_cfg.get("enabled", False)) or ({"P1", "N1"} <= {str(c).upper() for c in cfg.components})
    if p2p_enabled and ("P1" in collapsed_results) and ("N1" in collapsed_results):
        n1_roi_names = p2p_cfg.get("roi") or (components_cfg.get("N1", {}) or {}).get("rois") or ["N1_L", "N1_R"]
        n1_roi_channels: list[str] = []
        for rn in n1_roi_names:
            if rn in electrodes_cfg:
                n1_roi_channels.extend(electrodes_cfg[rn])
        n1_roi_channels = list(dict.fromkeys(n1_roi_channels))

        p1_win = (float(collapsed_results["P1"]["window_start_ms"]), float(collapsed_results["P1"]["window_end_ms"]))
        n1_win = (float(collapsed_results["N1"]["window_start_ms"]), float(collapsed_results["N1"]["window_end_ms"]))

        p2p_rows = []
        p2p_by_label = {}
        p1_lat_by = {}
        n1_lat_by = {}
        p1_amp_by = {}
        n1_amp_by = {}
        p2p_curves = {}
        p2p_times_ms = None
        for set_name, evoked_list in set_name_to_evokeds.items():
            if not evoked_list:
                continue
            import mne

            gav = mne.grand_average(evoked_list)
            times_ms = (gav.times * 1000.0).astype(float)
            if p2p_times_ms is None:
                p2p_times_ms = times_ms
            roi_ev = gav.copy().pick_channels(n1_roi_channels, ordered=False)
            roi_curve = (roi_ev.data.mean(axis=0) * 1e6).astype(float)
            from erp.measures import compute_peak_to_peak_metrics

            metrics = compute_peak_to_peak_metrics(
                signal=roi_curve,
                times_ms=times_ms,
                p1_window_ms=p1_win,
                n1_window_ms=n1_win,
                p_polarity="positive",
                n_polarity="negative",
            )
            p2p_curves[set_name] = roi_curve
            p2p_by_label[set_name] = float(metrics["p2p_amp"])
            p1_lat_by[set_name] = float(metrics["p1_lat_ms"])
            n1_lat_by[set_name] = float(metrics["n1_lat_ms"])
            p1_amp_by[set_name] = float(metrics["p1_amp"])
            n1_amp_by[set_name] = float(metrics["n1_amp"])
            p2p_rows.append(
                {
                    "analysis_id": analysis_id,
                    "condition": set_name,
                    "p1_amp": float(metrics["p1_amp"]),
                    "n1_amp": float(metrics["n1_amp"]),
                    "p2p_amp": float(metrics["p2p_amp"]),
                    "p1_lat_ms": float(metrics["p1_lat_ms"]),
                    "n1_lat_ms": float(metrics["n1_lat_ms"]),
                    "roi_channels": ";".join(n1_roi_channels),
                }
            )

        if p2p_rows:
            p2p_csv = Path(tables_dir) / "peak_to_peak_measurements.csv"
            with open(p2p_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(p2p_rows[0].keys()))
                writer.writeheader()
                writer.writerows(p2p_rows)

            condition_cfg_map = {str(s.get("name")): s for s in sets}
            colors_map = {
                name: str(condition_cfg_map[name].get("color"))
                for name in p2p_curves.keys()
                if name in condition_cfg_map and condition_cfg_map[name].get("color")
            }
            linestyles_map = {
                name: str(condition_cfg_map[name].get("linestyle"))
                for name in p2p_curves.keys()
                if name in condition_cfg_map and condition_cfg_map[name].get("linestyle")
            }

            fig = make_peak_to_peak_figure(
                curves_by_label=p2p_curves,
                times_ms=p2p_times_ms if p2p_times_ms is not None else times_ms,
                p2p_by_label=p2p_by_label,
                p1_lat_by_label=p1_lat_by,
                n1_lat_by_label=n1_lat_by,
                p1_amp_by_label=p1_amp_by,
                n1_amp_by_label=n1_amp_by,
                colors=colors_map,
                linestyles=linestyles_map,
                xlim_ms=xlim_ms,
                ylimit_uv=ylimit_uv,
                hline_color=str(p2p_cfg.get("hline_color", "#000000")),
                hline_style=str(p2p_cfg.get("hline_style", ":")),
                epochs_by_label=set_name_to_total_epochs,
                title=f"{analysis_id}: P1↔N1 peak-to-peak ({response})",
                subtitle="Using N1 electrode montage (bilateral POT)",
            )
            out_path = Path(plots_dir) / f"{analysis_id}-P1_N1_peak_to_peak.png"
            save_figure(fig, str(out_path), dpi=int(cfg.plots.get("dpi", 300)), bbox_inches="tight")
            plt.close(fig)
            saved_figs.append(str(Path(out_path).resolve()))

    # Save condition_measurements + qc_summary + collapsed JSON + run_metrics
    if condition_rows:
        cond_csv = Path(tables_dir) / "condition_measurements.csv"
        with open(cond_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(condition_rows[0].keys()))
            writer.writeheader()
            writer.writerows(condition_rows)

    if qc_rows:
        qc_csv = Path(tables_dir) / "qc_summary.csv"
        with open(qc_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["subject", "set", "included", "epoch_count", "exclusion_reason"])
            writer.writeheader()
            writer.writerows(qc_rows)

    collapsed_payload = {
        "analysis_id": analysis_id,
        "date_analyzed": dt.datetime.now().isoformat(),
        "baseline_ms": list(baseline),
        "response_filter": response,
        "fal_fraction": 0.5,
        "n_subjects_total": len(fif_files),
        "n_evokeds_included": total_evokeds,
        "conditions": [s["name"] for s in sets],
        "components": {},
    }
    for comp, res in collapsed_results.items():
        collapsed_payload["components"][comp] = {
            "collapsed_peak_latency_ms": float(res.get("peak_latency_ms", float("nan"))),
            "peak_localizer_amplitude": float(res.get("peak_amplitude", float("nan"))),
            "fwhm_ms": float(res.get("fwhm_ms", float("nan"))),
            "window_start_ms": float(res.get("window_start_ms", float("nan"))),
            "window_end_ms": float(res.get("window_end_ms", float("nan"))),
            "search_range_ms": list(res.get("search_range_ms", [])) if "search_range_ms" in res else None,
            "method": str(res.get("method", "")),
            "n_subjects": int(res.get("n_subjects", 0)),
            "n_conditions": int(res.get("n_conditions", 0)),
        }
    (Path(tables_dir) / "collapsed_localizer_results.json").write_text(
        json.dumps(collapsed_payload, indent=2), encoding="utf-8"
    )

    duration_s = float(time.perf_counter() - t_start)
    metrics = {
        "analysis_id": analysis_id,
        "duration_seconds": round(duration_s, 3),
        "num_figures": len(saved_figs),
    }
    (Path(tables_dir) / "run_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

