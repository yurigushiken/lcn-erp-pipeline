# LCN ERP Pipeline

A user-friendly, YAML-driven ERP (Event-Related Potential) analysis pipeline for EEG studies at the LCN Lab.

This repo is designed so lab members can:
- clone/fork it
- drop preprocessed EEG epochs into `data/`
- define conditions in a YAML file
- run one command
- get plots, measurements, and stats outputs in `docs/`

## What this pipeline does (plain language)

If you have EEG data, you often want to answer questions like:
- “Do condition A and condition B produce different brain responses?”
- “At what time does a component (like N1 or P3b) peak?”
- “Is the effect reliable across subjects?”

This pipeline:
- loads preprocessed EEG **epochs** (short time windows around an event)
- groups trials into **conditions** (based on event codes and metadata)
- computes **ERP waveforms** (average across trials, then across subjects)
- finds ERP components using a **collapsed localizer** (to avoid circular analysis)
- measures each subject’s amplitude/latency inside those unbiased windows
- runs statistics (LMM-first, plus ANOVA/pairwise) and saves a report
- writes all outputs into `docs/` (plots/tables/stats)

If you are new to ERPs:
- ERP = averaged EEG response time-locked to an event
- Component (P1/N1/P3b/…) = a characteristic bump in the ERP waveform
- Collapsed localizer = pick windows using data collapsed across conditions (avoids “double dipping”)

## Requirements

- **Python/Conda**: Python 3.12 via conda (recommended)
- **Environment name**: `lcn-erp-pipeline`
- **Input format**: MNE Epochs `.fif` files with **embedded** `epochs.metadata`
- **Electrode net**: EGI 128-channel HydroCel
- **Montage file**: `assets/net/AdultAverageNet128_v1.sfp`

## Quick Start (detailed)

### 1) Clone the repo

```powershell
git clone <your-repo-url>
cd lcn-erp-pipeline
```

### 2) Set up the conda environment

```powershell
conda env update -f environment.yml
conda activate lcn-erp-pipeline
```

### 3) Put your data in `data/`

Create a folder for your study and copy your `.fif` epoch files into it.

Example:

```
data/
  autism_study_preprocessed/
    sub-01_preprocessed-epo.fif
    sub-02_preprocessed-epo.fif
    ...
```

### 4) Create a new analysis config (YAML)

Copy the template:

```powershell
Copy-Item configs/analyses/_template.yaml configs/analyses/autism_mmn.yaml
```

Edit `configs/analyses/autism_mmn.yaml`:
- set `dataset.root` to your data folder
- define your conditions under `selection.condition_sets`
- update `outputs.*` to a unique name for your analysis

### 5) Run the analysis

```powershell
python scripts/run_analysis.py --config configs/analyses/autism_mmn.yaml
```

Optional: also save “publication” overlay-only figures (no topography row):

```powershell
python scripts/run_analysis.py --config configs/analyses/autism_mmn.yaml
```

Disable `*-no_topo.png` generation (rare):

```powershell
python scripts/run_analysis.py --config configs/analyses/autism_mmn.yaml --no-save-no-topo
```

### 6) Run statistics

Edit `configs/statistics-default.yaml`:
- `input_csv`: point to the `subject_measurements.csv` from your analysis
- `output_dir`: where stats outputs should go

Then run:

```powershell
python scripts/run_statistics.py --config configs/statistics-default.yaml
```

### 7) Browse results

```powershell
Explore outputs directly in folders under `docs/assets/`:

- `docs/assets/plots/<analysis_id>/` (PNG figures)
- `docs/assets/tables/<analysis_id>/` (CSV/JSON tables)
- `docs/assets/stats/<analysis_id>/` (statistics outputs, after running stats)
```

## Project structure

```
lcn-erp-pipeline/
  assets/
    net/
      AdultAverageNet128_v1.sfp
      hydrocel_net.jpg
  configs/
    analyses/
      _template.yaml
      example.yaml
    components.yaml
    electrodes.yaml
    statistics-default.yaml
  data/
    (your EEG epochs go here; not committed)
  docs/
    ADAPTING_FOR_NEW_STUDY.md
    TUTORIAL.md
    assets/
      plots/
      tables/
      stats/
  scripts/
    run_analysis.py
    run_statistics.py
  src/
    erp/
      (pipeline library code)
  tests/
    (unit + integration tests)
  environment.yml
  pytest.ini
  README.md
```

## For new studies (how to fork and adapt)

Most new studies only require editing YAML files.

### What you usually change

- Create one or more new analysis YAMLs in `configs/analyses/`
- Point `dataset.root` to your study folder under `data/`
- Define your condition sets (names + codes + optional metadata filters)
- Change `outputs.plots_dir/tables_dir` to a unique analysis name

### What you sometimes change

- `configs/components.yaml` if you need a new component (e.g., MMN)
- `configs/electrodes.yaml` if your ROI electrode list differs

### What you should not change (unless directed)

- Python code in `src/erp/` (keeps analyses reproducible and consistent)
- Output naming conventions (helps lab-wide consistency)

For a full checklist, see `docs/ADAPTING_FOR_NEW_STUDY.md`.

## Output reference (what each file means)

All outputs go into `docs/`.

### Plots (PNG)

In `docs/assets/plots/<analysis_id>/`:

- `<analysis_id>-collapsed_localizer.png`
  - One subplot per component
  - Shows search range, detected peak, and FWHM window
  - This is the “scientific justification” for the measurement window

- `<analysis_id>-<component>.png` (e.g., `...-N1.png`)
  - ERP waveforms per condition in the ROI
  - Vertical lines: per-condition latency marker (Peak or FAL)
  - Horizontal lines: per-condition amplitude marker (Peak or Mean)
  - Includes topographies at the measurement window, with ROI sensors highlighted

- `<analysis_id>-<component>-no_topo.png` (optional, publication-only)
  - ERP overlay only (no topographies)
  - Saved by default (disable with `--no-save-no-topo`)
  - Not referenced anywhere automatically (intended for manual use in figures/manuscripts)

- `<analysis_id>-P1_N1_peak_to_peak.png` (optional)
  - P1↔N1 peak-to-peak measured on the N1 ROI waveform
  - Shows waveforms + dotted horizontal lines at P1 and N1 levels

### Tables (CSV/JSON)

In `docs/assets/tables/<analysis_id>/`:

- `subject_measurements.csv`
  - One row per **subject × component × condition**
  - Primary input for statistics

- `condition_measurements.csv`
  - Aggregated (grand-average) measurements per **component × condition**
  - Useful for quick inspection and sanity checks

- `qc_summary.csv`
  - Per-subject inclusion/exclusion for each condition set

- `collapsed_localizer_results.json`
  - Detected peak latency and FWHM window per component

- `run_metrics.json`
  - Runtime + metadata about the run

### Statistics outputs

In `docs/assets/stats/<analysis_id>/`:

- `lmm_<component>_<dv>.json`
  - LMM-first primary analysis (may include a graceful error payload if the fit is singular)

- `anova_<component>_<dv>.csv`
  - Repeated-measures ANOVA (supplementary)

- `pairwise_<component>_<dv>.csv`
  - Pairwise t-tests with correction
  - Includes **Cohen’s d** and **CI bounds** when available

- `STATISTICAL_REPORT.md`
  - A readable report with results ordered: LMM → ANOVA → Pairwise

- `plots/`
  - Boxplots, violins, and effect-size plots with CI bars + thresholds

## Troubleshooting (common issues)

### “No FIF files found”

- Check `dataset.root` and `dataset.pattern` in your analysis YAML.
- Ensure your data files match the naming pattern.

### “Epochs metadata is required…”

- This pipeline expects `epochs.metadata` embedded in each `.fif`.
- Confirm by loading one file in Python and checking `epochs.metadata.columns`.

### “Required metadata column missing: 'Condition'” (or your response field)

- Your `.fif` metadata must contain:
  - the condition field (default `Condition`)
  - the response field (default `Target.ACC`) if using ACC1/ACC0
- Fix by updating `selection.response_field` (and/or your preprocessing pipeline).

### “None of the requested condition codes matched…”

- Your `selection.condition_sets[*].conditions` do not match `epochs.metadata['Condition']`.
- Print unique values of `Condition` to confirm what codes exist.

### Montage errors / unmatched channel labels

- Ensure your `.fif` channel names are EGI-style (E1…E128) and the montage path is correct.
- Default montage: `assets/net/AdultAverageNet128_v1.sfp`

### LMM “Singular matrix”

- This can happen with small sample sizes or near-identical data across conditions.
- The pipeline retries multiple optimizers; if it still fails, the LMM JSON will include an error field.
- In these cases, rely more heavily on the ANOVA/pairwise results and consult the PI.

## More RA-friendly guides

- `docs/TUTORIAL.md` (walkthrough: first analysis end-to-end)
- `docs/ADAPTING_FOR_NEW_STUDY.md` (forking + checklist for new studies)

