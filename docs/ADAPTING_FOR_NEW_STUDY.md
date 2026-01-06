# Adapting This Repo for a New Study (RA Checklist)

This guide is for RAs adapting the pipeline for a new project (e.g., an autism ERP/MMN study).

## 1) Fork the repo

1. Fork on GitHub (or duplicate internally).
2. Clone your fork locally.
3. Create a fresh conda env:

```powershell
conda env update -f environment.yml
conda activate lcn-erp-pipeline
```

## 2) Put your data in `data/`

Create a study folder:

```
data/
  autism_study_preprocessed/
    sub-01_preprocessed-epo.fif
    sub-02_preprocessed-epo.fif
```

## 3) Create a new analysis config (copy the template)

```powershell
Copy-Item configs/analyses/_template.yaml configs/analyses/autism_mmn.yaml
```

Edit these fields:

- **dataset**
  - `root`: your folder under `data/`
  - `pattern`: file naming glob
- **selection**
  - `response`: `ALL` vs `ACC1`
  - `response_field`: accuracy column in `epochs.metadata`
  - `condition_sets`: define your conditions
- **components**
  - keep P1/N1/P3b/Fz or change as needed
- **outputs**
  - set a unique `page`, `plots_dir`, and `tables_dir`

## 4) Checklist: defining `condition_sets`

Your `condition_sets` determine what is compared in plots and stats.

### If your study has three conditions X, Y, Z

```yaml
selection:
  condition_sets:
    - name: X
      conditions: ["X"]
      color: "#1f77b4"
    - name: Y
      conditions: ["Y"]
      color: "#ff7f0e"
    - name: Z
      conditions: ["Z"]
      color: "#2ca02c"
```

### If your condition is defined by metadata columns (recommended if you have many event codes)

Example (hypothetical):

```yaml
selection:
  condition_sets:
    - name: Deviant
      conditions: []  # optional: leave empty to rely on metadata_filters
      metadata_filters:
        trial_type: ["DEV"]
    - name: Standard
      metadata_filters:
        trial_type: ["STD"]
```

Important:
- `metadata_filters` keys must exactly match columns in `epochs.metadata`.

## 5) When you need to modify `configs/components.yaml`

Add a new component when:
- Your study targets a component not listed (e.g., **MMN**)

You must define:
- `window_ms`: a priori search window for collapsed localizer (literature-based)
- `rois`: which ROI(s) to average (must exist in `configs/electrodes.yaml`)
- `localizer.polarity`: `positive` or `negative`

Example (MMN is typically negative, ~100–250 ms; confirm with your PI/RA lead):

```yaml
components:
  MMN:
    window_ms: [100, 250]
    rois: [Fz]
    anatomical_region: "Frontocentral"
    localizer:
      method: roi
      roi_names: [Fz]
      polarity: negative
```

## 6) When you need to modify `configs/electrodes.yaml`

Modify this only if:
- Your component ROI differs (e.g., MMN ROI includes FCz + neighbors)
- You want a different posterior ROI for N1/P1

Rules of thumb:
- Keep ROIs small and motivated by literature.
- Don’t cherry-pick electrodes after seeing the data.

## 7) Run and validate

Run analysis:

```powershell
python scripts/run_analysis.py --config configs/analyses/autism_mmn.yaml
```

Then check:
- `docs/assets/tables/autism_mmn/subject_measurements.csv` exists
- Plots look reasonable (ERP shapes and topographies)

Run stats by updating `configs/statistics-default.yaml` `input_csv` and `output_dir`, then:

```powershell
python scripts/run_statistics.py --config configs/statistics-default.yaml
```

## 8) “What should I change vs not change?”

Change for a new study:
- `configs/analyses/*.yaml` (yes)
- `configs/components.yaml` / `configs/electrodes.yaml` (only if needed)
- `configs/statistics-default.yaml` paths and chosen DVs/tests

Avoid changing:
- Python source in `src/erp/` unless you have a clear, reviewed reason
- Plot naming conventions and output filenames (keeps reproducibility)

