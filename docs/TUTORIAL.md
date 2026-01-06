# Tutorial: Creating Your First Analysis

This tutorial is written for RAs with basic Python familiarity.

## 1) What you need before you start

- **Preprocessed EEG epochs** saved as MNE Epochs `.fif` files
- Each `.fif` should include `epochs.metadata` with at least:
  - `Condition` (your event/condition label or code)
  - an accuracy field (e.g., `Target.ACC`, `Accuracy`, `correct`) if you want ACC1/ACC0 filtering
- The repo assumes **EGI 128-channel HydroCel** (montage at `assets/net/AdultAverageNet128_v1.sfp`)

## 2) Clone and set up the environment

```powershell
git clone <your-fork-url>
cd lcn-erp-pipeline
conda env update -f environment.yml
conda activate lcn-erp-pipeline
```

## 3) Put your data in `data/`

Create a folder for your study, then copy your `.fif` files into it.

Example:

```
data/
  autism_study_preprocessed/
    sub-01_preprocessed-epo.fif
    sub-02_preprocessed-epo.fif
    ...
```

## 4) Create your analysis YAML (the only thing you *must* edit)

Copy the template:

```powershell
Copy-Item configs/analyses/_template.yaml configs/analyses/autism_mmn.yaml
```

Edit `configs/analyses/autism_mmn.yaml`:

- Set `dataset.root` to your data folder
- Update `selection.condition_sets` to match your study conditions
- Choose `components` you care about (e.g., add `MMN` later if needed)
- Set `outputs.*` to a unique name, like:
  - `docs/assets/plots/autism_mmn`
  - `docs/assets/tables/autism_mmn`

## 5) Run the analysis

```powershell
python scripts/run_analysis.py --config configs/analyses/autism_mmn.yaml
```

What to expect:
- PNG plots in `docs/assets/plots/<analysis_id>/`
- Tables (CSV/JSON) in `docs/assets/tables/<analysis_id>/`

## 6) Run statistics

Update `configs/statistics-default.yaml`:
- `input_csv`: point to the analysis output `subject_measurements.csv`
- `output_dir`: choose a matching stats folder

Then run:

```powershell
python scripts/run_statistics.py --config configs/statistics-default.yaml
```

This produces:
- `docs/assets/stats/<analysis_id>/STATISTICAL_REPORT.md`
- ANOVA, pairwise tests, LMM, and plots (boxplots/violins/effect sizes)

## 7) View results

Open outputs directly from the filesystem:

- `docs/assets/plots/<analysis_id>/` (PNG figures)
- `docs/assets/tables/<analysis_id>/` (CSV/JSON tables)
- `docs/assets/stats/<analysis_id>/` (statistics outputs, after running stats)

## 8) How to interpret the key outputs (high-level)

- **Collapsed localizer figure**: shows how each component window was chosen (avoids circularity).
- **Component ERP plots**: each condition waveform + topographies at the chosen window.
- **subject_measurements.csv**: one row per subject × component × condition; used for stats.
- **STATISTICAL_REPORT.md**: narrative summary of LMM-first results, then ANOVA/pairwise.

## 9) Common “first run” checklist

- If you see “No FIF files found”: check `dataset.root` and `dataset.pattern`.
- If you see “Required metadata column missing”: verify your `epochs.metadata` columns.
- If plots look empty: check your `condition_sets` codes match `metadata['Condition']`.

