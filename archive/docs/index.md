# LCN ERP Pipeline Outputs (archived)

This file used to provide a browsable index of analysis outputs via a local web server.

Original contents:

# LCN ERP Pipeline Outputs

This folder is where the pipeline writes results.

## Viewing results

After running an analysis, start a local server:

```powershell
python -m http.server 8000 --directory docs
```

Then open `http://localhost:8000`.

## Analyses index (auto-generated)

This table is updated by `scripts/run_analysis.py` when you run an analysis.
Each analysis has a small landing page in `docs/analysis/<analysis_id>.md` that embeds its plots from `docs/assets/`.

<!-- AUTO-GENERATED START -->
| Analysis | CollapsedLocalizer | N1 |
| [increasing_vs_decreasing](analysis/increasing_vs_decreasing.md) | [![img](assets/plots/increasing_vs_decreasing/increasing_vs_decreasing-collapsed_localizer.png)](assets/plots/increasing_vs_decreasing/increasing_vs_decreasing-collapsed_localizer.png) | [![img](assets/plots/increasing_vs_decreasing/increasing_vs_decreasing-N1.png)](assets/plots/increasing_vs_decreasing/increasing_vs_decreasing-N1.png) |
| [example](analysis/example.md) | [![img](assets/plots/example/example-collapsed_localizer.png)](assets/plots/example/example-collapsed_localizer.png) | [![img](assets/plots/example/example-N1.png)](assets/plots/example/example-N1.png) |
<!-- AUTO-GENERATED END -->

