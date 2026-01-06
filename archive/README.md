## Archive

This folder holds files we removed from the active pipeline to keep the repo simpler for lab-wide use.

### What was archived (and why)
- **`docs/index.md` + `docs/analysis/*.md`**: previously used to browse results via a local web server (`python -m http.server`).
- **`src/erp/report.py`**: code that generated those markdown pages and auto-updated the index.

The pipeline outputs we still rely on are unchanged:
- `docs/assets/plots/<analysis_id>/`
- `docs/assets/tables/<analysis_id>/`
- `docs/assets/stats/<analysis_id>/`

