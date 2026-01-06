from __future__ import annotations

"""
Archived module.

This file previously generated:
- per-analysis markdown pages under `docs/analysis/`
- an auto-updated `docs/index.md` grid for browsing outputs via a local web server

The active pipeline no longer uses this module (by design: simpler repo / folder-hunting is OK).
"""

import os
from pathlib import Path
from typing import Dict, List, Optional


def write_analysis_page(
    page_path: str,
    *,
    title: str,
    figure_paths: Optional[List[str]] = None,
    notes: Optional[List[str]] = None,
) -> None:
    os.makedirs(os.path.dirname(page_path), exist_ok=True)
    page_dir = Path(page_path).parent
    with open(page_path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        if figure_paths:
            for p in figure_paths:
                try:
                    rel_path = Path(os.path.relpath(Path(p), start=page_dir)).as_posix()
                except Exception:
                    rel_path = str(p).replace("\\", "/")
                f.write(f"![figure]({rel_path})\n\n")
        if notes:
            f.write("## Notes\n\n")
            for n in notes:
                f.write(f"- {n}\n")


_START = "<!-- AUTO-GENERATED START -->"
_END = "<!-- AUTO-GENERATED END -->"


def ensure_index_template(index_path: str) -> None:
    p = Path(index_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if not p.exists():
        p.write_text(_default_index_template(), encoding="utf-8")
        return

    text = p.read_text(encoding="utf-8")
    if _START in text and _END in text:
        return

    appended = text.rstrip() + "\n\n" + _default_autogen_block() + "\n"
    p.write_text(appended, encoding="utf-8")


def update_index_grid(index_path: str, analysis_id: str, component_to_image: Dict[str, str]) -> None:
    ensure_index_template(index_path)

    p = Path(index_path)
    text = p.read_text(encoding="utf-8")

    start_i = text.find(_START)
    end_i = text.find(_END)
    if start_i == -1 or end_i == -1 or end_i <= start_i:
        raise ValueError("Index template missing AUTO-GENERATED markers")

    before = text[: start_i + len(_START)]
    block = text[start_i + len(_START) : end_i]
    after = text[end_i:]

    rows = [ln.strip() for ln in block.splitlines() if ln.strip().startswith("|")]

    def _row_key(line: str) -> str:
        parts = [p.strip() for p in line.strip("|").split("|")]
        return parts[0] if parts else ""

    rows_by_id = {_row_key(r): r for r in rows if _row_key(r)}
    rows_by_id[analysis_id] = _format_row(analysis_id, component_to_image)

    merged = [rows_by_id[k] for k in sorted(rows_by_id.keys())]
    new_block = "\n" + "\n".join(merged) + ("\n" if merged else "\n")
    p.write_text(before + new_block + after, encoding="utf-8")


def _format_row(analysis_id: str, component_to_image: Dict[str, str]) -> str:
    page = f"analysis/{analysis_id}.md"

    def _thumb(rel: str) -> str:
        rel = rel.replace("\\", "/")
        return f"[![img]({rel})]({rel})"

    cl = component_to_image.get("collapsed_localizer") or component_to_image.get("collapsed") or ""
    n1 = component_to_image.get("N1") or ""
    cl_cell = _thumb(cl) if cl else ""
    n1_cell = _thumb(n1) if n1 else ""
    return f"| [{analysis_id}]({page}) | {cl_cell} | {n1_cell} |"


def _default_autogen_block() -> str:
    return "\n".join([_START, "| Analysis | CollapsedLocalizer | N1 |", _END])


def _default_index_template() -> str:
    return "\n".join(
        [
            "# LCN ERP Analyses",
            "",
            "This index is updated automatically by `scripts/run_analysis.py`.",
            "",
            _default_autogen_block(),
            "",
        ]
    )

