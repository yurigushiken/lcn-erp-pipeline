from __future__ import annotations

from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_make_component_figure_include_topomaps_false_renders_overlay_only():
    import sys

    sys.path.insert(0, str(_repo_root() / "src"))
    from erp.plots import make_component_figure

    curves_by_label = {"A": [0.0, 1.0, 0.0], "B": [0.0, 0.5, 0.0]}
    sem_by_label = {"A": [0.0, 0.1, 0.0], "B": [0.0, 0.1, 0.0]}
    times_ms = [0.0, 10.0, 20.0]

    fig = make_component_figure(
        curves_by_label=curves_by_label,
        sem_by_label=sem_by_label,
        times_ms=times_ms,
        topomap_by_label={},  # ignored when include_topomaps=False
        info=None,  # ignored when include_topomaps=False
        title="Demo",
        subtitle="Sub",
        include_topomaps=False,
    )

    # Overlay-only figure should have a single axes.
    assert len(fig.axes) == 1

