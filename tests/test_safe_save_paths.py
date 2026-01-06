from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _import_save_figure():
    import sys

    sys.path.insert(0, str(_repo_root() / "src"))
    from erp.plots import save_figure

    return save_figure


def test_save_figure_handles_windows_like_paths(tmp_path: Path):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])

    save_figure = _import_save_figure()

    rel_dir = tmp_path / "docs" / "assets" / "plots" / "demo"
    rel_path = rel_dir / "demo-P3b.png"
    path_str = str(rel_path).replace("/", "\\")  # windows-like

    save_figure(fig, path_str, dpi=80)
    assert rel_path.exists()


def test_save_figure_retries_on_errno22(tmp_path: Path, monkeypatch):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [1, 0])

    save_figure = _import_save_figure()

    calls = {"n": 0}
    orig_savefig = fig.savefig

    def flaky_savefig(fname, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError(22, "Invalid argument")
        return orig_savefig(fname, *args, **kwargs)

    monkeypatch.setattr(fig, "savefig", flaky_savefig)

    out = tmp_path / "docs" / "assets" / "plots" / "demo" / "demo-P3b.png"
    save_figure(fig, str(out), dpi=80)
    assert out.exists()

