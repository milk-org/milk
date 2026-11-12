"""O1 acceptance tests (py): milk-show mlat draws hardwlatency.dat and redraws on 'r'.

See specs/011-o1-steps-runstart-mlat/brief.md. Agg backend: no window opens.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("milk.show.cli")  # parked until O1 lands: P1 deletes this line

import matplotlib

matplotlib.use("Agg")

from click.testing import CliRunner
from matplotlib.backend_bases import KeyEvent

from milk.show.cli import milk_show, mlat_figure


def _write_dat(datadir: Path, peak_us: float) -> None:
    """hardwlatency.dat as mlat writes it: frame, dt [us], value; one '#' line per iteration."""
    lines = []
    for it in range(3):
        for k in range(1, 20):
            dt = (k - 2) * 1000.0
            lines.append(f"{k - 2}   {dt:10.2f}     {1.0 if dt == peak_us else 0.01:g}")
        lines.append(f"# {it:5d}  {peak_us * 1e-6:8.6f}")
    (datadir / "hardwlatency.dat").write_text("\n".join(lines) + "\n")


def _peak(fig) -> float:
    x, y = fig.axes[0].lines[0].get_data()
    return float(x[y.argmax()])


def test_mlat_figure_plots_value_against_dt(tmp_path):
    _write_dat(tmp_path, 3000.0)
    fig = mlat_figure(tmp_path)
    assert len(fig.axes[0].lines[0].get_xdata()) == 3 * 19
    assert _peak(fig) == 3000.0


def test_r_redraws_from_the_current_file(tmp_path):
    _write_dat(tmp_path, 3000.0)
    fig = mlat_figure(tmp_path)
    _write_dat(tmp_path, 5000.0)
    assert _peak(fig) == 3000.0  # nothing changes until asked
    fig.canvas.callbacks.process(
        "key_press_event", KeyEvent("key_press_event", fig.canvas, "r")
    )
    assert _peak(fig) == 5000.0


def test_cli_mlat_shows_the_figure(tmp_path, monkeypatch):
    _write_dat(tmp_path, 3000.0)
    shown = []
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: shown.append(1))
    result = CliRunner().invoke(milk_show, ["mlat", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert shown == [1]
    assert (
        CliRunner().invoke(milk_show, ["mlat", str(tmp_path / "nope")]).exit_code == 2
    )
