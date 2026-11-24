"""mokashow rm / cm: mode browser over rootdir/calib. Agg backend: no window opens."""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from astropy.io import fits
from click.testing import CliRunner
from matplotlib.backend_bases import KeyEvent

from moka.show.cli import mokashow
from moka.show.modes import modes_figure, stats


def _cube(n, shape, seed=0):
    return np.random.default_rng(seed).normal(size=(n, *shape)).astype(np.float32)


def _write(calib, kind, n=5, cm_header=None, sv=None):
    calib.mkdir(exist_ok=True)
    wfs, dm = _cube(n, (4, 6), 1), _cube(n, (3, 3), 2)
    if kind == "rm":
        fits.writeto(calib / "RMmodesWFS.fits", wfs, overwrite=True)
        fits.writeto(calib / "RMmodesDM.fits", dm, overwrite=True)
    else:
        header = fits.Header(cm_header or {})
        hdus = [fits.PrimaryHDU(wfs, header)]
        if sv is not None:
            hdus.append(fits.ImageHDU(sv, name="SINGVALS"))
        fits.HDUList(hdus).writeto(calib / "CMmodesWFS.fits", overwrite=True)
        fits.writeto(calib / "CMmodesDM.fits", dm, header, overwrite=True)
    return wfs, dm


def _key(fig, key):
    fig.canvas.callbacks.process(
        "key_press_event", KeyEvent("key_press_event", fig.canvas, key)
    )


def _shown(fig, i):
    return np.asarray(fig.axes[i].images[0].get_array())


def test_stats_over_all_pixels_or_the_mask():
    frame = np.array([[1.0, -1.0], [3.0, 5.0]])
    assert stats(frame) == "rms 3  p2v 6\nmean 2  std 2.24"
    assert stats(frame, np.array([[1, 1], [0, 0]])) == "rms 1  p2v 2\nmean 0  std 1"


@pytest.mark.parametrize("kind", ["rm", "cm"])
def test_slider_selects_the_mode_of_both_panels(tmp_path, kind):
    wfs, dm = _write(tmp_path, kind)
    fig = modes_figure(tmp_path, kind)
    np.testing.assert_array_equal(_shown(fig, 0), wfs[0])
    fig.moka.slider.set_val(3)
    np.testing.assert_array_equal(_shown(fig, 0), wfs[3])
    np.testing.assert_array_equal(_shown(fig, 1), dm[3])
    assert fig.axes[1].get_xlabel() == stats(dm[3])


def test_play_steps_and_wraps_at_the_fps_of_the_box(tmp_path):
    _write(tmp_path, "rm", n=3)
    fig = modes_figure(tmp_path, "rm")
    m = fig.moka
    m.fps.set_val("25")  # set_val also submits
    assert m.timer.interval == 40
    m.fps.set_val("-1")  # refused: back to the current rate
    assert (m.timer.interval, m.fps.text) == (40, "25")
    m.play._observers.process("clicked", None)
    assert m.play.label.get_text() == "Pause"
    seen = []
    for _ in range(4):
        m.tick()
        seen.append(int(m.slider.val))
    assert seen == [1, 2, 0, 1]
    m.play._observers.process("clicked", None)
    assert m.play.label.get_text() == "Play"


def test_masks_restrict_the_stats_and_toggle_by_key(tmp_path):
    wfs, dm = _write(tmp_path, "rm")
    fig = modes_figure(tmp_path, "rm")
    assert fig.moka.check.get_status() == [False, False]  # no mask file
    _key(fig, "d")
    assert fig.moka.check.get_status() == [False, False]  # nothing to turn on
    assert fig.axes[1].get_xlabel() == stats(dm[0])
    mask = np.zeros((3, 3), np.float32)
    mask[1:, 1:] = 1
    fits.writeto(tmp_path / "dmmask.fits", mask)
    fits.writeto(tmp_path / "wfsmask.fits", np.ones((2, 2), np.float32))  # wrong shape
    _key(fig, "r")
    assert fig.moka.check.get_status() == [False, True]
    assert fig.axes[1].get_xlabel() == stats(dm[0], mask) + "  (mask)"
    _key(fig, "d")
    assert fig.axes[1].get_xlabel() == stats(dm[0])
    _key(fig, "r")  # the operator's choice survives a reload
    assert fig.moka.check.get_status() == [False, False]


def test_r_rereads_the_cubes_and_the_mode_count(tmp_path):
    _write(tmp_path, "rm", n=5)
    fig = modes_figure(tmp_path, "rm")
    fig.moka.slider.set_val(4)
    wfs, _ = _write(tmp_path, "rm", n=3)
    _key(fig, "r")
    assert (fig.moka.slider.valmax, fig.moka.slider.val) == (2, 2)
    np.testing.assert_array_equal(_shown(fig, 0), wfs[2])


def test_cm_plots_the_singular_values_and_the_truncation(tmp_path):
    sv = np.array([10.0, 5.0, 1.0, 1e-3, 1e-9])
    header = {"CMFLAVOR": "tikhonov", "CMREG": 0.01, "CMNCUT": 4, "CMNKEPT": 4}
    _write(tmp_path, "cm", cm_header=header, sv=sv)
    fig = modes_figure(tmp_path, "cm")
    ax = fig.axes[2]
    line, reg, ncut = ax.lines
    np.testing.assert_allclose(line.get_ydata(), sv / 10)
    assert reg.get_ydata()[0] == pytest.approx(0.01)
    assert ncut.get_xdata()[0] == 3.5  # between kept 0..3 and cut 4
    assert ax.get_title() == "tikhonov  -r 0.01  -n 4  kept 4"


def test_cm_without_singvals_still_shows_the_modes(tmp_path):
    wfs, _ = _write(tmp_path, "cm")
    fig = modes_figure(tmp_path, "cm")
    assert fig.axes[2].get_title() == "no SINGVALS HDU"
    np.testing.assert_array_equal(_shown(fig, 0), wfs[0])


@pytest.mark.parametrize("kind", ["rm", "cm"])
def test_cli_shows_the_figure_or_says_what_is_missing(
    make_tmp_pipeline, monkeypatch, kind
):
    pp = make_tmp_pipeline()
    shown = []
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: shown.append(1))
    args = [kind, "-p", str(pp.root_folder)]
    result = CliRunner().invoke(mokashow, args)
    assert result.exit_code == 1 and "No " in result.output
    _write(pp.root_folder / "calib", kind, sv=np.ones(5))
    result = CliRunner().invoke(mokashow, args)
    assert result.exit_code == 0, result.output
    assert shown == [1]


def test_a_single_mode_cube_plays_in_place(tmp_path):
    wfs, _ = _write(tmp_path, "rm", n=1)
    fig = modes_figure(tmp_path, "rm")
    fig.moka.tick()
    assert fig.moka.slider.val == 0
    assert fig.axes[0].get_title() == "RMmodesWFS.fits[0]"
