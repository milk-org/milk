"""The takeref and takedark plans, the TakeRef and TakeDark tasks: real aol0_ streams, a writer thread."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest
from astropy.io import fits
from pyMilk.interfacing.shm import SHM

from moka.errors import TaskCannotRunError, TaskDidNotSucceedError
from moka.operate.plans import PLANS, Step
from moka.operate.tasks import TASKS

SHAPE = (4, 3)  # symcode 0: numpy (x, y)


@pytest.fixture
def tr(make_tmp_pipeline, pp_env):
    """pp with both streams, mlat's framerate of 100 Hz, and a write(frames) helper."""
    pp = make_tmp_pipeline()
    for name in ("aol0_imWFS1", "aol0_wfsref", "aol0_wfsrefc"):
        SHM(name, np.zeros(SHAPE, np.float32), symcode=0).close()
    datadir = pp.work_folder / "fps.basic_mlat.datadir"
    datadir.mkdir(parents=True)
    (datadir / "cacaovars.bash").write_text(
        "export CACAO_WFSFRATE=100.0\nexport CACAO_LATENCYHARDWFR=2.5\n"
    )

    def write(
        frames: int, value=lambda k: k, stream="aol0_imWFS1", dtype=np.float32
    ) -> threading.Thread:
        def run():
            shm = SHM(stream, symcode=0)
            time.sleep(0.2)  # the task flushes the semaphore first
            for k in range(frames):
                shm.set_data(np.full(SHAPE, value(k), dtype))
                time.sleep(0.002)
            shm.close()

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        return thread

    pp.write = write
    pp.do = lambda **params: pp.task_do(TASKS["TakeRef"](pp, None, params))
    return pp


def _ref(name="aol0_wfsref") -> np.ndarray:
    shm = SHM(name, symcode=0)
    data = shm.get_data()
    shm.close()
    return data


def test_plan_steps_and_its_n_t_options():
    plan = PLANS["takeref"]
    assert plan.steps == (Step("RunStart", ("acquWFS",)), Step("TakeRef"))
    nframes, seconds = plan.params
    assert (nframes.name, nframes.type, nframes.short) == ("nframes", int, "n")
    assert (seconds.name, seconds.type, seconds.short) == ("seconds", float, "t")


def test_n_averages_that_many_frames_as_float32(tr):
    tr.write(10)
    tr.do(nframes=4)
    ref = _ref()
    assert ref.dtype == np.float32
    np.testing.assert_allclose(ref, 1.5)  # (0+1+2+3)/4
    np.testing.assert_array_equal(_ref("aol0_wfsrefc"), ref)
    path = tr.root_folder / "calib" / "wfsref.fits"
    np.testing.assert_array_equal(fits.getdata(path), ref.T)
    header = fits.getheader(path)
    assert (header["NAXIS1"], header["NAXIS2"], header["NFRAMES"]) == (*SHAPE, 4)


def test_t_uses_mlat_framerate(tr):
    tr.write(30, lambda k: 1.0 if k < 10 else 100.0)
    tr.do(seconds=0.1)  # 10 frames at 100 Hz
    np.testing.assert_allclose(_ref(), 1.0)


@pytest.mark.parametrize(
    "params, reason",
    [
        ({"nframes": 3, "seconds": 1.0}, "-n and -t are mutually exclusive"),
        ({"nframes": 0}, "-n 0 is below 1"),
        ({"seconds": 0.0}, "-t 0.0 is not positive"),
    ],
)
def test_refusals(tr, params, reason):
    with pytest.raises(TaskCannotRunError, match=reason):
        tr.do(**params)


def test_t_needs_mlat_but_n_does_not(tr):
    (tr.work_folder / "fps.basic_mlat.datadir" / "cacaovars.bash").unlink()
    with pytest.raises(TaskCannotRunError, match="run `moka mlat` or give -n"):
        tr.do()
    tr.write(5)
    tr.do(nframes=2)


def test_no_frame_fails(tr, monkeypatch):
    monkeypatch.setattr("moka.calibs.takeref.FRAME_TIMEOUT", 0.1)
    with pytest.raises(TaskDidNotSucceedError, match="after 0 of 2"):
        tr.do(nframes=2)


def test_refuses_without_wfsrefc(tr):
    SHM("aol0_wfsrefc", symcode=0).destroy()
    with pytest.raises(TaskCannotRunError, match="stream aol0_wfsrefc does not exist"):
        tr.do(nframes=1)


def test_takedark_plan_shares_takeref_params():
    plan = PLANS["takedark"]
    assert plan.steps == (Step("TakeDark"),)
    assert plan.params is PLANS["takeref"].params


def test_takedark_averages_wfsim_and_creates_wfsdark(tr):
    SHM("aol0_wfsim", np.zeros(SHAPE, np.uint16), symcode=0).close()
    tr.write(10, stream="aol0_wfsim", dtype=np.uint16)
    tr.task_do(TASKS["TakeDark"](tr, None, {"nframes": 4}))
    dark = _ref("aol0_wfsdark")
    assert dark.dtype == np.float32
    np.testing.assert_allclose(dark, 1.5)
    path = tr.root_folder / "calib" / "wfsdark.fits"
    np.testing.assert_array_equal(fits.getdata(path), dark.T)


def test_takedark_refuses_a_wfsdark_of_another_shape(tr):
    SHM("aol0_wfsim", np.zeros(SHAPE, np.uint16), symcode=0).close()
    SHM("aol0_wfsdark", np.zeros((2, 2), np.float32), symcode=0).close()
    with pytest.raises(TaskCannotRunError, match=r"aol0_wfsdark is \(2, 2\)"):
        tr.task_do(TASKS["TakeDark"](tr, None, {"nframes": 1}))
