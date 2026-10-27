from __future__ import annotations

import os, time

import pytest

pmp = pytest.mark.parametrize

from pyMilk.interfacing.fps import FPS, FPSDoesntExistError
from pyMilk.interfacing.shm import SHM
import numpy as np

from milk.compute_unit import ComputeUnit


class StreamDelayComputeUnit(ComputeUnit):

    def __init__(self, fpsname: str = "streamdelay") -> None:
        super().__init__("milk-fpsexec-mem-streamdelay", fpsname)


def func_fpsinit(pinfo: bool = True) -> StreamDelayComputeUnit:
    # As fixture
    compu = StreamDelayComputeUnit()
    assert compu.fps is None

    compu.fpsinit(pinfo)
    assert compu.fps

    if pinfo:
        assert compu.procinfo
        assert "procinfo.enabled" in compu.fps
        assert compu.fps["procinfo.enabled"]
    else:
        assert compu.procinfo is None
        assert not "procinfo.enabled" in compu.fps

    # Configure
    # default imin, imout, delaysec = 0.1 ms, naive = OFF, timebuffsize = 1000

    return compu


def tp(call):
    # Call the callable and silently catch any exception - good for fixturized cleanups
    try:
        call()
    except:
        pass


def func_compute_unit_cleanup(s: ComputeUnit):
    tp(s.runstop)
    tp(s.confstop)
    if s.fps:
        tp(s.fps.tmux_stop)
        tp(s.fps.destroy)


@pytest.fixture
def fixt_fpsinit_pinfo():
    s = func_fpsinit(True)
    yield s
    func_compute_unit_cleanup(s)


@pytest.fixture
def fixt_fpsinit_nopinfo():
    s = func_fpsinit(False)
    yield s
    func_compute_unit_cleanup(s)


# ADD AFTER THIS LINE


def test_fps_lifecycle(fixt_fpsinit_pinfo, tmux: bool = False):
    compu: StreamDelayComputeUnit = fixt_fpsinit_pinfo
    fps = compu.fps
    assert fps

    fps["naive_mode"] = True
    fps["procinfo.loopcntMax"] = -1
    fps["procinfo.triggermode"] = 3
    fps["procinfo.triggersname"] = "imin"

    compu.confstart(tmux=tmux)
    time.sleep(0.1)
    assert fps.conf_isrunning()

    arr = np.random.randn(30, 40)
    in_shm = SHM("imin", arr * 0)

    compu.runstart(tmux=tmux)
    time.sleep(1.0)
    assert fps.run_isrunning()  # Wait this should NOT have asserted.

    out_shm = SHM("imout")

    assert out_shm.get_data(True, timeout=0.1, return_none_on_timeout=True) is None
    in_shm.set_data(arr)
    arr2 = out_shm.get_data(True)
    np.testing.assert_allclose(arr, arr2)

    out_shm.destroy()


@pmp("loopmode", [0, 1, 2])
@pmp("tmux", (False, True))
def test_loops(fixt_fpsinit_pinfo, loopmode: int, tmux: bool):
    compu: StreamDelayComputeUnit = fixt_fpsinit_pinfo
    assert compu.fps
    compu.fps["naive_mode"] = True
    compu.fps["procinfo.triggersname"] = "imin"
    compu.fps["in_name"] = "imin"
    # This should be enough and loops should autoconf the semaphore trigger freerun

    compu.confstart()
    time.sleep(0.1)
    assert compu.fps.conf_isrunning()

    arr = np.random.randn(30, 40)
    in_shm = SHM("imin", arr * 0)

    if loopmode == 0:
        compu.runstart(tmux=tmux)
    elif loopmode == 1:
        compu.runstart(tmux=tmux, loops=True)
    elif loopmode == 2:
        compu.runstart(tmux=tmux, loopd=0.01)
    else:
        raise ValueError(f"loopmode [0,1,2] - got {loopmode}")

    time.sleep(1.0)

    in_shm.set_data(in_shm.get_data())
    time.sleep(0.1)
    in_shm.set_data(in_shm.get_data())
    out_shm = SHM("imout")

    if loopmode == 0:
        for _ in range(100):
            print(compu.fps.run_isrunning())
            time.sleep(0.01)
        assert not compu.fps.run_isrunning()
        assert compu.fps["procinfo.loopcntMax"] == 1
        assert out_shm.md.cnt0 == 1
    elif loopmode == 1:
        assert compu.fps.run_isrunning()
        assert compu.fps["procinfo.loopcntMax"] == -1
        assert out_shm.md.cnt0 == 2
    elif loopmode == 2:
        assert compu.fps.run_isrunning()
        assert compu.fps["procinfo.loopcntMax"] == -1
        assert out_shm.md.cnt0 >= 50

    out_shm.destroy()


def test_loopd(): ...


def test_raises_on_missing_fps(fixt_fpsinit_pinfo):
    compu: StreamDelayComputeUnit = fixt_fpsinit_pinfo
    assert compu.fps

    compu.fps.destroy()

    with pytest.raises(FPSDoesntExistError):
        compu._trylink_fps(True)


def test_str(fixt_fpsinit_pinfo):
    compu: StreamDelayComputeUnit = fixt_fpsinit_pinfo
    s = str(compu)
    # Note that there are ANSI color chars everywhere in that string.
    assert "FPS Name" in s
    assert "streamdelay" in s
    assert "procinfo.cset" in s

    # Test confstart, runstart, runstop, confstop
    # test both in tmux and with subprocesses

    # What are confstop and runstop supposed to do when there's no tmux ?
    ...
