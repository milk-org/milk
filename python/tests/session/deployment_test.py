from __future__ import annotations

import os, time

import pytest

pmp = pytest.mark.parametrize

from pyMilk.interfacing.fps import FPS, FPSDoesntExistError
from pyMilk.interfacing.shm import SHM
import numpy as np

from milk.session import ComputeSession


class StreamDelayComputeSession(ComputeSession):

    def __init__(self, fpsname: str = "streamdelay") -> None:
        super().__init__("milk-fpsexec-mem-streamdelay", fpsname)


def func_fpsinit(pinfo: bool = True) -> StreamDelayComputeSession:
    # As fixture
    session = StreamDelayComputeSession()
    assert session.fps is None

    session.fpsinit(pinfo)
    assert session.fps

    if pinfo:
        assert session.procinfo
        assert "procinfo.enabled" in session.fps
        assert session.fps["procinfo.enabled"]
    else:
        assert session.procinfo is None
        assert not "procinfo.enabled" in session.fps

    # Configure
    # default imin, imout, delaysec = 0.1 ms, naive = OFF, timebuffsize = 1000

    return session


def tp(call):
    # Call the callable and silently catch any exception - good for fixturized cleanups
    try:
        call()
    except:
        pass


def func_session_cleanup(s: ComputeSession):
    tp(s.runstop)
    tp(s.confstop)
    if s.fps:
        tp(s.fps.tmux_stop)
        tp(s.fps.destroy)


@pytest.fixture
def fixt_fpsinit_pinfo():
    s = func_fpsinit(True)
    yield s
    func_session_cleanup(s)


@pytest.fixture
def fixt_fpsinit_nopinfo():
    s = func_fpsinit(False)
    yield s
    func_session_cleanup(s)


# ADD AFTER THIS LINE


def test_fps_lifecycle(fixt_fpsinit_pinfo, tmux: bool = False):
    session: StreamDelayComputeSession = fixt_fpsinit_pinfo
    fps = session.fps
    assert fps

    fps["naive_mode"] = True
    fps["procinfo.loopcntMax"] = -1
    fps["procinfo.triggermode"] = 3
    fps["procinfo.triggersname"] = "imin"

    session.confstart(tmux=tmux)
    time.sleep(0.1)
    assert fps.conf_isrunning()

    arr = np.random.randn(30, 40)
    in_shm = SHM("imin", arr * 0)

    session.runstart(tmux=tmux)
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
    session: StreamDelayComputeSession = fixt_fpsinit_pinfo
    assert session.fps
    session.fps["naive_mode"] = True
    session.fps["procinfo.triggersname"] = "imin"
    session.fps["in_name"] = "imin"
    # This should be enough and loops should autoconf the semaphore trigger freerun

    session.confstart()
    time.sleep(0.1)
    assert session.fps.conf_isrunning()

    arr = np.random.randn(30, 40)
    in_shm = SHM("imin", arr * 0)

    if loopmode == 0:
        session.runstart(tmux=tmux)
    elif loopmode == 1:
        session.runstart(tmux=tmux, loops=True)
    elif loopmode == 2:
        session.runstart(tmux=tmux, loopd=0.01)
    else:
        raise ValueError(f"loopmode [0,1,2] - got {loopmode}")

    time.sleep(1.0)

    in_shm.set_data(in_shm.get_data())
    time.sleep(0.1)
    in_shm.set_data(in_shm.get_data())
    out_shm = SHM("imout")

    if loopmode == 0:
        for _ in range(100):
            print(session.fps.run_isrunning())
            time.sleep(0.01)
        assert not session.fps.run_isrunning()
        assert session.fps["procinfo.loopcntMax"] == 1
        assert out_shm.md.cnt0 == 1
    elif loopmode == 1:
        assert session.fps.run_isrunning()
        assert session.fps["procinfo.loopcntMax"] == -1
        assert out_shm.md.cnt0 == 2
    elif loopmode == 2:
        assert session.fps.run_isrunning()
        assert session.fps["procinfo.loopcntMax"] == -1
        assert out_shm.md.cnt0 >= 50

    out_shm.destroy()


def test_loopd(): ...


def test_raises_on_missing_fps(fixt_fpsinit_pinfo):
    session: StreamDelayComputeSession = fixt_fpsinit_pinfo
    assert session.fps

    session.fps.destroy()

    with pytest.raises(FPSDoesntExistError):
        session._trylink_fps(True)


def test_str(fixt_fpsinit_pinfo):
    session: StreamDelayComputeSession = fixt_fpsinit_pinfo
    s = str(session)
    # Note that there are ANSI color chars everywhere in that string.
    assert "FPS Name" in s
    assert "streamdelay" in s
    assert "procinfo.cset" in s

    # Test confstart, runstart, runstop, confstop
    # test both in tmux and with subprocesses

    # What are confstop and runstop supposed to do when there's no tmux ?
    ...
