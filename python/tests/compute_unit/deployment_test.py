from __future__ import annotations

import pytest

pmp = pytest.mark.parametrize

from pyMilk.interfacing.fps import FPS, FPSDoesntExistError
from pyMilk.interfacing.shm import SHM
import numpy as np

from milk.operate.compute_unit import ComputeUnit

from tests.conftestaux.compute_unit import fixt_fpsinit_factory
from tests.conftestaux.polling import poll_until_or_fail


class StreamDelayComputeUnit(ComputeUnit):

    def __init__(self, fpsname: str = "streamdelay") -> None:
        super().__init__("milk-fpsexec-mem-streamdelay", fpsname)


fixt_fpsinit_pinfo = fixt_fpsinit_factory(StreamDelayComputeUnit, pinfo=True)


# ADD AFTER THIS LINE


def test_fps_lifecycle(fixt_fpsinit_pinfo, tmux: bool = False):
    compu: StreamDelayComputeUnit = fixt_fpsinit_pinfo
    fps = compu.fps
    assert fps

    fps["naive_mode"] = True
    fps["procinfo.loopcntMax"] = -1
    fps["procinfo.triggermode"] = 3
    # runstop is only noticed when the semaphore wait times out (default 2 s).
    fps["procinfo.triggertimeout"] = 0.1
    fps["procinfo.triggersname"] = "imin"

    compu.confstart(tmux=tmux)
    poll_until_or_fail(fps.conf_isrunning, t=5.0, msg="conf not running")

    arr = np.random.randn(30, 40)
    in_shm = SHM("imin", arr * 0)

    compu.runstart(tmux=tmux)
    poll_until_or_fail(fps.run_isrunning, t=5.0, msg="run not running")

    out_shm = SHM("imout")

    # Negative window: no output before any input.
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
    compu.fps["procinfo.triggertimeout"] = 0.1
    # This should be enough and loops should autoconf the semaphore trigger freerun

    compu.confstart()
    poll_until_or_fail(compu.fps.conf_isrunning, t=5.0, msg="conf not running")

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

    poll_until_or_fail(compu.fps.run_isrunning, t=5.0, msg="run not running")
    out_shm = SHM("imout")

    def primed() -> bool:
        # Triggers posted before the run loop is ready may be flushed: repost until one lands.
        if out_shm.md.cnt0 >= 1:
            return True
        in_shm.set_data(arr)
        return False

    poll_until_or_fail(primed, t=5.0, interval=0.01, msg="no output cnt0 >= 1")

    if loopmode == 0:
        poll_until_or_fail(
            lambda: not compu.fps.run_isrunning(), t=5.0, msg="run still running"
        )
        assert compu.fps["procinfo.loopcntMax"] == 1
        assert out_shm.md.cnt0 == 1
    elif loopmode == 1:
        c = out_shm.md.cnt0
        in_shm.set_data(arr)
        poll_until_or_fail(
            lambda: out_shm.md.cnt0 == c + 1, t=5.0, msg=f"cnt0 never reached {c + 1}"
        )
        assert compu.fps.run_isrunning()
        assert compu.fps["procinfo.loopcntMax"] == -1
    elif loopmode == 2:
        poll_until_or_fail(
            lambda: out_shm.md.cnt0 >= 50, t=5.0, msg="cnt0 never reached 50"
        )
        assert compu.fps.run_isrunning()
        assert compu.fps["procinfo.loopcntMax"] == -1

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
