from __future__ import annotations
import pytest

import os
from milk.compute_unit import ComputeUnit


def func_fpsinit(compu_cls: type[ComputeUnit], pinfo: bool = True) -> ComputeUnit:
    # As fixture
    compu = compu_cls()
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


def tp(callable):
    try:
        callable()
    except:
        pass


import time


def func_compute_unit_cleanup(compu: ComputeUnit):
    tp(compu.runstop)
    tp(compu.confstop)
    if compu.fps:
        tp(compu.fps.tmux_stop)
        # Need to wait until not RUNisrunning to get there...
        compu.fps.run_stop(timeoutsync=5.0)
        time.sleep(0.1)
        with pytest.raises(ProcessLookupError):
            os.kill(compu.fps.fps.md().runpid, 0)
        with pytest.raises(ProcessLookupError):
            os.kill(compu.fps.fps.md().confpid, 0)

        tp(compu.fps.destroy)


def fixt_fpsinit_factory(compu_cls: type[ComputeUnit], pinfo: bool = True):
    @pytest.fixture
    def fixt_fpsinit():
        s = func_fpsinit(compu_cls, pinfo)
        yield s
        func_compute_unit_cleanup(s)

    return fixt_fpsinit
