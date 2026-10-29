from __future__ import annotations
import pytest

import os
from milk.operate.compute_unit import ComputeUnit

from tests.conftestaux.polling import poll_until_or_fail


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


def pid_is_dead(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


def func_compute_unit_cleanup(compu: ComputeUnit):
    tp(compu.runstop)
    tp(compu.confstop)
    if compu.fps:
        tp(compu.fps.tmux_stop)
        md = compu.fps.fps.md()
        for pid in (md.runpid, md.confpid):
            if pid == 0:  # never started
                continue
            poll_until_or_fail(lambda: pid_is_dead(pid), t=1.0, msg=f"pid {pid} alive")

        tp(compu.fps.destroy)


def fixt_fpsinit_factory(compu_cls: type[ComputeUnit], pinfo: bool = True):
    @pytest.fixture
    def fixt_fpsinit():
        s = func_fpsinit(compu_cls, pinfo)
        yield s
        func_compute_unit_cleanup(s)

    return fixt_fpsinit
