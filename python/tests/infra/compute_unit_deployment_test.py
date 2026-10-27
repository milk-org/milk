from __future__ import annotations

import os, time

import pytest

from pyMilk.interfacing.fps import FPS
from pyMilk.interfacing.shm import SHM
import numpy as np

from milk.compute_unit import ComputeUnit

from .compute_unit_fixtures import fixt_fpsinit_factory


class StreamDelayCU(ComputeUnit):

    def __init__(self, fpsname: str = "streamdelay") -> None:
        super().__init__("milk-fpsexec-mem-streamdelay", fpsname)


fixt_fpsinit_pinfo_streamdelay = fixt_fpsinit_factory(StreamDelayCU, pinfo=True)


def test_fps_lifecycle(fixt_fpsinit_pinfo_streamdelay):
    compu: StreamDelayCU = fixt_fpsinit_pinfo_streamdelay
    fps = compu.fps
    assert fps

    fps["naive_mode"] = True
    fps["procinfo.loopcntMax"] = -1
    fps["procinfo.triggermode"] = 3
    fps["procinfo.triggersname"] = "imin"

    compu.confstart()
    time.sleep(0.1)
    assert fps.conf_isrunning()

    arr = np.random.randn(30, 40)
    in_shm = SHM("imin", arr * 0)

    compu.runstart(tmux=False)
    time.sleep(1.0)
    assert fps.run_isrunning()  # Wait this should NOT have asserted.

    out_shm = SHM("imout")

    assert (
        out_shm.get_data(
            True, checkSemAndFlush=True, timeout=0.1, return_none_on_timeout=True
        )
        is None
    )
    in_shm.set_data(arr)
    arr2 = out_shm.get_data(True)
    np.testing.assert_allclose(arr, arr2)


def test_lifecycle():
    # Test confstart, runstart, runstop, confstop
    # test both in tmux and with subprocesses

    # What are confstop and runstop supposed to do when there's no tmux ?
    ...


def test_fps(): ...
