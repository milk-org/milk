from __future__ import annotations

import pytest
import time, os

from pyMilk.interfacing.fps import FPS
from pyMilk.interfacing.pinfo import ProcessInfo

from milk.session import ComputeSession
from .session_fixtures import fixt_fpsinit_factory


class FPSTestCU(ComputeSession):
    def __init__(self, fpsname: str = "fpssynctest") -> None:
        super().__init__("milk-fpsexec-fpstest-fpssynctest", fpsname)


fixt_fpsinit_pinfo_fpstest = fixt_fpsinit_factory(FPSTestCU, pinfo=True)


def test_fpssynctest(fixt_fpsinit_pinfo_fpstest):
    session: FPSTestCU = fixt_fpsinit_pinfo_fpstest
    fps = session.fps
    assert fps

    fps["procinfo.loopcntMax"] = -1
    fps["procinfo.triggermode"] = 4  # delay
    fps["procinfo.triggerdelay"] = 0.01

    session.confstart()
    time.sleep(0.1)
    assert fps.conf_isrunning()
    session.runstart()
    time.sleep(0.1)
    assert fps.run_isrunning()

    pinfo = ProcessInfo()
    pinfo.link(
        f"{os.environ['MILK_PROC_DIR']}/proc.fpssynctest.{fps.fps.md().runpid}.shm"
    )

    time.sleep(0.1)
    assert pinfo.loopcnt > 10

    # Find a way to assert that the counter has spun by about ~100
