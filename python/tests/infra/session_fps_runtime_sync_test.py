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

# fmt: off
FPS_TEST_PARAMS = [
    # (parameter_name, parameter_type, default_value, another_value)
    # HAS TO match defaults in fpstest/fps_sync_test.c
    ("p_int32",       "FPTYPE_INT32",             123,                    999),
    ("p_uint32",      "FPTYPE_UINT32",            456,                    999),
    ("p_int64",       "FPTYPE_INT64",             789,                    -42),
    ("p_uint64",      "FPTYPE_UINT64",            101112,                 55),
    ("p_float32",     "FPTYPE_FLOAT32",           3.14,                   1.5),
    ("p_float64",     "FPTYPE_FLOAT64",           2.718,                  9.81),
    ("p_onoff",       "FPTYPE_ONOFF",             0,                      1),
    ("p_pid",         "FPTYPE_PID",               1000,                   2000),
    ("p_timespec",    "FPTYPE_TIMESPEC",          1709424000.123456789,   123456789.987654321),
    ("p_streamname",  "FPTYPE_STREAMNAME",        "cam01",                "cam02"),
    ("p_filename",    "FPTYPE_FILENAME",          "data.txt",             "other.txt"),
    ("p_fitsfile",    "FPTYPE_FITSFILENAME",      "image.fits",           "other.fits"),
    ("p_execfile",    "FPTYPE_EXECFILENAME",      "run_me.sh",            "other.sh"),
    ("p_dirname",     "FPTYPE_DIRNAME",           "/tmp",                 "/var/tmp"),
    ("p_string",      "FPTYPE_STRING",            "hello",                "world"),
    #("p_process",     "FPTYPE_PROCESS",           "process_a",            "process_b"),
    ("p_fpsname",     "FPTYPE_FPSNAME",           "otherfps",             "anotherfps"),
    ("p_strnotstrm",  "FPTYPE_STRING_NOT_STREAM", "not_a_stream",         "still_not_a_stream"),
]
# fmt: on


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

    def busywait_until_incr(lastvalue: int):
        while pinfo.loopcnt == lastvalue:
            ...

    last_pinfo_cnt = pinfo.loopcnt
    for param_name, _, base_value, target_value in FPS_TEST_PARAMS:
        value = fps[param_name]
        if isinstance(value, int):
            assert value == base_value
        elif isinstance(value, float):
            assert value == pytest.approx(base_value)
        elif isinstance(value, str):
            assert value == base_value
        else:
            assert False

    # Find a way to assert that the counter has spun by about ~100
