from __future__ import annotations

import shutil

import pytest

from pyMilk.interfacing.fps import FPS, ProcessInfoFps

MILK_BINARY = "milk-fpsexec-mem-streamdelay"

pytestmark = pytest.mark.skipif(
    shutil.which(MILK_BINARY) is None, reason=f"{MILK_BINARY} not in PATH"
)


@pytest.fixture
def fixt_fpsinit_from_milk_pinfo():

    import subprocess, shlex

    proc = subprocess.run(shlex.split(f"{MILK_BINARY} -procinfo fpsinit"))
    if proc.returncode != 0:
        pytest.skip(f"{MILK_BINARY} exited with code {proc.returncode}.")

    f = FPS("streamdelay")

    yield f

    f.destroy()


@pytest.fixture
def fixt_fpsinit_from_milk_no_pinfo():

    import subprocess, shlex

    proc = subprocess.run(shlex.split(f"{MILK_BINARY} fpsinit"))
    if proc.returncode != 0:
        pytest.skip(f"{MILK_BINARY} exited with code {proc.returncode}.")

    f = FPS("streamdelay")

    yield f

    f.destroy()


def test_procinfo_present(fixt_fpsinit_from_milk_pinfo: FPS):
    assert fixt_fpsinit_from_milk_pinfo.procinfo is not None
    assert isinstance(fixt_fpsinit_from_milk_pinfo.procinfo, ProcessInfoFps)


def test_procinfo_absent(fixt_fpsinit_from_milk_no_pinfo: FPS):
    assert fixt_fpsinit_from_milk_no_pinfo.procinfo is None


@pytest.mark.parametrize(
    "attr, value",
    [
        ("RTprio", 42),
        ("cset", "system"),
        ("taskset", "0-3"),
        ("NBthread", 4),
        ("enabled", True),
        ("loopcntMax", 100),
        ("triggermode", 3),
        ("triggersname", "streamdelay_out"),
        ("MeasureTiming", True),
        ("semindexrequested", 2),
        ("triggerdelay", 0.5),
        ("triggertimeout", 1.5),
    ],
)
def test_procinfo_get_set(fixt_fpsinit_from_milk_pinfo: FPS, attr: str, value):
    pinfo = fixt_fpsinit_from_milk_pinfo.procinfo
    assert pinfo is not None

    setattr(pinfo, attr, value)
    assert getattr(pinfo, attr) == value
