"""Unit starts and stops from milk-CTRL and fpsCTRL (unit-lifecycle Phase 2, §7 items 18 and 30).

The TUIs run the milk-fps-* tools in the background, so a unit that takes seconds to start never
freezes the screen. The unit is `ltunit` (tests/cli/unit_lifecycle_test.py), built here with milk-cc
and set to sleep 2 s before its loop starts.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

import pytest

from tests.cli.conftest import _install_dir, _isolated_shm_env, _run
from tests.cli.unit_lifecycle_test import LTUNIT_C

pytestmark = pytest.mark.timeout(90)


@pytest.fixture(scope="module")
def ltunit(tmp_path_factory) -> str:
    """Path of milk-fpsexec-ltunit, built once by milk-cc."""
    d = tmp_path_factory.mktemp("ltunit")
    (d / "ltunit.c").write_text(LTUNIT_C)
    env, shm_dir = _isolated_shm_env()
    try:
        res = _run(
            [str(_install_dir() / "bin" / "milk-cc"), "ltunit.c"], env, 120, cwd=str(d)
        )
    finally:
        shutil.rmtree(shm_dir, ignore_errors=True)
    exe = d / "milk-fpsexec-ltunit"
    if res.returncode != 0 or not exe.exists():
        pytest.skip(f"milk-cc could not build the test unit: {res.stderr[-400:]}")
    return str(exe)


@pytest.fixture
def slow_unit(tui_env, entities, ltunit):
    """slow_unit(NAME): a unit whose loop starts 2 s after its process; its workers die with the test."""
    env, shm_dir = tui_env
    entities.make_stream("lt_in", 4, 4)

    def make(name: str) -> None:
        subprocess.run(
            [ltunit, "-n", name, "fpsinit", ".delay=2"],
            env=env,
            check=True,
            capture_output=True,
            timeout=30,
        )

    yield make
    marker = f"MILK_SHM_DIR={shm_dir}".encode()
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            penv = Path(f"/proc/{d}/environ").read_bytes().split(b"\0")
        except OSError:
            continue
        if b"MILK_UNIT_WORKER=1" in penv and marker in penv:
            with contextlib.suppress(ProcessLookupError):
                os.kill(int(d), signal.SIGKILL)


def test_fpsctrl_unit_keys_never_block(legacy_pane, slow_unit, tui_env):
    """Item 18: R starts a slow unit in the background: the screen answers a key before the unit is
    ready. Ctrl+a follows the unit's output, Ctrl-C there stops it, E erases it."""
    _, shm_dir = tui_env
    slow_unit("tu1")
    p = legacy_pane("milk-fpsCTRL", x=160, y=40)
    p.wait_for(r"tu1")
    p.keys("R")
    t0 = time.monotonic()
    p.wait_for(r"tu1 runstart \.\.\.", timeout=2)
    p.keys("h")
    p.wait_for(r"MODE: HELP", timeout=2)
    assert time.monotonic() - t0 < 1.5, "the screen waited for the unit"
    p.wait_for(r"tu1 run: ready after", timeout=10)
    p.keys("F2")
    p.wait_for(r"MODE: CTRL")
    p.keys("C-a")
    p.wait_for(r"OUTPUT tu1 run: running")
    p.keys("C-c")
    p.wait_for(r"tu1 run: stopped")
    p.keys("Escape")
    p.wait_for(r"MODE: CTRL")
    p.keys("E")
    p.wait_for(r"FPS 'tu1' removed")
    assert not (Path(shm_dir) / "tu1.fps.shm").exists()
    p.keys("x")
    assert p.exit_code() == 0


def test_milkctrl_unit_actions_never_block(milkctrl, slow_unit, tui_env):
    """Items 18 and 30: r starts a slow unit without freezing the screen; e on the unit's process entry
    stops the unit through its stop (marker in run.out); erase removes it."""
    _, shm_dir = tui_env
    slow_unit("tu2")
    m = milkctrl()
    m.keys("c")
    m.wait_for(r"CONTROL")
    m.keys("F5")
    m.wait_for(r"tu2\s+stopped")
    m.keys("r")
    t0 = time.monotonic()
    m.wait_for(r'FPS "tu2": runstart \.\.\.', timeout=2)
    m.keys("F4")
    m.wait_for(r"PROC › Procs", timeout=2)
    assert time.monotonic() - t0 < 1.5, "the screen waited for the unit"
    m.wait_for(r"tu2 run: ready after", timeout=10)
    m.wait_for(r"●\s+tu2\s")
    m.keys("e")
    m.wait_for(r"tu2 run: stopped \(request")
    assert "stop: request" in (Path(shm_dir) / "tu2.fps.d" / "run.out").read_text()
    m.keys("F5")
    m.wait_for(r"tu2\s+exited")
    m.keys("C-e")
    m.wait_for(r"CONFIRM")
    m.keys("y")
    m.wait_for(r"FPS 'tu2' removed")
    assert not (Path(shm_dir) / "tu2.fps.shm").exists()
    assert m.quit() == 0
