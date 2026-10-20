"""milk-procCTRL: cursor and selection keyed by process identity (W7 #9, pc-1, pc-2)."""

from __future__ import annotations

import subprocess
import time

import pytest

from tests.tui.conftest import _install_dir

pytestmark = pytest.mark.timeout(60)


@pytest.fixture
def scanner(tui_env):
    env, _ = tui_env
    p = subprocess.Popen(
        [str(_install_dir() / "bin" / "milk-procCTRL-scan"), "-r", "20"],
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(0.5)
    yield p
    p.kill()
    p.wait()


def _proc_popen(entities, pid):
    return next(p for p in entities.procs if p.pid == pid)


def test_sigterm_reaches_highlighted_process(legacy_pane, entities, scanner):
    pid = entities.make_proc("termme")
    p = legacy_pane("milk-procCTRL")
    p.wait_for(r"termme")
    time.sleep(0.5)  # one frame for the cursor to settle on its identity
    p.keys("T")
    deadline = time.monotonic() + 5
    while _proc_popen(entities, pid).poll() is None and time.monotonic() < deadline:
        time.sleep(0.1)
    assert (
        _proc_popen(entities, pid).poll() is not None
    ), "SIGTERM did not reach the cursor process"
    p.keys("x")
    assert p.exit_code() == 0


def test_marked_crashed_entry_is_not_signalled(legacy_pane, entities, scanner):
    """T acts on the marks; a marked entry whose process died is skipped, nothing else is hit."""
    a = entities.make_proc("aaa_marked")
    b = entities.make_proc("bbb_other")
    p = legacy_pane("milk-procCTRL")
    p.wait_for(r"bbb_other")
    time.sleep(0.5)
    p.keys("Space")
    p.wait_for(r"\* .*(aaa_marked|bbb_other)")
    marked_a = any("* " in l and "aaa_marked" in l for l in p.lines())
    dead, other = (a, b) if marked_a else (b, a)
    entities.kill(dead)
    p.wait_for(r"CRASH")
    p.keys("T")
    time.sleep(1.0)
    assert _proc_popen(entities, other).poll() is None, "T hit an unmarked process"
    p.keys("x")
    assert p.exit_code() == 0
