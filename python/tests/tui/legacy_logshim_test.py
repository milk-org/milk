"""milk-logshim-ctrl and milk-streamFITSlog drive the logger FPS without the fpsCTRL FIFO (W7 #7)."""

from __future__ import annotations

import subprocess

import pytest

from tests.tui.conftest import _install_dir
from tests.tui.milkctrl_fps_identity_test import _get_param

pytestmark = pytest.mark.timeout(90)


def _logger(entities):
    return entities.make_fps(
        "streamFITSlog-lsx",
        "saveON:ONOFF:0",
        "lastcubeON:ONOFF:0",
        "cubesize:INT64:100",
    )


def _run(env, *argv):
    return subprocess.run(
        [str(_install_dir() / "bin" / argv[0]), *argv[1:]],
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_logshim_on_stat_kill(tui_env, entities):
    env, _ = tui_env
    path = _logger(entities)
    r = _run(env, "milk-logshim-ctrl", "on", "lsx")
    assert r.returncode == 0, r.stderr
    assert _get_param(env, "streamFITSlog-lsx", "saveON") in ("1", "True", "ON")
    r = _run(env, "milk-logshim-ctrl", "stat", "lsx")
    assert "LIVE" in r.stdout and "FIFO" not in r.stdout
    r = _run(env, "milk-logshim-ctrl", "kill", "lsx")
    assert r.returncode == 0, r.stdout + r.stderr
    assert not path.exists()


def test_streamfitslog_set_and_on(tui_env, entities):
    env, _ = tui_env
    _logger(entities)
    r = _run(env, "milk-streamFITSlog", "-z", "64", "lsx", "set")
    assert r.returncode == 0, r.stdout + r.stderr
    assert _get_param(env, "streamFITSlog-lsx", "cubesize") == "64"
    r = _run(env, "milk-streamFITSlog", "lsx", "on")
    assert r.returncode == 0, r.stdout + r.stderr
    assert _get_param(env, "streamFITSlog-lsx", "saveON") in ("1", "True", "ON")
