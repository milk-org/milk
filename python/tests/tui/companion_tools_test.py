"""Companion tools sharing milk-CTRL's data layer decode states with the shared labels (gap-1)."""

from __future__ import annotations

import subprocess

import pytest

pytestmark = pytest.mark.timeout(60)


def _info(env, name):
    return subprocess.run(
        ["milk-procinfo-info", name],
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    ).stdout


def test_procinfo_info_labels(entities, tui_env):
    env, _ = tui_env
    entities.make_proc("lblproc")
    out = _info(env, "lblproc")
    assert "RUNNING" in out
    assert "CTRL val          : RUN" in out
    assert "IMMEDIATE (0)" in out


def test_crashed_process_never_shows_unknown(milkctrl, entities):
    entities.make_proc("aaa_cursor")
    pid = entities.make_proc("crashme")
    entities.kill(pid)
    m = milkctrl()
    m.keys("F4")
    m.wait_for(r"crashme.*CRSH")
    assert "??" not in m.screen()
    assert m.quit() == 0
