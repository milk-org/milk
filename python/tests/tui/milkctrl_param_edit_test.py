"""Parameter edits are parsed and checked before writing; refusals say why (W2 C9, gap-2)."""

from __future__ import annotations

import os

import pytest

from tests.tui.milkctrl_fps_identity_test import _edit_gain, _get_param

pytestmark = pytest.mark.timeout(60)


def _open(milkctrl, entities, spec):
    entities.make_fps("pedit", spec)
    m = milkctrl()
    m.keys("c")
    m.wait_for(r"CONTROL")
    m.keys("F5")
    m.wait_for(r"gain\s+\S+\s+")
    return m


@pytest.mark.parametrize(
    "spec,value,reason",
    [
        ("gain:FLOAT64:0.5", "1.5x", r"not a number"),
        ("gain:FLOAT64:0.5", "inf", r"out of range"),
        ("gain:INT64:3", "12abc", r"not a number"),
    ],
)
def test_bad_values_are_not_written(milkctrl, entities, tui_env, spec, value, reason):
    env, _ = tui_env
    m = _open(milkctrl, entities, spec)
    before = _get_param(env, "pedit", "gain")
    _edit_gain(m, value)
    m.wait_for(rf"not written: {reason}")
    assert _get_param(env, "pedit", "gain") == before
    m.keys("Escape")
    assert m.quit() == 0


def test_write_protected_file_is_reported(milkctrl, entities, tui_env):
    env, shm = tui_env
    m = _open(milkctrl, entities, "gain:FLOAT64:0.5")
    os.chmod(os.path.join(shm, "pedit.fps.shm"), 0o444)
    _edit_gain(m, "2.0")
    m.wait_for(r"not written: FPS shm file is write-protected for you")
    os.chmod(os.path.join(shm, "pedit.fps.shm"), 0o644)  # pyMilk opens read-write
    assert float(_get_param(env, "pedit", "gain")) == pytest.approx(0.5)
    m.keys("Escape")
    assert m.quit() == 0
