"""milk-fpsCTRL stopgaps: action keys without a live FPS are no-ops (W7 #6, fpsctrl-d2)."""

from __future__ import annotations

import time

import pytest

pytestmark = pytest.mark.timeout(60)


def test_action_keys_without_fps(legacy_pane):
    p = legacy_pane("milk-fpsCTRL")
    time.sleep(1.5)
    for k in ["u", "y", "r", "T", "Enter", "C-e", "Space"]:
        p.keys(k)
        time.sleep(0.2)
    time.sleep(0.5)
    assert p.alive()
    p.keys("x")
    assert p.exit_code() == 0
