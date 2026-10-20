"""F7 loop member tables follow the entities when the UI re-sorts them (arch-1, robust-12, links-9)."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.timeout(60)


def _ring(entities):
    for s in ["aa1", "aa2", "lpA", "lpB", "zz9"]:
        entities.make_stream(s)
    entities.make_fps("fpsA", "streamin:STREAMNAME:lpB", "streamout:STREAMNAME:lpA")
    entities.make_fps("fpsB", "streamin:STREAMNAME:lpA", "streamout:STREAMNAME:lpB")


def _check_f7(m):
    m.keys("F7")
    m.wait_for(r"lpA\s+F32")
    screen = m.screen()
    assert "lpB" in screen
    assert "aa1" not in screen and "aa2" not in screen and "zz9" not in screen


def test_f7_members_survive_sort(milkctrl, entities):
    """Both directions: whichever one differs from the scan order exposes stale indices."""
    _ring(entities)
    m = milkctrl(y=40)
    for arrow in ["▼", "▲"]:
        m.keys("F3")
        m.wait_for(r"zz9")
        m.keys("[")
        m.wait_for(rf"NAME{arrow}")
        _check_f7(m)
    assert m.quit() == 0
