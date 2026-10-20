"""Colour depth is honoured at flush time; NO_COLOR / TERM=dumb give monochrome (gap-3)."""

from __future__ import annotations

import re
import time

import pytest

pytestmark = pytest.mark.timeout(60)


def _sgr(m):
    time.sleep(0.5)
    return " ".join(re.findall(r"\x1b\[([0-9;]*)m", m.screen_ansi()))


def test_truecolor(milkctrl):
    m = milkctrl()
    assert "38;2;" in _sgr(m)
    assert m.quit() == 0


def test_256_colors_without_colorterm(milkctrl):
    m = milkctrl(env_extra={"COLORTERM": ""})
    s = _sgr(m)
    assert "38;2;" not in s and "48;2;" not in s
    assert "38;5;" in s
    assert m.quit() == 0


@pytest.mark.parametrize("extra", [{"NO_COLOR": "1"}, {"TERM": "dumb"}])
def test_monochrome(milkctrl, entities, extra):
    entities.make_stream("monosel")
    m = milkctrl(env_extra=extra)
    m.wait_for(r"monosel")
    codes = {
        c
        for seq in re.findall(r"\x1b\[([0-9;]*)m", m.screen_ansi())
        for c in seq.split(";")
    }
    colour = {
        c for c in codes if c.isdigit() and (30 <= int(c) <= 49 or 90 <= int(c) <= 107)
    }
    assert not colour, codes
    assert "7" in codes  # the selected row is drawn in reverse video
    assert m.quit() == 0
