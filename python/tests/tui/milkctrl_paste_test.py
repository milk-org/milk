"""Bracketed paste: pasted text reaches text prompts only (robust-7)."""

from __future__ import annotations

import time

import pytest

pytestmark = pytest.mark.timeout(60)


def test_pasted_keys_do_not_act(milkctrl):
    m = milkctrl()
    m.paste("cq\n")  # would toggle CONTROL and quit if typed
    m.wait_for(r"Pasted text ignored")
    time.sleep(0.3)
    assert m.alive()
    assert "READ ONLY" in m.screen()
    assert m.quit() == 0


def test_paste_into_filter_prompt(milkctrl, entities):
    for s in ["pasta", "other"]:
        entities.make_stream(s)
    m = milkctrl()
    m.keys("F3")
    m.wait_for(r"other")
    m.keys("/")
    m.paste("pas\nq")  # the newline must not submit; 'q' is text
    m.keys("Enter")
    m.wait_for(r"STREAMS regex filter applied: /pasq/")
    assert m.quit() == 0
