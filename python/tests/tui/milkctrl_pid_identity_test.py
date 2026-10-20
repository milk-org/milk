"""Process control checks identity: dead rows are not signalled; long names resolve (robust-3, control-15)."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.timeout(60)

LONG = "p" * 52  # beyond the 40 characters OV_PROC.name keeps


def _ctrl_procs(milkctrl, name):
    m = milkctrl()
    m.keys("c")
    m.wait_for(r"CONTROL")
    m.keys("F4")
    m.wait_for(name)
    return m


def test_dead_row_is_not_signalled(milkctrl, entities):
    entities.kill(entities.make_proc("deadrow"))
    m = _ctrl_procs(milkctrl, r"deadrow.*CRSH")
    m.keys("k")
    m.wait_for(r"is not running; not signalled")
    assert m.quit() == 0


def test_ctrlval_reaches_process_with_long_name(milkctrl, entities):
    entities.make_proc(LONG)
    m = _ctrl_procs(milkctrl, LONG[:12])
    m.keys("p")  # CTRLval pause through the registry's full name
    m.wait_for(r"\": Pause")
    m.wait_for(r"PAUS|STAL")  # the fake loop stops counting
    m.keys("p")
    m.wait_for(r"\": Resume")
    assert m.quit() == 0
