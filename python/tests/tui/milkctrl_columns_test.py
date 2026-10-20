"""Column tables: a header and its rows come from one definition, so they line up."""

from __future__ import annotations

import re

import pytest

pytestmark = pytest.mark.timeout(60)


def _end(line: str, pattern: str) -> int:
    hit = re.search(pattern, line)
    assert hit, f"{pattern!r} not in {line!r}"
    return hit.end()


def test_proc_rows_align_with_header(milkctrl, entities):
    """Right-aligned cells end where their header label ends, also after STAT and CPU% (the old drift)."""
    pid = entities.make_proc("alignp")
    m = milkctrl()
    m.keys("F4")
    m.wait_for(r"alignp.*%")
    lines = m.lines()
    hdr = next(x for x in lines if "NAME" in x and "STAT" in x)
    row = next(x for x in lines if "alignp" in x)
    assert _end(hdr, r"PID") == _end(row, rf"\b{pid}\b")
    assert _end(hdr, r"STAT") == _end(row, r"\b(RUN|STAL|CRSH|STOP|PAUS|INIT)\b")
    assert _end(hdr, r"CPU%") == _end(row, r"\d+\.\d%")
    assert m.quit() == 0


def test_stream_rows_align_with_header(milkctrl, entities):
    """Stream cells end where their right-aligned header label ends."""
    entities.make_stream("aligns")
    m = milkctrl()
    m.keys("F3")
    m.wait_for(r"aligns")
    lines = m.lines()
    hdr = next(x for x in lines if "NAME" in x and "INODE" in x)
    row = next(x for x in lines if "aligns" in x)
    assert _end(hdr, r"TYP") == _end(row, r"\b[A-Z]+\d+\b")
    assert _end(hdr, r"SIZE") == _end(row, r"\b\d+x\d+\b")
    assert _end(hdr, r"INODE") == _end(row, r"\b\d{3,}\b")
    assert m.quit() == 0
