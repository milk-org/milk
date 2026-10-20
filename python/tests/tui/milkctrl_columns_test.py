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


def test_sort_keys_reach_the_table(milkctrl, entities):
    """S sorts by Hz, [ reverses, > moves to the next sortable column, t hides the cursor column."""
    entities.make_stream("sortk")
    m = milkctrl()
    m.keys("F3")
    m.wait_for(r"sortk")
    m.keys("S")
    m.wait_for(r"Hz▲")
    m.keys("[")
    m.wait_for(r"Hz▼")
    m.keys(">")
    m.wait_for(r"MB/s▼")
    m.keys("t")  # column cursor on A: collapsed to one column
    m.wait_for(r"\u00bbNAME")  # no separator after a collapsed column
    assert m.quit() == 0


def test_narrow_columns_scroll_into_view(milkctrl, entities):
    """At 80 columns low-priority columns are dropped (›); moving the column cursor scrolls them in."""
    entities.make_stream("scrolls")
    m = milkctrl(x=80, y=24)
    m.keys("F3")
    m.wait_for(r"scrolls")
    hdr = next(x for x in m.lines() if "NAME" in x)
    assert "RPID" not in hdr and "›" in hdr
    for _ in range(
        11
    ):  # A, NAME, then the scrollable columns in table order up to RPID
        m.keys("S-Right")
    m.wait_for(r"‹A\s+NAME.*RPID")
    m.keys("S-Right")  # wraps to the first column, unscrolled
    m.wait_for(r"║ A\s+NAME.*›")
    assert m.quit() == 0
