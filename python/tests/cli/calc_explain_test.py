"""
Tests for `@calc --explain` and the one-time "statement not kept" note of milk.calc units (P7 of the
2026-10-27 engine review): every statement a unit would evaluate as text each tick is named once at
compile time with the limit or construct that stopped fusion, and `--explain` reports per statement
whether it is kept (fused pass, merged pass, scalar tape) or evaluated as text and why.
"""

from __future__ import annotations

import pytest

from milk.cliwrap import HAVE_CLI
from tests.cli.conftest import milk_bash, value

pytestmark = [
    pytest.mark.skipif(not HAVE_CLI, reason="MILK compiled without CLI support"),
    pytest.mark.timeout(120),
]

HELPERS = r"""
until_file() { for _ in $(seq 100); do [ -e "$MILK_SHM_DIR/$1.im.shm" ] && return 0; sleep 0.05; done; return 1; }
stop() { kill -INT "$1" 2>/dev/null; for _ in $(seq 60); do kill -0 "$1" 2>/dev/null || break; sleep 0.05; done
         kill -0 "$1" 2>/dev/null && echo "still running"; wait "$1" 2>/dev/null; }
mem.mk2Dim "@S:cam" 8 8 >/dev/null; @calc "cam = 2"
"""


def _line_after(out: str, stmt: str) -> str:
    """The report line that follows statement @stmt."""
    lines = out.splitlines()
    for i, ln in enumerate(lines):
        if ln.strip().endswith(stmt) and i + 1 < len(lines):
            return lines[i + 1].strip()
    raise AssertionError(f"no report line for {stmt!r} in:\n{out}")


def test_explain_reports_how_each_statement_runs(tmp_path):
    """Fused pass, merged pass, scalar tape, def, and two text fallbacks with their reasons."""
    res = milk_bash(
        'mem.mk2Dim "@S:cam" 64 64 >/dev/null; mem.mk2Dim dark 64 64 >/dev/null; @calc "cam = 3; dark = 1; g = 2.0; n = 0"\n'
        '@calc --explain "out = (cam - dark) * g; local t = out + 1; out2 = t * 2; m = imean(out); '
        'z = cplx(out, out); s = out[::2b]; x = n > 0 ? 1/n : -1; def f(a) = a*2; y = f(out)"; echo "rc=$?"',
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert value(res.stdout, "rc") == "0", out
    first = _line_after(res.stdout, "out = (cam - dark) * g")
    assert first.startswith("kept: fused pass over 64x64x1") and "float32" in first, out
    assert first.endswith("writes out out2"), out
    assert (
        _line_after(res.stdout, "local t = out + 1")
        == "kept, merged into the fused pass of statement 1"
    ), out
    assert (
        _line_after(res.stdout, "out2 = t * 2")
        == "kept, merged into the fused pass of statement 1"
    ), out
    assert _line_after(res.stdout, "m = imean(out)").startswith(
        "kept: scalar tape: 1 op, 1 reduction"
    ), out
    z = _line_after(res.stdout, "z = cplx(out, out)")
    assert z.startswith("NOT kept") and "complex destination" in z, out
    s = _line_after(res.stdout, "s = out[::2b]")
    assert s.startswith("NOT kept") and "out[::2b] is not a view" in s, out
    assert _line_after(res.stdout, "x = n > 0 ? 1/n : -1").startswith(
        "kept: scalar tape"
    ), out
    assert (
        _line_after(res.stdout, "def f(a) = a*2") == "def, inlined where it is called"
    ), out
    assert _line_after(res.stdout, "y = f(out)").startswith(
        "kept: fused pass over 64x64x1"
    ), out
    assert "6 statements kept, 2 evaluated as text" in res.stdout, out


def test_explain_names_limits_and_compile_errors(tmp_path):
    """More than four targets names FUSE_MAXDST; a program that does not compile is an error."""
    res = milk_bash(
        'mem.mk2Dim im 8 8 >/dev/null; @calc "im = 1"\n'
        '@calc --explain "a1, a2, a3, a4, a5 = im, im, im, im, im"; echo "rc1=$?"\n'
        'mem.mk2Dim small 4 4 >/dev/null; @calc "small = 1"\n'
        '@calc --explain "bad = im + small" 2>err.log; echo "rc2=$?"',
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert value(res.stdout, "rc1") == "0", out
    assert "more than 4 targets (FUSE_MAXDST)" in res.stdout, out
    assert value(res.stdout, "rc2") != "0", out
    assert "milk.calc:" in (tmp_path / "err.log").read_text() + res.stdout, out


def test_unit_notes_each_text_statement_once(tmp_path):
    """A unit says once, at compile, which statement it evaluates as text and why; kept ones are silent."""
    (tmp_path / "prog.calc").write_text(
        "out = cam * 2\nz = cplx(out, out)\nok = out + 1\n"
    )
    res = milk_bash(
        HELPERS
        + "milk-fpsexec-milk-calc -n p7 -t cam -f prog.calc fpsinit >/dev/null\n"
        "milk-fpsexec-milk-calc -n p7 run >run.log 2>&1 & pid=$!\n"
        "sleep 0.6\n"
        'for v in 2 3 4 5 6; do @calc "cam = $v"; sleep 0.15; done; sleep 0.3\n'
        '@calc -v o "imean(ok)"; echo "ok=$o"; echo "error=[$(@get f.p7.unit.error)]"; stop $pid',
        tmp_path,
    )
    log = (tmp_path / "run.log").read_text()
    out = res.stdout + res.stderr + log
    assert value(res.stdout, "error") == "[]" and value(res.stdout, "ok") == "13", out
    notes = [ln for ln in log.splitlines() if "is not kept" in ln]
    assert len(notes) == 1, out
    assert (
        "statement 2 is not kept" in notes[0] and "z = cplx(out, out)" in notes[0]
    ), out
    assert "complex destination" in notes[0], out
