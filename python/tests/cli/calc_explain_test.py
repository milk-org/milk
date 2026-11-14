"""`@calc --explain`: how each statement of a program would run, without running it.

Purpose:     to tune a unit, a user must see whether each statement became a fused pass, a pass
             merged into an earlier one, a scalar statement or a library step, and which limit
             applies.
Does:        checks the report line of each kind (fused pass with its shape, precision and thread
             count, merged pass, scalar statement, def), the last line, that no word of the
             removed engine is printed, the note for more than four targets, a compile
             error, that --explain changes no image and keeps its defs and history rings out of
             the session, and that a unit whose statements are all kept prints no "is not kept"
             note.
Covers:      mcalc_prog_explain() (src/calc/plan/mc_explain.c), which the session runs for
             `@calc --explain` (src/cli/libmilkscript/server/milk_server_eval.c);
             milk-fpsexec-milk-calc.
Uses:        conftest.py: milk_bash, value, CAM, UNIT_HELPERS.
See also:    docs/guide/calculator.md, "Fusion & Cost Model"; calc_superinstructions_test.py.
"""

from __future__ import annotations

import re

import pytest

from milk.cliwrap import HAVE_CLI
from tests.cli.conftest import (
    milk_bash,
    value,
    CAM,
    UNIT_HELPERS,
)

pytestmark = [
    pytest.mark.skipif(not HAVE_CLI, reason="MILK compiled without CLI support"),
    pytest.mark.timeout(120),
]


def _line_after(out: str, stmt: str) -> str:
    """The report line that follows statement @stmt."""
    lines = out.splitlines()
    for i, ln in enumerate(lines):
        if ln.strip().endswith(stmt) and i + 1 < len(lines):
            return lines[i + 1].strip()
    raise AssertionError(f"no report line for {stmt!r} in:\n{out}")


def test_explain_reports_how_each_statement_runs(tmp_path):
    """Fused pass, merged pass, scalar statement and def; `cplx` and a binned slice are fused too."""
    res = milk_bash(
        'mem.mk2Dim "@S:cam" 64 64 >/dev/null; mem.mk2Dim dark 64 64 >/dev/null; @calc "cam = 3; dark = 1; g = 2.0; n = 0"\n'
        '@calc --explain "out = (cam - dark) * g; local t = out + 1; out2 = t * 2; m = imean(out); '
        'z = cplx(out, out); s = out[::2b]; x = n > 0 ? 1/n : -1; def f(a) = a*2; y = f(out)"; echo "rc=$?"',
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert value(res.stdout, "rc") == "0", out
    first = _line_after(res.stdout, "out = (cam - dark) * g")
    assert first.startswith("fused pass over 64x64x1") and "float32" in first, out
    assert ", 1 thread)" in first, out  # 4096 pixels: below the size that gets a team
    assert first.endswith("writes out out2"), out
    assert (
        _line_after(res.stdout, "local t = out + 1")
        == "merged into the fused pass of statement 1"
    ), out
    assert (
        _line_after(res.stdout, "out2 = t * 2")
        == "merged into the fused pass of statement 1"
    ), out
    assert _line_after(res.stdout, "m = imean(out)").startswith(
        "scalar statement: 1 operation, 1 reduction"
    ), out
    z = _line_after(res.stdout, "z = cplx(out, out)")
    s = _line_after(res.stdout, "s = out[::2b]")
    assert z.startswith("fused pass over 64x64x1") and "complex float32" in z, out
    # out does not exist until statement 1 runs, and --explain runs nothing
    assert (
        s.startswith("fused pass over a shape set at run time") and "float32" in s
    ), out
    assert _line_after(res.stdout, "x = n > 0 ? 1/n : -1").startswith(
        "scalar statement"
    ), out
    assert (
        _line_after(res.stdout, "def f(a) = a*2") == "def, inlined where it is called"
    ), out
    assert _line_after(res.stdout, "y = f(out)").startswith(
        "fused pass over 64x64x1"
    ), out
    assert re.search(r"8 statements in \d+ steps?$", res.stdout, re.M), out
    assert not re.search(r"kept|tape|as text|FUSE_", res.stdout), out


def test_explain_names_limits_and_compile_errors(tmp_path):
    """More than four targets get one pass each; a program that does not compile is an error."""
    res = milk_bash(
        'mem.mk2Dim im 8 8 >/dev/null; @calc "im = 1"\n'
        '@calc --explain "a1, a2, a3, a4, a5 = im, im, im, im, im"; echo "rc1=$?"\n'
        '@calc --explain "bad = im +" 2>err.log; echo "rc2=$?"',
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert value(res.stdout, "rc1") == "0", out
    assert "more than 4 targets: one pass each" in res.stdout, out
    assert value(res.stdout, "rc2") != "0", out
    assert "milk: calc error:" in (tmp_path / "err.log").read_text() + res.stdout, out


def test_unit_notes_each_text_statement_once(tmp_path):
    """A unit whose statements are all kept (`cplx` included) prints no "is not kept" note."""
    (tmp_path / "prog.calc").write_text(
        "out = cam * 2\nz = cplx(out, out)\nok = out + 1\n"
    )
    res = milk_bash(
        UNIT_HELPERS
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
    assert len(notes) == 0, out


def test_explain_does_not_run(tmp_path):
    res = milk_bash(
        CAM + '@calc --explain "cam = cam + 100" >/dev/null; echo "rc=$?"\n'
        '@calc -v m "imean(cam)"; echo "m=$m"',
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert value(res.stdout, "rc") == "0", out
    assert value(res.stdout, "m") == "2", out


def test_explain_leaves_defs_and_rings_alone(tmp_path):
    res = milk_bash(
        CAM
        + '@calc --explain "def hid(x) = x * 2; y = hid(cam)" >/dev/null; echo "x1=$?"\n'
        "@calc 'hid(2)' 2>/dev/null; echo \"hid=$?\"\n"
        '@calc --explain "hist(cam, 4); y = cam - prev(cam)" 2>&1 |'
        ' grep -c "only in a milk.calc unit" | sed "s/^/unit=/"\n'
        '@exists s.cam_hist && echo "ring=yes" || echo "ring=no"',
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert value(res.stdout, "x1") == "0", out
    assert value(res.stdout, "hid") == "2", out  # the def stayed out of the session
    assert value(res.stdout, "unit") == "1", out
    assert value(res.stdout, "ring") == "no", out
