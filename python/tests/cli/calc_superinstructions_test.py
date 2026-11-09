"""Short arithmetic statements compile to one fused pass and match the per-operator values.

Purpose:     the fused C pass makes per-frame arithmetic fast; a common statement must stay one
             pass, and compute what evaluating each operator on its own computes.
Does:        checks with --explain that seven statements each compile to one fused pass reading
             two images; compares 13 expressions (float and u16 inputs, slices, a broadcast row)
             with the same forced through the per-operator path by binned slices; runs a unit
             whose local, read by the next two statements, is merged into their pass.
Covers:      the planner and the JIT (src/calc/plan/mc_plan.c, mc_views.c: fusion and local
             elision; src/calc/jit/mc_emit.c, mc_cc.c); src/calc/plan/mc_explain.c;
             milk-fpsexec-milk-calc.
Uses:        conftest.py: milk_bash, value.
See also:    docs/guide/calculator.md, "Fusion & Cost Model"; calc_int_float_path_test.py.
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
"""


def _report(out: str, stmt: str) -> str:
    lines = out.splitlines()
    for i, ln in enumerate(lines):
        if ln.strip().endswith(stmt) and i + 1 < len(lines):
            return lines[i + 1].strip()
    raise AssertionError(f"no report line for {stmt!r} in:\n{out}")


def test_peephole_shortens_the_tape(tmp_path):
    """Each statement on its own compiles to a single fused pass reported by `--explain`."""
    cases = [
        ("o1 = a*2.5 + b*0.5", "1 operand), writes o1", "2 operands), writes o1"),
        ("o2 = a + b", "2 operands), writes o2", "2 operands), writes o2"),
        ("o3 = (a - b) * 2", "2 operands), writes o3", "2 operands), writes o3"),
        ("o4 = 2 / a - b", "2 operands), writes o4", "2 operands), writes o4"),
        (
            "o5 = a*2.5 + b*0.5 - a/4 + 1",
            "2 operands), writes o5",
            "2 operands), writes o5",
        ),
        ("o6 = (a + 1) * (b - 2)", "2 operands), writes o6", "2 operands), writes o6"),
        ("o7 = a * b + 1", "2 operands), writes o7", "2 operands), writes o7"),
    ]
    script = 'mem.mk2Dim a 64 64 >/dev/null; mem.mk2Dim b 64 64 >/dev/null; @calc "a = 1; b = 2"\n'
    for st, _, _ in cases:
        script += f'@calc --explain "{st}"\n'
    res = milk_bash(script, tmp_path)
    out = res.stdout + res.stderr
    for st, _, suffix in cases:
        rep = _report(res.stdout, st)
        assert rep.startswith(
            "kept: fused pass over 64x64x1 (float32, 1D,"
        ) and rep.endswith(suffix), (
            st,
            out,
        )


def test_superinstructions_match_fallback(tmp_path):
    """Float and u16 inputs and views: fused values equal the slice-forced fallback."""
    # (fused expression, the same forced through the per-operator path with binned slices)
    exprs = [
        ("a*2.5 + b*0.5", "a[::1b]*2.5 + b[::1b]*0.5"),
        ("a + b", "a[::1b] + b[::1b]"),
        ("(a - b) * 2", "(a[::1b] - b[::1b]) * 2"),
        ("2 / a - b", "2 / a[::1b] - b[::1b]"),
        ("a*2.5 + b*0.5 - a/4 + 1", "a[::1b]*2.5 + b[::1b]*0.5 - a[::1b]/4 + 1"),
        ("u*2.5 + b*0.5", "u[::1b]*2.5 + b[::1b]*0.5"),
        ("u - a", "u[::1b] - a[::1b]"),
        ("a / u", "a[::1b] / u[::1b]"),
        ("b*3 - a*u", "b[::1b]*3 - a[::1b]*u[::1b]"),
        ("sin(a) + exp(b*0.01)", "sin(a[::1b]) + exp(b[::1b]*0.01)"),
        ("a[0:31,:]*2 + b[0:31,:]", "a[0:31::1b,:]*2 + b[0:31::1b,:]"),
        ("a[0:31,:] - b[32:63,:] * 0.5", "a[0:31::1b,:] - b[32:63::1b,:] * 0.5"),
        ("row * 2 + a", "row[::1b] * 2 + a[::1b]"),
    ]
    lines = [
        "mem.mk2Dim a 64 48 >/dev/null; mem.mk2Dim b 64 48 >/dev/null; mem.mk2Dim row 64 1 >/dev/null\n"
        '@calc "a = 1 + 5 * rand(a, 3); b = 2 + 5 * rand(b, 5); row = rand(row, 7); u = u16(a * 1000)"\n'
    ]
    for i, (e, s_) in enumerate(exprs):
        lines.append(
            f'@calc "f{i} = {e}"; @calc "s{i} = {s_}"; e=err; '
            f'@calc -v e "imax(abs(f{i} - s{i})) / (imax(abs(s{i})) + 1)"; echo "e{i}=$e"\n'
        )
    res = milk_bash("".join(lines), tmp_path)
    out = res.stdout + res.stderr
    for i, (e, _) in enumerate(exprs):
        v = value(res.stdout, f"e{i}")
        assert v not in (None, "err", ""), (e, out)
        assert float(v) < 1e-6, (e, v, out)


def test_merged_statement_reading_the_previous_output(tmp_path):
    """`local t = a*2; out = t*3 + b` merges into one pass; the fused read of `t` is unfused into its
    register slot, so `out` is right and `t` is not written."""
    (tmp_path / "prog.calc").write_text(
        "local t = a * 2\nout = t * 3 + b\nout2 = b - t\n"
    )
    res = milk_bash(
        HELPERS
        + 'mem.mk2Dim "@S:a" 16 16 >/dev/null; mem.mk2Dim "@S:b" 16 16 >/dev/null; @calc "a = 2; b = 5"\n'
        "milk-fpsexec-milk-calc -n sp -t a -f prog.calc fpsinit >/dev/null\n"
        "milk-fpsexec-milk-calc -n sp run >run.log 2>&1 & pid=$!\n"
        'sleep 0.6; @calc "a = 3"; sleep 0.3\n'
        '@calc -v o,o2 "imean(out); imean(out2)"; echo "o=$o,$o2"; echo "error=[$(@get f.sp.unit.error)]"\n'
        '@calc --explain "local t = a * 2; out = t * 3 + b; out2 = b - t" | grep -c "merged into"; stop $pid',
        tmp_path,
    )
    out = res.stdout + res.stderr + (tmp_path / "run.log").read_text()
    assert value(res.stdout, "error") == "[]", out
    assert value(res.stdout, "o") == "23,-1", out
