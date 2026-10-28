"""
Regression tests for the 2026-10-27 calculator engine review fixes (C1-C4).

C1: a `local` that no later statement reads is merged into the next statement's pass without leaving
its value on the vector stack (the next statement's output used to get the dead local). C2: a `local`
is written to memory when a statement already merged into an earlier pass, or a `def` body, reads it
(the text scan used to miss both). C3: an axis-0 or middle-axis reduction read as an operand is placed
on its own axis, so `im - imean(im, 0)` subtracts the row means of a non-square image and is not
transposed on a square one, while a plain assignment keeps the squeezed shape. C4: a division by zero
inside the untaken branch of `?:` or `where()` is selected away instead of failing the statement.
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


def _unit(tmp_path, name, prog, probes, init=""):
    """Run @prog as unit @name on cam, feed cam = 2 then 5, print @probes, stop; return stdout+log."""
    (tmp_path / "prog.calc").write_text(prog)
    opt_init = f' -i "{init}"' if init else ""
    res = milk_bash(
        HELPERS
        + f"milk-fpsexec-milk-calc -n {name} -t cam -f prog.calc{opt_init} fpsinit >/dev/null\n"
        f"milk-fpsexec-milk-calc -n {name} run >run.log 2>&1 & pid=$!\n"
        "sleep 0.6\n"
        'for v in 2 2 5 5 5; do @calc "cam = $v"; sleep 0.15; done; sleep 0.3\n'
        + probes
        + f'\necho "error=[$(@get f.{name}.unit.error)]"; stop $pid',
        tmp_path,
    )
    log = (tmp_path / "run.log").read_text()
    return res, res.stdout + res.stderr + log


def test_C1_elided_unread_local_does_not_replace_next_output(tmp_path):
    """`local tmp = cam - 1; out = cam * 3` fuse into one pass; out is 3*cam, not tmp."""
    res, out = _unit(
        tmp_path,
        "c1",
        "local tmp1 = cam - 1\nout1 = cam * 3\n",
        '@calc -v o1 "imean(out1)"; echo "o1=$o1"',
    )
    assert value(res.stdout, "error") == "[]", out
    assert value(res.stdout, "o1") == "15", out


def test_C2_local_read_through_def_is_written(tmp_path):
    """A `def` body reading the local keeps it materialized: out3 sees this frame's tmp3."""
    res, out = _unit(
        tmp_path,
        "c2a",
        "def g(x) = x + tmp3\nlocal tmp3 = cam - 1\nout3b = cam * 2\nout3 = g(cam)\n",
        '@calc -v o3,o3b "imean(out3); imean(out3b)"; echo "o3=$o3,$o3b"',
    )
    assert value(res.stdout, "error") == "[]", out
    assert value(res.stdout, "o3") == "9,10", out


def test_C2_local_read_by_merged_statement_is_written(tmp_path):
    """A reader folded into an earlier pass (STMT_NOP) still counts: b2 sees the previous frame's tmp2."""
    res, out = _unit(
        tmp_path,
        "c2b",
        "a2 = cam * 2\nb2 = tmp2 + 1\nlocal tmp2 = cam - 1\nout2b = cam * 2\n",
        '@calc -v b2,o2b "imean(b2); imean(out2b)"; echo "b2=$b2,$o2b"',
        init="local tmp2 = cam * 0",
    )
    assert value(res.stdout, "error") == "[]", out
    assert value(res.stdout, "b2") == "5,10", out


def test_C3_axis0_reduction_is_placed_on_its_axis(tmp_path):
    """Row means come off every row of a 16x8 image and of an 8x8 one; shapes of plain results stay squeezed."""
    res = milk_bash(
        'mem.mk2Dim im3 16 8 >/dev/null; @calc "im3 = iy(im3)*10 + ix(im3) + urand(im3, 3)"\n'
        '@calc "d0 = im3 - imean(im3, 0)"; echo "rc0=$?"\n'
        '@calc -v s0,n0 "imax(abs(itot(d0, 0))); irms(d0)"; echo "s0=$s0"; echo "n0=$n0"\n'
        '@calc "d0r = imean(im3, 0) - im3"; @calc -v e0r "imax(abs(d0r + d0))"; echo "e0r=$e0r"\n'
        '@calc "p0 = imean(im3, 0)"; echo "sh0=$(@get s.p0.naxis):$(@get s.p0.xsize)"\n'
        '@calc "d1 = im3 - imean(im3, 1)"; @calc -v s1 "imax(abs(itot(d1, 1)))"; echo "s1=$s1"\n'
        'mem.mk2Dim sq 8 8 >/dev/null; @calc "sq = iy(sq)*10 + ix(sq) + urand(sq, 5)"\n'
        '@calc "dsq = sq - imean(sq, 0)"; @calc -v ssq "imax(abs(itot(dsq, 0)))"; echo "ssq=$ssq"\n'
        'mem.mk3Dim cube3 6 4 3 >/dev/null; @calc "cube3 = urand(cube3, 7)"\n'
        '@calc "c1 = cube3 - imean(cube3, 1)"; echo "rc1=$?"; @calc -v sc1 "imax(abs(itot(c1, 1)))"; echo "sc1=$sc1"\n'
        '@calc "m1 = imean(cube3, 1)"; echo "shm1=$(@get s.m1.naxis):$(@get s.m1.xsize)x$(@get s.m1.ysize)"\n'
        'mem.mk2Dim m1e 6 3 >/dev/null; @calc "m1e = imean(cube3, 1)"; @calc -v e1e "imax(abs(m1e - m1))"; echo "e1e=$e1e"\n'
        '@calc "c0 = cube3 - imean(cube3, 0)"; @calc -v sc0 "imax(abs(itot(c0, 0)))"; echo "sc0=$sc0"',
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert value(res.stdout, "rc0") == "0" and value(res.stdout, "rc1") == "0", out
    assert (
        float(value(res.stdout, "s0")) < 1e-3 and float(value(res.stdout, "n0")) > 1.0
    ), out
    assert float(value(res.stdout, "e0r")) < 1e-5, out
    assert value(res.stdout, "sh0") == "1:8", out
    assert float(value(res.stdout, "s1")) < 1e-3, out
    assert float(value(res.stdout, "ssq")) < 1e-3, out
    assert (
        float(value(res.stdout, "sc1")) < 1e-4
        and float(value(res.stdout, "sc0")) < 1e-4
    ), out
    assert value(res.stdout, "shm1") == "2:6x3", out
    assert float(value(res.stdout, "e1e")) < 1e-6, out


def test_C4_zero_divisor_in_untaken_branch_is_selected_away(tmp_path):
    """Kept scalar and image selects with a zero divisor in the other branch run every tick."""
    res, out = _unit(
        tmp_path,
        "c4",
        "n4 = _n % 2\nx4 = n4 > 0 ? 7 / n4 : -1\nout4 = where(n4 > 0, cam / n4, -1)\nout4s = cam * 0 + x4\n"
        "k4 = _n % 2 == 0 ? 1 : 1 / (_n % 2)\nout4k = cam * 0 + k4\n",
        '@calc -v o4,o4s,o4k "imean(out4); imean(out4s); imean(out4k)"; echo "o4=$o4,$o4s,$o4k"\n'
        'echo "cnt=$(( $(@get s.out4.cnt0) ))"',
    )
    assert value(res.stdout, "error") == "[]", out
    o4, o4s, o4k = value(res.stdout, "o4").split(",")
    assert o4 in ("5", "-1") and o4s in ("7", "-1") and o4k == "1", out
    assert (o4 == "5") == (o4s == "7"), out
    assert int(value(res.stdout, "cnt")) >= 5, out


def _ext_blk(name, params):
    """Source of a Tier-1 block kernel @name with parameter X-macro lines @params."""
    return (
        '#include "milk_calc_ext.h"\n'
        f"#define {name.upper()}_PARAMS(X) \\\n" + " \\\n".join(params) + "\n"
        f"MILK_CALC_PARAMS({name}, {name.upper()}_PARAMS);\n"
        f"static void {name}_f(float *restrict out, const float *restrict in, const struct {name}_par *P, int len)\n"
        "{ for (int i = 0; i < len; i++) out[i] = in[i] * 2.0f; }\n"
        f'MILK_CALC_EXT_EXPORT(MILK_CALC_REG_BLK("{name}", {name}_f, {name}, 4));\n'
    )


def test_C5_parameter_tables_the_packer_cannot_hold_are_refused(tmp_path):
    """17 parameters, or a string field, used to load silently and pack past the 256-byte buffer."""
    (tmp_path / "big17.c").write_text(
        _ext_blk(
            "big",
            [
                f'    X(".p{i}", p{i}, FPTYPE_FLOAT64, {i}.0, FPFLAG_DEFAULT_INPUT, "p{i}")'
                for i in range(17)
            ],
        )
    )
    (tmp_path / "strp.c").write_text(
        _ext_blk(
            "strp",
            [
                '    X(".tag", tag, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT, "tag")',
                '    X(".g", g, FPTYPE_FLOAT64, 2.0, FPFLAG_DEFAULT_INPUT, "gain")',
            ],
        )
    )
    (tmp_path / "ok16.c").write_text(
        _ext_blk(
            "ok",
            [
                f'    X(".q{i}", q{i}, FPTYPE_FLOAT32, {i}.0, FPFLAG_DEFAULT_INPUT, "q{i}")'
                for i in range(16)
            ],
        )
    )
    res = milk_bash(
        'mem.mk2Dim a5 4 4 >/dev/null; @calc "a5 = 3"\n'
        '@calc -l big17.c "o5 = big(a5)" 2>err17.log; echo "rc17=$?"\n'
        '@calc -l strp.c "o5s = strp(a5)" 2>errs.log; echo "rcstr=$?"\n'
        '@calc -l ok16.c "o5k = ok(a5)" 2>errok.log; echo "rcok=$?"\n'
        '@calc -v m "imean(o5k)"; echo "ok=$m"',
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert value(res.stdout, "rc17") != "0" and value(res.stdout, "rcstr") != "0", out
    assert "at most 16" in (tmp_path / "err17.log").read_text(), out
    assert "cannot pack" in (tmp_path / "errs.log").read_text(), out
    assert value(res.stdout, "rcok") == "0" and value(res.stdout, "ok") == "6", out


def test_C6_stencils_on_broadcast_or_sliced_operands_match_eager(tmp_path):
    """A stencil on a 2-D operand broadcast into a cube, or on a slice, used to index the whole image
    by iteration coordinates (reading past it); such programs now leave the fused path. The eager
    medfilt used a window size the tokenizer never set."""
    res = milk_bash(
        "mem.mk2Dim a6 8 6 >/dev/null; mem.mk3Dim cube6 8 6 3 >/dev/null\n"
        '@calc "a6 = urand(a6, 11); cube6 = urand(cube6, 12)"\n'
        '@calc "s6 = shift(a6, 1, 0); m6 = medfilt3(a6); m5 = medfilt5(a6)"\n'
        'chk() { @calc -v e "imax(abs($1[:,:,$2] - cube6[:,:,$2] - $3))"; echo "$1=$e"; }\n'
        '@calc "o1 = shift(a6, 1, 0) + cube6"; echo "rc1=$?"; chk o1 2 s6\n'
        '@calc "o2 = medfilt3(a6) + cube6"; echo "rc2=$?"; chk o2 1 m6\n'
        '@calc "o3 = cube6 + medfilt5(a6)"; echo "rc3=$?"; chk o3 1 m5\n'
        '@calc "o4 = medfilt3(a6) + m6[::1b]"; @calc -v e4 "imax(abs(o4 - 2 * m6))"; echo "e4=$e4"\n'
        '@calc "def f6(x) = shift(x, 1, 0)"; @calc "o5 = f6(a6[2:5,:])"; echo "rc5=$?"\n'
        '@calc "t6 = a6[2:5::1b,:]; r5 = shift(t6, 1, 0)"; @calc -v e5 "imax(abs(o5 - r5))"; echo "e5=$e5"\n'
        '@calc "o6 = medfilt3(a6[2:5,:])"; echo "rc6=$?"; @calc "r6 = medfilt3(t6)"; @calc -v e6 "imax(abs(o6 - r6))"; echo "e6=$e6"',
        tmp_path,
    )
    out = res.stdout + res.stderr
    for k in ("rc1", "rc2", "rc3", "rc5", "rc6"):
        assert value(res.stdout, k) == "0", out
    for k in ("o1", "o2", "o3", "e4", "e5", "e6"):
        assert float(value(res.stdout, k)) < 1e-6, out


def test_C4_unguarded_zero_divisor_still_fails(tmp_path):
    """Outside a select the zero divisor is still an error, in the shell and in a unit."""
    res = milk_bash(
        HELPERS
        + '@calc "n0 = 0"; @calc "bad4 = cam / n0" 2>/dev/null; echo "badrc=$?"\n'
        '@calc "ok4 = n0 > 0 ? 1 / n0 : -1"; @calc -v y "ok4"; echo "y=$y"',
        tmp_path,
    )
    assert value(res.stdout, "badrc") != "0", res.stdout + res.stderr
    assert value(res.stdout, "y") == "-1", res.stdout + res.stderr
    res, out = _unit(
        tmp_path,
        "c4u",
        "n4u = _n * 0\nbad4u = cam / n4u\n",
        'echo "has=$(@get s.bad4u.cnt0)"',
    )
    assert value(res.stdout, "error") != "[]", out
