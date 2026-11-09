"""How calculator statements run: fusion, locals, axis reductions, selects, casts and NaN.

Purpose:     a statement must give the same values however it runs (one fused pass, merged
             with the next statement, per operator, kept in a unit), and keep running when an
             input changes shape or type.
Does:        checks reductions that read a value written earlier in the block; locals elided
             or kept in a unit; axis reductions on non-square images and cubes; a zero divisor
             in an untaken branch (no error) and outside a select (an error); stencils on
             broadcast or sliced operands; re-planning after an input is re-created; the JIT's
             math and its `/=` rollback; exact `%`, `fmod` and `wrap_pi`; casts, masks and
             broadcasts; argmax and argmin in float and double; NaN in imin, imax, lut and
             interp2; hypot overflow; the `torn` field of `s.<name>.stats`; kept scalar
             statements.
Covers:      the calculator: src/calc/plan/ (mc_plan.c, mc_views.c), src/calc/jit/ (mc_emit.c,
             mc_cc.c), src/calc/run/ (mc_exec.c, mc_prestage.c, mc_reduce.c), src/calc/mc_eval.c;
             milk-fpsexec-milk-calc (src/calc/units/milk_unit_calc.c); the stats property
             (src/prop/milk_shm_prop_stream.c).
Uses:        conftest.py: milk_bash, _run_bash, value, CAM, UNIT_HELPERS.
See also:    docs/guide/calculator.md, "Fusion & Cost Model" and "Compute Units Without C";
             fused_views_test.py.
"""

from __future__ import annotations

import pytest
import math
import re
import struct

from milk.cliwrap import HAVE_CLI
from tests.cli.conftest import CAM, UNIT_HELPERS, _run_bash, milk_bash, value

pytestmark = [
    pytest.mark.skipif(not HAVE_CLI, reason="MILK compiled without CLI support"),
    pytest.mark.timeout(120),
]


def _unit(tmp_path, name, prog, probes, init=""):
    """Run @prog as unit @name on cam, feed cam = 2 then 5, print @probes, stop; return stdout+log."""
    (tmp_path / "prog.calc").write_text(prog)
    opt_init = f' -i "{init}"' if init else ""
    res = milk_bash(
        UNIT_HELPERS
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


def test_nan_in_lut_and_interp2_does_not_crash():
    res = _run_bash("""
        mem.mk2Dim "@S:rim" 4 4
        @calc "rim = -2.0"
        mem.mk2Dim "@S:tbl" 8 1
        @calc "tbl = ix(tbl); nanim = rim * sqrt(-1.0); lout = lut(nanim, tbl, 0, 1); iout = interp2(rim, nanim, nanim)"
        echo "NAN_OK"
    """)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "NAN_OK" in res.stdout


def test_fused_reduction_reads_this_block_value(tmp_path):
    """In one block, `imean(y)` sees the `y` written by the statement before it, not the old one."""
    res = milk_bash(
        CAM + '@calc "y = cam * 0"\n'
        '@calc "for k = 1:1 { y = cam * 3; z = y - imean(y) }"; echo "rc=$?"\n'
        '@calc -v zm "imean(z)"; echo "zm=$zm"',
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert value(res.stdout, "rc") == "0", out
    assert value(res.stdout, "zm") == "0", out


def test_image_hypot_does_not_overflow(tmp_path):
    res = milk_bash(
        'mem.mk2Dim a 4 4 >/dev/null; @calc "a = 1e20"; @calc "h = hypot(a, a)"\n'
        '@calc -v hm "imax(h)"; echo "hm=$hm"',
        tmp_path,
    )
    hm = value(res.stdout, "hm")
    assert hm is not None, res.stdout + res.stderr
    assert float(hm) == pytest.approx(math.sqrt(2.0) * 1e20, rel=1e-6)


def test_mod_and_wrap_pi_stay_exact(tmp_path):
    res = milk_bash(
        "mem.mk2Dim a 16 16 >/dev/null; mem.mk2Dim y 16 16 >/dev/null; @config digits 17 >/dev/null\n"
        '@calc "a = (rand(a, 1) - 0.5) * 2000; y = (rand(y, 2) + 0.01) * 3"\n'
        '@calc "m = a % y; f = fmod(a, y); w = wrap_pi(a)"\n'
        "for k in 0 17 101 202 255; do x=$((k % 16)); j=$((k / 16)); "
        'echo "px=$(@calc -v r "a[$x,$j]"; echo $r) $(@calc -v r "y[$x,$j]"; echo $r) '
        '$(@calc -v r "m[$x,$j]"; echo $r) $(@calc -v r "f[$x,$j]"; echo $r) $(@calc -v r "w[$x,$j]"; echo $r)"; done\n'
        '@calc -v big "1e20 % 3"; echo "big=$big"',
        tmp_path,
    )
    out = res.stdout + res.stderr
    f32 = lambda v: struct.unpack("f", struct.pack("f", v))[0]
    rows = re.findall(r"^px=(.*)$", res.stdout, re.M)
    assert len(rows) == 5, out
    for row in rows:
        a, y, m, f, w = map(float, row.split())
        ref = f32(math.fmod(a, y))
        assert m == ref and f == ref, (row, ref, out)
        assert w == f32(math.remainder(a, f32(2 * math.pi))), (row, out)
    assert value(res.stdout, "big") == "nan", out


def test_casts_masks_and_broadcasts_keep_values(tmp_path):
    vals = "300 -2 2.7 -2.7 70000.5 -70000.5 2147483904 nan inf"
    res = milk_bash(
        'mem.mk2Dim a 8 8 >/dev/null; mem.mk2Dim "@TF64:d" 8 8 >/dev/null\n'
        f'for v in {vals}; do @calc "a = $v; d = $v"\n'
        '  @calc "b = u8(a); c = i16(a); m = a & 0xF0; e = u16(d); g = d & 0xFFF"\n'
        '  echo "img=$v $(@calc "b[0,0]") $(@calc "c[0,0]") $(@calc "m[0,0]") $(@calc "e[0,0]") $(@calc "g[0,0]")"\n'
        '  echo "num=$v $(@calc "u8($v)") $(@calc "i16($v)") $(@calc "$v & 0xF0" 2>/dev/null || echo 0) '
        '$(@calc "u16($v)") $(@calc "$v & 0xFFF" 2>/dev/null || echo 0)"; done\n'
        "mem.mk2Dim A 64 8 >/dev/null; mem.mk2Dim col 1 8 >/dev/null; mem.mk2Dim U 64 8 >/dev/null; mem.mk2Dim V 64 8 >/dev/null\n"
        '@calc "A = ix(A, 0, 0); col = iy(col, 0, 0) * 100; C = A + col; U = 1; V = 2"; echo "bc=$(@calc "C[5, 3]")"\n'
        '@calc "U, V = V, U"; @calc "U = U + A; V = V - A"; echo "uv=$(@calc "U[5, 0]") $(@calc "V[5, 0]")"',
        tmp_path,
    )
    out = res.stdout + res.stderr
    img = {
        r.split()[0]: r.split()[1:] for r in re.findall(r"^img=(.*)$", res.stdout, re.M)
    }
    num = {
        r.split()[0]: r.split()[1:] for r in re.findall(r"^num=(.*)$", res.stdout, re.M)
    }
    assert len(img) == 9 and img == num, out
    assert value(res.stdout, "bc") == "305", out
    assert value(res.stdout, "uv") == "7 -4", out


def test_float_argext_matches_double(tmp_path):
    res = milk_bash(
        'mem.mk2Dim w 100 50 >/dev/null; mem.mk2Dim "@TF64:wd" 100 50 >/dev/null\n'
        'for s in 1 2 3; do @calc "w = floor(rand(w, $s) * 20) - 10"\n'
        '  [ $s = 2 ] && @calc "w = -inf; w[1, 0] = nan; w[60, 40] = -3.4028234663852886e38"\n'
        '  [ $s = 3 ] && @calc "w = nan"\n'
        '  @calc "wd = f64(w)"; echo "f$s=$(@calc "iargmax(w); iargmin(w); iamax(w)" | tr "\n" " ")"\n'
        '  echo "d$s=$(@calc "iargmax(wd); iargmin(wd); iamax(wd)" | tr "\n" " ")"; done',
        tmp_path,
    )
    out = res.stdout + res.stderr
    for s in "123":
        assert value(res.stdout, f"f{s}") == value(res.stdout, f"d{s}"), out
    assert value(res.stdout, "f2").split()[0] == "4060", out
    assert value(res.stdout, "f3").split() == ["-1", "-1", "-1"], out


def test_elided_unread_local_does_not_replace_next_output(tmp_path):
    """`local tmp = cam - 1; out = cam * 3` fuse into one pass; out is 3*cam, not tmp."""
    res, out = _unit(
        tmp_path,
        "c1",
        "local tmp1 = cam - 1\nout1 = cam * 3\n",
        '@calc -v o1 "imean(out1)"; echo "o1=$o1"',
    )
    assert value(res.stdout, "error") == "[]", out
    assert value(res.stdout, "o1") == "15", out


def test_local_read_through_def_is_written(tmp_path):
    """A `def` body reading the local keeps it materialized: out3 sees this frame's tmp3."""
    res, out = _unit(
        tmp_path,
        "c2a",
        "def g(x) = x + tmp3\nlocal tmp3 = cam - 1\nout3b = cam * 2\nout3 = g(cam)\n",
        '@calc -v o3,o3b "imean(out3); imean(out3b)"; echo "o3=$o3,$o3b"',
    )
    assert value(res.stdout, "error") == "[]", out
    assert value(res.stdout, "o3") == "9,10", out


def test_local_read_by_merged_statement_is_written(tmp_path):
    """A reader folded into an earlier pass still counts: b2 sees the previous frame's tmp2."""
    res, out = _unit(
        tmp_path,
        "c2b",
        "a2 = cam * 2\nb2 = tmp2 + 1\nlocal tmp2 = cam - 1\nout2b = cam * 2\n",
        '@calc -v b2,o2b "imean(b2); imean(out2b)"; echo "b2=$b2,$o2b"',
        init="local tmp2 = cam * 0",
    )
    assert value(res.stdout, "error") == "[]", out
    assert value(res.stdout, "b2") == "5,10", out


def test_axis0_reduction_is_placed_on_its_axis(tmp_path):
    """Row means come off every row of a 16x8 image and of an 8x8 one; shapes of plain results stay squeezed."""
    res = milk_bash(
        'mem.mk2Dim im3 16 8 >/dev/null; @calc "im3 = iy(im3)*10 + ix(im3) + rand(im3, 3)"\n'
        '@calc "d0 = im3 - imean(im3, 0)"; echo "rc0=$?"\n'
        '@calc -v s0,n0 "imax(abs(itot(d0, 0))); irms(d0)"; echo "s0=$s0"; echo "n0=$n0"\n'
        '@calc "d0r = imean(im3, 0) - im3"; @calc -v e0r "imax(abs(d0r + d0))"; echo "e0r=$e0r"\n'
        '@calc "p0 = imean(im3, 0)"; echo "sh0=$(@get s.p0.naxis):$(@get s.p0.xsize)"\n'
        '@calc "d1 = im3 - imean(im3, 1)"; @calc -v s1 "imax(abs(itot(d1, 1)))"; echo "s1=$s1"\n'
        'mem.mk2Dim sq 8 8 >/dev/null; @calc "sq = iy(sq)*10 + ix(sq) + rand(sq, 5)"\n'
        '@calc "dsq = sq - imean(sq, 0)"; @calc -v ssq "imax(abs(itot(dsq, 0)))"; echo "ssq=$ssq"\n'
        'mem.mk3Dim cube3 6 4 3 >/dev/null; @calc "cube3 = rand(cube3, 7)"\n'
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


def test_zero_divisor_in_untaken_branch_is_selected_away(tmp_path):
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


def test_stencils_on_broadcast_or_sliced_operands_match_eager(tmp_path):
    """A stencil on a 2-D operand broadcast into a cube, or on a slice, used to index the whole image
    by iteration coordinates (reading past it); such programs now leave the fused path. The eager
    medfilt used a window size the tokenizer never set."""
    res = milk_bash(
        "mem.mk2Dim a6 8 6 >/dev/null; mem.mk3Dim cube6 8 6 3 >/dev/null\n"
        '@calc "a6 = rand(a6, 11); cube6 = rand(cube6, 12)"\n'
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


def test_unguarded_zero_divisor_still_fails(tmp_path):
    """Outside a select the zero divisor is still an error, in the shell and in a unit."""
    res = milk_bash(
        UNIT_HELPERS
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


def test_jit_live_stream_datatype_and_shape_replan(tmp_path):
    """Kept programs and cached sub-programs automatically re-plan when an input image changes datatype or shape."""
    res = milk_bash(
        'mem.mk2Dim s2 8 8 >/dev/null; @calc "s2 = 3.0"\n'
        'for _ in 1 2; do @calc "o2 = s2 * 2.0 + 1.0; sh2 = shift(s2 + 1.0, 1, 0)"; done\n'
        '@calc -v m1,ms1 "imean(o2); imean(sh2)"; echo "f32=$m1:$ms1:$(@get s.o2.xsize)"\n'
        # Recreate s2 as 16x16 float64
        'mem.rmshmim s2 >/dev/null; mem.mk2Dim s2 16 16 >/dev/null; @calc "s2 = f64(s2) + 4.25"\n'
        '@calc "o2 = s2 * 2.0 + 1.0; sh2 = shift(s2 + 1.0, 1, 0)"\n'
        '@calc -v m2,ms2 "imean(o2); imean(sh2)"; echo "f64=$m2:$ms2:$(@get s.o2.xsize)"\n'
        # Recreate s2 as 12x10 uint16
        'mem.rmshmim s2 >/dev/null; mem.mk2Dim s2 12 10 >/dev/null; @calc "s2 = u16(s2) + 10"\n'
        '@calc "o2 = s2 * 2.0 + 1.0; sh2 = shift(s2 + 1.0, 1, 0)"\n'
        '@calc -v m3,ms3 "imean(o2); imean(sh2)"; echo "u16=$m3:$ms3:$(@get s.o2.xsize)x$(@get s.o2.ysize)"',
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert value(res.stdout, "f32") == "7:4:8", out
    assert value(res.stdout, "f64") == "9.5:5.25:16", out
    assert value(res.stdout, "u16") == "21:11:12x10", out


def test_jit_differential_math_and_multipass_batch(tmp_path):
    """Differential test for libmvec SIMD transcendentals, NaN/negative-pow, vectorized /= rollback, and batch compilation."""
    cache_dir = tmp_path / "jit_cache"
    cache_dir.mkdir()
    res = milk_bash(
        f'export MILK_CALC_CACHE="{cache_dir}"\n'
        "mem.mk2Dim a 16 12 >/dev/null; mem.mk2Dim b 16 12 >/dev/null\n"
        '@calc "a = (abs(ix(a)) + 1.0) * 0.1; b = (abs(iy(b)) + 1.0) * 0.05"\n'
        # 1. Multi-pass batch compile (3 distinct passes) + verify reuse from disk cache
        '@calc "t1 = a * 2.5 + b; t2 = shift(t1, 1, 0) * 1.5; c_batch = shift(t2, 0, 1) - a"\n'
        '@calc "t2_only = shift(t1, 1, 0) * 1.5"\n'
        '@calc -v eb "imax(abs(t2 - t2_only))"; echo "eb=$eb"\n'
        # 2. Transcendentals & negative-base pow (no -ffast-math finite-only / exp(y*log(x)) breakage)
        '@calc "tr = sin(a) * sin(a) + cos(a) * cos(a)"\n'
        '@calc -v etr "imax(abs(tr - 1.0))"; echo "etr=$etr"\n'
        '@calc "neg = a * 0.0 - 2.0; p3 = pow(neg, 3.0)"\n'
        '@calc -v mp3 "imean(p3)"; echo "mp3=$mp3"\n'
        # 3. Vectorized /= fast path and atomic rollback when last pixel of divisor is 0
        '@calc "c_div = a + 0.0"; @calc "c_div /= b"; @calc "ref_div = a / b"\n'
        '@calc -v ediv "imax(abs(c_div - ref_div))"; echo "ediv=$ediv"\n'
        '@calc "zdiv = b + 0.0; zdiv[15, 11] = 0.0; c_rol = a + 5.0"\n'
        '@calc "c_rol /= zdiv" 2>/dev/null; echo "zrc=$?"\n'
        '@calc -v erol "imax(abs(c_rol - (a + 5.0)))"; echo "erol=$erol"\n'
        # 4. Sliced fused views + compound sub-expressions
        '@calc "c_sl = a * 0.0"\n'
        '@calc "c_sl[2:14, 1:10] = a[2:14, 1:10] * 3.0 + b[2:14, 1:10]"\n'
        '@calc -v esl "imax(abs(c_sl[2:14, 1:10] - (a[2:14, 1:10] * 3.0 + b[2:14, 1:10])))"; echo "esl=$esl"',
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert float(value(res.stdout, "eb")) < 1e-6, out
    assert float(value(res.stdout, "etr")) < 1e-5, out
    assert value(res.stdout, "mp3") == "-8", out
    assert float(value(res.stdout, "ediv")) < 1e-6, out
    assert value(res.stdout, "zrc") != "0", out
    assert float(value(res.stdout, "erol")) == 0.0, out
    assert float(value(res.stdout, "esl")) < 1e-6, out


def test_imin_imax_skip_nan(tmp_path):
    """A NaN pixel is skipped whatever its position and the image size; all-NaN gives NaN."""
    res = milk_bash(
        'mem.mk2Dim "@S:a" 8 8 >/dev/null; @calc "a = ix(a) + 10; a[0,0] = sqrt(-1)"\n'
        '@calc -v mn,mx "imin(a); imax(a)"; echo "small=$mn,$mx"\n'
        'mem.mk2Dim "@S:big" 1100 1000 >/dev/null\n'
        '@calc "big = ix(big) + 5000; big[0,0] = sqrt(-1); big[5,5] = sqrt(-1)"\n'
        '@calc -v mn,mx "imin(big); imax(big)"; echo "big=$mn,$mx"\n'
        '@calc "r = imin(a, 1); c = imax(a, 0)"; @calc -v r0,c0 "r[0]; c[0]"; echo "axis=$r0,$c0"\n'
        'mem.mk2Dim "@S:nn" 4 4 >/dev/null; @calc "nn = sqrt(-1)"\n'
        '@calc -v mn,mx "imin(nn); imax(nn)"; echo "allnan=$mn,$mx"',
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert value(res.stdout, "small") == "6,13", out  # ix is centred: -4+10 .. 3+10
    assert value(res.stdout, "big") == "4450,5549", out  # -550+5000 .. 549+5000
    assert value(res.stdout, "axis") == "6,13", out  # row 0 and column 0 hold the NaN
    assert all(
        v.lstrip("-") == "nan" for v in value(res.stdout, "allnan").split(",")
    ), out


def test_stats_listing_has_torn_flag(tmp_path):
    res = milk_bash(
        'mem.mk2Dim "@S:c" 8 8 >/dev/null; @calc "c = 3"\n'
        '@get -A st s.c.stats; echo "torn=${st[torn]} mean=${st[mean]}"\n'
        'echo "lines=$(@get s.c.stats | wc -l)"',
        tmp_path,
    )
    assert value(res.stdout, "torn") == "0 mean=3", res.stdout + res.stderr
    assert value(res.stdout, "lines") == "8", res.stdout + res.stderr


def test_kept_scalar_statement_value_is_stable(tmp_path):
    """The second run of a scalar statement (kept) gives the text path's value, integer or not."""
    res = milk_bash(
        'mem.mk2Dim "@S:b" 8 8 >/dev/null; @calc "b = 1; b[0,0] = 0"\n'
        '@calc "n = icount(b) + 1"; @calc -v v1 "n"; @calc "n = icount(b) + 1"; @calc -v v2 "n"\n'
        '@calc "d = icount(b) / 2"; @calc "d = icount(b) / 2"; @calc -v v3 "d"\n'
        '@calc "q = n > 3 && d < 100"; @calc "q = n > 3 && d < 100"; @calc -v v4 "q"\n'
        'echo "vals=$v1,$v2,$v3,$v4"',
        tmp_path,
    )
    assert value(res.stdout, "vals") == "64,64,31.5,1", res.stdout + res.stderr
