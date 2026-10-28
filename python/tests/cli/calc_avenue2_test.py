"""
Regression tests for the milk.calc / @calc Avenue 2 scripting language features
and Track A bytecode compiler upgrades:
- A1: Multi-reductions (up to 4 per expression), sub-expression deduplication,
      and reentrant nested reductions (`irms(im - imean(im))`).
- A2/A3: Multi-threaded float-fast-path axis reductions and compute-weighted OMP scaling.
- A4/B2: Cross-statement Paradigm (1) cache-resident block fusion and `local` / `L:` / `@L:`
         process-private scratch images.
- B1: Fused bitwise & integer operators (`&`, `|`, `^^`, `~`, `<<`, `>>`).
- B3: Option 2 brace-delimited `if { ... } elif { ... } else { ... }` and
      `for k = a:b[:s] { ... }` loops.
- B4: Live SHM entity bindings (`s.`, `f.`, `p.`) in expressions and LHS assignments.
- B5: 2D relative neighbor stencils (`shift(im, dx[, dy])`) including in-place aliasing safety.
- B6: Multi-output unpacking (`a, b = expr1, expr2` and `m, s[, lo, hi] = istats(im)`).
- B7: Inline user functions (`def f(x, a) = <expr>`) on scalars and images.
- B8: 1D lookup table linear interpolation (`lut(expr, tbl[, x0, dx])`).
"""

from __future__ import annotations

import pytest

from milk.cliwrap import HAVE_CLI
from tests.cli.conftest import milk_bash, value

pytestmark = [
    pytest.mark.skipif(not HAVE_CLI, reason="MILK compiled without CLI support"),
    pytest.mark.timeout(120),
]


def test_multi_and_nested_reductions_and_istats(tmp_path):
    """A1 & B6: Multi-reductions in one expression, nested reductions, and istats() unpacking."""
    res = milk_bash(
        "mem.mk2Dim a 16 16 >/dev/null\n"
        '@calc "a = ix(a) * 1.5 - iy(a) * 0.7 + 3.2"\n'
        '@calc "z = (a - imean(a)) / istd(a)"\n'
        '@calc "n = (a - imin(a)) / (imax(a) - imin(a))"\n'
        '@calc -v zm,zs,nmin,nmax "imean(z); istd(z); imin(n); imax(n)"\n'
        '@calc -v nest,ref "irms(a - imean(a)); istd(a)"\n'
        '@calc "mu, sig, lo, hi = istats(a)"\n'
        '@calc -v vmu,vsig,vlo,vhi "mu; sig; lo; hi"\n'
        '@calc -v rmu,rsig,rlo,rhi "imean(a); istd(a); imin(a); imax(a)"\n'
        '@calc "mu2, sig2 = istats(a * 2)"\n'
        '@calc -v vmu2,vsig2 "mu2; sig2"\n'
        'echo "z=$zm,$zs"\n'
        'echo "n=$nmin,$nmax"\n'
        'echo "nest=$nest,$ref"\n'
        'echo "istats=$vmu,$vsig,$vlo,$vhi"\n'
        'echo "ref=$rmu,$rsig,$rlo,$rhi"\n'
        'echo "istats2=$vmu2,$vsig2"',
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert res.returncode == 0, out
    zm, zs = [float(x) for x in value(res.stdout, "z").split(",")]
    assert abs(zm) < 1e-5 and abs(zs - 1.0) < 1e-5, out
    nmin, nmax = [float(x) for x in value(res.stdout, "n").split(",")]
    assert abs(nmin - 0.0) < 1e-6 and abs(nmax - 1.0) < 1e-6, out
    nest, ref = [float(x) for x in value(res.stdout, "nest").split(",")]
    assert abs(nest - ref) < 1e-5, out
    ist = [float(x) for x in value(res.stdout, "istats").split(",")]
    rfs = [float(x) for x in value(res.stdout, "ref").split(",")]
    for u, v in zip(ist, rfs):
        assert abs(u - v) < 1e-5, out
    mu2, sig2 = [float(x) for x in value(res.stdout, "istats2").split(",")]
    assert abs(mu2 - 2.0 * rfs[0]) < 1e-4 and abs(sig2 - 2.0 * rfs[1]) < 1e-4, out


def test_bitwise_operators_scalars_and_images(tmp_path):
    """B1: Bitwise &, |, ^^, ~, <<, >> on scalars and images, including >2^24 precision."""
    res = milk_bash(
        '@calc -v b1,b2,b3,b4,b5,b6 "0xff & 0x0f; 0xf0 | 0x05; 0xff ^^ 0x0f; ~0; 1 << 10; 1024 >> 3"\n'
        '@calc -v big "((1 << 28) + 7) & ((1 << 28) + 3)"\n'
        '@calc -v badsh "(1 << 70) + (1 << -3)"\n'
        "mem.mk2Dim m1 8 8 >/dev/null; mem.mk2Dim m2 8 8 >/dev/null\n"
        '@calc "m1 = u16(ix(m1) + 4); m2 = u16(iy(m2) + 4)"\n'
        '@calc "mand = m1 & 3; mor = m1 | 8; mxor = m1 ^^ m2; msh = (m1 & 3) << 2"\n'
        '@calc -v v0,v1,v2,v3 "mand[3,2]; mor[3,2]; mxor[3,2]; msh[3,2]"\n'
        'echo "sc=$b1,$b2,$b3,$b4,$b5,$b6"\n'
        'echo "big=$big"\n'
        'echo "badsh=$badsh"\n'
        'echo "im=$v0,$v1,$v2,$v3"',
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert res.returncode == 0, out
    assert value(res.stdout, "sc") == "15,245,240,-1,1024,128", out
    assert int(value(res.stdout, "big")) == ((1 << 28) + 3), out
    assert value(res.stdout, "badsh") == "0", out
    # At [3,2] on 8x8 centred at (4,4): ix = -1 -> m1 = 3; iy = -2 -> m2 = 2
    # mand = 3 & 3 = 3; mor = 3 | 8 = 11; mxor = 3 ^ 2 = 1; msh = (3 & 3) << 2 = 12
    assert value(res.stdout, "im") == "3,11,1,12", out


def test_brace_if_elif_else_and_for_loops(tmp_path):
    """B3: Option 2 C-style brace blocks for if/elif/else and bounded for loops."""
    res = milk_bash(
        'mem.mk2Dim im 8 8 >/dev/null; @calc "im = 1"\n'
        '@calc "mode = 2; if mode == 1 { im = im + 10 } elif mode == 2 { im = im + 20 } else { im = im + 30 }"\n'
        '@calc -v v1 "im[0,0]"\n'
        '@calc "acc = 0; for k = 1:5 { acc = acc + k * k }"\n'
        '@calc -v vacc "acc"\n'
        '@calc "sstep = 0; for k = 10:2:-2 { sstep = sstep + k }"\n'
        '@calc -v vsstep "sstep"\n'
        '@calc "for k = 1:4 { if k % 2 == 0 { im = im + k } else { im = im - 1 } }"\n'
        '@calc -v v2 "im[0,0]"\n'
        'echo "v1=$v1"\n'
        'echo "acc=$vacc"\n'
        'echo "sstep=$vsstep"\n'
        'echo "v2=$v2"',
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert res.returncode == 0, out
    assert value(res.stdout, "v1") == "21", out
    assert value(res.stdout, "acc") == "55", out  # 1 + 4 + 9 + 16 + 25
    assert value(res.stdout, "sstep") == "30", out  # 10 + 8 + 6 + 4 + 2
    # v2 starts at 21; k=1 (-1), k=2 (+2), k=3 (-1), k=4 (+4) => 21 + 4 = 25
    assert value(res.stdout, "v2") == "25", out


def test_shift_stencil_and_inplace_aliasing(tmp_path):
    """B5: 2D neighbor stencil shift(im, dx, dy) with edge clamping and self-aliasing protection."""
    res = milk_bash(
        "mem.mk2Dim a 8 6 >/dev/null\n"
        '@calc "a = (ix(a) + 4) + 10 * (iy(a) + 3)"\n'
        '@calc "sx = shift(a, 1, 0); sy = shift(a, 0, -1)"\n'
        '@calc -v p_in,p_edge,py_in,py_edge "sx[2,3]; sx[7,3]; sy[2,3]; sy[2,0]"\n'
        # Test in-place self-aliasing: a = shift(a, 1, 0) must match sx everywhere
        '@calc "b = a; b = shift(b, 1, 0)"\n' '@calc -v err_alias "imax(abs(b - sx))"\n'
        # 5-point discrete Laplacian on a quadratic surface f(x,y) = x^2 + y^2 has interior value -4
        '@calc "q = ix(a)*ix(a) + iy(a)*iy(a)"\n'
        '@calc "lap = 4*q - shift(q, 1, 0) - shift(q, -1, 0) - shift(q, 0, 1) - shift(q, 0, -1)"\n'
        '@calc -v lap_c "lap[4,3]"\n'
        'echo "st=$p_in,$p_edge,$py_in,$py_edge"\n'
        'echo "alias=$err_alias"\n'
        'echo "lap=$lap_c"',
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert res.returncode == 0, out
    # a[x,y] = x + 10*y. sx[2,3] = a[3,3] = 33; sx[7,3] = a[clamp(8),3] = a[7,3] = 37
    # sy[2,3] = a[2,2] = 22; sy[2,0] = a[2,clamp(-1)] = a[2,0] = 2
    assert value(res.stdout, "st") == "33,37,22,2", out
    assert float(value(res.stdout, "alias")) == 0.0, out
    assert float(value(res.stdout, "lap")) == -4.0, out


def test_user_defined_functions_and_lut(tmp_path):
    """B7 & B8: Inline def f(x, ...) = expr user functions and 1D lut() interpolation."""
    res = milk_bash(
        '@calc "def clamp_pos(x, thr, hi) = where(x > thr, min(x, hi), 0)"\n'
        '@calc "def sq_plus(x, c) = x * x + c"\n'
        '@calc -v sc "sq_plus(4, 3)"\n'
        "mem.mk2Dim a 8 8 >/dev/null\n"
        '@calc "a = ix(a)"\n'
        '@calc "b = clamp_pos(a, 0, 2); c = sq_plus(a + 1, 5)"\n'
        '@calc "cref = (a + 1) * (a + 1) + 5"\n'
        '@calc -v b_neg,b_one,b_hi,cerr "b[2,0]; b[5,0]; b[7,0]; imax(abs(c - cref))"\n'
        # 1D LUT: tbl has 5 entries [10, 20, 40, 80, 160]
        "mem.mk2Dim tbl 5 1 >/dev/null\n"
        '@calc "tbl = 10 * (2 ^ (ix(tbl) + 2))"\n'
        '@calc "u = zeros(4, 1); u[0,0] = -1; u[1,0] = 0.5; u[2,0] = 2.25; u[3,0] = 99"\n'
        '@calc "lu = lut(u, tbl)"\n'
        '@calc -v l0,l1,l2,l3 "lu[0,0]; lu[1,0]; lu[2,0]; lu[3,0]"\n'
        # Scaled LUT with x0 = 100, dx = 10
        '@calc "us = u * 10 + 100; lus = lut(us, tbl, 100, 10)"\n'
        '@calc -v lserr "imax(abs(lus - lu))"\n'
        'echo "sc=$sc"\n'
        'echo "udef=$b_neg,$b_one,$b_hi,$cerr"\n'
        'echo "lut=$l0,$l1,$l2,$l3"\n'
        'echo "lserr=$lserr"',
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert res.returncode == 0, out
    assert value(res.stdout, "sc") == "19", out
    # ix(a) at x=2 is -2 -> 0; x=5 is 1 -> 1; x=7 is 3 -> clamped to 2
    assert value(res.stdout, "udef") == "0,1,2,0", out
    # tbl = [10, 20, 40, 80, 160]
    # u=-1 -> 10; u=0.5 -> 15; u=2.25 -> 40 + 0.25*(80-40) = 50; u=99 -> 160
    assert value(res.stdout, "lut") == "10,15,50,160", out
    assert float(value(res.stdout, "lserr")) == 0.0, out


def test_shm_live_bindings_and_multi_assign_and_local_fusion(tmp_path):
    """A4, B2, B4, B6: SHM property reads/writes, multi-image unpacking, and local scratch fusion."""
    (tmp_path / "fusion.calc").write_text(
        "local tmp = cam * 2 + 1\n"
        "out = tmp + shift(cam, 1, 0)\n"
        "p, q = cam + 10, cam * 3\n"
    )
    res = milk_bash(
        'mem.mk2Dim "@S:cam" 8 8 >/dev/null; @calc "cam = ix(cam) + 4"\n'
        '@calc "s.cam.kw.GAIN = 2.5"\n'
        '@calc -v nx,c0,g "s.cam.xsize; s.cam.cnt0; s.cam.kw.GAIN"\n'
        '@calc "u1, u2 = cam + 1, cam * 2"\n'
        '@calc -v uerr "imax(abs(u2 - 2 * (u1 - 1)))"\n'
        "milk-fpsexec-milk-calc -n ftest -t cam -f fusion.calc fpsinit >/dev/null\n"
        "milk-fpsexec-milk-calc -n ftest run >run.log 2>&1 & pid=$!\n"
        "sleep 0.5\n"
        '@calc "cam = ix(cam) + 5"; sleep 0.2\n'
        "kill -INT $pid 2>/dev/null; wait $pid 2>/dev/null\n"
        '@calc -v o0,p0,q0 "out[2,2]; p[2,2]; q[2,2]"\n'
        # Verify local tmp was NOT exported to /dev/shm
        '@get s.tmp.xsize >/dev/null 2>&1; echo "tmpsh=$?"\n'
        'echo "shm=$nx,$g"\n'
        'echo "uerr=$uerr"\n'
        'echo "unit=$o0,$p0,$q0"',
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert res.returncode == 0, out
    assert value(res.stdout, "shm") == "8,2.5", out
    assert float(value(res.stdout, "uerr")) == 0.0, out
    # At [2,2], ix = -2 -> cam = 3; shift(cam,1,0)[2,2] = cam[3,2] = 4
    # tmp = 3*2 + 1 = 7; out = 7 + 4 = 11; p = 13; q = 9
    assert value(res.stdout, "unit") == "11,13,9", out
    assert value(res.stdout, "tmpsh") != "0", out
