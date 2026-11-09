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
from tests.cli.conftest import (
    milk_bash,
    value,
    CAM,
    _run_bash,
    milk_cli_script,
)

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
        '@calc "tbl = 10 * pow(2, ix(tbl) + 2)"\n'
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


def test_negative_index_assigns_the_last_pixel():
    res = _run_bash("""
        mem.mk2Dim "@S:pim" 8 8
        @calc "pim = 4.0"
        @calc "pim[-1, -1] = 10.0"
        [[ "$(@calc "pim[7, 7]")" == "10" ]]
        echo "NEG_OK"
    """)
    assert res.returncode == 0, res.stderr
    assert "NEG_OK" in res.stdout


def test_imag_and_arg_of_a_real_image():
    res = _run_bash("""
        mem.mk2Dim "@S:rim" 4 4
        @calc "rim = -2.0; im_i = imag(rim); im_a = arg(rim)"
        [[ "$(@calc "imean(im_i)")" == "0" ]]
        [[ "$(@calc "abs(imean(im_a) - 3.141592653589793) < 1e-5")" == "1" ]]
        echo "IMAG_OK"
    """)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "IMAG_OK" in res.stdout


def test_strings_are_values_not_names_or_math(tmp_path):
    res = milk_bash(
        'myvar=SECRETVALUE; echo "lit=$(@calc \'"myvar"\')"\n'
        '@calc "abc = 42"; echo "len=$(@calc \'strlen("abc")\')"\n'
        '@calc \'s = "1+2"\'; echo "s=$(@calc s)"',
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert value(res.stdout, "lit") == "myvar", out
    assert value(res.stdout, "len") == "3", out
    assert value(res.stdout, "s") == "1+2", out


def test_nan_comparisons_follow_ieee(tmp_path):
    """Shell-side scalars and server-side expressions (they read an image) agree."""
    res = milk_bash(
        CAM + 'echo "eq=$(@calc "nan == nan")"; echo "ne=$(@calc "nan != nan")"\n'
        'echo "ge=$(@calc "nan >= 1")"\n'
        'echo "srv_eq=$(@calc "(imean(cam) + nan) == (imean(cam) + nan)")"\n'
        'echo "srv_ge=$(@calc "imean(cam) + nan >= 1")"',
        tmp_path,
    )
    out = res.stdout + res.stderr
    for key, want in (
        ("eq", "0"),
        ("ne", "1"),
        ("ge", "0"),
        ("srv_eq", "0"),
        ("srv_ge", "0"),
    ):
        assert value(res.stdout, key) == want, f"{key}: {out}"


def test_long_identifier_is_an_error_not_a_crash(tmp_path):
    res = milk_bash(
        CAM + "L=$(printf 'x%.0s' {1..600})\n"
        '@calc "cam + $L" 2>err.txt; echo "rc=$?"\n'
        '@calc -v cm "imean(cam * 3)"; echo "cm=$cm"',
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert value(res.stdout, "rc") not in (None, "0"), out
    assert "identifier too long" in (tmp_path / "err.txt").read_text(), out
    assert value(res.stdout, "cm") == "6", out


def test_calc_text_is_checked_then_run_in_order(tmp_path):
    res = milk_bash(
        CAM + '@calc "a1 = 1; a2 = = 2" 2>/dev/null; echo "syn=$?"\n'
        'echo "a1=[$(@calc a1 2>/dev/null)]"\n'
        '@calc "s.cam.kw.K = 0; for k = 1:3 { s.cam.kw.K += 1 }"; echo "K=$(@get s.cam.kw.K)"\n'
        "@calc 'k = 0 # first\nif imin(cam) > 5 {\n  k = 1\n}\nelif imin(cam) > 1 { k = 2 }\n"
        'else { k = 3 }\'; echo "k=$(@calc k)"',
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert value(res.stdout, "syn") == "2", out
    assert value(res.stdout, "a1") == "[]", out
    assert value(res.stdout, "K") == "3", out  # read live at each iteration
    assert value(res.stdout, "k") == "2", out


def test_dashed_stream_names(tmp_path):
    res = milk_bash(
        CAM + 'mem.mk2Dim "@S:my-cam" 6 4 >/dev/null\n'
        "echo \"local=$(@calc 's.my-cam.xsize - 1') sub=$(@calc 's.cam.xsize-1')\"\n"
        '@calc "o = cam * 0 + s.my-cam.xsize"; echo "srv=$(@calc \'imean(o)\')"',
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert value(res.stdout, "local") == "5 sub=7", out
    assert value(res.stdout, "srv") == "6", out


def test_def_calls_are_inlined(tmp_path):
    res = milk_bash(
        CAM + '@calc "cam = ix(cam)"\n'
        '@calc "def avg(x) = imean(x); def nrm(x) = x / imax(x); def two(x) = x + x"\n'
        "echo \"avg=$(@calc 'avg(cam + 1)')\"\n"
        '@calc "o = nrm(cam)"; echo "nrm=$(@calc \'imax(o)\')"\n'
        '@calc "r = two(rand(cam))" 2>e1.txt; echo "rand=$?"\n'
        '@calc "def sqrt(x) = x" 2>e2.txt; echo "builtin=$?"\n'
        "@calc 'if 1 { def zz(x) = x }' 2>e3.txt; echo \"nested=$?\"",
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert value(res.stdout, "avg") == "0.5", out  # ix(cam) + 1 on 8 columns: -3 .. 4
    assert value(res.stdout, "nrm") == "1", out
    assert value(res.stdout, "rand") == "2", out
    assert "draws random numbers" in (tmp_path / "e1.txt").read_text(), out
    assert value(res.stdout, "builtin") == "2", out
    assert "built-in function" in (tmp_path / "e2.txt").read_text(), out
    assert value(res.stdout, "nested") == "2", out


def test_reserved_constants_and_per_shell_digits(tmp_path):
    res = milk_bash(
        CAM + '@calc "pi = 3" 2>e.txt; echo "assign=$?"\n'
        "pi=3; echo \"bash=$(@calc 'pi * 2; y = 1')\"\n"
        "mem.mk2Dim pi 4 4 >/dev/null; echo \"image=$(@calc 'pi * 2 + imean(cam) * 0')\"\n"
        "@config digits 6 >/dev/null; echo \"d6=$(@calc 'imean(cam) / 3')\"\n"
        "( unset MILK_DPDIGITS; echo \"d15=$(@calc 'imean(cam) / 3')\" )",
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert value(res.stdout, "assign") == "2", out
    assert "is a constant" in (tmp_path / "e.txt").read_text(), out
    assert value(res.stdout, "bash") == "6.28318530717959", out
    assert value(res.stdout, "image") == "6.28318530717959", out
    assert value(res.stdout, "d6") == "0.666667", out
    assert value(res.stdout, "d15") == "0.666666666666667", out


def test_stored_variables_keep_every_digit(tmp_path):
    """A variable keeps the exact value assigned, whatever the printed digits."""
    res = milk_bash(
        '@calc "x = 1/3"; @test "x == 1/3"; echo "exact=$?"\n'
        '@config digits 4 >/dev/null; @calc "y = 1/3"; @calc -v y3 "y * 3"; echo "y3=$y3"\n'
        '@test "y == 1/3"; echo "exact4=$?"',
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert value(res.stdout, "exact") == "0", out
    assert value(res.stdout, "y3") == "1", out
    assert value(res.stdout, "exact4") == "0", out


def test_order_statistics_and_argmax_rules(tmp_path):
    res = milk_bash(
        'mem.mk2Dim v 8 1 >/dev/null; @calc "v = [5, 1, 8, 3, 7, 2, 6, 4]"\n'
        'mem.mk2Dim m 8 1 >/dev/null; @calc "m = [1, 0, 1, 0, 1, 0, 0, 0]"\n'
        '@calc -v a,b,c,d "imedian(v); perc(v, 0.25); imedian(v, m); perc(i16(v), 1)"\n'
        'printf "med=%s\\np25=%s\\nmmed=%s\\npmax=%s\\n" "$a" "$b" "$c" "$d"\n'
        'mem.mk2Dim w 6 4 >/dev/null; @calc "w = -inf"; @calc "x0, y0 = iargmax(w)"\n'
        '@calc "w = 3; w[4, 2] = 9; w[1, 3] = 9; w[0, 0] = nan; x1, y1 = iargmax(w)"\n'
        '@calc "w = -w; x2, y2 = iargmin(w)"\n'
        '@calc -v a,b,c,d,e,f "x0; y0; x1; y1; x2; y2"; echo "arg=$a,$b,$c,$d,$e,$f"',
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert value(res.stdout, "med") == "5", out  # upper middle of 8 values
    assert value(res.stdout, "p25") == "3", out
    assert value(res.stdout, "mmed") == "7", out  # 5, 8, 7 under the mask
    assert value(res.stdout, "pmax") == "8", out
    assert (
        value(res.stdout, "arg") == "0,0,4,2,4,2"
    ), out  # all -inf; first of two extremes; NaN skipped


def test_conditions_are_numbers(tmp_path):
    res = milk_bash(
        CAM + '@test \'"abc"\' 2>e1.txt; echo "str=$?"; @test nan; echo "nan=$?"\n'
        '@calc "t = 0"; @calc "if nan: t = 1"; @calc \'if "abc": t = 2\' 2>/dev/null; echo "ifrc=$?"\n'
        '@calc -v t "t"; echo "t=$t"; @assert \'"abc"\' 2>/dev/null; echo "assert=$?"',
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert value(res.stdout, "str") == "2", out
    assert (
        "a condition is a number, not a string" in (tmp_path / "e1.txt").read_text()
    ), out
    assert value(res.stdout, "nan") == "1", out
    assert value(res.stdout, "ifrc") != "0", out
    assert value(res.stdout, "t") == "0", out
    assert value(res.stdout, "assert") == "2", out


def test_name_lookup_order_and_slice_end(tmp_path):
    res = milk_bash(
        'mem.mk2Dim qi 4 4 >/dev/null; @calc "qi = 1"; qi=5; @calc -v a "imean(qi)"; echo "img_bash=$a"\n'
        'unset qi; @calc "cv = 3"; mem.mk2Dim cv 4 4 >/dev/null; @calc -v b "imean(cv)"; echo "img_calc=$b"\n'
        '@calc "bv = 2"; bv=9; @calc -v c "bv"; echo "bash_calc=$c"; unset bv\n'
        'mem.mk2Dim sl 8 8 >/dev/null; @calc "end = 3; sd = sl[0:end, :]"; @get -v n l.sd.xsize\n'
        '@calc -v e "end * 2"; echo "end=$n,$e"',
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert value(res.stdout, "img_bash") == "1", out
    assert value(res.stdout, "img_calc") == "0", out
    assert value(res.stdout, "bash_calc") == "9", out
    assert value(res.stdout, "end") == "8,6", out


def test_numbers_and_names(tmp_path):
    long80 = "a" * 80
    res = milk_bash(
        CAM + '@calc "w = f64(cam * 0) + pi; wn = wrap_pi(-w)"\n'
        '@calc -v w1,w2,w3 "wrap_pi(pi); wrap_pi(-pi); wn[0,0]"; echo "wrap=$w1,$w2,$w3"\n'
        '@calc -v big,hex "99999999999999999999; 0x10000000000000000"; echo "big=$big,$hex"\n'
        f'@calc "{long80} = 5" 2>e1.txt; echo "long=$?"; @calc "{long80[:79]} = 5"; echo "ok79=$?"\n'
        '@calc -v tr,rms,sd "trunc(-2.7); irms(cam); istd(cam)"; echo "fn=$tr,$rms,$sd"',
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert value(res.stdout, "wrap") == ",".join(["3.14159265358979"] * 3), out
    assert value(res.stdout, "big") == "1e+20,1.84467440737096e+19", out
    assert value(res.stdout, "long") == "2", out
    assert (
        "identifier too long (max 79 characters)" in (tmp_path / "e1.txt").read_text()
    ), out
    assert value(res.stdout, "ok79") == "0", out
    assert value(res.stdout, "fn") == "-2,2,0", out


def test_error_status_position_and_def_checks(tmp_path):
    res = milk_bash(
        "mem.mk2Dim ea 4 4 >/dev/null; mem.mk2Dim eb 8 8 >/dev/null\n"
        '@calc "c = ea + eb" 2>/dev/null; s1=$?; @calc "c = nosuch + 1" 2>/dev/null; s2=$?\n'
        '@calc "c = sqrt(1, 2)" 2>/dev/null; s3=$?; @test "imean(ea + eb) > 0" 2>/dev/null; s4=$?\n'
        'echo "status=$s1,$s2,$s3,$s4"\n'
        "@calc $'q1 = 1\\nq2 = (2 +' 2>e1.txt; @calc $'q1 = 1\\nq2 = ea + eb' 2>e2.txt\n"
        '@calc "def bad(x) = x + nosuch" 2>e3.txt; echo "def=$?"\n'
        '@calc "def good(x) = x * 2 + imean(ea)"; @calc -v g "good(1)"; echo "good=$g"',
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert value(res.stdout, "status") == "1,2,2,2", out
    assert "(line 2, column" in (tmp_path / "e1.txt").read_text(), out
    assert "(line 2) in 'q2 = ea + eb'" in (tmp_path / "e2.txt").read_text(), out
    assert value(res.stdout, "def") == "2", out
    assert (
        "def 'bad': undefined identifier 'nosuch'" in (tmp_path / "e3.txt").read_text()
    ), out
    assert value(res.stdout, "good") == "2", out


def test_a_variable_holds_what_was_assigned_last(tmp_path):
    res = milk_bash(
        '@calc \'t = 5; t = "xyz"; s = "abc"; s = 5; q = "5"\'\n'
        "echo \"vals=$(@calc t) $(@calc s) $(@calc 'q + 1')\"\n"
        'milk-fpsexec-milk-calc -n us -p "nm:str=abc g:float=1" -e \'nm = "xyz"; g = 3\' exec >/dev/null 2>&1\n'
        'echo "unit=$(@get f.us.nm) $(@get f.us.g)"\n'
        'milk-fpsexec-milk-calc -n ut -p "g:float=1" -e \'g = "xyz"\' exec 2>&1 | '
        'grep -c "g is a number parameter: it cannot take a string" | sed "s/^/refused=/"',
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert value(res.stdout, "vals") == "xyz 5 51", out
    assert value(res.stdout, "unit") == "xyz 3", out
    assert value(res.stdout, "refused") == "1", out


def test_zero_divisors_error_and_post_nothing(tmp_path):
    """`a /= b` with a zero pixel is an error that leaves `a` as it was; `a = b / x` with x == 0
    neither changes `a` nor posts a frame."""
    res = milk_bash(
        'mem.mk2Dim "@S:a" 4 4 >/dev/null; mem.mk2Dim "@S:b" 4 4 >/dev/null\n'
        '@calc "a = 8; b = where(ix(b) == 0, 0, 2)"\n'
        "c0=$(@get s.a.cnt0)\n"
        '@calc "a /= b" 2>err.log; echo "rc=$?"; echo "amean=$(@get s.a.mean)"\n'
        'x=0; @calc "a = b / x" 2>>err.log; echo "rc2=$?"\n'
        'echo "dcnt=$(( $(@get s.a.cnt0) - c0 ))"',
        tmp_path,
    )
    err = (tmp_path / "err.log").read_text()
    assert value(res.stdout, "rc") not in (None, "0"), res.stdout + err
    assert "zero" in err.lower(), err
    assert value(res.stdout, "amean") == "8", res.stdout + err
    assert value(res.stdout, "rc2") not in (None, "0"), res.stdout + err
    assert value(res.stdout, "dcnt") == "0", res.stdout + err


def test_image_conditions_are_errors(tmp_path):
    """An image is neither true nor false: @test, @assert, @waitfor and `if` fail (status 2) with a hint,
    and @waitfor at once rather than at its timeout. imin()/imax() of the image are the conditions.
    """
    res = milk_cli_script(
        'mem.mk2Dim "@S:im" 4 4 >/dev/null; @calc "im = 0"\n'
        '@test "im > 1" 2>t.err; echo "test=$?"\n'
        '@assert "im > 1" 2>/dev/null; echo "assert=$?"\n'
        's=$SECONDS; @waitfor "im > 1" -t 5 2>/dev/null; echo "waitfor=$? waited=$((SECONDS - s))"\n'
        '@calc "if im > 1: x = 1" 2>/dev/null; echo "if=$?"\n'
        '@test "imax(im > 1)"; echo "any=$?"\n'
        '@test "imin(im < 1)"; echo "all=$?"',
        tmp_path,
        timeout=30,
    )
    out = res.stdout + res.stderr
    assert (
        "imin(...) for all pixels, imax(...) for any"
        in (tmp_path / "t.err").read_text()
    ), out
    assert value(res.stdout, "test") == "2", out
    assert value(res.stdout, "assert") not in (None, "0"), out
    assert value(res.stdout, "waitfor") in ("2 waited=0", "2 waited=1"), out
    assert value(res.stdout, "if") == "2", out
    assert value(res.stdout, "any") == "1", out
    assert value(res.stdout, "all") == "0", out


def test_failed_image_assignments_are_errors(tmp_path):
    """`u, v = a, b` assigns an existing image as `v = b` does: a stream of another size is an error that
    leaves it as it was, a number fills it. An image assigned to a variable is an error. Both used to
    print "Cannot change name" and return 0. A block's error is printed once."""
    res = milk_cli_script(
        'mem.mk2Dim "@S:im" 4 4 >/dev/null; @calc "im = 2"\n'
        'mem.mk2Dim "@S:v" 2 2 >/dev/null; @calc "v = 7"\n'
        '@calc "u, v = im + 1, im * 2" 2>e1.txt; echo "mismatch=$?"; echo "v=$(@get s.v.xsize) $(@get s.v.mean)"\n'
        'mem.mk2Dim "@S:w" 4 4 >/dev/null\n'
        '@calc "u, w = im + 1, im + 5"; echo "same=$? $(@get s.w.mean)"\n'
        '@calc "u, w = 3, 4"; echo "numbers=$? $(@get s.w.mean)"\n'
        '@calc "x = 5"; @calc "x = im * 3" 2>e2.txt; echo "var=$?"; @calc -v y x; echo "x=$y"\n'
        '@calc "if 1: y = nosuchname + 1" 2>e3.txt; echo "block=$?"',
        tmp_path,
    )
    out = res.stdout + res.stderr
    for f in ("e1.txt", "e2.txt", "e3.txt"):
        out += f"\n--- {f}\n" + (tmp_path / f).read_text()
    assert value(res.stdout, "mismatch") == "1", out  # a failure, not a usage error
    assert "Dimension mismatch" in (tmp_path / "e1.txt").read_text(), out
    assert value(res.stdout, "v") == "2 7", out
    assert value(res.stdout, "same") == "0 7", out
    assert value(res.stdout, "numbers") == "0 4", out
    assert value(res.stdout, "var") == "2", out
    assert (
        "'x' is a variable (calculator or bash)" in (tmp_path / "e2.txt").read_text()
    ), out
    assert value(res.stdout, "x") == "5", out
    assert "Cannot change name" not in out, out
    assert value(res.stdout, "block") == "2", out
    assert (tmp_path / "e3.txt").read_text().count("undefined identifier") == 1, out
