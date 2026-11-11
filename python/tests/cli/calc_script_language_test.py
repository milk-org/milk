"""The calculator language: reductions, operators, control flow, functions, names and errors.

Purpose:     scripts and units rely on these rules: what a name refers to, what a condition is,
             and how numbers, strings, images and errors behave, in @calc and in units.
Does:        checks multiple, nested and istats reductions; bitwise operators; brace if/elif/else
             blocks and for loops; shift and in-place aliasing; def and lut; entity properties,
             multiple assignment and local images in a unit; negative indices; imag and arg of a
             real image; strings and NaN comparisons; identifier limits, dashed stream names,
             name lookup and `end`; reserved constants and per-shell digits; that a text is
             checked before it runs; order statistics and argmax; number and image conditions;
             exit statuses and error positions; zero divisors; failed image assignments; that
             an argument beyond those a function reads is an error.
Covers:      the calculator (src/calc: front/, run/mc_scalar_eval.c, run/mc_scalar_ops.c,
             mc_eval.c, mc_func.c, and plan/ and jit/ for image statements); @calc, @test,
             @assert and @waitfor (src/cli/libmilkscript: client/, server/milk_server_eval.c);
             milk-fpsexec-milk-calc.
Uses:        conftest.py: milk_bash, _run_bash, milk_cli_script, value, CAM.
See also:    docs/guide/calculator.md, "Calculator (`@calc`, `@test`, `@pipe`)", "Operators" and
             "Functions"; docs/guide/scripting.md, "Errors and Exit Status".
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
    """Multi-reductions in one expression, nested reductions, and istats() unpacking."""
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
    """Bitwise &, |, ^^, ~, <<, >> on scalars and images, including >2^24 precision."""
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
    """C-style brace blocks for if/elif/else and bounded for loops."""
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
    """2D neighbor stencil shift(im, dx, dy) with edge clamping and self-aliasing protection."""
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
    """Inline def f(x, ...) = expr user functions and 1D lut() interpolation."""
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
    """SHM property reads/writes, multi-image unpacking, and local scratch fusion."""
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


def test_surplus_arguments_are_errors(tmp_path):
    """A function takes the arguments `@calc --list` shows: the BLAS calls read no transpose or
    triangle flag (a transposed operand is `A.T`) and `zeros` no type, and ignored them.
    """
    res = milk_bash(
        'mem.mk2Dim A 2 2 >/dev/null; mem.mk2Dim C 2 2 >/dev/null; @calc "A = 2; C = 0"\n'
        'mem.mk2Dim x 2 1 >/dev/null; mem.mk2Dim y 2 1 >/dev/null; @calc "x = 1; y = 0"\n'
        '@calc \'gemm(1, A, A, 0, C, "T", "N")\' 2>e1; echo "gemm=$?"\n'
        '@calc \'gemv(1, A, x, 0, y, "T")\' 2>e2; echo "gemv=$?"\n'
        '@calc \'syrk(1, A, 0, C, "T")\' 2>e3; echo "syrk=$?"\n'
        '@calc \'symv(1, A, x, 0, y, "L")\' 2>e4; echo "symv=$?"\n'
        '@calc "z = zeros(4, 3, 2, 9)" 2>e5; echo "zeros=$?"\n'
        '@calc -v t "gemm(1, A.T, A, 0, C); itot(C)"; echo "ok=$? $t"\n'
        '@calc --list gemm | grep -qF "gemm(a, A, B, b, C)" && echo "list=ok"',
        tmp_path,
    )
    out = res.stdout + res.stderr
    for k, f in (
        ("gemm", "e1"),
        ("gemv", "e2"),
        ("syrk", "e3"),
        ("symv", "e4"),
        ("zeros", "e5"),
    ):
        assert value(res.stdout, k) == "2", out
        assert "wrong number of arguments" in (tmp_path / f).read_text(), out
    assert value(res.stdout, "ok") == "0 32", out
    assert value(res.stdout, "list") == "ok", out


def test_stream_prefix_letters_are_not_variable_names(tmp_path):
    """`S` and `L` name images only: as a number, `im[S:n]` would read `S:n` as a stream."""
    res = milk_bash(
        'mem.mk2Dim im 4 4 >/dev/null; @calc "im = 2"\n'
        + "".join(
            f'@calc "{stmt}" 2>>e.txt; echo "{key}=$?"\n'
            for key, stmt in (
                ("var", "S = 3"),
                ("multi", "x, L = 1, 2"),
                ("loop", "for L = 1:3 { y = L }"),
                ("def", "def f(S) = S + 1"),
            )
        )
        + 'milk-fpsexec-milk-calc -n up -p "S:int=0" -e "S = 1" fpsinit 2>>e.txt; echo "par=$?"\n'
        '@calc "L = im + 1; S = L * 2; S:shr = im * 2; L:loc = im + 1"; echo "img=$?"\n'
        '@calc -v t "itot(S) + itot(L) + itot(S:shr) + itot(L:loc)"; echo "tot=$t"',
        tmp_path,
    )
    out = res.stdout + res.stderr + (tmp_path / "e.txt").read_text()
    for key in ("var", "multi", "loop", "def", "par"):
        assert value(res.stdout, key) != "0", key + "\n" + out
    assert (tmp_path / "e.txt").read_text().count("is a stream prefix") == 4, out
    # 16 pixels each of 6, 3, 4 and 3
    assert value(res.stdout, "img") == "0" and value(res.stdout, "tot") == "256", out


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


def test_audit_regressions_and_edge_cases(tmp_path):
    """Covers edge cases across lexer, resolver, planner, views, reductions, linalg, and stacking:
    - malformed radix/float literals (`0x`, `0b2`, `0o8`, `1.2.3`)
    - `def` with >4 arguments (up to 8)
    - `linspace` + `zeros` shape propagation, `end` on newly created images
    - self-aliased overlapping view assignment (`v[0:6] = v[1:7]`, `w[::-1] = w`)
    - non-contiguous views in `norm`, `asum`, `icount`, and masked reductions with strided masks
    - integer destination streams (`@STU16:`) for `cumsum`, `rprof`, and axis reductions
    - `iargmax` / `iargmin` 2D unpacking on all-NaN images, `INT64_MIN % -1` and shift bounds
    - `hstack` with >8 arguments and linalg validation (`solve` negative reg, `trsv` bad flag)
    """
    res = milk_bash(
        # 1. Malformed numeric literals
        '@calc "0x" 2>/dev/null; echo "hex_empty=$?"\n'
        '@calc "0b2" 2>/dev/null; echo "bin_bad=$?"\n'
        '@calc "0o8" 2>/dev/null; echo "oct_bad=$?"\n'
        '@calc "1.2.3" 2>/dev/null; echo "flt_bad=$?"\n'
        # 2. Multi-arg def (>4 args) and where()
        '@calc "def poly6(a, b, c, d, e, f) = a + 2*b + 3*c + 4*d + 5*e + 6*f"\n'
        'echo "poly6=$(@calc "poly6(1, 1, 1, 1, 1, 1)")"\n'
        'echo "where_sc=$(@calc "where(1, 42, 99)")"\n'
        # 3. Shape propagation for created images + `end` in same multi-stmt @calc + linspace
        '@calc "L = linspace(0, 7, 8) + zeros(8); S = L[1:end-1]"\n'
        'echo "lin=$(@calc "itot(S)")"\n'
        # 4. Self-aliased overlapping slice assignments (1D shift and reversal)
        '@calc "v = [i for i = 0:7]; v[0:6] = v[1:7]"; @calc -a va v\n'
        'echo "shift1d=${va[*]}"\n'
        '@calc "w = [i for i = 0:3]; w[::-1] = w"; @calc -a wa w\n'
        'echo "rev1d=${wa[*]}"\n'
        # 5. Non-contiguous views in norm, asum, icount, and strided mask in imean/istats
        '@calc "M = reshape([i - 8 for i = 0:15], 4, 4); Mc = M[::2, ::2] + 0"\n'
        'echo "dnorm=$(@calc "abs(norm(M[::2, ::2]) - norm(Mc))")"\n'
        'echo "dasum=$(@calc "abs(asum(M[::2, ::2]) - asum(Mc))")"\n'
        'echo "dcnt=$(@calc "icount(M[::2, ::2]) - icount(Mc)")"\n'
        '@calc "msk = where(M > -5, 1, 0); msk_c = msk[::2, ::2] + 0"\n'
        'echo "dmsk=$(@calc "abs(imean(Mc, msk[::2, ::2]) - imean(Mc, msk_c))")"\n'
        # 6. Integer destination streams for cumsum and axis reductions
        'mem.mk2Dim "@STU16:u16_cs" 4 4 >/dev/null\n'
        'mem.mk2Dim "@STU16:u16_ax" 4 1 >/dev/null\n'
        '@calc "src4 = zeros(4, 4) + 3; u16_cs = cumsum(src4, 0); u16_ax = itot(src4, 1)"\n'
        'echo "u16_cs=$(@calc "itot(u16_cs)") u16_ax=$(@calc "itot(u16_ax)")"\n'
        # 7. All-NaN iargmax/iargmin 2D unpacking and INT64_MIN % -1 + shift safety
        '@calc "nan_im = zeros(4, 4) + nan; rx, ry = iargmax(nan_im); nx, ny = iargmin(nan_im)"\n'
        'echo "nan_arg=$(@calc rx),$(@calc ry),$(@calc nx),$(@calc ny)"\n'
        'echo "i64_safe=$(@calc "(-9223372036854775807 - 1) % -1")"\n'
        'echo "sh_safe=$(@calc "(1 << 64) + (8 >> -1)")"\n'
        # 8. hstack with 10 arguments (>8) and linalg validation errors
        '@calc "o1 = zeros(2, 1) + 1; H10 = hstack(o1, o1, o1, o1, o1, o1, o1, o1, o1, o1)"\n'
        'echo "h10=$(@calc "itot(H10)")"\n'
        '@calc "A2 = eye(3); b2 = zeros(3, 1) + 1; x2 = solve(A2, b2, -0.1)" 2>/dev/null; echo "solve_neg=$?"\n'
        '@calc "A2 = eye(3); b2 = zeros(3) + 1; x2 = trsv(A2, b2, \\"BAD\\")" 2>/dev/null; echo "tri_bad=$?"\n',
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert value(res.stdout, "hex_empty") == "2", out
    assert value(res.stdout, "bin_bad") == "2", out
    assert value(res.stdout, "oct_bad") == "2", out
    assert value(res.stdout, "flt_bad") == "2", out
    assert value(res.stdout, "poly6") == "21", out
    assert value(res.stdout, "where_sc") == "42", out
    assert value(res.stdout, "lin") == "21", out
    assert value(res.stdout, "shift1d") == "1 2 3 4 5 6 7 7", out
    assert value(res.stdout, "rev1d") == "3 2 1 0", out
    assert float(value(res.stdout, "dnorm")) < 1e-6, out
    assert float(value(res.stdout, "dasum")) < 1e-6, out
    assert value(res.stdout, "dcnt") == "0", out
    assert float(value(res.stdout, "dmsk")) < 1e-6, out
    assert value(res.stdout, "u16_cs") == "120 u16_ax=48", out
    assert value(res.stdout, "nan_arg") == "-1,-1,-1,-1", out
    assert value(res.stdout, "i64_safe") == "0", out
    assert value(res.stdout, "sh_safe") == "0", out
    assert value(res.stdout, "h10") == "20", out
    assert value(res.stdout, "solve_neg") not in (None, "0"), out
    assert value(res.stdout, "tri_bad") not in (None, "0"), out


def test_colon_before_a_call_is_a_separator(tmp_path):
    """`:` before `name(` ends a slice bound, a range bound or a ?: branch; only a C extension
    takes a `:label`."""
    res = milk_bash(
        'mem.mk2Dim im 8 8 >/dev/null; @calc "im = ix(im) + 4; k = 2; n = 5"\n'
        '@calc -v a "itot(im[k:min(n, 6), 0])"; echo "slice=$a"\n'
        '@calc -v b "k > 1 ? k:sqrt(16)"; echo "tern=$b"\n'
        '@calc "acc = 0; for i = k:floor(5.5) { acc = acc + i }"; @calc -v c "acc"; echo "range=$c"',
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert value(res.stdout, "slice") == "14", out  # pixels x = 2..5 hold 2..5
    assert value(res.stdout, "tern") == "2", out
    assert value(res.stdout, "range") == "14", out


def test_whole_image_steps_store_integers_as_passes_do(tmp_path):
    """Into an integer stream a value is truncated toward zero and wraps, whatever computes it."""
    res = milk_bash(
        "mem.mk2Dim src 4 4 >/dev/null; mem.mk2Dim neg 4 4 >/dev/null\n"
        '@calc "src = 300.7; neg = -1.5"\n'
        'for n in p1 c1 p2 c2; do mem.mk2Dim "@STU8:$n" 4 4 >/dev/null; done\n'
        '@calc "p1 = src + 0; c1 = fftshift(src); p2 = neg + 0; c2 = fftshift(neg)"\n'
        '@calc -v a,b,c,d "p1[0,0]; c1[0,0]; p2[0,0]; c2[0,0]"; echo "v=$a,$b,$c,$d"\n'
        'echo "dt=$(@get s.c1.dtype)"',
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert value(res.stdout, "dt") == "UINT8", out
    assert value(res.stdout, "v") == "44,44,255,255", out
