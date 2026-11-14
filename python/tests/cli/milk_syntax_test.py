"""The milk script language: keywords, module commands and their arguments, @{...}, bindings, @calc.

Purpose:     a script relies on these rules wherever it runs (milk-cli, milk-script, or bash after
             `source milk.bash`): the keyword set, how module commands take their arguments, how
             values are bound to shell variables, and what the calculator computes.
Does:        keywords and namespaces (`milkls`), `@unload`, `@module load --as`, qualified command
             names only; command arguments (`-h` and `@help`, `.name=value`, `--save`, `--load`,
             `--default`, `--no-run`, number forms read alike everywhere, a too-long argument
             refused); modules loaded on first use, and a libmilkbash.so that links neither
             cfitsio nor OpenMP; @{...}; the bindings of `@get` and `@calc` (-v, -a, -A);
             `@defer`, `@set --restore`, `@restore --undo`, `@sleep`, `@waitfor` on an expression,
             `@args` (also in a sourced file); calculator reductions, math, complex, BLAS, LAPACK
             and random functions, also against the module commands that compute the same
             images; the property, modifier and keyword lists against docs/guide/scripting.md;
             a failed `@set` prints its error once; the options of the keywords: an unknown
             one and a missing value are usage errors, `--opt=value`, the help and the
             completion list come from the keyword table.
Covers:      the bash builtins and hooks of libmilkbash.so (libmilkscript/bash/:
             milk_bash_builtins.c, milk_bash_hooks.c, milk_bash_defer.c, milk_bash_translate.c),
             the client (libmilkscript/client/milk_client_ops.c, milk_client_stream.c), the
             keyword table (core/milk_keywords.c), argument rules (core/milk_cmd_args.c,
             core/CLIcore_checkargs.c, src/engine/libfps/params/fps_args.c), module loading
             (core/milk_builtin_module.c, milk_catalog_autoload() in core/milk_catalog_build.c,
             server/milk_server_admin.c), the calculator (src/calc), scripts/milk.bash.
Uses:        conftest.py: milk_bash, milk_cli_script, milk_cli_env, translate, events, value,
             _run_bash, _env, _install_dir, CAM, UNIT_HELPERS.
See also:    docs/guide/scripting.md, "Keywords", "Module Commands", "Arguments", "Entities";
             docs/guide/calculator.md, "Functions".
"""

from __future__ import annotations

import os
import pty
import re
import select
import subprocess
import time
from pathlib import Path

import pytest

from milk.cliwrap import HAVE_CLI
from tests.cli.conftest import (
    _env,
    _install_dir,
    _run,
    events,
    milk_bash,
    milk_cli_env,
    milk_cli_script,
    milk_script,
    translate,
    value,
    CAM,
    UNIT_HELPERS,
    _run_bash,
)

pytestmark = [
    pytest.mark.skipif(not HAVE_CLI, reason="MILK compiled without CLI support"),
    pytest.mark.timeout(60),
]


def test_keywords_namespaces_and_module_registration(tmp_path):
    res = milk_bash(
        "milkls --keywords | awk '{print \"@\" $1}' | sort > kw.txt\n"
        "{ compgen -A function @; compgen -A builtin @; } | sort > fn.txt\n"
        "diff kw.txt fn.txt >/dev/null && echo same=1\n"
        "milkls --namespaces | awk '{print $1}' | tr '\\n' ' '; echo\n"
        "@unload >/dev/null 2>&1\n"
        'echo "after=$(type -t milkcall)"',
        tmp_path,
    )
    assert value(res.stdout, "same") == "1", res.stdout + res.stderr
    assert value(res.stdout, "after") == "", res.stdout
    tokens = res.stdout.split()
    assert "seq" not in tokens and "procinfo" not in tokens

    res2 = milk_cli_script(
        '@module load milkpsf --as s >/dev/null 2>&1; echo "rc=$?"\n'
        '@module load milkpsf --as clu >/dev/null; echo "ok=$?"\n'
        'milk.ci >/dev/null; echo "ci=$?"\n'
        '@ci >/dev/null 2>&1; echo "old=$?"',
        tmp_path,
    )
    assert value(res2.stdout, "rc") not in (None, "0")
    assert value(res2.stdout, "ok") == "0"
    assert value(res2.stdout, "ci") == "0"
    assert value(res2.stdout, "old") == "127"


def test_help_and_per_call_arguments(tmp_path):
    res = milk_cli_script(
        'iofits.loadfits -h > a.txt; echo "rc=$?"\n'
        "@help iofits.loadfits > b.txt\n"
        'mem.mk2Dim b >/dev/null 2>&1; echo "miss=$?"\n'
        'mem.mk2Dim c abc 4 >/dev/null 2>&1; echo "bad=$?"\n'
        'mem.mk2Dim .ysize=5 d .xsize=3 >/dev/null; echo "named=$(@get s.d.xsize),$(@get s.d.ysize)"\n'
        'mem.mk2Dim e 4 6 --save=p1 >/dev/null && mem.mk2Dim f --load=p1 >/dev/null; echo "preset=$(@get s.f.ysize)"\n'
        'mem.mk2Dim --default .xsize=3 .ysize=7 && mem.mk2Dim g >/dev/null; echo "def=$(@get s.g.ysize)"\n'
        'mem.mk2Dim h 4 4 --no-run >/dev/null; @exists s.h || echo "norun=1"',
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert value(res.stdout, "rc") == "0", out
    assert (tmp_path / "a.txt").read_text() == (tmp_path / "b.txt").read_text()
    assert value(res.stdout, "miss") == "2" and value(res.stdout, "bad") == "2", out
    assert value(res.stdout, "named") == "3,5", out
    assert value(res.stdout, "preset") == "6" and value(res.stdout, "def") == "7", out
    assert value(res.stdout, "norun") == "1", out


def test_autoload_and_thin_client(tmp_path):
    """A session starts with the core modules: an installed module loads at the first use of
    one of its commands, and search, completion and help know it before that.
    """
    env, _ = _env()
    env.pop("MILKCLI_ADD_LIBS", None)
    res = milk_cli_env(
        'echo "m0=$(@module list | grep -c image_gen)"\n'
        'echo "k=$(@help -k gaussian | grep -c imgen.mkgauss)"\n'
        'echo "c=$(milkcomplete imgen.mkg cmd | grep -c imgen.mkgauss)"\n'
        'echo "m1=$(@module list | grep -c image_gen)"\n'
        'imgen.mkgauss gau 16 16 3 1 >/dev/null; echo "rc=$?"\n@exists s.gau && echo made=1\n'
        'echo "m2=$(@module list | grep -c image_gen)"\n'
        'echo "h=$(@help psf | grep -c psfseqmeas)"\n'
        'echo "hc=$(@help imgfilt.gaussfilt | grep -c gaussfilt)"',
        tmp_path,
        env,
    )
    out = res.stdout + res.stderr
    assert value(res.stdout, "rc") == "0" and value(res.stdout, "made") == "1", out
    got = {k: value(res.stdout, k) for k in ("m0", "k", "c", "m1", "m2")}
    assert got == {"m0": "0", "k": "1", "c": "1", "m1": "0", "m2": "1"}, out
    assert int(value(res.stdout, "h")) >= 1 and int(value(res.stdout, "hc")) >= 1, out

    ldd_out = subprocess.run(
        ["ldd", str(_install_dir() / "lib" / "libmilkbash.so")],
        capture_output=True,
        text=True,
    ).stdout
    assert "cfitsio" not in ldd_out and "gomp" not in ldd_out, ldd_out


def test_inline_expansion_and_calc_parity(tmp_path):
    assert 'milkget -v x "s.cam.xsize"' in translate("x=@{s.cam.xsize}\n")
    res = milk_cli_script(
        "mem.mk2Dim ia 6 4 >/dev/null\n"
        'x=@{s.ia.xsize}; echo "a=$x rc=$?"\n'
        'x=old; x=@{s.nosuch.xsize}; echo "b=[$x] rc=$?"\n'
        'i=1; i=@{i + 1}; echo "d=$i"\n'
        '@config digits 4 >/dev/null; @calc -v v "1/3"; echo "v=$v"\n'
        '@calc "one = 1"; @calc -v c1 "pow(2, 10) + sqrt(16)"; @calc -v c2 "(pow(2, 10) + sqrt(16)) * one"\n'
        'echo "par=$c1,$c2"',
        tmp_path,
    )
    assert value(res.stdout, "a") == "6 rc=0"
    assert value(res.stdout, "b") == "[] rc=1"
    assert value(res.stdout, "d") == "2"
    assert value(res.stdout, "v") == "0.3333"
    assert value(res.stdout, "par") == "1028,1028"


def test_S1_multi_binding_stats_and_arrays(tmp_path):
    res = milk_cli_script(
        "mem.mk2Dim sa 5 3 >/dev/null\n"
        '@get nx=s.sa.xsize ny=s.sa.ysize; echo "p=$nx,$ny"\n'
        'mem.mk2Dim st 4 4 >/dev/null\n@calc "st = 2"\n'
        "@get -A g s.st.stats\n"
        "@calc -v a,b 'itot(st); imax(st)'\n"
        "@calc -a px 'st * 3'\n"
        'echo "res=${g[tot]} $a $b ${#px[@]} ${px[0]}"',
        tmp_path,
    )
    assert value(res.stdout, "p") == "5,3"
    assert value(res.stdout, "res") == "32 32 2 16 6"


def test_S2_scoped_defer_and_restore(tmp_path):
    res = milk_cli_script(
        "mem.mk2Dim rs 4 4 >/dev/null\n@set s.rs.kw.GAIN=1 s.rs.kw.MODE=closed\n"
        "@defer echo D:top\n"
        "f() { @defer echo D:f1; @set --restore s.rs.kw.GAIN=5; @defer echo D:f2; "
        'echo "D:in $(@get s.rs.kw.GAIN)"; }\n'
        'f; echo "D:out $(@get s.rs.kw.GAIN)"',
        tmp_path,
    )
    assert events(res.stdout) == ["in 5", "f2", "f1", "out 1", "top"]


def test_S3_coordinates_where_and_masked_reductions(tmp_path):
    res = milk_cli_script(
        "mem.mk2Dim b 4 3 >/dev/null\n@calc 'b = ix(b)'\n"
        "@calc -a px 'ix(b)'; echo \"ix=${px[*]}\"\n"
        "@calc -a w 'where(b > 0, sqrt(b), -1)'; echo \"where=${w[*]}\"\n"
        "@calc -v m1,m2 'imean(b, b > -1); itot(b, b > 0)'; echo \"m=$m1 $m2\"",
        tmp_path,
    )
    assert value(res.stdout, "ix") == "-2 -1 0 1 -2 -1 0 1 -2 -1 0 1"
    assert value(res.stdout, "where") == " ".join(["-1 -1 -1 1"] * 3)
    assert value(res.stdout, "m") == "0.5 3"


def test_S4_S5_sleep_and_waitfor_expressions(tmp_path):
    res = milk_bash(
        "unset t; for ((i=0;i<5;i++)); do @sleep --hz 500 t; done\n"
        'd0=$t; @sleep --every 0.01 t; echo "step=$(( t - d0 ))"\n'
        'mem.mk2Dim s.ws 8 8 >/dev/null; @calc "ws = 0"\n'
        '( sleep 0.08; @calc "ws = 5" ) &\n'
        '@waitfor "imean(ws) > 3.5" -t 3; echo "w=$?"; wait',
        tmp_path,
    )
    assert value(res.stdout, "step") == "10000000"
    assert value(res.stdout, "w") == "0"


def test_S6_S7_args_user_commands_and_undo(tmp_path):
    res = milk_bash(
        "m() { @args 'cam:str:camera' 'n:int=100' 'g:float=0.5' -- \"$@\"; echo \"D:$cam $n $g\"; }\n"
        "m s.cam .n=50\n"
        "mem.mk2Dim s.ua 4 4 >/dev/null\n"
        "@set s.ua.kw.G=1\n@set s.ua.kw.G=2\n@set s.ua.kw.G=3\n"
        '@restore --undo; echo "D:u1=$(@get s.ua.kw.G)"\n'
        '@restore --undo; echo "D:u2=$(@get s.ua.kw.G)"',
        tmp_path,
    )
    assert events(res.stdout) == ["s.cam 50 0.5", "u1=2", "u2=1"]


def test_calc_fused_reductions_math_and_complex_kernels(tmp_path):
    res = milk_cli_script(
        "mem.mk2Dim a 4 4 >/dev/null; mem.mk2Dim b 4 4 >/dev/null\n"
        "@calc 'a = ix(a); b = iy(a)'\n"
        "@calc -v tot,mn,sd,vr 'itot(a - b); imean(a + b); irms(a - b); ivar(a - b)'\n"
        "@calc -v mn0,mx0,cnt,dt 'imin(a + b); imax(a + b); icount((a >= 0) && (b >= 0)); dot(a + 1, b + 1)'\n"
        "@calc -v mtot,mstd 'itot(a, (a > -2) && (b == 0)); istd(a, b == 0)'\n"
        "@calc 'c = where((a >= 0) || (b >= 0), posi(a) + sign(b), -10)'\n"
        "@calc -v ctot 'itot(c)'\n"
        "@calc 'h = hypot(a, b); r = ir(a); th = itheta(a); at = atan2(b, a)'\n"
        "@calc -v hdiff,tdiff 'imax(abs(h - r)); imax(abs(th - at))'\n"
        "@calc 'm = tanh(sinh(a * 0.1)) + cosh(b * 0.1) + exp2(log2(abs(a) + 1)) + erf(a * 0.5) + erfc(b * 0.5)'\n"
        "@calc -v mfin 'isfinite(itot(m))'\n"
        "@calc 'z1 = expi(a * 0.25); z2 = cplx(a, b); z3 = (z1 * conj(z1)) + (z2 / z1) * z1 - z2'\n"
        "@calc -v zre,zim,zarg 'imean(real(z3)); imax(abs(imag(z3))); imax(abs(arg(expi(a * 0.2)) - a * 0.2))'\n"
        'echo "red=$tot $mn $mn0 $mx0 $cnt $dt"\n'
        'echo "mred=$mtot $ctot"\n'
        'echo "coord=$hdiff $tdiff $mfin"\n'
        'echo "cplx=$zre $zim $zarg"\n'
        'echo "sd=$sd"\n'
        'echo "vr=$vr"\n'
        'echo "mstd=$mstd"',
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert value(res.stdout, "red") == "0 -1 -4 2 4 4", out
    assert value(res.stdout, "mred") == "0 -36", out
    assert value(res.stdout, "coord") == "0 0 1", out
    zre, zim, zarg = [float(x) for x in value(res.stdout, "cplx").split()]
    assert zre == pytest.approx(1.0, abs=1e-5)
    assert zim == pytest.approx(0.0, abs=1e-5)
    assert zarg == pytest.approx(0.0, abs=1e-5)
    assert float(value(res.stdout, "sd")) == pytest.approx(
        (40.0 / 16.0) ** 0.5, rel=1e-5
    )
    assert float(value(res.stdout, "vr")) == pytest.approx(40.0 / 16.0, rel=1e-5)
    assert float(value(res.stdout, "mstd")) == pytest.approx(1.25**0.5, rel=1e-5)


def test_calc_blas_levels_1_2_3_and_algebraic_fusion(tmp_path):
    res = milk_cli_script(
        "@calc 'v = vec(1, -4, 2, 3); x = vec(1, 2, 3); y = vec(4, 5, 6); xs = vec(1, 2, 3); vy = vec(4, 5, 6)'\n"
        "@calc -v l1,l2,idx,fl1 'asum(v); norm(v); iamax(v); asum(v * 2)'\n"
        "@calc 'axpy(2.0, x, vy); scal(3.0, xs)'\n"
        "@calc -v vy_tot,vs_tot 'itot(vy); itot(xs)'\n"
        "@calc 'I3 = eye(3); I23 = eye(2, 3); D = diag(x); D1 = diag(vec(1, 2), 1)'\n"
        "@calc 'dx = diag(D); dx1 = diag(D1, 1)'\n"
        "@calc -v trI,trD,dx_err,d1_err 'trace(I3); trace(D); imax(abs(dx - x)); imax(abs(dx1 - vec(1, 2)))'\n"
        "@calc 'A = reshape(vec(1, 2, 3, 4, 5, 6), 3, 2); u = vec(2, -1)'\n"
        "@calc 'gv = x * 1.0; gemv(2.0, A, u, 0.5, gv)'\n"
        "@calc 'gv_ref = 2.0 * matmul(A, u) + 0.5 * x'\n"
        "@calc -v gv_err 'imax(abs(gv - gv_ref))'\n"
        "@calc 'G = outer(x, u); G2 = A * 1.0; ger(2.0, x, u, G2)'\n"
        "@calc -v ger_err 'imax(abs(G2 - (2.0 * G + A)))'\n"
        "@calc 'S = eye(3); syrk(1.0, A, 0.0, S); ST = eye(2); syrk(1.0, transpose(A), 0.0, ST)'\n"
        "@calc -v syrk_err,syrkt_err 'imax(abs(S - matmul(A, transpose(A)))); imax(abs(ST - matmul(transpose(A), A)))'\n"
        "@calc 'sv = y * 1.0; symv(1.5, S, x, -0.5, sv); sv_ref = 1.5 * matmul(S, x) - 0.5 * y'\n"
        "@calc -v sv_err 'imax(abs(sv - sv_ref))'\n"
        "@calc 'Gm = eye(3); gemm(2.0, A, transpose(A), -0.5, Gm); Gm_ref = 2.0 * S - 0.5 * eye(3)'\n"
        "@calc -v gemm_err 'imax(abs(Gm - Gm_ref))'\n"
        "@calc 'C = reshape(vec(1, 1, 1, 1, 1, 1, 1, 1, 1), 3, 3)'\n"
        "@calc 'C = 2.0 * matmul(A, transpose(A)) - 0.5 * C'\n"
        "@calc 'C_ref = 2.0 * S - 0.5 * reshape(vec(1, 1, 1, 1, 1, 1, 1, 1, 1), 3, 3)'\n"
        "@calc -v fgemm_err 'imax(abs(C - C_ref))'\n"
        "@calc 'dm = vec(10, 20); wfs = vec(1, 2, 3); Cmat = transpose(A)'\n"
        "@calc 'dm_ref = 0.9 * dm - 0.2 * matmul(Cmat, wfs)'\n"
        "@calc 'dm = 0.9 * dm - 0.2 * matmul(Cmat, wfs)'\n"
        "@calc -v fgemv_err 'imax(abs(dm - dm_ref))'\n"
        "@calc 'R = A * 1.0; R += 0.5 * outer(x, u); R_ref = A + 0.5 * G'\n"
        "@calc -v fger_err 'imax(abs(R - R_ref))'\n"
        "@calc 'K = eye(2); K = 2.0 * matmul(transpose(A), A) + 3.0 * K; K_ref = 2.0 * ST + 3.0 * eye(2)'\n"
        "@calc -v fsyrk_err 'imax(abs(K - K_ref))'\n"
        'echo "l1=$l1 $idx $fl1 $vy_tot $vs_tot"\n'
        'echo "l2=$l2"\n'
        'echo "bldr=$trI $trD $dx_err $d1_err"\n'
        'echo "errs=$gv_err $ger_err $syrk_err $syrkt_err $sv_err $gemm_err $fgemm_err $fgemv_err $fger_err $fsyrk_err"',
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert value(res.stdout, "l1") == "10 1 20 27 18", out
    assert float(value(res.stdout, "l2")) == pytest.approx(30.0**0.5, rel=1e-5)
    assert value(res.stdout, "bldr") == "3 6 0 0", out
    errs = [float(x) for x in value(res.stdout, "errs").split()]
    for e in errs:
        assert e == pytest.approx(0.0, abs=1e-5), out


def test_calc_lapack_solvers_and_inverses(tmp_path):
    res = milk_cli_script(
        "@calc 'U = reshape(vec(2, 0, 0, 1, 3, 0, -1, 2, 4), 3, 3); b = vec(1, 5, 8)'\n"
        '@calc \'xu = trsv(U, b, "U"); xut = trsv(transpose(U), b, "U")\'\n'
        "@calc -v trsv_err1,trsv_err2 'imax(abs(matmul(U, xu) - b)); imax(abs(matmul(transpose(U), xut) - b))'\n"
        '@calc \'B2 = reshape(vec(1, 5, 8, 2, -1, 4), 3, 2); Xm = trsm(U, B2, "U", "N", 2.0)\'\n'
        "@calc -v trsm_err 'imax(abs(matmul(U, Xm) - 2.0 * B2))'\n"
        "@calc 'A = reshape(vec(4, 1, 2, 1, 5, 1, 2, 1, 3), 3, 3)'\n"
        "@calc 'xg = gesv(A, b); xp = posv(A, b); xs = solve(A, b); Ai = inv(A)'\n"
        "@calc -v gesv_err,posv_err,solve_err,inv_err 'imax(abs(matmul(A, xg) - b)); imax(abs(matmul(A, xp) - b)); imax(abs(matmul(A, xs) - b)); imax(abs(matmul(A, Ai) - eye(3)))'\n"
        "@calc 'Atall = reshape(vec(1, 2, 3, 4, 1, -1, 2, 0), 4, 2); btall = vec(2, 1, 5, 4)'\n"
        "@calc 'xls = gels(Atall, btall); xne = posv(matmul(transpose(Atall), Atall), matmul(transpose(Atall), btall))'\n"
        "@calc -v gels_tall_err 'imax(abs(xls - xne))'\n"
        "@calc 'Awide = transpose(Atall); bwide = vec(3, 5); xmn = gels(Awide, bwide)'\n"
        "@calc -v gels_wide_err 'imax(abs(matmul(Awide, xmn) - bwide))'\n"
        "@calc 'xridge = solve(Atall, btall, 0.25); xridge_ref = posv(matmul(transpose(Atall), Atall) + 0.25 * eye(2), matmul(transpose(Atall), btall))'\n"
        "@calc -v ridge_tall_err 'imax(abs(xridge - xridge_ref))'\n"
        "@calc 'xridge_w = solve(Awide, bwide, 0.25); xridge_wref = matmul(transpose(Awide), posv(matmul(Awide, transpose(Awide)) + 0.25 * eye(2), bwide))'\n"
        "@calc -v ridge_wide_err 'imax(abs(xridge_w - xridge_wref))'\n"
        "@calc 'Ptall = pinv(Atall); Pwide = pinv(Awide)'\n"
        "@calc -v pinv_t1,pinv_t2 'imax(abs(matmul(Atall, matmul(Ptall, Atall)) - Atall)); imax(abs(matmul(Ptall, Atall) - eye(2)))'\n"
        "@calc -v pinv_w1,pinv_w2 'imax(abs(matmul(Awide, matmul(Pwide, Awide)) - Awide)); imax(abs(matmul(Awide, Pwide) - eye(2)))'\n"
        'echo "tr=$trsv_err1 $trsv_err2 $trsm_err"\n'
        'echo "sq=$gesv_err $posv_err $solve_err $inv_err"\n'
        'echo "ls=$gels_tall_err $gels_wide_err $ridge_tall_err $ridge_wide_err"\n'
        'echo "pinv=$pinv_t1 $pinv_t2 $pinv_w1 $pinv_w2"',
        tmp_path,
    )
    out = res.stdout + res.stderr
    for key in ("tr", "sq", "ls", "pinv"):
        vals = [float(x) for x in value(res.stdout, key).split()]
        for v in vals:
            assert v == pytest.approx(0.0, abs=1e-4), f"{key}: {vals}\n{out}"


def test_calc_simd_random_generators(tmp_path):
    res = milk_cli_script(
        "mem.mk2Dim u 256 256 >/dev/null; mem.mk2Dim g 256 256 >/dev/null\n"
        "@calc -v sr0,sr1,sg0 'rand(42); rand(42); randn(42)'\n"
        "@calc 'u = rand(u, 123); u2 = rand(u, 123); u3 = rand(u)'\n"
        "@calc -v umin,umax,umean,ustd,urep,udiff 'imin(u); imax(u); imean(u); istd(u); imax(abs(u - u2)); imax(abs(u - u3))'\n"
        "@calc 'g = 10.0 + 2.5 * randn(g, 99); g2 = 10.0 + 2.5 * randn(g, 99); g3 = randn(g)'\n"
        "@calc -v gmean,gstd,grep,gdiff 'imean(g); istd(g); imax(abs(g - g2)); imax(abs(g3 - randn(g)))'\n"
        'echo "s=$sr0 $sr1 $sg0"\n'
        'echo "u=$umin $umax $umean $ustd $urep $udiff"\n'
        'echo "g=$gmean $gstd $grep $gdiff"',
        tmp_path,
    )
    out = res.stdout + res.stderr
    sr0, sr1, _ = [float(x) for x in value(res.stdout, "s").split()]
    assert sr0 == pytest.approx(sr1, abs=1e-12), out
    assert 0.0 <= sr0 < 1.0, out
    umin, umax, umean, ustd, urep, udiff = [
        float(x) for x in value(res.stdout, "u").split()
    ]
    assert umin >= 0.0 and umax < 1.0, out
    assert umean == pytest.approx(0.5, abs=0.01), out
    assert ustd == pytest.approx((1.0 / 12.0) ** 0.5, abs=0.01), out
    assert urep == 0.0 and udiff > 0.1, out
    gmean, gstd, grep, gdiff = [float(x) for x in value(res.stdout, "g").split()]
    assert gmean == pytest.approx(10.0, abs=0.05), out
    assert gstd == pytest.approx(2.5, abs=0.05), out
    assert grep == 0.0 and gdiff > 0.5, out


def test_random_generators_seeds_and_normal_tails(tmp_path):
    """Every distinct seed is another image (beyond 32 bits, fractional, and beyond the 24 bits
    a float32 pass holds); rand and randn of one seed are unrelated; rand(seed) and randn(seed)
    are pixel 0 of the double image; randn is normal out to its tails, which a sum of uniforms
    is not.
    """
    res = milk_cli_script(
        "mem.mk2Dim a 256 256 >/dev/null; mem.mk2Dim b 256 256 >/dev/null\n"
        "mem.mk2Dim g 2048 2048 >/dev/null\n"
        'df() { @calc "a = $1; b = $2"; @calc -v r \'imax(abs(a - b))\'; echo "$3=$r"; }\n'
        "df 'rand(a, 1)' 'rand(b, 4294967297)' big\n"
        "df 'rand(a, 0.25)' 'rand(b, 0.5)' frac\n"
        "df 'rand(a, 16777216)' 'rand(b, 16777217)' f24\n"
        "df 'randn(a, 16777216)' 'randn(b, 16777217)' n24\n"
        "df 'rand(a, 7)' 'rand(b, 7)' same\n"
        "@calc 'a = rand(a, 5) - 0.5; b = randn(b, 5)'\n"
        "@calc -v c 'imean(a * b) / (istd(a) * istd(b))'; echo \"corr=$c\"\n"
        "@calc 'd = f64(zeros(16, 16)); d = rand(d, 42)'\n"
        "@calc -v p0,s0 'd[0,0]; rand(42)'; echo \"px=$p0 $s0\"\n"
        "@calc 'd = randn(d, 42)'\n"
        "@calc -v p0,s0 'd[0,0]; randn(42)'; echo \"pn=$p0 $s0\"\n"
        "@calc 'g = randn(g, 3)'\n"
        "@calc -v m,s,k,t,hi 'imean(g); istd(g); imean(pow(g, 4)) / pow(ivar(g), 2); "
        "icount(abs(g) > 4.3); imax(abs(g))'\n"
        'echo "g=$m $s $k $t $hi"',
        tmp_path,
    )
    out = res.stdout + res.stderr
    for k in ("big", "frac", "f24", "n24"):
        assert float(value(res.stdout, k)) > 0.5, f"{k}: {out}"
    assert float(value(res.stdout, "same")) == 0.0, out
    assert abs(float(value(res.stdout, "corr"))) < 0.02, out
    for k, tol in (("px", 0.0), ("pn", 1e-12)):  # log and cos differ in their last bit
        p0, s0 = [float(x) for x in value(res.stdout, k).split()]
        assert abs(p0 - s0) <= tol, f"{k}: {out}"
    m, s, kurt, tail, hi = [float(x) for x in value(res.stdout, "g").split()]
    assert abs(m) < 0.003 and abs(s - 1.0) < 0.003, out  # 4.2 million samples
    assert abs(kurt - 3.0) < 0.02, out  # 2.8 for a sum of six uniforms
    assert 30 <= tail <= 130 and hi > 4.3, out  # about 72 samples beyond 4.3 sigma


def test_calc_and_module_kernel_unification(tmp_path):
    res = milk_cli_script(
        "imgen.mkdist d_mod 16 16 8 8 >/dev/null\n"
        "imgen.mkslopexy sl_mod 16 16 1.5 -0.5 >/dev/null\n"
        "imgen.im2coord d_mod 0 cx_mod >/dev/null\n"
        "imgen.im2coord d_mod 1 cy_mod >/dev/null\n"
        "@calc 'd_calc = ir(d_mod, 8, 8)'\n"
        "@calc 'sl_calc = 1.5 * ix(d_mod, 8, 8) - 0.5 * iy(d_mod, 8, 8)'\n"
        "@calc 'cx_calc = ix(d_mod, 0, 0); cy_calc = iy(d_mod, 0, 0)'\n"
        "@calc -v e_d,e_sl,e_cx,e_cy 'imax(abs(d_mod - d_calc)); imax(abs(sl_mod - sl_calc)); imax(abs(cx_mod - cx_calc)); imax(abs(cy_mod - cy_calc))'\n"
        "@calc 'mA = reshape(vec(1, 2, 3, 4, 5, 6), 3, 2)'\n"
        "imgbasic.imswapaxis2D mA mAt_mod >/dev/null\n"
        "@calc 'mAt_calc = transpose(mA)'\n"
        "@calc -v e_tr 'imax(abs(mAt_mod - mAt_calc))'\n"
        "@calc 'cube = reshape(vec(1, 2, 3, 4, 5, 6, 7, 8), 2, 2, 2)'\n"
        "imgbasic.cubecollapse cube cc_mod >/dev/null\n"
        "@calc 'cc_calc = itot(cube, 2)'\n"
        "@calc -v e_cc 'imax(abs(cc_mod - cc_calc))'\n"
        'echo "unif=$e_d $e_sl $e_cx $e_cy $e_tr $e_cc"',
        tmp_path,
    )
    out = res.stdout + res.stderr
    vals = [float(x) for x in value(res.stdout, "unif").split()]
    assert vals == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], out


def test_calc_v_quoted_expression(tmp_path):
    res = milk_cli_script(
        '@calc -v r "1+2"\necho "r=$r"\n@calc -v s \'2*3\'\necho "s=$s"', tmp_path
    )
    assert value(res.stdout, "r") == "3"
    assert value(res.stdout, "s") == "6"


def test_calc_v_error_on_unknown_image(tmp_path):
    res = milk_cli_script(
        '@calc -v r "imean(nosuchimage)"\necho "rc=$? r=$r"', tmp_path
    )
    assert "r=imean(nosuchimage)" not in res.stdout
    assert "rc=0" not in res.stdout


def test_qualified_names_only(tmp_path):
    body = 'mem.listvar >/dev/null 2>&1; echo "{0}q=$?"\nlistvar >/dev/null 2>&1; echo "{0}b=$?"\n'
    (tmp_path / "sub.milk").write_text(body.format("s"))
    res = milk_cli_script(body.format("t") + "@source sub.milk", tmp_path)
    for k, v in (("tq", "0"), ("tb", "127"), ("sq", "0"), ("sb", "127")):
        assert value(res.stdout, k) == v, res.stdout


def test_args_in_sourced_file_binds_caller_variable(tmp_path):
    sub = tmp_path / "sourced_args.milk"
    sub.write_text('@args "val:int=42:test int" -- "$@"\n')
    res = _run_bash(f"""
        @source "{sub}" 99
        [[ "$val" == "99" ]]
        echo "ARGS_OK"
    """)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "ARGS_OK" in res.stdout


@pytest.mark.timeout(120)
def test_long_module_argument_fails_instead_of_being_cut(tmp_path):
    res = milk_bash(
        "mem.mk2Dim im 4 4 >/dev/null\n"
        "P=\"$PWD/$(printf 'd%.0s' {1..220})/out.fits\"\n"
        'iofits.saveFITS im "$P" --no-run 2>err.txt; echo "rc=$?"',
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert value(res.stdout, "rc") == "1", out
    assert "longer than 199 characters" in (tmp_path / "err.txt").read_text(), out


@pytest.mark.timeout(120)
def test_argument_values_and_words_read_alike(tmp_path):
    res = milk_bash(
        CAM
        + 'mem.mk2Dim "@S:a1" 010 0x8 >/dev/null; echo "m1=$(@get s.a1.xsize),$(@get s.a1.ysize)"\n'
        'mem.mk2Dim "@S:a2" 1e1 0b11 >/dev/null; echo "m2=$(@get s.a2.xsize),$(@get s.a2.ysize)"\n'
        'mem.mk2Dim "@S:a3" 1.5 4 2>/dev/null; echo "m3=$?"\n'
        'C="milk-fpsexec-milk-calc -n av1"\n'
        '$C -t cam -p "n:int=0 b:bool=off" -e "out_av = cam * 0 + n + b" fpsinit .n=010 .b=true '
        '>/dev/null; echo "init=$?"\n'
        "$C run >run.log 2>&1 & pid=$!\n"
        'sleep 0.6; @calc "cam = 3"; sleep 0.3; @calc -v o "imean(out_av)"; echo "unit=$o"\n'
        "kill -INT $pid; wait $pid\n"
        '$C set .n=1 .n=2 >/dev/null 2>e1; echo "twice=$?"\n'
        '$C set -- -5 >/dev/null; echo "dashes=$?,$(@get f.av1.n)"\n'
        '@set f.av1.n=0b101; @set f.av1.b=false; echo "set=$(@get f.av1.n),$(@get f.av1.b)"\n'
        'h() { @args "a:int" "b:int" -- "$@"; echo "h=$a,$b"; }; h 1 .a=2\n'
        'g() { @args "n:int=5" -- "$@"; echo "g=$n"; }; g .n=.\n'
        'k() { @args "n:int" -- "$@"; echo "k=$n"; }; k 0o17',
        tmp_path,
    )
    out = res.stdout + res.stderr + (tmp_path / "run.log").read_text()
    assert value(res.stdout, "m1") == "10,8" and value(res.stdout, "m2") == "10,3", out
    assert value(res.stdout, "m3") == "2", out
    assert value(res.stdout, "init") == "0", out
    assert (
        value(res.stdout, "unit") == "11"
    ), out  # 010 is 10, and the bool reached the unit
    assert value(res.stdout, "twice") == "2", out
    assert "given twice" in (tmp_path / "e1").read_text(), out
    assert value(res.stdout, "dashes") == "0,-5", out
    assert value(res.stdout, "set") == "5,OFF", out
    assert value(res.stdout, "h") == "2,1", out  # tags first, then the positional words
    assert value(res.stdout, "g") == "5" and value(res.stdout, "k") == "15", out


@pytest.mark.timeout(120)
def test_property_modifier_and_keyword_tables_agree(tmp_path):
    from pathlib import Path

    res = milk_bash(
        'mem.mk2Dim "@S:pl" 4 4 >/dev/null; names=$(@get s.pl.list); echo "list=$names"\n'
        'for p in $names; do @get "s.pl.$p" >/dev/null 2>&1 || echo "unreadable=$p"; done\n'
        'echo "mods=$(milkcomplete @TC arg:mem.mk2Dim:1 | tr "\\n" " ")"\n'
        'echo "kw=$(milkls --keywords | cut -d" " -f1 | tr "\\n" " ")"\n'
        "@help > help.txt; milk-call > usage.txt 2>&1",
        tmp_path,
    )
    out = res.stdout + res.stderr
    names = value(res.stdout, "list").split()
    assert names[:5] == ["xsize", "ysize", "zsize", "naxis", "nelement"], out
    assert "owner_pid" in names and "ownerPID" not in names, out
    assert "unreadable=" not in res.stdout, out
    docs = Path(__file__).resolve().parents[3] / "docs" / "guide" / "scripting.md"
    text = docs.read_text()
    bullet = text[text.index("- streams (`s.`, `l.`") : text.index("- FPS (`f.`)")]
    assert [n for n in names if f"`{n}`" not in bullet] == [], bullet
    assert value(res.stdout, "mods").split() == ["@TC64:", "@TC128:"], out
    kws = value(res.stdout, "kw").split()
    assert len(kws) == 27, out
    section = text[text.index("## Keywords") : text.index("## Module Commands")]
    table = "".join(ln for ln in section.splitlines(True) if ln.startswith("|"))
    assert sorted(kws) == sorted(re.findall(r"`@(\w+)`", table)), table
    helptext = (tmp_path / "help.txt").read_text()
    assert [k for k in kws if f"@{k}" not in helptext] == [], helptext
    assert "batch" not in (tmp_path / "usage.txt").read_text(), out


@pytest.mark.timeout(120)
def test_failed_set_prints_its_error_once(tmp_path):
    """An @set that fails in the shell is not sent again to the server, which printed the same error a
    second time. An @set on a server-local image still goes to the server, with no error.
    """
    specs = [
        "f.se.procinfo.loopcntMax=abc",
        "f.se.nosuchparam=1",
        "f.nosuch.x=1",
        "f.se.procinfo.triggermode=often",
        "s.nosuch.kw.A=1",
        "p.nosuch.ctrl=pause",
        "s.cam.nosuchprop=1",
    ]
    res = milk_bash(
        UNIT_HELPERS
        + 'milk-fpsexec-milk-calc -n se -e "o = cam" fpsinit >/dev/null\n'
        + "".join(
            f'@set "{s}" >set{i}.log 2>&1; echo "rc{i}=$?"\n'
            for i, s in enumerate(specs)
        )
        + 'mem.mk2Dim "@L:limg" 4 4 >/dev/null; @set s.limg.kw.A=1 >lim.log 2>&1; echo "lrc=$?"\n'
        'echo "lim=$(@get l.limg.kw.A)"',
        tmp_path,
    )
    for i, spec in enumerate(specs):
        lines = (tmp_path / f"set{i}.log").read_text().splitlines()
        assert value(res.stdout, f"rc{i}") not in (None, "0"), (
            spec + res.stdout + res.stderr
        )
        assert len(lines) == 1, f"{spec}: {lines}"
    assert value(res.stdout, "lrc") == "0", res.stdout + res.stderr
    assert (tmp_path / "lim.log").read_text() == ""
    assert value(res.stdout, "lim") == "1", res.stdout + res.stderr


def test_keyword_options_come_from_one_table(tmp_path):
    """Every keyword reads its options with milk_kw_parse(), from its list in the keyword table:
    the same errors everywhere, and help and completion show what the parser takes."""
    bad = (
        "@grab --bogus cam 1 x",
        "@grab cam 1 x -t",
        "@waitfor cam --bogus",
        "@waitany cam --bogus",
        "@sleep --bogus",
        "@exists --bogus cam",
        "@get --bogus cam.xsize",
        "@set --bogus",
        "@ls --bogus",
        "@log --bogus",
        "@replay --dt 0.1 cam out",
        "@snapshot --bogus",
        "@restore --bogus x.json",
        "@module list --bogus",
        "@session ls --bogus",
        "@session end --json",
        "@help --bogus",
        "@source --bogus",
    )
    script = '@calc "cam = zeros(8, 8)"\n'
    script += "".join(
        f'{c} >/dev/null 2>>err.txt; echo "b{i}=$?"\n' for i, c in enumerate(bad)
    )
    script += (
        '@grab --timeout=1 cam g1 && @grab cam -t 1 g2 && echo "eq=$?"\n'
        '@calc -v v "-3 + 1"; echo "neg=$v"\n'
        '@log --help | grep -c "commands slower than" | sed "s/^/help=/"\n'
        'echo "comp=$(milkcomplete --t "arg:@waitfor:2:cam" | tr "\\n" " ")"\n'
        'echo "hid=$(milkcomplete --f "arg:@set:1:@set" | tr "\\n" " ")"'
    )
    res = milk_bash(script, tmp_path)
    out = res.stdout + res.stderr
    assert [value(res.stdout, f"b{i}") for i in range(len(bad))] == ["2"] * len(
        bad
    ), out
    err = (tmp_path / "err.txt").read_text()
    assert (
        "@grab: unknown option --bogus" in err and "@grab: -t needs a value" in err
    ), err
    assert "@session: end does not take --json" in err and "usage: @waitany" in err, err
    assert value(res.stdout, "eq") == "0" and value(res.stdout, "neg") == "-2", out
    assert value(res.stdout, "help") == "1", out
    assert value(res.stdout, "comp").split() == ["--timeout"], out
    assert value(res.stdout, "hid") == "", out  # --flush-undo is internal: not offered
