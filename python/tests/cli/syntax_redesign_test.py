"""
Consolidated regression tests for the milk syntax redesign and Phase C enhancements.
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
        '@module load milkclustering --as s >/dev/null 2>&1; echo "rc=$?"\n'
        '@module load milkclustering --as clu >/dev/null; echo "ok=$?"\n'
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
    env, _ = _env()
    env["MILKCLI_ADD_LIBS"] = "none"
    res = milk_cli_env(
        'imgen.mkgauss gau 16 16 3 1 >/dev/null; echo "rc=$?"\n@exists s.gau && echo made=1',
        tmp_path,
        env,
    )
    assert value(res.stdout, "rc") == "0" and value(res.stdout, "made") == "1"

    ldd_out = subprocess.run(
        ["ldd", str(_install_dir() / "lib" / "libmilkbash.so")],
        capture_output=True,
        text=True,
    ).stdout
    assert "cfitsio" not in ldd_out and "gomp" not in ldd_out, ldd_out


def test_inline_expansion_and_calc_parity(tmp_path):
    assert "milkget -v x s.cam.xsize" in translate("x=@{s.cam.xsize}\n")
    res = milk_cli_script(
        "mem.mk2Dim ia 6 4 >/dev/null\n"
        'x=@{s.ia.xsize}; echo "a=$x rc=$?"\n'
        'x=old; x=@{s.nosuch.xsize}; echo "b=[$x] rc=$?"\n'
        'i=1; i=@{i + 1}; echo "d=$i"\n'
        '@config digits 4 >/dev/null; @calc -v v "1/3"; echo "v=$v"\n'
        '@calc "one = 1"; @calc -v c1 "2^10 + sqrt(16)"; @calc -v c2 "(2^10 + sqrt(16)) * one"\n'
        'echo "par=$c1,$c2"',
        tmp_path,
    )
    assert value(res.stdout, "a") == "6 rc=0"
    assert value(res.stdout, "b") == "[] rc=1"
    assert value(res.stdout, "d") == "2"
    assert value(res.stdout, "v") == "0.3333"
    assert value(res.stdout, "par") == "1028,1028"


def test_modernize_script_rewrites(tmp_path):
    exe = str(_install_dir() / "bin" / "milk-script")
    env, _ = _env()
    f = tmp_path / "s.milk"
    f.write_text("@ci\n@dpdigits 4\n@ x = 1\necho $x\n")
    assert _run([exe, "--check", str(f)], env, 10).returncode == 1
    assert _run([exe, "--modernize", str(f)], env, 10).returncode == 0
    assert f.read_text() == 'milk.ci\n@config digits 4\n@calc -v x "1"\necho $x\n'
    assert _run([exe, "--check", str(f)], env, 10).returncode == 0


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
        "@calc -v mtot,mstd 'itot(a, (a > -2) && (b == 0)); irms(a, b == 0)'\n"
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
        "@calc -v l1,l2,idx,fl1 'asum(v); nrm2(v); iamax(v); asum(v * 2)'\n"
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
