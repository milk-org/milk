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

from milk.cliwrap import HAVE_CLI, MILK_CLI_EXEC
from tests.cli.bash_engine_review_test import (
    _env,
    _install_dir,
    _run,
    milk_cli_script,
    milk_script,
    translate,
    value,
)

pytestmark = [
    pytest.mark.skipif(not HAVE_CLI, reason="MILK compiled without CLI support"),
    pytest.mark.timeout(60),
]


def milk_bash(
    script: str,
    tmp_path: Path,
    extra_env: dict[str, str] | None = None,
    timeout: float = 30,
):
    """Run @script in plain bash after `source milk.bash` (no -e), with an isolated HOME and SHM dir."""
    env, _ = _env()
    if extra_env:
        env.update(extra_env)
    full = f'source "{_install_dir() / "bin" / "milk.bash"}"\n{script}\n'
    return _run(["/bin/bash", "-c", full], env, timeout, cwd=str(tmp_path))


def milk_cli_env(script: str, tmp_path: Path, env: dict[str, str], timeout: float = 20):
    """`milk-cli -s` with a caller-provided environment."""
    f = tmp_path / "s.milk"
    f.write_text(script + "\n")
    return _run([MILK_CLI_EXEC, "-s", str(f)], env, timeout, cwd=str(tmp_path))


def events(out: str) -> list[str]:
    return re.findall(r"^D:(.*)$", out, re.M)


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
