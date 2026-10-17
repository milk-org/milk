"""
Regression targets for the milk syntax redesign (plan decisions D6-D17, Phase A work packages).

Each test asserts the *target* behaviour and is marked ``xfail(strict=True)`` with the work package
that delivers it, so a fix flips it to XPASS -> failure until the marker is removed. The guards at the
end are not xfail: they must hold before, during and after the redesign.
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


def wp(package: str, reason: str):
    """Strict xfail tagged with the work package (L*) that delivers the behaviour."""
    return pytest.mark.xfail(strict=True, reason=f"[{package}] {reason}")


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


# ------------------------------------------------------------- L1a: fixes in our files


def test_L1_bad_regex_in_help_search(tmp_path):
    res = milk_cli_script(
        "@help -k '[' >/dev/null 2>&1; echo \"rc=$?\"\necho after=1", tmp_path
    )
    assert value(res.stdout, "after") == "1", res.stdout + res.stderr
    assert value(res.stdout, "rc") not in (None, "0")


def test_L1_unload_disables_all_builtins(tmp_path):
    res = milk_bash(
        "@unload >/dev/null 2>&1\nfor b in milksleep milktranslate milkcloexec milkcall; do "
        'echo "$b=$(type -t $b)"; done',
        tmp_path,
    )
    for b in ("milksleep", "milktranslate", "milkcloexec", "milkcall"):
        assert value(res.stdout, b) == "", res.stdout


def test_L1_server_local_in_milk_cli(tmp_path):
    res = milk_cli_script(
        'mem.mk2Dim l.keep 4 4 >/dev/null\n@server local >/dev/null 2>&1; echo "lrc=$?"\n'
        '@exists l.keep && echo still=1\n@server stop >/dev/null 2>&1; echo "src=$?"\n'
        "@exists l.keep && echo alive=1",
        tmp_path,
    )
    assert value(res.stdout, "lrc") == "0", res.stdout + res.stderr
    assert value(res.stdout, "still") == "1", res.stdout + res.stderr
    assert value(res.stdout, "src") not in (None, "0"), res.stdout
    assert value(res.stdout, "alive") == "1", res.stdout + res.stderr


def test_L1_milkrc_not_run_for_scripts(tmp_path):
    env, _ = _env()
    home = Path(env["HOME"])
    (home / ".milkrc").write_text('touch "$HOME/rc_ran"\n')
    milk_cli_env("echo hi", tmp_path, env)
    assert not (home / "rc_ran").exists()


def test_L1_cwd_milklib_not_loaded(tmp_path):
    (tmp_path / "milklib").mkdir()
    (tmp_path / "milklib" / "libbogusmod.so").write_text("not a library\n")
    (tmp_path / "lib").mkdir()
    (tmp_path / "lib" / "libbogusmod.so").write_text("not a library\n")
    res = milk_cli_script("echo ok=1", tmp_path)
    assert value(res.stdout, "ok") == "1"
    assert "bogusmod" not in res.stdout + res.stderr


# ------------------------------------------------------------- L2: keyword table


def test_L2_bash_at_functions_are_keywords(tmp_path):
    """The @ words of bash are exactly the keywords: builtins for the builtin ones, functions for the rest."""
    res = milk_bash(
        "milkls --keywords | awk '{print \"@\" $1}' | sort > kw.txt\n"
        "{ compgen -A function @; compgen -A builtin @; } | sort > fn.txt\n"
        'milkls --keywords | awk \'$2 == "builtin" {print "@" $1}\' | sort > bi.txt\n'
        "compgen -A builtin @ | sort | diff - bi.txt >/dev/null && echo builtins=1\n"
        'echo "kw=$(wc -l < kw.txt)"\ndiff kw.txt fn.txt >/dev/null && echo same=1',
        tmp_path,
    )
    assert int(value(res.stdout, "kw") or 0) >= 20, res.stdout + res.stderr
    assert value(res.stdout, "same") == "1", res.stdout + res.stderr
    assert value(res.stdout, "builtins") == "1", res.stdout + res.stderr


# ------------------------------------------------------------- L3: registry


def test_L3_namespace_aliases_dropped(tmp_path):
    res = milk_bash(
        "milkls --namespaces | awk '{print $1}' | tr '\\n' ' '; echo", tmp_path
    )
    names = res.stdout.split()
    assert "seq" not in names and "procinfo" not in names, res.stdout
    assert {"s", "stream", "l", "local", "f", "fps", "p", "proc", "q", "fpsseq"} <= set(
        names
    )


def test_L3_no_hardcoded_aliases(tmp_path):
    res = milk_cli_script(
        "mem.mk2Dim a 4 4 >/dev/null\n"
        'milk-call call iofits.savefits a o.fits >/dev/null 2>&1; echo "rc=$?"',
        tmp_path,
    )
    assert value(res.stdout, "rc") == "127", res.stdout
    assert not (tmp_path / "o.fits").exists()


def test_L3_reserved_short_name_rejected(tmp_path):
    res = milk_cli_script(
        '@module load milkclustering --as s >/dev/null 2>&1; echo "rc=$?"\n'
        '@module load milkclustering --as clu >/dev/null; echo "ok=$?"\n'
        'echo "t=$(type -t clu.cubeclust)"',
        tmp_path,
    )
    assert value(res.stdout, "rc") not in (None, "0"), res.stdout
    assert value(res.stdout, "ok") == "0", res.stdout + res.stderr
    assert value(res.stdout, "t") == "function", res.stdout


def test_L3_milk_module(tmp_path):
    res = milk_cli_script(
        'milk.ci >/dev/null; echo "rc=$?"\nmilk.timing on >/dev/null; echo "t=$?"',
        tmp_path,
    )
    assert value(res.stdout, "rc") == "0", res.stdout + res.stderr
    assert value(res.stdout, "t") == "0", res.stdout + res.stderr


def test_L6_old_engine_names_removed(tmp_path):
    res = milk_cli_script('@ci >/dev/null 2>&1; echo "old=$?"', tmp_path)
    assert value(res.stdout, "old") == "127", res.stdout


def test_L3_milk_script_has_engine_commands(tmp_path):
    res = milk_script(
        'milk.ci >/dev/null; echo "ci=$?"\n@help >/dev/null; echo "help=$?"', tmp_path
    )
    assert value(res.stdout, "ci") == "0", res.stdout + res.stderr
    assert value(res.stdout, "help") == "0", res.stdout + res.stderr


# ------------------------------------------------------------- L4a: help


def test_L4_dash_h_on_module_command(tmp_path):
    res = milk_cli_script(
        'iofits.loadfits -h > h.txt; echo "rc=$?"\nmem.mk2Dim --help > h2.txt; echo "rc2=$?"\n'
        "mem.listim",
        tmp_path,
    )
    assert value(res.stdout, "rc") == "0", res.stdout + res.stderr
    assert value(res.stdout, "rc2") == "0", res.stdout + res.stderr
    assert "loadfits" in (tmp_path / "h.txt").read_text()
    assert "ERROR" not in res.stdout + res.stderr
    assert "-h" not in re.findall(r"^\s*\d+\s+(\S+)", res.stdout, re.M)


def test_L4_help_keyword_matches_dash_h(tmp_path):
    res = milk_cli_script(
        '@help iofits.loadfits > a.txt; echo "rc=$?"\niofits.loadfits -h > b.txt',
        tmp_path,
    )
    assert value(res.stdout, "rc") == "0", res.stdout + res.stderr
    a, b = (tmp_path / "a.txt").read_text(), (tmp_path / "b.txt").read_text()
    assert a and a == b


def test_L4_calc_long_help(tmp_path):
    res = milk_cli_script('@calc --help > c.txt; echo "rc=$?"', tmp_path)
    assert value(res.stdout, "rc") == "0", res.stdout + res.stderr
    assert "calc" in (tmp_path / "c.txt").read_text()


@pytest.mark.parametrize(
    "old, hint",
    [
        ("@hc mem.listim", "@help"),
        ("@hm mem", "@help"),
        ("@hb", "@help"),
        ("@m? mem", "@help"),
        ("@cmd? mem.listim", "@help"),
    ],
)
def test_L4_removed_help_words(tmp_path, old, hint):
    res = milk_cli_script(f'{old} >/dev/null; echo "rc=$?"', tmp_path)
    assert value(res.stdout, "rc") == "127", res.stdout
    assert hint in res.stderr


def test_L4_dash_h_in_milk_bash(tmp_path):
    res = milk_bash('iofits.loadfits -h > h.txt; echo "rc=$?"', tmp_path)
    assert value(res.stdout, "rc") == "0", res.stdout + res.stderr
    assert "loadfits" in (tmp_path / "h.txt").read_text()


# ------------------------------------------------------------- L5a: per-call arguments


def test_L5_missing_args_usage_error(tmp_path):
    res = milk_cli_script(
        'mem.mk2Dim a 8 8 >/dev/null\nmem.mk2Dim b >/dev/null 2>&1; echo "rc=$?"\n'
        "@exists s.b && echo made=1",
        tmp_path,
    )
    assert value(res.stdout, "rc") == "2", res.stdout
    assert value(res.stdout, "made") is None, res.stdout


@pytest.mark.parametrize("bad", ["-h", "abc", "1.5x"])
def test_L5_invalid_number(tmp_path, bad):
    res = milk_cli_script(
        f'mem.mk2Dim c {bad} 4 >/dev/null 2>&1; echo "rc=$?"\n@exists s.c && echo made=1',
        tmp_path,
    )
    assert value(res.stdout, "rc") == "2", res.stdout
    assert value(res.stdout, "made") is None, res.stdout


def test_L5_named_args(tmp_path):
    res = milk_cli_script(
        'mem.mk2Dim .ysize=5 d .xsize=3 >/dev/null; echo "rc=$?"\n'
        '@get -v x s.d.xsize; @get -v y s.d.ysize; echo "x=$x y=$y"',
        tmp_path,
    )
    assert value(res.stdout, "rc") == "0", res.stdout + res.stderr
    assert "x=3 y=5" in res.stdout


def test_L5_presets(tmp_path):
    res = milk_cli_script(
        'mem.mk2Dim e 4 6 --save=p1 >/dev/null; echo "rc=$?"\n'
        'mem.mk2Dim f --load=p1 >/dev/null; echo "rc2=$?"\n'
        '@get -v x s.f.xsize; @get -v y s.f.ysize; echo "x=$x y=$y"',
        tmp_path,
    )
    assert value(res.stdout, "rc") == "0" and value(res.stdout, "rc2") == "0", (
        res.stdout + res.stderr
    )
    assert "x=4 y=6" in res.stdout


def test_L5_session_default(tmp_path):
    res = milk_cli_script(
        'mem.mk2Dim --default .xsize=3 .ysize=7; echo "rc=$?"\n'
        'mem.mk2Dim g >/dev/null; @get -v y s.g.ysize; echo "y=$y"',
        tmp_path,
    )
    assert value(res.stdout, "rc") == "0", res.stdout + res.stderr
    assert value(res.stdout, "y") == "7", res.stdout


def test_L5_no_run(tmp_path):
    res = milk_cli_script(
        'mem.mk2Dim h 4 4 --no-run; echo "rc=$?"\n@exists s.h || echo absent=1',
        tmp_path,
    )
    assert value(res.stdout, "rc") == "0", res.stdout + res.stderr
    assert value(res.stdout, "absent") == "1", res.stdout


def test_L5_dot_negative_values_and_dashdash(tmp_path):
    res = milk_cli_script(
        'imgen.mkgauss g . 8 -2 -0.5 --no-run; echo "rc=$?"\n'
        'mem.mk2Dim --no-run -- --default 4 4; echo "rc2=$?"',
        tmp_path,
    )
    assert value(res.stdout, "rc") == "0" and value(res.stdout, "rc2") == "0", (
        res.stdout + res.stderr
    )
    assert ".ysize=8 .a=-2 .amp=-0.5" in res.stdout
    assert "mem.mk2Dim .out_name=--default .xsize=4 .ysize=4" in res.stdout


def test_L5_removed_tag_value_form_hints(tmp_path):
    res = milk_cli_script('mem.mk2Dim .xsize 4 2>&1; echo "rc=$?"', tmp_path)
    assert value(res.stdout, "rc") == "2", res.stdout
    assert ".xsize=VALUE" in res.stdout


def test_L5_equals_in_value_is_literal(tmp_path):
    res = milk_cli_script(
        'x=1\nmem.mk2Dim x=7 4 4 >/dev/null; echo "rc=$? x=$x"\n@exists s.x=7 && echo made=1',
        tmp_path,
    )
    assert "rc=0 x=1" in res.stdout, res.stdout + res.stderr
    assert value(res.stdout, "made") == "1", res.stdout


# ------------------------------------------------------------- L6a/L7: catalog, autoload


def test_L6_module_commands_are_functions(tmp_path):
    script = 'echo "t=$(type -t iofits.loadfits)"'
    assert value(milk_bash(script, tmp_path).stdout, "t") == "function"
    assert value(milk_cli_script(script, tmp_path).stdout, "t") == "function"
    assert value(milk_script(script, tmp_path).stdout, "t") == "function"


def test_L6_bare_short_names_removed(tmp_path):
    res = milk_bash(
        'echo "t=$(type -t @loadfits)"\n@loadfits x.fits im; echo "rc=$?"', tmp_path
    )
    assert value(res.stdout, "t") == "", res.stdout
    assert value(res.stdout, "rc") == "127", res.stdout
    assert "iofits.loadfits" in res.stderr


def test_L6_unknown_dotted_word_starts_no_server(tmp_path):
    res = milk_bash(
        "setup.py >/dev/null 2>&1\n"
        'if ps -eo args | grep -F -- "--server $_MILK_LOCAL_SOCK" | grep -qv grep; '
        "then echo server=1; else echo server=0; fi",
        tmp_path,
    )
    assert value(res.stdout, "server") == "0", res.stdout + res.stderr


def test_L6_no_configurable_sigil(tmp_path):
    res = milk_bash(
        'echo "c=$(type -t ,calc)" "a=$(type -t @calc)"',
        tmp_path,
        {"MILKCLI_SIGIL": ","},
    )
    assert "c= " in res.stdout + " " and "a=builtin" in res.stdout, res.stdout


def test_L6_source_spawns_no_process(tmp_path):
    env, _ = _env()
    mb = _install_dir() / "bin" / "milk.bash"
    _run(["/bin/bash", "-c", f'source "{mb}"'], env, 30)  # warm caches
    trace = tmp_path / "trace"
    _run(
        [
            "strace",
            "-f",
            "-qq",
            "-e",
            "trace=clone,clone3,fork,vfork",
            "-o",
            str(trace),
            "/bin/bash",
            "-c",
            f'source "{mb}"',
        ],
        env,
        30,
    )
    forks = [
        ln
        for ln in trace.read_text().splitlines()
        if re.search(r"\b(clone3?|v?fork)\(", ln)
    ]
    assert forks == [], "\n".join(forks[:10])


def test_L7_autoload(tmp_path):
    env, _ = _env()
    env["MILKCLI_ADD_LIBS"] = "none"  # core modules only: imgen must be autoloaded
    script = 'imgen.mkgauss gau 16 16 3 1 >/dev/null; echo "rc=$?"\n@exists s.gau && echo made=1'
    (tmp_path / "ms.milk").write_text(script + "\n")
    runs = (
        milk_cli_env(script, tmp_path, env),
        _run(
            [
                str(_install_dir() / "bin" / "milk-script"),
                "-q",
                str(tmp_path / "ms.milk"),
            ],
            env,
            20,
            cwd=str(tmp_path),
        ),
    )
    for res in runs:
        assert value(res.stdout, "rc") == "0", res.stdout + res.stderr
        assert value(res.stdout, "made") == "1", res.stdout


def test_L7_runtime_load_reaches_other_shells(tmp_path):
    # commands the catalog lacks (a module loaded under another name) become functions in every shell
    mb = _install_dir() / "bin" / "milk.bash"
    res = milk_bash(
        f'bash -c \'source "{mb}" --server l7t; @module load milkclustering --as clu2 >/dev/null; echo "a=$?"\'\n'
        f'bash -c \'source "{mb}" --server l7t; mem.listim >/dev/null; '
        f'echo "b=$(type -t clu2.cubeclust)"; @server stop >/dev/null 2>&1\'',
        tmp_path,
    )
    assert value(res.stdout, "a") == "0", res.stdout + res.stderr
    assert value(res.stdout, "b") == "function", res.stdout + res.stderr


# ------------------------------------------------------------- L8: keyword consolidation


@pytest.mark.parametrize(
    "old, hint",
    [
        ("@mload milkfft", "@module load"),
        ("@usleep 10", "@sleep"),
        ("@batch 'a = 1'", "@pipe"),
        ("@dpdigits 4", "@config"),
        ("@streamlist", "@ls"),
        ("include_once x.milk", "@source --once"),
    ],
)
def test_L8_removed_words(tmp_path, old, hint):
    res = milk_cli_script(f'{old} >/dev/null; echo "rc=$?"', tmp_path)
    assert value(res.stdout, "rc") == "127", res.stdout
    assert hint in res.stderr


def test_L8_module_keyword(tmp_path):
    res = milk_cli_script(
        '@module load milkfft; echo "rc=$?"\n@module list | grep -c fft', tmp_path
    )
    assert value(res.stdout, "rc") == "0", res.stdout + res.stderr
    assert res.stdout.strip().splitlines()[-1] != "0"


def test_L8_config_digits(tmp_path):
    res = milk_cli_script(
        '@config digits 4; echo "rc=$?"\n@calc -v v \'1/3\'; echo "v=$v"', tmp_path
    )
    assert value(res.stdout, "rc") == "0", res.stdout + res.stderr
    assert value(res.stdout, "v") == "0.3333", res.stdout


def test_L8_sleep(tmp_path):
    res = milk_cli_script('@sleep 0.01; echo "rc=$?"', tmp_path)
    assert value(res.stdout, "rc") == "0", res.stdout + res.stderr


def test_L8_repl_only_keyword(tmp_path):
    res = milk_bash('@history >/dev/null; echo "rc=$?"', tmp_path)
    assert value(res.stdout, "rc") == "2", res.stdout
    assert "REPL" in res.stderr


def test_L8_source_once(tmp_path):
    (tmp_path / "inc.milk").write_text("echo inc\n")
    res = milk_cli_script("@source --once inc.milk\n@source --once inc.milk", tmp_path)
    assert res.stdout.count("inc") == 1, res.stdout + res.stderr


# ------------------------------------------------------------- L9: syntax


@pytest.mark.parametrize(
    "src",
    [
        "curl -d @f.data.json http://x\n",
        'echo "HEAD@{1}"\n',
        "echo @s.cam.xsize\n",
        "echo ${@s.cam.xsize}\n",
        "echo a |> b\n",
        "@ls cam*\n",
    ],
)
def test_L9_translator_leaves_text(src):
    assert translate(src) == src


def test_L9_inline_entity_fast_path():
    out = translate("x=@{s.cam.xsize}\n")
    assert "milkget -v x s.cam.xsize" in out and "calc" not in out, out


def test_L9_inline_assignment_keeps_substitution_semantics(tmp_path):
    """NAME=@{...} binds NAME without a substitution: same value, status, failure and scoping."""
    res = milk_cli_script(
        "mem.mk2Dim ia 6 4 >/dev/null\n"
        'x=@{s.ia.xsize}; echo "a=$x rc=$?"\n'
        'x=old; x=@{s.nosuch.xsize}; echo "b=[$x] rc=$?"\n'
        "false && x=@{s.ia.xsize} || echo c=or\n"
        'i=1; i=@{i + 1}; echo "d=$i"\n'
        'x=old; x=@{zz = 3}; echo "e=[$x]"\n'
        'f() { local x; x=@{1+2}; echo "f=$x"; }; unset x; f; echo "g=${x-unset}"\n'
        'x="@{s.ia.ysize}"; echo "h=$x"',
        tmp_path,
    )
    got = {k: value(res.stdout, k) for k in "abcdefgh"}
    assert got == {
        "a": "6 rc=0",
        "b": "[] rc=1",
        "c": "or",
        "d": "2",
        "e": "[]",
        "f": "3",
        "g": "unset",
        "h": "4",
    }, (
        res.stdout + res.stderr
    )
    assert res.stderr.count("inline expansion failed") == 1, res.stderr


@pytest.mark.parametrize("pre", ["export MILK_STRICT=1", "set -e"])
def test_L9_inline_assignment_failure_stops_script(tmp_path, pre):
    res = milk_cli_script(f"{pre}\nx=@{{s.nosuch.xsize}}\necho reached", tmp_path)
    assert "reached" not in res.stdout and res.returncode == 1, res.stdout + res.stderr


@pytest.mark.parametrize(
    "old, hint",
    [
        ("@ x = 1", "@calc"),
        ("@f.loop.gain=0.2", "@set"),
        ("@s.cam.xsize", "@get"),
    ],
)
def test_L9_statement_forms_removed(tmp_path, old, hint):
    res = milk_cli_script(f'{old} >/dev/null; echo "rc=$?"', tmp_path)
    assert value(res.stdout, "rc") not in (None, "0"), res.stdout
    assert hint in res.stderr


def test_L9_plain_source_untranslated(tmp_path):
    (tmp_path / "f.milk").write_text('echo "v=@{1+2}"\n')
    res = milk_cli_script("source f.milk\n@source f.milk", tmp_path)
    assert re.findall(r"^v=(.*)$", res.stdout, re.M) == ["@{1+2}", "3"], res.stdout


# ------------------------------------------------------------- L10: variables


def test_L10_no_implicit_push_into_args(tmp_path):
    res = milk_cli_script(
        'n=8\nmem.mk2Dim im3 n 4 >/dev/null 2>&1; echo "rc=$?"\n@exists s.im3 && echo made=1',
        tmp_path,
    )
    assert value(res.stdout, "rc") == "2", res.stdout
    assert value(res.stdout, "made") is None, res.stdout


def test_L10_no_implicit_write_back(tmp_path):
    res = milk_cli_script("@calc 'zz = 3'\necho \"zz=${zz-unset}\"", tmp_path)
    assert value(res.stdout, "zz") == "unset", res.stdout


def test_L10_modernize_notes_calc_list(tmp_path):
    f = tmp_path / "s.milk"
    f.write_text('@calc "a = 1; b = 2"\necho "$b"\n')
    env, _ = _env()
    res = _run(
        [str(_install_dir() / "bin" / "milk-script"), "--check", str(f)], env, 10
    )
    assert (
        res.returncode == 1 and "'b' is set in calc only now" in res.stdout + res.stderr
    ), res.stdout


# ------------------------------------------------------------- L12: thin client


CALC_PARITY = [
    "1+2",
    "7/2",
    "7%3",
    "-7%3",
    "7%-3",
    "2^10",
    "2^-1",
    "2^63",
    "(-2)^3",
    "-2^2",
    "--3",
    "2^3^2",
    "1-2-3",
    "10/4*2",
    "2*(3+4)*-1",
    "9223372036854775807+1",
    "3*1.5",
    "1e3",
    "1.5e-3*2",
    ".5+.25",
    "1<2",
    "2<=2",
    "3>4",
    "1==1.0",
    "1!=2",
    "1&&0",
    "0||2",
    "!0",
    "!3+1",
    "~5",
    "6&3",
    "6|3",
    "6^^3",
    "1<<4",
    "256>>4",
    "1<<64",
    "1?2:3",
    "0?2:0?4:5",
    "1+0?7:8",
    "0&&(1/0)",
    "1||(1/0)",
    "sqrt(16)",
    "sin(0.5)",
    "cos(0)",
    "exp(1)",
    "ln(10)",
    "log(1000)",
    "tan(1)",
    "atan(1)",
    "cbrt(27)",
    "ceil(1.2)",
    "floor(-1.2)",
    "asin(1)",
    "acos(0)",
    "posi(-1)+posi(2)",
    "abs(-3)",
    "fabs(-2.5)",
    "round(2.5)",
    "atan2(1,1)",
    "fmod(7.5,2)",
    "min(3,4)",
    "max(3,4.5)",
    "trunc(5,0,3)",
    "0x1f+0o17+0b101",
    "sqrt(-1)",
    "1e308*10",
    "1/3",
]


def test_L12_local_calc_matches_engine(tmp_path):
    """The in-shell evaluator gives what the engine gives (`* one` sends the expression to the engine)."""
    exprs = CALC_PARITY + [
        "1/3",
        "exp(1)",
        "1e-7/3",
    ]  # the last three after `@config digits 4`
    lines = ['@calc "one = 1"']
    for i, e in enumerate(exprs):
        lines.append(f"@calc -v a '{e}'; @calc -v b '({e}) * one'; echo \"P{i}|$a|$b\"")
    lines.insert(len(CALC_PARITY) + 1, "@config digits 4 >/dev/null")
    res = milk_cli_script("\n".join(lines), tmp_path)
    got = re.findall(r"^P(\d+)\|(.*)\|(.*)$", res.stdout, re.M)
    assert len(got) == len(exprs), res.stdout + res.stderr
    assert [(exprs[int(i)], a, b) for i, a, b in got if a != b] == []
    assert got[-3][1] == "0.3333", got[-3]


def test_L12_scalar_calc_starts_no_server(tmp_path):
    res = milk_bash(
        "@calc -v x '2^10 + sqrt(16)'; @test '1 < 2' && echo t=1; echo \"x=$x\"\n"
        'if ps -eo args | grep -F -- "--server $_MILK_LOCAL_SOCK" | grep -qv grep; '
        "then echo server=1; else echo server=0; fi",
        tmp_path,
    )
    assert (value(res.stdout, "x"), value(res.stdout, "t")) == ("1028", "1"), (
        res.stdout + res.stderr
    )
    assert value(res.stdout, "server") == "0", res.stdout + res.stderr


def test_L12_thin_client():
    out = subprocess.run(
        ["ldd", str(_install_dir() / "lib" / "libmilkbash.so")],
        capture_output=True,
        text=True,
    ).stdout
    assert "cfitsio" not in out and "gomp" not in out, out


# ------------------------------------------------------------- migration tool (golden)

MODERNIZE_CASES = [
    # (old, new): every removed spelling has a rewrite rule (one per work package)
    ("@ci\n", "milk.ci\n"),
    ("cli.timing on\n", "milk.timing on\n"),
    ("@savefits im out.fits\n", "iofits.saveFITS im out.fits\n"),
    ("@get seq.loop.status\n", "@get q.loop.status\n"),
    ('echo "@seq.loop.status seq.txt"\n', 'echo "@{q.loop.status} seq.txt"\n'),
    ("@ls 'procinfo.*'\n", "@ls 'p.*'\n"),
    ("echo seq.notanentity\n", "echo seq.notanentity\n"),
    ("mem.mk2Dim .xsize 4\n", "mem.mk2Dim --default .xsize=4\n"),
    (
        "imgformat.cred_ql_utr ..triggermode 1\n",
        "imgformat.cred_ql_utr ..triggermode 1\n",
    ),
    ("mem.mk2Dim .xsize=4 a 3\n", "mem.mk2Dim .xsize=4 a 3\n"),
    ("@mk2Dim a 4 4\n", "mem.mk2Dim a 4 4\n"),
    ("x=$(@info.imstats a)\n", "x=$(info.imstats a)\n"),
    ("@get s.a.xsize\n@L:x\n", "@get s.a.xsize\n@L:x\n"),
    ("@dpdigits 4\n", "@config digits 4\n"),
    ("@usleep 2000\n", "@sleep 0.002\n"),
    ("@fpsset loop gain 0.25\n", "@set f.loop.gain=0.25\n"),
    ("@procctl loop pause\n", "@set p.loop.ctrl=pause\n"),
    ("@procwait loop RUNNING 5\n", "@waitfor p.loop.status=RUNNING -t 5\n"),
    ("@mloadas milkfft f2\n", "@module load milkfft --as f2\n"),
    ("@streamcat -a pix s.cam\n", "@grab --stdout -a pix s.cam\n"),
    ("include_once lib.milk\n", "@source --once lib.milk\n"),
    ('@batch "a = 1; b = 2"\n', '@pipe "a = 1; b = 2"\n'),
    ("exitCLI\n", "exit\n"),
    ("@ x = 1\n", '@calc "x = 1"\n'),
    ("@f.loop.gain=0.2\n", "@set f.loop.gain=0.2\n"),
    ("@s.cam.xsize\n", "@get s.cam.xsize\n"),
    ('echo "@s.cam.cnt0"\n', 'echo "@{s.cam.cnt0}"\n'),
    ("echo ${@s.cam.xsize}\n", "echo @{s.cam.xsize}\n"),
    ("mem.listim _g_*\n", 'mem.listim "_g_*"\n'),
    ('@calc "x = @s.a.xsize * 2"\n', '@calc "x = s.a.xsize * 2"\n'),
    ("source lib.milk\n", "@source lib.milk\n"),
    ("@s.cam.kw.step\n", "@get s.cam.kw.step\n"),
    ("@p.loop.pause\n", "@set p.loop.pause\n"),
    ("@f.ao.gain==3\n", "@f.ao.gain==3\n"),
    ('echo "v=@{1+2}"\n', 'echo "v=@{1+2}"\n'),
    ("@ x = 1\necho $x\n", '@calc -v x "1"\necho $x\n'),
    ("@calc 'n = 2 * k'\necho \"${n}\"\n", "@calc -v n '2 * k'\necho \"${n}\"\n"),
    ('@calc "m *= 2 + 1"\necho $m\n', '@calc -v m "m * (2 + 1)"\necho $m\n'),
    ('@calc "c1 = 3"\nmem.rm c1\n', '@calc "c1 = 3"\nmem.rm c1\n'),
]


@pytest.mark.parametrize("old, new", MODERNIZE_CASES)
def test_L3_modernize_golden(tmp_path, old, new):
    exe = str(_install_dir() / "bin" / "milk-script")
    env, _ = _env()
    f = tmp_path / "s.milk"
    f.write_text(old)
    chk = _run([exe, "--check", str(f)], env, 10)
    assert chk.returncode == (0 if old == new else 1), chk.stdout + chk.stderr
    assert _run([exe, "--modernize", str(f)], env, 10).returncode == 0
    assert f.read_text() == new
    assert _run([exe, "--check", str(f)], env, 10).returncode == 0  # idempotent


# ------------------------------------------------------------- L13: @{...} at an interactive prompt


def bash_interactive(
    lines: list[str], tmp_path: Path, extra_env: dict[str, str] | None = None
) -> str:
    """Type @lines into an interactive bash (milk.bash sourced) on a pty; returns what it printed."""
    env, _ = _env()
    env.update(TERM="dumb", PS1="$ ", INPUTRC="/dev/null", **(extra_env or {}))
    pid, fd = pty.fork()
    if pid == 0:
        os.chdir(tmp_path)
        os.execve("/bin/bash", ["/bin/bash", "--norc", "--noprofile", "-i"], env)
    out = b""
    try:
        for i, ln in enumerate(
            [f'source "{_install_dir() / "bin" / "milk.bash"}"'] + lines + ["exit"]
        ):
            os.write(fd, (ln + "\r").encode())
            end = time.time() + (1.5 if i == 0 else 0.4)
            while time.time() < end:
                if select.select([fd], [], [], 0.05)[0]:
                    try:
                        out += os.read(fd, 65536)
                    except OSError:
                        break
    finally:
        os.close(fd)
        os.waitpid(pid, 0)
    return out.decode(errors="replace").replace("\r", "")


def test_L13_inline_at_interactive_prompt(tmp_path):
    out = bash_interactive(
        [
            'echo "a=@{1+2}"',
            "@config inline on",
            'echo "b=@{1+2}"',
            "echo 'c=@{1+2}'",
            'echo "d=$(echo @{2*3})"',
            "@config inline",
            "@config inline off",
            'echo "e=@{1+2}"',
        ],
        tmp_path,
    )
    assert re.findall(r"^([a-e])=(.*)$", out, re.M) == [
        ("a", "@{1+2}"),
        ("b", "3"),
        ("c", "@{1+2}"),
        ("d", "6"),
        ("e", "@{1+2}"),
    ], out
    assert "inline on" in out, out


def test_L13_inline_without_builtins(tmp_path):
    out = bash_interactive(
        ["@config inline on", 'echo "b=@{1+2}"'], tmp_path, {"MILK_NO_BUILTINS": "1"}
    )
    assert re.findall(r"^b=(.*)$", out, re.M) == ["3"], out


def test_L13_inline_needs_interactive_shell(tmp_path):
    res = milk_bash('@config inline on; echo "rc=$?"', tmp_path)
    assert value(res.stdout, "rc") == "2" and "interactive" in res.stderr, (
        res.stdout + res.stderr
    )


# ------------------------------------------------------------- guards (must always hold)


def test_L0_calc_reads_entities_and_bash_vars(tmp_path):
    res = milk_cli_script(
        "mem.mk2Dim ce 6 4 >/dev/null\nn=8\n@calc -v v 's.ce.xsize * 2 + n'; echo \"v=$v\"",
        tmp_path,
    )
    assert value(res.stdout, "v") == "20", res.stdout + res.stderr


# Per-op ceilings (µs) at ~2.5x the 2026-10-17 baseline (after the @keyword builtins); MILK_PERF_SCALE
# relaxes them on slow hosts.
PERF_OPS = [
    (1000, "@get -v x s.pbim.xsize", 7),
    (1000, "@get -v x s.pbsh.xsize", 7),
    (1000, "x=@{s.pbim.xsize}", 7),
    (300, ': "@{s.pbim.xsize}"', 40),
    (1000, "@set s.pbim.kw.FOO=1", 13),
    (1000, "@calc -v y 1+2", 5),
    (1000, "y=@{1+2}", 6),
    (100, "mem.listim >/dev/null", 200),
]


def test_L0_latency_ceilings(tmp_path):
    lines = ["mem.mk2Dim pbim 64 64 >/dev/null", "mem.mk2Dim s.pbsh 64 64 >/dev/null"]
    for (
        n,
        op,
        _,
    ) in (
        PERF_OPS
    ):  # best of 3 runs (0.1 µs steps), so a busy host does not fail the guard
        lines.append(
            f"best=999999; for r in 1 2 3; do t0=$EPOCHREALTIME; for ((i=0;i<{n};i++)); do {op}; done; "
            f"t1=$EPOCHREALTIME; t=$(( (${{t1/./}}-${{t0/./}})*10/{n} )); (( t < best )) && best=$t; done"
        )
        lines.append('echo "PERF $best"')
    res = milk_cli_script("\n".join(lines), tmp_path, timeout=60)
    got = [int(x) / 10 for x in re.findall(r"^PERF (\d+)$", res.stdout, re.M)]
    assert len(got) == len(PERF_OPS), res.stdout + res.stderr
    scale = float(os.environ.get("MILK_PERF_SCALE", "1"))
    slow = [
        (op, us, cap) for (_, op, cap), us in zip(PERF_OPS, got) if us > cap * scale
    ]
    assert not slow, slow
    us = {op: t for (_, op, _), t in zip(PERF_OPS, got)}
    assert (
        us["x=@{s.pbim.xsize}"] <= 1.5 * us["@get -v x s.pbim.xsize"]
    ), us  # plan §2.10 budget
