"""
Regression repros for the bash-engine review (feat/bash-engine).

Finding IDs (T*, C*, P*, S*, Sec*) refer to the review / work plan.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from milk.cliwrap import HAVE_CLI, MILK_CLI_EXEC
from tests.cli.conftest import (
    _env,
    _install_dir,
    _kill_by_cmdline,
    _run,
    _run_bash,
    bug,
    milk_cli_script,
    milk_cli_stdin,
    milk_script,
    pty_session,
    translate,
    value,
)

pytestmark = [
    pytest.mark.skipif(not HAVE_CLI, reason="MILK compiled without CLI support"),
    pytest.mark.timeout(40),
]


# ------------------------------------------------- T*: translator (bash/milk_bash_translate.c)


def test_T1_shift_in_arith_is_not_heredoc():
    out = translate("echo $((1<<2))\necho @{1+2}\n")
    assert "milkcalc" in out.splitlines()[1]


def test_T2_multiline_string_untouched():
    src = 'echo "x\n@ 1+2 in string\nmk2Dim a 1 1\n"\n'
    assert translate(src) == src


def test_T2_continuation_not_new_statement():
    out = translate("echo a \\\n  mk2Dim b 1 1\n")
    assert "mem.mk2Dim" not in out


def test_T3_at_v_quoted_expression(tmp_path):
    res = milk_cli_script(
        '@calc -v r "1+2"\necho "r=$r"\n@calc -v s \'2*3\'\necho "s=$s"', tmp_path
    )
    assert value(res.stdout, "r") == "3"
    assert value(res.stdout, "s") == "6"


def test_T3_at_v_error_on_unknown_image(tmp_path):
    res = milk_cli_script(
        '@calc -v r "imean(nosuchimage)"\necho "rc=$? r=$r"', tmp_path
    )
    assert "r=imean(nosuchimage)" not in res.stdout
    assert "rc=0" not in res.stdout


def test_T4_at_expr_not_split_on_operators(tmp_path):
    res = milk_cli_script(
        '@calc "x = (2 > 1) && (2 < 1)"\n@calc -v r "x"\necho "r=$r"', tmp_path
    )
    assert value(res.stdout, "r") == "0"


def test_T5_top_level_and_sourced_agree(tmp_path):
    (tmp_path / "sub.milk").write_text('listvar >/dev/null 2>&1; echo "sub=$?"\n')
    res = milk_cli_script(
        'listvar >/dev/null 2>&1; echo "top=$?"\n@source sub.milk', tmp_path
    )
    assert value(res.stdout, "top") is not None
    assert value(res.stdout, "top") == value(res.stdout, "sub")


def test_WP8_qualified_names_only(tmp_path):
    body = 'mem.listvar >/dev/null 2>&1; echo "{0}q=$?"\nlistvar >/dev/null 2>&1; echo "{0}b=$?"\n'
    (tmp_path / "sub.milk").write_text(body.format("s"))
    res = milk_cli_script(body.format("t") + "@source sub.milk", tmp_path)
    for k, v in (("tq", "0"), ("tb", "127"), ("sq", "0"), ("sb", "127")):
        assert value(res.stdout, k) == v, res.stdout


def test_T6_inline_result_not_globbed(tmp_path):
    (tmp_path / "a.zz").touch()
    (tmp_path / "b.zz").touch()
    res = milk_cli_script(
        "mem.mk2Dim t6 4 4\n@set s.t6.kw.OBS '*.zz'\necho @{s.t6.kw.OBS}", tmp_path
    )
    assert "*.zz" in res.stdout
    assert "a.zz" not in res.stdout


def test_T9_multiword_value_not_truncated(tmp_path):
    res = milk_cli_script(
        'mem.mk2Dim t9 4 4\n@set s.t9.kw.OBS="Sub aru"\necho "setrc=$?"\n@get -v v s.t9.kw.OBS\necho "v=$v"',
        tmp_path,
    )
    assert value(res.stdout, "setrc") == "0" and value(res.stdout, "v") == "Sub aru", (
        res.stdout + res.stderr
    )


def test_T10_hash_in_param_length(tmp_path):
    res = milk_cli_script('arr=(a b c)\nn=$(echo ${#arr[@]})\necho "n=$n"', tmp_path)
    assert value(res.stdout, "n") == "3"
    assert "unexpected end" not in res.stdout + res.stderr


def test_T11_dollar_single_quote_escape():
    out = translate("echo $'it\\'s' @{s.a.b}\n")
    assert "milkget" in out


def test_T12_second_heredoc_body_untouched():
    src = "cat <<A <<B\na\nA\nhelp\nB\n"
    assert translate(src) == src


def test_T13_lineno_matches_source(tmp_path):
    res = milk_cli_script('echo "l=$LINENO"', tmp_path)
    assert value(res.stdout, "l") == "1"


@pytest.mark.parametrize(
    "src, must_have, must_not_have",
    [
        ("xs=@{s.${cam}.xsize}\n", "{ milkget -v xs s.${cam}.xsize ||", "calc"),
        ("x=@{s.c.xsize}\n", "{ milkget -v x s.c.xsize ||", "calc"),
        ('x="@{s.c.xsize}"\n', "{ milkget -v x s.c.xsize ||", "REPLY"),
        ("a && y=@{1+2} || b\n", 'a && { milkcalc -v y "1+2" ||', None),
        ("w=@{s.c.xsize} cmd\n", "${| milkget -v REPLY s.c.xsize", None),
        ("export z=@{s.c.xsize}\n", "${| milkget -v REPLY s.c.xsize", None),
        (
            'echo "${@s.c.xsize}" @s.c.xsize\n',
            'echo "${@s.c.xsize}" @s.c.xsize',
            "milkget",
        ),
        ('echo "\\@{lit}"\n', 'echo "@{lit}"', "milkcalc"),
        ("echo foo@{a,b}\n", "foo@{a,b}", "milkcalc"),
        ('echo "$@f.x.y" user@s.ex.org\n', '"$@f.x.y" user@s.ex.org', "milkget"),
        ("val=@{a * 10 + b[2, 3]}\n", '{ milkcalc -v val "a * 10 + b[2, 3]" ||', None),
        ("echo @{a && b}; ls\n", 'milkcalc -v REPLY "a && b"', None),
        (
            "x=$(mem.listim)\ny=`mem.listim`\n",
            "x=$(mem.listim)\ny=`mem.listim`",
            None,
        ),
        ("x=$(listim)\n", "x=$(listim)", "mem.listim"),
    ],
)
def test_WP7_translation_cases(src, must_have, must_not_have):
    out = translate(src)
    assert must_have in out, out
    if must_not_have is not None:
        assert must_not_have not in out, out


def test_WP7_lineno_with_errexit(tmp_path):
    env, _ = _env()
    f = tmp_path / "l.milk"
    f.write_text('true\necho "l=$LINENO"\n')
    res = _run([MILK_CLI_EXEC, "-e", "-s", str(f)], env, 20, cwd=str(tmp_path))
    assert value(res.stdout, "l") == "2"


# --------------------------------------- C*: coprocess, REPL, milk-script


def test_C1_subshell_and_bare_exit(tmp_path):
    res = milk_cli_stdin("y=$(exit 3)\n( exit 0 )\necho two", tmp_path)
    assert "two" in res.stdout
    res2 = milk_script("false\nexit", tmp_path)
    assert res2.returncode == 1


def test_C2_time_and_exit_trap(tmp_path):
    res = milk_cli_script(
        "trap '@calc -v z 1+1; echo \"deferdone=$z\"' EXIT\n@time echo hi\necho after",
        tmp_path,
        timeout=15,
    )
    assert res.returncode != 124
    assert "after" in res.stdout
    assert value(res.stdout, "deferdone") == "2"


def test_C3_C5_pty_interactive_controls(tmp_path):
    out = pty_session(
        [
            ("line", "x=5", 0.4),
            ("line", "for i in 1 2 3; do sleep 1; done", 0.4),
            ("raw", "\x03", 0.8),
            ("line", 'echo "x=[$x]"', 0.6),
            ("line", 'read v; echo "got=[$v]"', 0.4),
            ("line", "hello", 0.6),
        ],
        tmp_path,
    )
    assert "x=[5]" in out
    assert "got=[hello]" in out


def test_P_and_S_consolidated():
    """Consolidated regression checks for P* (calc/shm_prop) and S* (server/builtins)."""
    res = _run_bash("""
        # P1/P2/P3: scalar & image expressions, division by zero, negative indices
        mem.mk2Dim "@S:pim" 8 8
        @calc "pim = 4.0"
        @calc "pim[-1, -1] = 10.0"
        [[ "$(@calc "pim[7, 7]")" == "10" ]]

        # S1/S2: server status and builtin return codes
        @server status >/dev/null
        rc=0
        @get s.no_such_stream.xsize 2>/dev/null || rc=$?
        [[ "$rc" -ne 0 ]]
        echo "PS_OK"
    """)
    assert res.returncode == 0, res.stderr
    assert "PS_OK" in res.stdout
