"""How milk lines become bash: the @{...} translator and the bash co-process that runs them.

Purpose:     the translator rewrites only @{...} and its escaped form; heredocs, quotes,
             arithmetic, globbing and line numbers must reach bash as written, or scripts break
             in ways that are hard to trace.
Does:        checks that `<<` in arithmetic or in @{...} is not a heredoc; that strings,
             continuation lines and a second heredoc body stay as written; that `$'...'`,
             `${#...}`, a quoted @calc text and a `case` pattern that is a bash keyword are read
             as bash reads them; that an @{...} result is neither globbed nor split; that $LINENO
             matches the source, also with -e; that top-level and @source'd lines agree; the bash
             each @{...} form becomes; that git's `@{-1}` and an `@{` right after a letter are
             left alone.
Covers:      milk_translate_shorthand_to_bash() and the bash scanner it uses
             (src/cli/libmilkscript/bash/milk_bash_translate.c, milk_shell_lex.c), through
             `milk-call translate`; milk-cli -s and -e, which run the translated lines in their
             bash co-process.
Uses:        conftest.py: translate, milk_cli_script, _run_bash, _env, _run, value.
See also:    docs/guide/scripting.md, "The Language at a Glance"; bash_integration_test.py.
"""

from __future__ import annotations

import pytest

from milk.cliwrap import (
    HAVE_CLI,
    MILK_CLI_EXEC,
)
from tests.cli.conftest import _env, _run, _run_bash, milk_cli_script, translate, value

pytestmark = [
    pytest.mark.skipif(not HAVE_CLI, reason="MILK compiled without CLI support"),
    pytest.mark.timeout(40),
]


def test_shift_in_arith_is_not_heredoc():
    out = translate("echo $((1<<2))\necho @{1+2}\n")
    assert "milkcalc" in out.splitlines()[1]


def test_multiline_string_untouched():
    src = 'echo "x\n@ 1+2 in string\nmk2Dim a 1 1\n"\n'
    assert translate(src) == src


def test_continuation_not_new_statement():
    out = translate("echo a \\\n  mk2Dim b 1 1\n")
    assert "mem.mk2Dim" not in out


def test_calc_expression_not_split_on_operators(tmp_path):
    res = milk_cli_script(
        '@calc "x = (2 > 1) && (2 < 1)"\n@calc -v r "x"\necho "r=$r"', tmp_path
    )
    assert value(res.stdout, "r") == "0"


def test_top_level_and_sourced_agree(tmp_path):
    (tmp_path / "sub.milk").write_text('listvar >/dev/null 2>&1; echo "sub=$?"\n')
    res = milk_cli_script(
        'listvar >/dev/null 2>&1; echo "top=$?"\n@source sub.milk', tmp_path
    )
    assert value(res.stdout, "top") is not None
    assert value(res.stdout, "top") == value(res.stdout, "sub")


def test_inline_result_not_globbed(tmp_path):
    (tmp_path / "a.zz").touch()
    (tmp_path / "b.zz").touch()
    res = milk_cli_script(
        "mem.mk2Dim t6 4 4\n@set s.t6.kw.OBS '*.zz'\necho @{s.t6.kw.OBS}", tmp_path
    )
    assert "*.zz" in res.stdout
    assert "a.zz" not in res.stdout


def test_inline_entity_path_is_not_a_file_pattern(tmp_path):
    """`@{s.t7.kw.*}` reaches milk as written: bash must not match it against file names (with
    failglob the line failed), nor split a `$n` in it."""
    (tmp_path / "s.t7.kw.zz").touch()
    res = milk_cli_script(
        "shopt -s failglob\nmem.mk2Dim t7 4 4\n@set s.t7.kw.OBS night\n"
        'n="t7"\necho "all=@{s.t7.kw.*}"\necho one=@{s.$n.kw.OBS}',
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert value(res.stdout, "all") == "OBS", out  # kw.* lists the keyword names
    assert value(res.stdout, "one") == "night", out


def test_multiword_value_not_truncated(tmp_path):
    res = milk_cli_script(
        'mem.mk2Dim t9 4 4\n@set s.t9.kw.OBS="Sub aru"\necho "setrc=$?"\n@get -v v s.t9.kw.OBS\necho "v=$v"',
        tmp_path,
    )
    assert value(res.stdout, "setrc") == "0" and value(res.stdout, "v") == "Sub aru", (
        res.stdout + res.stderr
    )


def test_hash_in_param_length(tmp_path):
    res = milk_cli_script('arr=(a b c)\nn=$(echo ${#arr[@]})\necho "n=$n"', tmp_path)
    assert value(res.stdout, "n") == "3"
    assert "unexpected end" not in res.stdout + res.stderr


def test_dollar_single_quote_escape():
    out = translate("echo $'it\\'s' @{s.a.b}\n")
    assert "milkget" in out


def test_second_heredoc_body_untouched():
    src = "cat <<A <<B\na\nA\nhelp\nB\n"
    assert translate(src) == src


def test_lineno_matches_source(tmp_path):
    res = milk_cli_script('echo "l=$LINENO"', tmp_path)
    assert value(res.stdout, "l") == "1"


@pytest.mark.parametrize(
    "src, must_have, must_not_have",
    [
        ("xs=@{s.${cam}.xsize}\n", '{ milkget -v xs "s.${cam}.xsize" ||', "calc"),
        ("x=@{s.c.xsize}\n", '{ milkget -v x "s.c.xsize" ||', "calc"),
        ('x="@{s.c.xsize}"\n', '{ milkget -v x "s.c.xsize" ||', "REPLY"),
        ("a && y=@{1+2} || b\n", 'a && { milkcalc -v y "1+2" ||', None),
        ("w=@{s.c.xsize} cmd\n", '${| milkget -v REPLY "s.c.xsize"', None),
        ("export z=@{s.c.xsize}\n", '${| milkget -v REPLY "s.c.xsize"', None),
        ("echo @{s.dm*.xsize}\n", '${| milkget -v REPLY "s.dm*.xsize"', None),
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
def test_translation_cases(src, must_have, must_not_have):
    out = translate(src)
    assert must_have in out, out
    if must_not_have is not None:
        assert must_not_have not in out, out


def test_lineno_with_errexit(tmp_path):
    env, _ = _env()
    f = tmp_path / "l.milk"
    f.write_text('true\necho "l=$LINENO"\n')
    res = _run([MILK_CLI_EXEC, "-e", "-s", str(f)], env, 20, cwd=str(tmp_path))
    assert value(res.stdout, "l") == "2"


def test_shift_inside_inline_calc_is_not_heredoc():
    out = translate("x=@{1 << 4}\necho @{2 + 3}\n")
    assert 'milkcalc -v x "1 << 4"' in out
    assert 'milkcalc -v REPLY "2 + 3"' in out or 'milkcalc "2 + 3"' in out


def test_case_pattern_may_be_a_bash_keyword():
    res = _run_bash("""
        x="if"
        case $x in
            if) matched=1 ;;
            done) matched=2 ;;
        esac
        [[ "$matched" == "1" ]]
        echo "CASE_OK"
    """)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "CASE_OK" in res.stdout


@pytest.mark.timeout(120)
def test_integer_braces_are_left_to_git(tmp_path):
    """`@{-1}` and `@{3}` (git's forms) are passed to bash as written, in words and assignments. An `@{`
    right after a letter is not expanded either, unless the expansion is quoted."""
    res = milk_cli_script(
        "echo a=@{-1} @{3}\n"
        'v=@{-2}; echo "b=$v"\n'
        'w="@{-1}"; echo "c=$w"\n'
        "echo d=img@{1+1}\n"
        'echo e=img"@{1+1}"',
        tmp_path,
    )
    assert value(res.stdout, "a") == "@{-1} @{3}", res.stdout + res.stderr
    assert value(res.stdout, "b") == "@{-2}", res.stdout + res.stderr
    assert value(res.stdout, "c") == "@{-1}", res.stdout + res.stderr
    assert value(res.stdout, "d") == "img@{1+1}", res.stdout + res.stderr
    assert value(res.stdout, "e") == "img2", res.stdout + res.stderr
