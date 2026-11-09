"""
Tab completion: bash and the milk-cli prompt use one completer (milk_complete), and
completing FPS names keeps one mapping of the FPS.
"""

from __future__ import annotations

import pytest

from milk.cliwrap import HAVE_CLI
from tests.cli.conftest import milk_bash, pty_session, value

pytestmark = [
    pytest.mark.skipif(not HAVE_CLI, reason="MILK compiled without CLI support"),
    pytest.mark.timeout(120),
]


def test_fps_completion_keeps_one_mapping(tmp_path):
    """Six TABs on `@get f.cmpl9.` leave at most one mapping of the FPS in any session process."""
    (tmp_path / "prog.calc").write_text("out9 = cam * 2\n")
    steps = [
        ("line", 'mem.mk2Dim "@S:cam" 8 8 >/dev/null', 0.5),
        (
            "line",
            "milk-fpsexec-milk-calc -n cmpl9 -t cam -f prog.calc fpsinit >/dev/null",
            1.5,
        ),
        ("raw", "@get f.cmpl9.", 0.3),
        *[("raw", "\t", 0.3)] * 6,
        ("raw", "\x15", 0.2),
        (
            "line",
            "m=0; for p in $(pgrep -s 0); do c=$(grep -c cmpl9.fps.shm /proc/$p/maps);"
            ' ((c > m)) && m=$c; done; echo "fpsmaps=$m"',
            1.0,
        ),
    ]
    out = pty_session(steps, tmp_path)
    assert value(out, "fpsmaps") == "1", out[-2000:]


def test_one_completer_for_bash_and_the_prompt(tmp_path):
    res = milk_bash(
        'echo "opt=$(milkcomplete --li "arg:@calc:1:@calc" | tr "\\n" " ")"\n'
        'echo "set=$(milkcomplete sy "arg:@config:1:@config" | tr "\\n" " ")"\n'
        'echo "pre=$(milkcomplete "" "arg:@config:3:preset" | tr "\\n" " ")"\n'
        'echo "fn=$(milkcomplete \'"a = sqr\' "arg:@calc:1:@calc")"',
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert value(res.stdout, "opt").split() == ["--lib", "--list"], out
    assert value(res.stdout, "set").split() == ["synhl", "syndiag"], out
    assert value(res.stdout, "pre").split() == ["minimal", "std", "ao"], out
    assert value(res.stdout, "fn") == '"a = sqrt(', out

    (tmp_path / "zzfile_abc.txt").write_text("hello-zz\n")
    steps = [
        ("raw", '@calc -v r "a1 = 2 + sqr', 0.3),
        ("raw", "\t", 0.5),
        ("line", '9)"; echo "r=$r"', 0.8),
        ("raw", 'cat "zzfile_a', 0.3),
        ("raw", "\t", 0.5),
        ("line", "", 0.5),
        ("line", 'mem.mk2Dim "@S:qcam" 4 4 >/dev/null', 0.6),
        ("raw", 'echo "xs=$(@get s.qcam.xsi', 0.3),
        ("raw", "\t", 0.5),
        ("line", ')"', 0.6),
        ("raw", 'echo "v=@{sqr', 0.3),
        ("raw", "\t", 0.5),
        ("line", '25)}"', 0.6),
    ]
    out = pty_session(steps, tmp_path)
    assert [value(out, k) for k in ("r", "xs", "v")] == ["5", "4", "5"], out[-2000:]
    assert "hello-zz" in out, out[-2000:]
