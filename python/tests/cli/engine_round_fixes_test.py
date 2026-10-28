"""
Plan 'the-new-milk-cli-calc' follow-up round: synchronised input reads (P7) and the L items.

- P7: every stream a kept statement reads is covered by a torn-frame guard; `_torn` counts re-runs.
- L10: `imin`/`imax` skip NaN pixels in the block kernels and the axis reductions.
- L5: `s.<im>.stats` carries a `torn` flag.
- L7: piped `milk-cli` input logs every line to the history, and a batch ending in calc mode is
  not held back by an open parenthesis.
- L12: a kept scalar statement keeps its value across the text and kept paths (its type is not
  observable from a script: bash holds the value as a string).
- milk-cli and milk-script derive MILK_INSTALLDIR from their own path when it is unset, so modules,
  milk.bash and the units they launch come from the install the binary belongs to.
"""

from __future__ import annotations

import os

import pytest

from milk.cliwrap import HAVE_CLI
from tests.cli.conftest import _install_dir, _run, milk_bash, value

pytestmark = [
    pytest.mark.skipif(not HAVE_CLI, reason="MILK compiled without CLI support"),
    pytest.mark.timeout(120),
]

STOP = (
    'stop() { kill -INT "$1" 2>/dev/null; for _ in $(seq 60); do kill -0 "$1" 2>/dev/null || break;'
    ' sleep 0.05; done; wait "$1" 2>/dev/null; }\n'
)


def test_imin_imax_skip_nan(tmp_path):
    """A NaN pixel is skipped whatever its position and the image size; all-NaN gives NaN."""
    res = milk_bash(
        'mem.mk2Dim "@S:a" 8 8 >/dev/null; @calc "a = ix(a) + 10; a[0,0] = sqrt(-1)"\n'
        '@calc -v mn,mx "imin(a); imax(a)"; echo "small=$mn,$mx"\n'
        'mem.mk2Dim "@S:big" 1100 1000 >/dev/null\n'
        '@calc "big = ix(big) + 5000; big[0,0] = sqrt(-1); big[5,5] = sqrt(-1)"\n'
        '@calc -v mn,mx "imin(big); imax(big)"; echo "big=$mn,$mx"\n'
        '@calc "r = imin(a, 1); c = imax(a, 0)"; @calc -v r0,c0 "r[0]; c[0]"; echo "axis=$r0,$c0"\n'
        'mem.mk2Dim "@S:nn" 4 4 >/dev/null; @calc "nn = sqrt(-1)"\n'
        '@calc -v mn,mx "imin(nn); imax(nn)"; echo "allnan=$mn,$mx"',
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert value(res.stdout, "small") == "6,13", out  # ix is centred: -4+10 .. 3+10
    assert value(res.stdout, "big") == "4450,5549", out  # -550+5000 .. 549+5000
    assert value(res.stdout, "axis") == "6,13", out  # row 0 and column 0 hold the NaN
    assert all(
        v.lstrip("-") == "nan" for v in value(res.stdout, "allnan").split(",")
    ), out


def test_stats_listing_has_torn_flag(tmp_path):
    res = milk_bash(
        'mem.mk2Dim "@S:c" 8 8 >/dev/null; @calc "c = 3"\n'
        '@get -A st s.c.stats; echo "torn=${st[torn]} mean=${st[mean]}"\n'
        'echo "lines=$(@get s.c.stats | wc -l)"',
        tmp_path,
    )
    assert value(res.stdout, "torn") == "0 mean=3", res.stdout + res.stderr
    assert value(res.stdout, "lines") == "8", res.stdout + res.stderr


def test_piped_lines_logged_and_calc_batch_not_held(tmp_path):
    res = milk_bash(
        'rm -f "$HOME/.milk_history_log"\n'
        "printf 'echo one\\necho two\\n@calc\\nz = 2 + 3\\n@calc\\necho \"z=$z\"\\n' | milk-cli -q >/dev/null 2>&1\n"
        'echo "entries=$(grep -c $\'\\tP\\t\' "$HOME/.milk_history_log")"\n'
        "( printf 'echo one\\n@calc\\ny = (3\\n'; sleep 1; printf '@calc\\necho end\\n' ) | milk-cli -q > held.txt 2>&1 &\n"
        'sleep 0.5; echo "early=$(grep -m1 . held.txt)"; wait',
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert value(res.stdout, "entries") == "6", out
    # before: the open parenthesis of the calc line held the whole batch, `one` included, until EOF
    assert value(res.stdout, "early") == "one", out
    assert "end" in (tmp_path / "held.txt").read_text(), out


def test_torn_counter_with_fast_producer(tmp_path):
    """A consumer reading a stream another unit rewrites at full speed re-runs torn passes and counts them."""
    res = milk_bash(
        STOP
        + 'mem.mk2Dim "@S:src" 128 128 >/dev/null; mem.mk2Dim "@S:out" 128 128 >/dev/null; @calc "src = 1"\n'
        'milk-fpsexec-milk-calc -n prod -t src -e "src = src + 1" fpsinit >/dev/null\n'
        'milk-fpsexec-milk-calc -n cons -t out -p "torn:int=0" -e "out = src * 2 + 1; torn = _torn" fpsinit >/dev/null\n'
        "milk-fpsexec-milk-calc -n cons run >cons.log 2>&1 & pc=$!\n"
        "sleep 0.4; mem.imsetsempost out -1 >/dev/null; sleep 0.4\n"
        'echo "alone=$(@get f.cons.torn)"\n'
        "milk-fpsexec-milk-calc -n prod run >prod.log 2>&1 & pp=$!\n"
        "sleep 0.4; mem.imsetsempost src -1 >/dev/null; sleep 0.8\n"
        'echo "with=$(@get f.cons.torn)"; echo "error=[$(@get f.cons.unit.error)]"\n'
        "stop $pp; stop $pc",
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert value(res.stdout, "alone") == "0", out
    assert int(value(res.stdout, "with") or 0) > 0, out
    assert value(res.stdout, "error") == "[]", out


def test_kept_scalar_statement_value_is_stable(tmp_path):
    """The second run of a scalar statement (kept) gives the text path's value, integer or not."""
    res = milk_bash(
        'mem.mk2Dim "@S:b" 8 8 >/dev/null; @calc "b = 1; b[0,0] = 0"\n'
        '@calc "n = icount(b) + 1"; @calc -v v1 "n"; @calc "n = icount(b) + 1"; @calc -v v2 "n"\n'
        '@calc "d = icount(b) / 2"; @calc "d = icount(b) / 2"; @calc -v v3 "d"\n'
        '@calc "q = n > 3 && d < 100"; @calc "q = n > 3 && d < 100"; @calc -v v4 "q"\n'
        'echo "vals=$v1,$v2,$v3,$v4"',
        tmp_path,
    )
    assert value(res.stdout, "vals") == "64,64,31.5,1", res.stdout + res.stderr


def _bare_env(tmp_path):
    """No install variables, a PATH without any milk bin: only the executable's own path is known."""
    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(tmp_path),
        "MILK_SHM_DIR": str(tmp_path),
        "MILK_PROC_DIR": str(tmp_path),
        "MILK_QUIET": "1",
        "TMUX_TMPDIR": os.environ.get("TMUX_TMPDIR", "/tmp"),
    }
    return env


def test_binaries_locate_their_own_install(tmp_path):
    """Run by absolute path with MILK_INSTALLDIR unset and no milk bin on PATH, milk-cli and milk-script
    export their own install, load its modules and resolve units from its bin."""
    inst = _install_dir().resolve()
    f = tmp_path / "s.milk"
    f.write_text(
        'echo "dir=$MILK_INSTALLDIR"; echo "unit=$(which milk-fpsexec-milk-calc)"\n'
        'mem.mk2Dim "@S:selfloc" 2 2 >/dev/null; echo "mk=$?"\n'
    )
    res = _run(
        [str(inst / "bin" / "milk-cli"), "-q", "-s", str(f)],
        _bare_env(tmp_path),
        20,
        cwd=str(tmp_path),
    )
    out = res.stdout + res.stderr
    assert value(res.stdout, "dir") == str(inst), out
    assert value(res.stdout, "unit") == str(
        inst / "bin" / "milk-fpsexec-milk-calc"
    ), out
    assert value(res.stdout, "mk") == "0", out
    res = _run(
        [str(inst / "bin" / "milk-script"), "-q", str(f)],
        _bare_env(tmp_path),
        20,
        cwd=str(tmp_path),
    )
    out = res.stdout + res.stderr
    assert value(res.stdout, "dir") == str(inst), out
    assert value(res.stdout, "unit") == str(
        inst / "bin" / "milk-fpsexec-milk-calc"
    ), out
    # a value set by the caller is kept
    env = _bare_env(tmp_path)
    env["MILK_INSTALLDIR"] = "/nonexistent"
    res = _run(
        [str(inst / "bin" / "milk-script"), "-q", str(f)], env, 20, cwd=str(tmp_path)
    )
    assert value(res.stdout, "dir") == "/nonexistent", res.stdout + res.stderr
