"""
Tests for the Phase C syntax enhancements (plan §8, decisions D18-D27), one section per work package.
"""

from __future__ import annotations

import re

import pytest

from milk.cliwrap import HAVE_CLI
from tests.cli.bash_engine_review_test import (
    _env,
    _run,
    milk_cli_script,
    pty_session,
    value,
)
from tests.cli.bash_integration_test import _install_dir
from tests.cli.syntax_redesign_test import milk_bash

pytestmark = [
    pytest.mark.skipif(not HAVE_CLI, reason="MILK compiled without CLI support"),
    pytest.mark.timeout(60),
]


# ------------------------------------------------------------- S1: binding several values


def test_S1_get_pairs_and_several_specs(tmp_path):
    res = milk_cli_script(
        "mem.mk2Dim sa 5 3 >/dev/null\n"
        '@get nx=s.sa.xsize ny=s.sa.ysize; echo "p=$nx,$ny"\n'
        "echo \"l=$(@get s.sa.xsize s.sa.ysize | tr '\\n' ,)\"\n"
        '@get -v x s.sa.xsize s.sa.ysize 2>/dev/null; echo "two=$?"\n'
        '@get 1x=s.sa.xsize 2>/dev/null; echo "bad=$?"',
        tmp_path,
    )
    got = [value(res.stdout, k) for k in ("p", "l", "two", "bad")]
    assert got == ["5,3", "5,3,", "2", "2"], res.stdout + res.stderr


def test_S1_stream_stats_match_the_calculator(tmp_path):
    res = milk_cli_script(
        'mem.mk2Dim st 7 5 >/dev/null\n@calc "st = 0"\n@calc "st[1,2] = 9"\n@calc "st[3,4] = -2.5"\n'
        "@get -A g s.st.stats\n"
        "@calc -v mn,mx,me,sd,va,to,co "
        "'imin(st); imax(st); imean(st); istd(st); ivar(st); itot(st); icount(st)'\n"
        'echo "get=${g[min]} ${g[max]} ${g[mean]} ${g[std]} ${g[var]} ${g[tot]} ${g[count]}"\n'
        'echo "calc=$mn $mx $me $sd $va $to $co"\n'
        'echo "one=$(@get s.st.mean)"',
        tmp_path,
    )
    got = [float(x) for x in (value(res.stdout, "get") or "").split()]
    ref = [float(x) for x in (value(res.stdout, "calc") or "").split()]
    assert len(ref) == 7 and got == pytest.approx(ref, rel=1e-12), (
        res.stdout + res.stderr
    )
    assert float(value(res.stdout, "one")) == pytest.approx(ref[2], rel=1e-12)


def test_S1_calc_binds_the_last_statements(tmp_path):
    res = milk_cli_script(
        'mem.mk2Dim cb 4 4 >/dev/null\n@calc "cb = 2"\n'
        "@calc -v a,b 'itot(cb); imax(cb)'; echo \"ab=$a,$b\"\n"
        "@calc -v r,p 'r1 = itot(cb); p1 = imax(cb)'; echo \"rp=$r,$p\"\n"
        "@calc -v x 'y = 3'; echo \"x=$x\"\n"
        "@calc -v l 'q = 1; q + 41'; echo \"l=$l\"\n"
        "@calc -v u,v,w '1+2; 3*4'; echo \"uvw=[$u],$v,$w\"\n"
        '@calc -v t \'st = "ab"\'; echo "t=$t"\n'
        "@calc -v e 'imean(cb' 2>/dev/null; echo \"e=$? ${e-unset}\"",
        tmp_path,
    )
    got = [value(res.stdout, k) for k in ("ab", "rp", "x", "l", "uvw", "t", "e")]
    assert got == ["32,2", "32,2", "3", "42", "[],3,12", "ab", "2 unset"], (
        res.stdout + res.stderr
    )


def test_S1_calc_pixels_into_an_array(tmp_path):
    res = milk_cli_script(
        'mem.mk2Dim cp 3 2 >/dev/null\n@calc "cp = 1"\n@calc "cp[2,1] = 5"\n'
        "@calc -a px 'cp * 2'; echo \"px=${px[*]}\"\n"
        'mem.mk2Dim big 256 256 >/dev/null\n@calc "big = 0.125"\n'
        "@calc -a bp 'big'; echo \"n=${#bp[@]} last=${bp[65535]}\"",
        tmp_path,
    )
    assert value(res.stdout, "px") == "2 2 2 2 2 10", res.stdout + res.stderr
    assert value(res.stdout, "n") == "65536 last=0.125", res.stdout + res.stderr


@pytest.mark.parametrize("env", [{}, {"MILK_NO_BUILTINS": "1"}])
def test_S1_plain_bash_with_and_without_builtins(tmp_path, env):
    res = milk_bash(
        "mem.mk2Dim pb 4 2 >/dev/null\n@calc 'pb = 1'\n"
        "@get nx=s.pb.xsize; @calc -v a,b 'itot(pb); 1+1'; @calc -a p 'pb'\n"
        'echo "r=$nx $a $b ${#p[@]}"',
        tmp_path,
        env,
    )
    assert value(res.stdout, "r") == "4 8 2 8", res.stdout + res.stderr


# ------------------------------------------------------------- S2: scoped @defer


def events(out: str) -> list[str]:
    return re.findall(r"^D:(.*)$", out, re.M)


FRONTENDS = ["builtins", "no-builtins", "milk-cli"]


def run_in(frontend: str, script: str, tmp_path):
    if frontend == "milk-cli":
        return milk_cli_script(script, tmp_path)
    env = {"MILK_NO_BUILTINS": "1"} if frontend == "no-builtins" else {}
    return milk_bash(script, tmp_path, env)


@pytest.mark.parametrize("frontend", FRONTENDS)
def test_S2_defer_belongs_to_its_function_or_file(tmp_path, frontend):
    (tmp_path / "lib.sh").write_text("@defer echo D:s1\necho D:sbody\n")
    (tmp_path / "lib.milk").write_text("@defer echo D:m1\necho D:mbody\n")
    res = run_in(
        frontend,
        "@defer echo D:top1\n"
        "g() { @defer echo D:g1; echo D:gbody; }\n"
        "f() { local v=fv; @defer 'echo \"D:f1 $v $1\"'; g; @defer echo D:f2; echo D:fbody; return 3; }\n"
        'f arg; echo "D:rc=$?"\n'
        "h() { echo D:h0; source ./lib.sh; echo D:hend; }\n"
        "h\n"
        "k() { echo D:k0; source ./lib.sh; }\n"
        "k\n"
        "@source lib.milk; echo D:after-src\n"
        "main() { @defer echo D:main1; echo D:mainbody; }\n"
        "main\n"
        "q() { @defer --exit echo D:x1; @defer echo D:q1; }\n"
        "q\n"
        "p() { @defer echo D:p1; ( r() { @defer echo D:r1; }; r; @defer --exit echo D:sx ); echo D:pmid; }\n"
        "p\n"
        "echo D:last",
        tmp_path,
    )
    assert events(res.stdout) == [
        "gbody", "g1", "fbody", "f2", "f1 fv arg", "rc=3",
        "h0", "sbody", "hend", "s1",
        "k0", "sbody", "s1",
        "mbody", "m1", "after-src",
        "mainbody", "main1",
        "q1",
        "r1", "sx", "pmid", "p1",
        "last", "x1", "top1",
    ], res.stdout + res.stderr  # fmt: skip


@pytest.mark.parametrize("env", [{}, {"MILK_NO_BUILTINS": "1"}])
def test_S2_defer_after_set_e_failure_runs_at_exit(tmp_path, env):
    res = milk_bash(
        "trap 'echo D:user-exit' EXIT\n"
        "@defer echo D:session\n"
        "e() { @defer echo D:e1; false; echo D:not-here; }\n"
        "set -e\ne\necho D:not-reached",
        tmp_path,
        env,
    )
    assert events(res.stdout) == ["e1", "session", "user-exit"], res.stdout + res.stderr


@pytest.mark.parametrize("frontend", FRONTENDS)
def test_S2_restore_values_at_scope_end(tmp_path, frontend):
    res = run_in(
        frontend,
        "mem.mk2Dim rs 4 4 >/dev/null\n@set s.rs.kw.GAIN=1 s.rs.kw.MODE=closed\n"
        "h() { @set --restore s.rs.kw.GAIN=5; @defer --restore s.rs.kw.MODE; @set s.rs.kw.MODE=open; "
        'echo "D:in $(@get s.rs.kw.GAIN) $(@get s.rs.kw.MODE)"; }\n'
        'h; echo "D:out $(@get s.rs.kw.GAIN) $(@get s.rs.kw.MODE)"\n'
        "gl() { @defer --restore 's.rs.kw.*'; @set s.rs.kw.GAIN=7 s.rs.kw.MODE=x; }\n"
        'gl; echo "D:glob $(@get s.rs.kw.GAIN) $(@get s.rs.kw.MODE)"\n'
        "@set --restore 's.r*.kw.GAIN=2' 2>/dev/null; echo \"D:rc=$?\"",
        tmp_path,
    )
    assert events(res.stdout) == [
        "in 5 open",
        "out 1 closed",
        "glob 1 closed",
        "rc=2",
    ], (
        res.stdout + res.stderr
    )


@pytest.mark.parametrize("env", [{}, {"MILK_NO_BUILTINS": "1"}])
def test_S2_on_scoped_stops_with_its_function(tmp_path, env):
    res = milk_bash(
        "mem.mk2Dim ow 4 4 >/dev/null\n"
        'w() { @on --scoped s.ow true >/dev/null; echo "D:in $(@on --list | wc -l)"; }\n'
        'w; sleep 0.3; echo "D:out $(@on --list | wc -l)"',
        tmp_path,
        env,
    )
    assert events(res.stdout) == ["in 1", "out 0"], res.stdout + res.stderr


@pytest.mark.parametrize("env", [{}, {"MILK_NO_BUILTINS": "1"}])
def test_S2_unload_runs_the_entries_and_gives_back_the_exit_trap(tmp_path, env):
    res = milk_bash(
        "trap 'echo D:user-exit' EXIT\n"
        "@defer echo D:session1\n@defer --exit echo D:session2\n"
        '@unload; echo "D:unloaded $(trap -p EXIT | grep -c user-exit)"',
        tmp_path,
        env,
    )
    assert events(res.stdout) == ["session2", "session1", "unloaded 1", "user-exit"], (
        res.stdout + res.stderr
    )


def test_S2_defer_help_and_usage(tmp_path):
    res = milk_bash(
        '@defer --help; @defer --bogus 2>/dev/null; echo "D:rc=$?"', tmp_path
    )
    assert "@defer [--exit]" in res.stdout, res.stdout + res.stderr
    assert events(res.stdout) == ["rc=2"], res.stdout + res.stderr


@pytest.mark.parametrize("env", [{}, {"MILK_NO_BUILTINS": "1"}])
def test_S2_interactive_ctrl_c_runs_the_skipped_entries(tmp_path, env):
    """Interactive bash: an interrupted function's entries run before the next prompt."""
    full_env, _ = _env()
    full_env.update(env)
    script = (
        f'source "{_install_dir() / "bin" / "milk.bash"}"\n'
        "f() { @defer echo D:cleanup; kill -INT $$; echo D:after-kill; }\n"
        "f\necho D:next\n"
    )
    res = _run(
        ["/bin/bash", "--norc", "-i"], full_env, 30, stdin=script, cwd=str(tmp_path)
    )
    assert events(res.stdout) == ["cleanup", "next"], res.stdout + res.stderr


def test_S2_milk_cli_ctrl_c_runs_the_skipped_entries(tmp_path):
    out = pty_session(
        [
            (
                "line",
                'f() { @defer echo "D:clean$((6*7))"; sleep 5; echo "D:after$((1+1))"; }',
                1.0,
            ),
            ("line", "f", 1.5),
            ("raw", "\x03", 2.0),
            ("line", 'echo "D:done$((2+3))"', 2.0),
        ],
        tmp_path,
    )
    assert "D:clean42" in out and "D:done5" in out and "D:after2" not in out, out
    assert out.index("D:clean42") < out.index("D:done5"), out


# ------------------------------------------------------------- S3: calculator


def test_S3_coordinate_images(tmp_path):
    res = milk_cli_script(
        "mem.mk2Dim b 4 3 >/dev/null\nmem.mk3Dim cu 4 3 2 >/dev/null\n"
        "@calc -a px 'ix(b)'; echo \"ix=${px[*]}\"\n"
        "@calc -a py 'iy(b)'; echo \"iy=${py[*]}\"\n"
        "@calc -a pc 'ix(b, 0, 0) + 10 * iy(b, 0, 0)'; echo \"c00=${pc[*]}\"\n"
        "@calc 'rr = ir(b)'; @calc 'th = itheta(b)'\n"
        "@calc -v r,t1,t2 'rr[0, 0]; th[3, 1]; th[2, 2]'; echo \"pts=$r $t1 $t2\"\n"
        "@calc 'f2 = ir(b) + cu * 0'\n"  # broadcast: the normal path
        "@calc -v d 'itot(f2) - 2 * itot(rr)'; echo \"d=$d\"\n"
        "@calc -v cx 'itot(ix(cu))'; echo \"cx=$cx\"",
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert value(res.stdout, "ix") == "-2 -1 0 1 -2 -1 0 1 -2 -1 0 1", out
    assert value(res.stdout, "iy") == "-1 -1 -1 -1 0 0 0 0 1 1 1 1", out
    assert value(res.stdout, "c00") == "0 1 2 3 10 11 12 13 20 21 22 23", out
    pts = [float(x) for x in (value(res.stdout, "pts") or "").split()]
    assert pts == pytest.approx([5**0.5, 0.0, 3.14159265 / 2], rel=1e-6), out
    assert float(value(res.stdout, "d")) == pytest.approx(0, abs=1e-5), out
    assert value(res.stdout, "cx") == "-12", out


def test_S3_where_and_ternary_select_per_pixel(tmp_path):
    res = milk_cli_script(
        "mem.mk2Dim b 4 3 >/dev/null\n@calc 'b = ix(b)'\nmem.mk3Dim cu 4 3 2 >/dev/null\n@calc 'cu = 5'\n"
        "@calc -a w 'where(b > 0, sqrt(b), -1)'; echo \"where=${w[*]}\"\n"
        "@calc -a q 'b > 0 ? sqrt(b) : -1'; echo \"tern=${q[*]}\"\n"
        "@calc -a n 'where(b >= 0, sqrt(b), 0)'; echo \"nan=${n[*]}\"\n"
        "@calc -a s 'where(b > 0, b[0:3, 0:2], 0)'; echo \"slice=${s[*]}\"\n"
        "@calc -a c 'where(b > 0, cu, 0)'; echo \"bcast=${c[*]}\"\n"
        "@calc -v k '1 ? 2 : no_such_image'; echo \"k=$k\"",
        tmp_path,
    )
    out = res.stdout + res.stderr
    row = "-1 -1 -1 1"
    assert value(res.stdout, "where") == " ".join([row] * 3), out
    assert value(res.stdout, "tern") == " ".join([row] * 3), out
    assert value(res.stdout, "nan") == " ".join(["0 0 0 1"] * 3), out
    assert value(res.stdout, "slice") == " ".join(["0 0 0 1"] * 3), out
    assert value(res.stdout, "bcast") == " ".join(["0 0 0 5"] * 6), out
    assert value(res.stdout, "k") == "2", out


def test_S3_masked_reductions(tmp_path):
    res = milk_cli_script(
        "mem.mk2Dim b 4 3 >/dev/null\n@calc 'b = ix(b)'\nmem.mk3Dim cu 4 3 2 >/dev/null\n"
        "@calc -v m1,m2,m3,m4,m5 'imean(b, b > -1); itot(b, b > 0); imax(b, b < 0); "
        "imedian(b, b != 0); perc(b, 0.9, b != 0)'\n"
        'echo "m=$m1 $m2 $m3 $m4 $m5"\n'
        "@calc -v e1,e2 'imean(b, b > 5); itot(b, b > 5)'; echo \"empty=$e1 $e2\"\n"
        "@calc -v s1,s2 'istd(b, b < 1); ivar(b, b < 1)'; echo \"sv=$s1 $s2\"\n"
        "@calc -v x 'icount(b, b)' 2>/dev/null; echo \"bad1=$?\"\n"
        "@calc -v x 'imean(b, cu)' 2>/dev/null; echo \"bad2=$?\"",
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert value(res.stdout, "m") == "0.5 3 -1 -1 1", out
    assert value(res.stdout, "empty") in ("nan 0", "-nan 0"), out
    s1, s2 = (float(x) for x in (value(res.stdout, "sv") or "0 0").split())
    vals = [-2, -1, 0] * 3
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    assert (s1, s2) == pytest.approx((var**0.5, var), rel=1e-9), out
    assert value(res.stdout, "bad1") == "2" and value(res.stdout, "bad2") == "2", out


# ------------------------------------------------------------- S4: paced loops


@pytest.mark.parametrize("env", [{}, {"MILK_NO_BUILTINS": "1"}])
def test_S4_sleep_every_keeps_the_deadline_in_a_variable(tmp_path, env):
    res = milk_bash(
        "unset t; t0=$EPOCHREALTIME\n"
        "for ((i=0;i<20;i++)); do @sleep --hz 200 t; done\n"
        't1=$EPOCHREALTIME; echo "ms=$(( (${t1/./}-${t0/./})/1000 ))"\n'
        'd0=$t; @sleep --every 0.01 t; echo "step=$(( t - d0 ))"\n'
        'sleep 0.06; @sleep --every 0.01 --late L t; echo "late=$L"\n'
        'f() { local lt; @sleep --hz 1000 lt; echo "local=${lt:+set}"; }; f\n'
        '@sleep --every 0 t 2>/dev/null; echo "e1=$?"\n'
        '@sleep --every 0.1 2>/dev/null; echo "e2=$?"\n'
        'bad=xyz; @sleep --every 0.1 bad 2>/dev/null; echo "e3=$?"\n'
        '@sleep --hz 10 1x 2>/dev/null; echo "e4=$?"',
        tmp_path,
        env,
    )
    out = res.stdout + res.stderr
    if (
        not env
    ):  # the fallback runs milk-call for each tick, which takes longer than 5 ms
        assert 95 <= int(value(res.stdout, "ms") or 0) <= 160, out
    assert value(res.stdout, "step") == "10000000", out
    assert int(value(res.stdout, "late") or -1) >= 4, out
    assert value(res.stdout, "local") == "set", out
    assert [value(res.stdout, f"e{i}") for i in range(1, 5)] == ["2"] * 4, out


# ------------------------------------------------------------- S5: events


@pytest.mark.parametrize("env", [{}, {"MILK_NO_BUILTINS": "1"}])
def test_S5_waitfor_expression_wakes_on_frames(tmp_path, env):
    res = milk_bash(
        'mem.mk2Dim s.ws 8 8 >/dev/null; @calc "ws = 0"\n'
        '( for i in 1 2 3 4 5 6; do sleep 0.1; @calc "ws = $i"; done ) &\n'
        't0=$EPOCHREALTIME; @waitfor "imean(ws) > 3.5" -t 5; rc=$?; t1=$EPOCHREALTIME\n'
        'echo "w=$rc $(( (${t1/./}-${t0/./})/1000 ))"; wait\n'
        '@waitfor "imean(ws) > 100" -t 0.3; echo "timeout=$?"\n'
        '@waitfor --next "s.ws.cnt0 >= 0" -t 0.3; echo "next=$?"\n'
        '( sleep 0.1; @calc "ws = 9" ) & @waitfor --next "s.ws.cnt0 >= 0" -t 3; echo "nextframe=$?"; wait\n'
        '@waitfor "no_such_name_q > 1" -t 1 2>/dev/null; echo "err=$?"\n'
        'mem.mk2Dim s.w2 4 4 >/dev/null; @calc "w2 = 0"\n'
        '( sleep 0.1; @calc "w2 = 2" ) & @waitfor "imax(ws) > 8 && imax(w2) > 1" -t 3; echo "two=$?"; wait',
        tmp_path,
        env,
        timeout=60,
    )
    out = res.stdout + res.stderr
    rc, ms = (value(res.stdout, "w") or "x 0").split()
    assert rc == "0" and 350 <= int(ms) < 1500, out
    assert value(res.stdout, "timeout") == "124", out
    assert (
        value(res.stdout, "next") == "124" and value(res.stdout, "nextframe") == "0"
    ), out
    assert value(res.stdout, "err") == "2" and "Cannot open shm" not in out, out
    assert value(res.stdout, "two") == "0", out


@pytest.mark.parametrize("env", [{}, {"MILK_NO_BUILTINS": "1"}])
def test_S5_on_when_level_hold_every(tmp_path, env):
    res = milk_bash(
        'mem.mk2Dim s.ow 4 4 >/dev/null; @calc "ow = 0"\n'
        '@on --when "imean(ow) > 2.5" -- echo edge >> edge.txt\n'
        '@on --when "imean(ow) > 2.5" --level -- echo level >> level.txt\n'
        '@on --when "imean(ow) > 2.5" --hold 0.25 -- echo held >> held.txt\n'
        "@on --when 'imean(ow) > 2.5' --hold 2 -- echo never >> never.txt\n"
        "@on --every 0.05 -- echo tick >> tick.txt\n"
        'echo "listed=$(@on --list | grep -c when:)"\n'
        "sleep 0.3\n"
        'for v in 1 2 3 4 5 6 0 0 5; do @calc "ow = $v"; sleep 0.1; done\n'
        "sleep 0.5; @on --clear\n"
        'for f in edge level held never tick; do echo "$f=$(cat $f.txt 2>/dev/null | wc -l)"; done\n'
        '@on --every abc -- true 2>/dev/null; echo "bad1=$?"\n'
        '@on --level s.ow true 2>/dev/null; echo "bad2=$?"',
        tmp_path,
        env,
        timeout=60,
    )
    out = res.stdout + res.stderr
    got = {
        k: int(value(res.stdout, k) or -1)
        for k in ("edge", "level", "held", "never", "tick")
    }
    assert value(res.stdout, "listed") == "4", out
    assert (got["edge"], got["level"], got["held"], got["never"]) == (2, 5, 2, 0), out
    assert 15 <= got["tick"] <= 40, out
    assert value(res.stdout, "bad1") == "2" and value(res.stdout, "bad2") == "2", out


# ------------------------------------------------------------- S6: @args, user commands


def test_S6_args_binds_checked_locals(tmp_path):
    res = milk_bash(
        "m() { @args 'cam:str:camera' 'n:int=100:frames' 'g:float=0.5' 'on:bool=off' -- \"$@\"; "
        'echo "D:$cam $n $g $on"; }\n'
        "m s.cam; m s.cam .n=50; m s.cam . 0.3 on; m .g=2 s.cam 1e3\n"
        'm 2>/dev/null; echo "D:rc=$?"\n'
        'm s.cam abc 2>/dev/null; echo "D:rc=$?"\n'
        'm s.cam 1 2 on 4 2>/dev/null; echo "D:rc=$?"\n'
        'm s.cam .nope=1 2>/dev/null; echo "D:rc=$?"\n'
        'm -h >/dev/null; echo "D:help=$?"\n'
        "m -- .dotted\n"
        'f() { m s.c 7; echo "D:n=${n-unset}"; }; f\n'
        "@args 'x:int' -- 5; echo \"D:top=$x\"",
        tmp_path,
    )
    assert events(res.stdout) == [
        "s.cam 100 0.5 0", "s.cam 50 0.5 0", "s.cam 100 0.3 1", "s.cam 1000 2 0",
        "rc=2", "rc=2", "rc=2", "rc=2", "help=0",
        ".dotted 100 0.5 0", "s.c 7 0.5 0", "n=unset", "top=5",
    ], res.stdout + res.stderr  # fmt: skip


def test_S6_args_without_builtins(tmp_path):
    res = milk_bash(
        "m() { local cam n; @args 'cam:str' 'n:int=3' -- \"$@\" || return; echo \"D:$cam $n\"; }\n"
        'm a; m a 9; m 2>/dev/null; echo "D:rc=$?"; m a x 2>&1 | head -1',
        tmp_path,
        {"MILK_NO_BUILTINS": "1"},
    )
    assert events(res.stdout) == ["a 3", "a 9", "rc=2"], res.stdout + res.stderr
    assert "m: n, not an int: x" in res.stdout, res.stdout + res.stderr


def test_S6_args_in_a_script(tmp_path):
    from tests.cli.bash_engine_review_test import milk_script

    script = "@args 'n:int:a count' 'mode:str=fast' -- \"$@\"\necho \"D:$n $mode\""
    ok = milk_script(script, tmp_path, args=["5", "slow"])
    usage = milk_script(script, tmp_path, args=["x"])
    hlp = milk_script(script, tmp_path, args=["-h"])
    assert events(ok.stdout) == ["5 slow"], ok.stdout + ok.stderr
    assert usage.returncode == 2 and "n, not an int: x" in usage.stderr, usage.stderr
    assert hlp.returncode == 0 and "<n> [mode]" in hlp.stdout and not events(hlp.stdout)


@pytest.mark.parametrize("env", [{}, {"MILK_NO_BUILTINS": "1"}])
def test_S6_user_commands(tmp_path, env):
    d = tmp_path / "cfg" / "milk" / "commands.d"
    d.mkdir(parents=True)
    (d / "ao.bash").write_text(
        "# Reset the gain of a loop\n"
        "ao.reset() {\n"
        "    @args 'loop:str:loop name' 'gain:float=0.1' -- \"$@\" || return\n"
        '    echo "D:reset $loop $gain"\n'
        "}\n\n"
        "# Count frames\n"
        'function ao.count { @args \'n:int=10\' -- "$@" || return; echo "D:count $n"; }\n'
    )
    res = milk_bash(
        "ao.reset l1 .gain=0.2; ao.count\n"
        "@help ao.reset | head -2\n"
        '@help ao | grep -c "user command" | sed "s/^/D:ns=/"\n'
        '@help -k "Count" | grep -c "ao.count" | sed "s/^/D:k=/"\n'
        'complete -p ao.reset >/dev/null 2>&1 && echo "D:complete"\n'
        '@unload; type ao.reset >/dev/null 2>&1 || echo "D:unloaded"',
        tmp_path,
        {"XDG_CONFIG_HOME": str(tmp_path / "cfg"), **env},
    )
    out = res.stdout + res.stderr
    assert events(res.stdout) == [
        "reset l1 0.2", "count 10", "ns=2", "k=1", "complete", "unloaded",
    ], out  # fmt: skip
    assert "ao.reset - Reset the gain of a loop" in res.stdout, out
    assert "Usage: ao.reset <loop> [gain]" in res.stdout, out


# ------------------------------------------------------------- S7: undo


@pytest.mark.parametrize("env", [{}, {"MILK_NO_BUILTINS": "1"}])
def test_S7_restore_undo(tmp_path, env):
    res = milk_bash(
        "mem.mk2Dim s.ua 4 4 >/dev/null; mem.mk2Dim s.ub 4 4 >/dev/null\n"
        "@set s.ua.kw.G=1 s.ub.kw.G=10\n@set s.ua.kw.G=2\n@set 's.u*.kw.G'=3\n@set s.ub.kw.G=4\n"
        "@set f.nosuch.run=start 2>/dev/null\n"  # an action: not recorded
        '@restore --undo -n 2 | sed "s/^/D:dry /"\n'
        'echo "D:$(@get s.ua.kw.G) $(@get s.ub.kw.G)"\n'
        '@restore --undo; echo "D:$(@get s.ua.kw.G) $(@get s.ub.kw.G)"\n'
        '@restore --undo; echo "D:$(@get s.ua.kw.G) $(@get s.ub.kw.G)"\n'
        "__MILK_NOLOG=1 @set s.ua.kw.G=99\n"
        '@restore --undo 2>/dev/null; echo "D:rc=$? $(@get s.ua.kw.G)"\n'
        '@restore --undo 0 2>/dev/null; echo "D:rc=$?"\n'
        "h() { @set --restore s.ub.kw.G=50; }; h\n"
        '@restore --undo -n 1 | sed "s/^/D:dry /"',
        tmp_path,
        env,
    )
    assert events(res.stdout) == [
        "dry s.ub.kw.G: 4 -> 3",
        "dry s.ua.kw.G: 3 -> 2",
        "dry s.ub.kw.G: 3 -> 10  (now 4: left as is)",
        "3 4", "3 3", "2 10",
        "rc=1 99", "rc=2",
        "dry s.ub.kw.G: 50 -> 10  (now 10: left as is)",
    ], res.stdout + res.stderr  # fmt: skip
