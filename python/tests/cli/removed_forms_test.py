"""Removed forms fail with a hint naming their replacement, and the install keeps none of them.

Purpose:     an old script or habit gets its replacement in the error message, not "command not
             found" or a silent wait; nothing of the removed parts stays installed or linked.
Does:        the sequencer (milk-seq, `seq.*`, and the `q.` namespace in @get, @exists, @waitfor,
             @calc and @ls); the tmux-era unit interface: milk-cli's unit arguments (_RUNSTART_,
             ...), the fpslist.txt deployment and tmux scripts, the FIFO and tmux headers,
             fpsCTRL's command FIFO options, `p.X.tmux` (now `p.X.unit`); the stream logger, which
             ran in tmux, runs as a unit; removed calculator spellings (ln, fabs, urand, grand,
             nrm2, trunc, _PI, `if ... end`, `for ... end`, --tokens); the keywords @running and
             `@session log`, and the option letters that had two meanings (@calc -l, @restore -n,
             @log -a, @assert -e).
Covers:      the rename table (libmilkscript/core/milk_renames.c, `milkls --rename`), the `q.`
             hint (src/prop/milk_entity_ns.h), the unit-argument check
             (libmilkscript/core/CLIcore_checkargs_fps.c), src/tui/fpsCTRL/milk-fpsCTRL.c,
             src/prop/milk_shm_prop_proc.c (p.X.unit), the calculator's hints (src/calc/mc_func.c,
             front/mc_parse.c, front/mc_resolve.c), milk-streamFITSlog and milk-logshim-ctrl
             (src/coremods/COREMOD_memory).
Uses:        conftest.py: milk_bash, milk_cli_script, value, CAM, _install_dir;
             unit_lifecycle_test.py: FUNCS and the autouse no_worker_left fixture.
See also:    docs/arch/unit_lifecycle.md, "Removed, with a hint"; docs/guide/scripting.md,
             "Migrating Old Scripts".
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
import re

from milk.cliwrap import HAVE_CLI
from tests.cli.conftest import (
    _install_dir,
    milk_bash,
    milk_cli_script,
    value,
    CAM,
)
from tests.cli.unit_lifecycle_test import (
    FUNCS,
    no_worker_left,
)  # noqa: F401 (autouse fixture)

pytestmark = [
    pytest.mark.skipif(not HAVE_CLI, reason="MILK compiled without CLI support"),
    pytest.mark.timeout(120),
]

GONE = "the sequencer (milk-seq) was removed"
GONE_SCRIPTS = (
    "milk-fpsinit",
    "milk-fpsmkcmd",
    "milk-fpslist-addentry",
    "tmuxsessionname",
    "tmuxkillall",
)
GONE_HEADERS = (
    "fps_tmux.h",
    "fps_execFPScmd.h",
    "fps_process_fpsCMDarray.h",
    "fps_processcmdline.h",
    "fps_read_fpsCMD_fifo.h",
)


def test_seq_commands_fail_with_hint(tmp_path: Path):
    """seq.* exit 127 naming a milk script; milk-seq has the same entry (PATH may hold an older
    install's binary, so the table is asked directly)."""
    res = milk_bash(
        'seq.list; echo "list=$?"\n'
        'seq.start loop -f a.seq; echo "start=$?"\n'
        'echo "bin=$(milkls --rename milk-seq)"',
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert value(res.stdout, "list") == "127", out
    assert value(res.stdout, "start") == "127", out
    assert "milk: seq.list was removed; use: a milk script" in res.stderr, out
    assert "milk: seq.start was removed; use: a milk script" in res.stderr, out
    assert "a milk script" in value(res.stdout, "bin"), out


def test_q_lookups_fail_with_hint(tmp_path: Path):
    """q. lookups fail at once (no wait until the timeout); milk-cli prints the hint once, not once
    in the prompt and again in its server."""
    res = milk_cli_script(
        '@get q.x.status; echo "get=$?"\n'
        '@exists q.x; echo "exists=$?"\n'
        '@waitfor q.x.status=IDLE -t 5; echo "wait=$?"\n'
        '@calc "q.x.status + 1"; echo "calc=$?"\n'
        'seq.list; echo "list=$?"\n'
        '@help seq; echo "help=$?"',
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert value(res.stdout, "get") == "1", out
    assert value(res.stdout, "exists") == "1", out
    assert value(res.stdout, "wait") == "1", out
    assert value(res.stdout, "calc") not in (None, "0"), out
    assert value(res.stdout, "list") == "127", out
    assert value(res.stdout, "help") == "0", out
    assert res.stderr.count(f"milk: q.x.status: {GONE}") == 1, out
    assert f"milk: q.x.status=IDLE: {GONE}" in res.stderr, out
    assert f"undefined identifier 'q.x.status': {GONE}" in res.stderr, out
    assert "Migrating Old Scripts" in res.stdout, out


def test_ls_lists_no_sequencer(tmp_path: Path):
    """@ls ignores a leftover sequencer state file and has no sequencer column."""
    res = milk_bash(
        ': > "$MILK_SHM_DIR/milkseq.oldq.shm"\n'
        "@ls; @ls -l; @ls -1\n"
        "echo ---\n"
        "@ls --json\n"
        "echo ---\n"
        "@ls 'q.*'; echo \"q=$?\"",
        tmp_path,
    )
    out = res.stdout + res.stderr
    plain, js, rest = res.stdout.split("---\n")
    assert "oldq" not in plain and "oldq" not in js, out
    assert not [k for k in json.loads(js) if "seq" in k.lower()], out
    assert value(rest, "q") == "1", out
    assert f"milk: q.*: {GONE}" in res.stderr, out


def test_install_has_no_sequencer():
    """No milk-seq file in the install, and milk-cli, milkbash and milk-fpsCTRL do not link it."""
    inst = _install_dir()
    for rel in (
        "bin/milk-seq",
        "bin/milk-seq-help",
        "lib/libmilkfpsseq.so",
        "lib/libmilksequencer.so",
        "share/milk/catalog.d/milksequencer.tsv",
        "include/libfpsseq",
        "include/sequencer",
    ):
        assert not (inst / rel).exists(), rel
    if shutil.which("readelf") is None:
        pytest.skip("readelf not found")
    elves = [inst / "bin/milk-cli", inst / "lib/libmilkscript.so"]
    elves += [
        f
        for f in (inst / "lib/libmilkbash.so", inst / "bin/milk-fpsCTRL")
        if f.exists()
    ]
    for f in elves:
        dyn = subprocess.run(
            ["readelf", "-d", str(f)], capture_output=True, text=True, check=True
        ).stdout
        assert "NEEDED" in dyn, f
        assert "fpsseq" not in dyn and "sequencer" not in dyn, f


def test_cli_unit_arguments_fail_with_hint(tmp_path: Path):
    """milk-cli no longer hosts units (docs/arch/unit_lifecycle.md, U2): _RUNSTART_, _TMUXSTART_
    and the others fail, naming the fpsexec binary."""
    res = milk_bash(
        'mem.streamdelay _RUNSTART_ sd1; echo "run=$?"\n'
        'mem.streamdelay _TMUXSTART_ sd1; echo "tmux=$?"\n'
        'mem.streamdelay _CONFSTOP_ sd1; echo "conf=$?"',
        tmp_path,
    )
    out = res.stdout + res.stderr
    for key, arg in (
        ("run", "_RUNSTART_"),
        ("tmux", "_TMUXSTART_"),
        ("conf", "_CONFSTOP_"),
    ):
        assert value(res.stdout, key) == "1", out
        assert (
            f"milk: mem.streamdelay {arg} was removed: a unit runs from its fpsexec binary"
            in res.stderr
        ), out


def test_removed_scripts_and_headers(tmp_path: Path):
    """The deployment and tmux scripts and the FIFO and tmux headers are gone from the install; in a
    milk shell each script fails (127) naming its replacement (PATH limited to this install: an older
    one may still have them)."""
    script = 'PATH="$MILK_INSTALLDIR/bin:/usr/bin:/bin"; hash -r\n' + "".join(
        f'{s}; echo "{s}=$?"\n' for s in GONE_SCRIPTS
    )
    res = milk_bash(script, tmp_path)
    out = res.stdout + res.stderr
    for s in GONE_SCRIPTS:
        assert value(res.stdout, s) == "127", out
        assert f"milk: {s} was removed; use: " in res.stderr, out
        assert not (_install_dir() / "bin" / s).exists(), s
    for h in GONE_HEADERS:
        assert not list(_install_dir().glob(f"include/**/{h}")), h


def test_fpsctrl_fifo_options_fail(tmp_path: Path):
    """fpsCTRL's -f/--fifo and -s/--stdio went with the command FIFO
    (docs/arch/unit_lifecycle.md, U12): status 2, naming the tools to use instead."""
    res = milk_bash(
        'milk-fpsCTRL -f nofifo; echo "f=$?"\nmilk-fpsCTRL --stdio; echo "s=$?"',
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert value(res.stdout, "f") == "2", out
    assert value(res.stdout, "s") == "2", out
    assert (
        res.stderr.count("were removed with the command FIFO: use milk-fps-set") == 2
    ), out


def test_process_unit_property(tmp_path: Path):
    """processinfo's tmuxname is unitname: p.X.unit names the unit a process runs; p.X.tmux fails
    with a hint."""
    res = milk_bash(
        'mem.mk2Dim "@S:cam" 8 8 >/dev/null; @calc "cam = 2"\n'
        'milk-fpsexec-milk-calc -n pu1 -t cam -e "o1 = cam * 2" fpsinit >/dev/null\n'
        'milk-fps-runstart pu1 >/dev/null; echo "start=$?"\n'
        'echo "unit=$(@get p.pu1.unit)"\n'
        '@get p.pu1.tmux 2>tmux.log; echo "tmux=$?"\n'
        'echo "json=$(@get --json p.pu1 | grep -c \'"unit":"pu1"\')"\n'
        "milk-fps-runstop pu1 >/dev/null",
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert value(res.stdout, "start") == "0", out
    assert value(res.stdout, "unit") == "pu1", out
    assert value(res.stdout, "tmux") == "1", out
    assert value(res.stdout, "json") == "1", out
    assert (
        "p.pu1.tmux was renamed p.pu1.unit" in (tmp_path / "tmux.log").read_text()
    ), out


def test_stream_logger_is_a_unit(tmp_path: Path):
    """milk-streamFITSlog runs its logger as a unit (milk-fpsexec-mem-streamFITSlog): pstart starts
    its conf and run processes, on writes FITS cubes, rstop and rstart stop and restart the run
    process; the script's kill and milk-logshim-ctrl kill remove the unit. (The log directory is
    relative to $MILK_SHM_DIR/FITSlogger: an FPS string holds 63 characters.)"""
    lg = "streamFITSlog-lg"
    res = milk_bash(
        FUNCS + 'mem.mk2Dim "@S:lg" 8 8 >/dev/null\n'
        'milk-streamFITSlog -D logs -z 4 lg pstart >pstart.log 2>&1; echo "pstart=$?"\n'
        f'echo "run=$(@get f.{lg}.run)"; echo "conf=$(@get f.{lg}.conf)"\n'
        "milk-streamFITSlog lg on >on.log 2>&1\n"
        'for i in $(seq 10); do @calc "lg = $i" >/dev/null; sleep 0.05; done\n'
        f"@waitfor f.{lg}.filecnt=2 -t 10 >/dev/null\n"
        'echo "fits=$(ls "$U"/FITSlogger/logs/*.fits 2>/dev/null | wc -l)"\n'
        'milk-streamFITSlog lg rstop >rstop.log 2>&1; echo "rstop=$?"\n'
        f'echo "stopped=$(@get f.{lg}.run)"\n'
        'milk-streamFITSlog lg rstart >rstart.log 2>&1; echo "rstart=$?"\n'
        f'echo "restarted=$(@get f.{lg}.run)"\n'
        'milk-streamFITSlog lg kill >kill.log 2>&1; echo "kill=$?"\n'
        f'echo "left1=$(ls "$U" | grep -c "^{lg}\\.fps")"\n'
        'milk-streamFITSlog -D logs -z 4 lg pstart >pstart2.log 2>&1; echo "pstart2=$?"\n'
        'milk-logshim-ctrl kill lg >ctrlkill.log 2>&1; echo "ctrlkill=$?"\n'
        f'echo "left2=$(ls "$U" | grep -c "^{lg}\\.fps")"',
        tmp_path,
        timeout=90,
    )
    out = (
        res.stdout
        + res.stderr
        + "".join(
            f"\n--- {p.name}\n{p.read_text()}" for p in sorted(tmp_path.glob("*.log"))
        )
    )
    assert value(res.stdout, "pstart") == "0", out
    assert value(res.stdout, "run") == "running", out
    assert value(res.stdout, "conf") == "running", out
    assert value(res.stdout, "fits") == "2", out
    assert value(res.stdout, "rstop") == "0", out
    assert value(res.stdout, "stopped") == "exited", out
    assert value(res.stdout, "rstart") == "0", out
    assert value(res.stdout, "restarted") == "running", out
    assert value(res.stdout, "kill") == "0", out
    assert value(res.stdout, "left1") == "0", out
    assert value(res.stdout, "pstart2") == "0", out
    assert value(res.stdout, "ctrlkill") == "0", out
    assert value(res.stdout, "left2") == "0", out


def test_removed_spellings_name_their_replacement(tmp_path):
    res = milk_bash(
        CAM + 'for t in "ln(1)" "fabs(-1)" "rand() + urand()" "grand()" "nrm2(cam)" '
        '"trunc(5, 0, 3)" "_PI"; do @calc -v z "$t" 2>>e.txt; echo "rc=$?"; done\n'
        "@calc $'if 1\\nz = 2\\nend' 2>>e.txt; @calc 'for k = 1:2; z = k; end' 2>>e.txt\n"
        "for o in --tokens; do @calc $o 2>>e.txt; milk-fpsexec-milk-calc $o 2>>e.txt; done; true",
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert set(re.findall(r"^rc=(\d+)$", res.stdout, re.M)) == {"2"}, out
    err = (tmp_path / "e.txt").read_text()
    for hint in (
        "use log",
        "use abs",
        "use rand",
        "use randn",
        "use norm",
        "use clamp(x, lo, hi)",
        "use pi, exp(1)",
        "'if ... end' was removed",
        "'for ... end' was removed",
        "@calc: --tokens was removed; use: --list",
        "milk.calc: '--tokens' was removed: use --list",
    ):
        assert hint in err, err


def test_removed_keywords_and_option_letters(tmp_path):
    """@running and @session log name their replacement; a one-letter option that had another
    meaning in another keyword is refused with its long form, and nothing runs: @restore -n
    does not apply the snapshot, which --dry-run only lists.
    """
    res = milk_bash(
        CAM + '@running s.cam 2>>e.txt; echo "run=$?"; @exists -r s.cam; echo "ex=$?"\n'
        '@session log 2>>e.txt; echo "slog=$?"; @log --all >/dev/null; echo "log=$?"\n'
        '@log -a 2>>e.txt; echo "la=$?"; @assert -e s.cam 2>>e.txt; echo "ae=$?"\n'
        '@assert --exists s.cam; echo "aex=$?"; @calc -l none.c "1" 2>>e.txt; echo "cl=$?"\n'
        '@set s.cam.kw.G=1; @snapshot -o b.snap "s.cam*" >/dev/null; @set s.cam.kw.G=2\n'
        '@restore -n b.snap 2>>e.txt; echo "rn=$? $(@get s.cam.kw.G)"\n'
        '@restore --undo -n 2>>e.txt; echo "un=$? $(@get s.cam.kw.G)"\n'
        '@restore --dry-run b.snap >dry.txt; echo "dry=$? $(@get s.cam.kw.G)"\n'
        '@restore b.snap >/dev/null; echo "app=$? $(@get s.cam.kw.G)"',
        tmp_path,
    )
    out = res.stdout + res.stderr
    err = (tmp_path / "e.txt").read_text()
    got = {
        k: value(res.stdout, k)
        for k in "run ex slog log la ae aex cl rn un dry app".split()
    }
    assert got == {
        "run": "127",
        "ex": "0",
        "slog": "2",
        "log": "0",
        "la": "2",
        "ae": "2",
        "aex": "0",
        "cl": "2",
        "rn": "2 2",
        "un": "2 2",
        "dry": "0 2",
        "app": "0 1",
    }, (
        out + err
    )
    for hint in (
        "@running was removed; use: @exists -r",
        "@session: log is @log",
        "@log: -a was removed; use: --all",
        "@assert: -e was removed; use: --exists",
        "@calc: -l was removed; use: --lib",
        "@restore: -n was removed; use: --dry-run",
    ):
        assert hint in err, err
    assert err.count("@restore: -n was removed") == 2, err
