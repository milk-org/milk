"""
Phase 3 of the unit-lifecycle plan: the tmux API, milk-cli's unit arguments, the command FIFO, the
fpslist.txt deployment scripts and the tmux fields are removed. Each removed form fails with a hint
naming its replacement, the install keeps none of them, and the stream logger, which ran in tmux,
is a unit.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from milk.cliwrap import HAVE_CLI
from tests.cli.conftest import _install_dir, milk_bash, value
from tests.cli.unit_lifecycle_test import (
    FUNCS,
    no_worker_left,
)  # noqa: F401 (autouse fixture)

pytestmark = [
    pytest.mark.skipif(not HAVE_CLI, reason="MILK compiled without CLI support"),
    pytest.mark.timeout(120),
]

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


def test_cli_unit_arguments_fail_with_hint(tmp_path: Path):
    """milk-cli no longer hosts units (U2): _RUNSTART_, _TMUXSTART_ and the others fail, naming the
    fpsexec binary."""
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
    """fpsCTRL's -f/--fifo and -s/--stdio went with the command FIFO (U12): status 2, naming the tools
    to use instead."""
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
