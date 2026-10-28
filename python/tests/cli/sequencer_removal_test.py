"""
Phase 1b of the unit-lifecycle plan (U19): milk-seq is removed. Its commands and the `q.` namespace
fail with a hint naming the scripting replacement (§7 item 34), and the install neither has nor
links the sequencer (item 35).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from milk.cliwrap import HAVE_CLI
from tests.cli.conftest import _install_dir, milk_bash, milk_cli_script, value

pytestmark = [
    pytest.mark.skipif(not HAVE_CLI, reason="MILK compiled without CLI support"),
    pytest.mark.timeout(60),
]

GONE = "the sequencer (milk-seq) was removed"


def test_seq_commands_fail_with_hint(tmp_path: Path):
    """Item 34: seq.* exit 127 naming a milk script; milk-seq has the same entry (PATH may hold an
    older install's binary, so the table is asked directly)."""
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
    """Item 34: q. lookups fail at once (no wait until the timeout); milk-cli prints the hint once,
    not once in the prompt and again in its server."""
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
    """Item 34: @ls ignores a leftover sequencer state file and has no sequencer column."""
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
    """Item 35: no milk-seq file in the install, and milk-cli, milkbash and milk-fpsCTRL do not link
    it."""
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
