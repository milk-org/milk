"""milk-CTRL command line: usage errors exit 2, valid lines reach the tty check (gap-10, gap-15)."""

from __future__ import annotations

import subprocess

import pytest

from tests.cli.bash_integration_test import _install_dir

pytestmark = pytest.mark.timeout(30)


def _run(env, *args):
    return subprocess.run(
        [str(_install_dir() / "bin" / "milk-CTRL"), *args],
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=10,
    )


@pytest.mark.parametrize(
    "argv",
    [
        ["-d"],
        ["-T"],
        ["--theme"],
        ["--bogus"],
        ["-z"],
        ["one", "two"],  # one PATTERN only
        ["a("],  # PATTERN is an extended regex
        ["--view", "bogus"],
        ["--view"],
        ["--interval", "20"],
        ["--interval", "0.01"],
        ["--interval", "fast"],
        ["--debug"],  # needs --log
        ["-d", "/nonexistent/milk-shm"],
        ["-T", "nosuchtheme"],
        ["--theme=nosuchtheme"],
    ],
)
def test_usage_errors_exit_2(tui_env, argv):
    env, _ = tui_env
    r = _run(env, *argv)
    assert r.returncode == 2, r.stderr
    assert "Error:" in r.stderr and "-h for usage" in r.stderr
    if "unknown theme" in r.stderr:
        assert "dark" in r.stderr and "matrix" in r.stderr


@pytest.mark.parametrize(
    "argv",
    [
        ["-T", "dracula"],
        ["--theme=dracula"],
        ["--theme", "night"],
        ["-T", "RED"],
        ["-T", "hacker"],
        ["-T", "sol-light"],
        ["-T", "Paper"],
        ["--view", "loops", "wfs|dm"],
        ["--interval", "0.5", "--shm-dir", "/tmp"],
        ["--log", "/dev/null", "--debug"],
    ],
)
def test_valid_command_lines_pass_parsing(tui_env, argv):
    env, _ = tui_env
    r = _run(env, *argv)
    assert r.returncode == 1 and "interactive terminal required" in r.stderr, r.stderr


def test_shm_dir_option_passes_parsing(tui_env, tmp_path):
    env, _ = tui_env
    r = _run(env, "-d", str(tmp_path))
    assert r.returncode == 1 and "interactive terminal required" in r.stderr, r.stderr


def test_help_exits_0(tui_env):
    env, _ = tui_env
    r = _run(env, "-hm")
    assert r.returncode == 0 and "milk-CTRL" in r.stdout


def test_unknown_env_theme_is_reported(milkctrl):
    m = milkctrl(env_extra={"MILK_CTRL_THEME": "nosuchtheme"})
    m.wait_for(r"MILK_CTRL_THEME=nosuchtheme is not a theme")
    assert m.quit() == 0


def test_view_and_pattern_open_filtered(milkctrl, entities):
    """--view opens that view; PATTERN filters its list (here the streams)."""
    for s in ["keepme1", "dropme1"]:
        entities.make_stream(s)
    m = milkctrl("--view", "streams", "keep")
    m.wait_for(r"\u258c F3:STRM \u2590")
    m.wait_for(r"keepme1")
    assert "dropme1" not in m.screen()
    assert m.quit() == 0


def test_interval_sets_the_scan_rate(milkctrl):
    m = milkctrl("--interval", "0.5")
    m.wait_for(r"2\.0Hz")
    assert m.quit() == 0


def test_log_mirrors_the_command_log(milkctrl, tmp_path):
    """--log appends the command log; --debug adds key events and scan timing."""
    log = tmp_path / "ctrl.log"
    m = milkctrl("--log", str(log), "--debug")
    m.wait_for(r"F2:DASH")
    m.keys("d")
    m.wait_for(r"Compact mode ON")
    assert m.quit() == 0
    text = log.read_text()
    assert "INFO  Shared memory directory:" in text
    assert "INFO  Compact mode ON" in text
    assert "DEBUG key 100" in text and "DEBUG scan " in text
