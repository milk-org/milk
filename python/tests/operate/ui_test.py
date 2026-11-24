"""moka.ui: colors, tags, table, log formatter, and the --color flag."""

from __future__ import annotations

import logging
import subprocess
import sys

import click
import pytest
from click.testing import CliRunner

from moka import ui
from moka.cli import ErrstackGroup
from moka.operate.cli import moka

ESC = "\x1b"


class FakeStream:
    def __init__(self, tty: bool) -> None:
        self.tty = tty

    def isatty(self) -> bool:
        return self.tty


@pytest.fixture
def probe():
    """Run a throwaway command under an ErrstackGroup; return (output, use_color on a tty, on a pipe)."""

    def run(*flags: str):
        seen = []

        @click.group(cls=ErrstackGroup)
        def group():
            pass

        @group.command()
        def cmd():
            seen.append((ui.use_color(FakeStream(True)), ui.use_color(FakeStream(False))))
            click.echo(click.style("x", fg="red"))

        result = CliRunner().invoke(group, [*flags, "cmd"])
        assert result.exit_code == 0, result.output
        return result.output, seen[0]

    return run


def test_no_ansi_by_default(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    result = CliRunner().invoke(moka, ["--help"])
    assert result.exit_code == 0 and ESC not in result.output


def test_color_always_and_never(probe, monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    output, (tty, pipe) = probe("--color=always")
    assert tty and pipe and ESC in output
    output, (tty, pipe) = probe("--color=never")
    assert not tty and not pipe and ESC not in output
    output, (tty, pipe) = probe()
    assert tty and not pipe and ESC not in output


def test_no_color_env(probe, monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    assert not ui.use_color(FakeStream(True))
    assert probe()[1] == (False, False)
    assert probe("--color=always")[1] == (True, True)
    monkeypatch.setenv("NO_COLOR", "")
    assert ui.use_color(FakeStream(True))


def test_table_aligns_with_ansi_codes():
    rows = [["a", "bb", "end"], [click.style("aaa", fg="red"), "b", "x"]]
    lines = ui.table(rows)
    assert [click.unstyle(line) for line in lines] == ["a    bb  end", "aaa  b   x"]
    assert not click.unstyle(lines[1]).endswith(" ")


def test_table_cut_and_last_column_unpadded():
    rows = [["name", "a long last column"], ["n", "short"]]
    lines = ui.table(rows, max_width=12)
    assert lines == ["name  a l...", "n     short"]
    assert all(len(click.unstyle(line)) <= 12 for line in lines)
    styled = ui.table([["a", click.style("b" * 30, fg="red")]], max_width=10)
    assert styled == ["a  bbbb..."]
    assert ui.table(rows)[1] == "n     short"


def test_table_rows_of_different_length_and_no_trailing_space():
    rows = [["h", "DONE", "CAN", "WHY"], ["x", "a very long last cell"], ["y", "no", "yes", ""]]
    assert ui.table(rows) == [
        "h  DONE  CAN  WHY",
        "x  a very long last cell",  # the last cell of a row does not widen its column
        "y  no    yes",
    ]


def test_highlight_ticks():
    assert ui.highlight_ticks("run `deploy` now", False) == "run `deploy` now"
    styled = ui.highlight_ticks("run `deploy` now", True)
    assert ESC in styled and click.unstyle(styled) == "run `deploy` now"


def _record(level: int) -> logging.LogRecord:
    return logging.LogRecord("moka", level, "f", 1, "msg `x`", None, None)


def test_log_formatter_color_off(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    assert ui.LogFormatter().format(_record(logging.WARNING)) == "WARNING: msg `x`"
    assert ui.LogFormatter().format(_record(logging.INFO)) == "INFO: msg `x`"


def test_log_formatter_color_on(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setattr(sys, "stderr", FakeStream(True))
    warning = ui.LogFormatter().format(_record(logging.WARNING))
    error = ui.LogFormatter().format(_record(logging.ERROR))
    assert "\x1b[33m" in warning and "\x1b[31m" in error
    assert click.unstyle(warning) == "WARNING: msg `x`"
    assert ui.LogFormatter().format(_record(logging.INFO)).startswith("INFO: msg ")


def test_ascii_everywhere(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setattr(sys, "stderr", FakeStream(True))
    for status in ui.TAGS:
        assert len(status) == 6
        for color in (True, False):
            assert click.unstyle(ui.tag(status, color)) == status
            assert status.isascii()
    rows = [[ui.tag(ui.OK, True), "a"], [ui.tag(ui.FAILED, True), "b"]]
    assert all(click.unstyle(line).isascii() for line in ui.table(rows, max_width=5))
    for level in (logging.INFO, logging.WARNING, logging.ERROR):
        assert click.unstyle(ui.LogFormatter().format(_record(level))).isascii()


def test_operate_cli_import_does_not_load_ui():
    code = "import sys, moka.operate.cli; sys.exit('moka.ui' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
