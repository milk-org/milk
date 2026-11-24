"""Colors and text layout for the ``moka`` and ``mokainit`` output. ASCII only.

NB: imports only click and the stdlib; keep it that way (CLI startup time).
"""

from __future__ import annotations

import logging
import os
import re
import sys

import click

ROLES = {
    "ok": {"fg": "green"},
    "fail": {"fg": "red"},
    "warn": {"fg": "yellow"},
    "name": {"fg": "cyan", "bold": True},
    "opt": {"fg": "yellow"},
    "arg": {"fg": "cyan"},
    "dim": {"dim": True},
}
# tag -> role: ok, already done, refused, failed
TAGS = {"[ ok ]": "ok", "[ -- ]": "dim", "[ NO ]": "warn", "[FAIL]": "fail"}
OK, DONE, REFUSED, FAILED = TAGS


def use_color(stream) -> bool:
    """--color (via the click context) beats NO_COLOR, which beats the stream being a tty."""
    ctx = click.get_current_context(silent=True)
    if ctx is not None and ctx.color is not None:
        return ctx.color
    return not os.environ.get("NO_COLOR") and stream.isatty()


def style(role: str, text: str, color: bool) -> str:
    return click.style(text, **ROLES[role]) if color else text


def tag(status: str, color: bool) -> str:
    return style(TAGS[status], status, color)


def highlight_ticks(msg: str, color: bool) -> str:
    if not color:
        return msg
    return re.sub(r"`([^`]*)`", lambda m: style("name", m[0], True), msg)


def table(rows: list, max_width: int | None = None) -> list[str]:
    """Aligned lines (two spaces between columns), cut with ``...``.

    The last cell of each row is unpadded and does not set a column width, so rows may differ in length.
    """
    widths: list[int] = []
    for row in rows:
        for i, cell in enumerate(row[:-1]):
            if i == len(widths):
                widths.append(0)
            widths[i] = max(widths[i], len(click.unstyle(cell)))
    lines = []
    for row in rows:
        line = "  ".join(
            c + " " * (w - len(click.unstyle(c))) for c, w in zip(row[:-1], widths)
        )
        line = (line + ("  " if len(row) > 1 else "") + row[-1]).rstrip()
        if max_width is not None and len(click.unstyle(line)) > max_width:
            plain = click.unstyle(line)  # loses the colors of the cut line
            line = plain[: max_width - 3] + "..."
        lines.append(line)
    return lines


class LogFormatter(logging.Formatter):
    """``LEVEL: message``, warnings yellow and errors red when color is on."""

    def __init__(self) -> None:
        super().__init__("%(levelname)s: %(message)s")

    def format(self, record: logging.LogRecord) -> str:
        color = use_color(sys.stderr)
        record.message = highlight_ticks(record.getMessage(), color)
        line = self.formatMessage(record)
        if record.levelno >= logging.ERROR:
            return style("fail", line, color)
        return style("warn", line, color) if record.levelno == logging.WARNING else line
