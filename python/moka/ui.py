"""Colors and text layout for the ``moka`` and ``mokainit`` output. ASCII only.

NB: imports only click and the stdlib; keep it that way (CLI startup time).
"""

from __future__ import annotations

import logging
import os
import re
import sys
import threading
import time

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


class PlanHandler(logging.StreamHandler):
    """The CLI's stderr: ``moka.plan`` records (see plans.run) as step lines; others by LogFormatter.

    On a terminal, the line of a running step shows its seconds, rewritten every ``tick``.
    """

    tick = 1.0
    STEP_TAGS = {"done": OK, "already": DONE, "refused": REFUSED, "failed": FAILED}

    def __init__(self) -> None:
        super().__init__()  # the current sys.stderr
        self.setFormatter(LogFormatter())
        self.width = 0  # of the widest step label of the plan
        self.failed_plan = False  # a plan verdict said it failed
        self._open = ""  # the step line being counted, terminal only
        self._stop = threading.Event()

    def close(self) -> None:
        self._stop.set()
        if self._open:  # interrupted mid-step: leave the cursor on a fresh line
            self.stream.write("\n")
            self._open = ""
        super().close()

    def _count(self, stop: threading.Event, start: float) -> None:
        while not stop.wait(self.tick):
            with self.lock:  # emit() holds it too
                if not stop.is_set():
                    seconds = int(time.monotonic() - start)
                    self.stream.write(f"\r{self._open} {seconds} s")
                    self.flush()

    def emit(self, record: logging.LogRecord) -> None:
        self._stop.set()
        p = getattr(record, "moka", {"event": None})
        color = use_color(self.stream)
        end = "\r\x1b[K" if self._open and p["event"] == "end" else "\n"
        if self._open:
            self.stream.write(end)
            self._open = ""
        if p["event"] is None:
            return super().emit(record)
        line = highlight_ticks(p["text"], color)
        if p["event"] == "plan":
            self.width = max(map(len, p["labels"]), default=0)
            line = f"{p['pipeline']}: {p['state']}"
        elif p["event"] == "final":
            self.failed_plan = not p["ok"]
            line = f"{tag(OK if p['ok'] else FAILED, color)} {line}"
        else:
            dots = style("dim", "." * (self.width - len(p["label"]) + 3), color)
            prefix = f"  [{p['i']:>{len(str(p['n']))}}/{p['n']}] {p['label']} {dots}"
            if p["event"] == "start":
                if self.stream.isatty():
                    self._open, self._stop = prefix, threading.Event()
                    self.stream.write(prefix)
                    self.flush()
                    args = (self._stop, time.monotonic())
                    threading.Thread(target=self._count, args=args, daemon=True).start()
                return
            line = f"{prefix} {tag(self.STEP_TAGS[p['outcome']], color)} {line}"
        self.stream.write(line + "\n")
        self.flush()
