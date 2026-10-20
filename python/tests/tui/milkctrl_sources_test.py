"""Source guards for milk-CTRL: approved symbols only, colours from the theme (ux-4, ux-17, arch-14)."""

from __future__ import annotations

import re
from pathlib import Path

OVERVIEW = Path(__file__).resolve().parents[3] / "src" / "cli" / "overview"

# Help-screen and parameter-type colour maps get theme slots in round 3
DEFERRED = (
    "overview_render_help",
    "overview_help_data",
    "fps_params",
    "detail_fps.c",
    "params_info",
)

EMOJI = re.compile(r"[\U0001F000-\U0001FFFF☀-✒✘-➿️ℹ⏸]")
ESCAPED_EMOJI = re.compile(r"\\xf0\\x9f|\\xe2\\x9a|\\xe2\\x84\\xb9|\\xe2\\x99")
NUMERIC_COLOUR = re.compile(r"ov_buf_(fg|bg)\(\s*[0-9]|\(ov_rgb_t\)\s*\{\s*[0-9]")


def _sources(sub=""):
    for ext in ("*.c", "*.h", "*.inc"):
        for p in (OVERVIEW / sub).rglob(ext):
            if "tests" not in p.parts:
                yield p


def test_no_emoji_in_overview():
    bad = []
    for p in _sources():
        for n, line in enumerate(p.read_text(errors="replace").splitlines(), 1):
            if EMOJI.search(line) or ESCAPED_EMOJI.search(line):
                bad.append(f"{p.name}:{n}: {line.strip()}")
    assert not bad, "\n".join(bad)


def test_no_numeric_colours_in_render():
    bad = []
    for p in _sources("render"):
        if any(d in p.name for d in DEFERRED) or p.name == "overview_theme.c":
            continue
        for n, line in enumerate(p.read_text().splitlines(), 1):
            if NUMERIC_COLOUR.search(line):
                bad.append(f"{p.name}:{n}: {line.strip()}")
    assert not bad, "\n".join(bad)
