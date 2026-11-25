#!/usr/bin/env python3
"""Run the moka tutorial page's own blocks and rewrite its outputs and images.

The page (``docs/quickstart/moka_tutorial.md``) is the source of the commands: an HTML
comment before a fenced block or an image link says what to do with it. The block or link
follows the marker directly or after one blank line (the mdformat hook inserts it); two
or more blank lines, or other content in between, is an error.

    <!-- tut:run id=ID [expect=N] -->   bash block: run it, exit code must be N (default 0)
    <!-- tut:out id=ID -->              text block: receives the output of tut:run ID
    <!-- tut:py id=ID -->               python block: run it, no output kept
    <!-- tut:tui id=ID cmd="..." size=COLSxROWS -->
                                        image link: capture cmd in a tmux pane, as SVG
    <!-- tut:fig id=ID -->              image link: mlat, rm or cm figure, as PNG
    <!-- tut:manual [id=ID] -->         any block: not run (ID may name an action below)

Modes:
    --check         no live system: markers, ids and image files
    --run [--from ID]
                    the full capture (milk, cacao, tmux, moka); see the docs-writer skill
    --callouts IMG  draws the numbered discs of the opening image onto IMG, saves desktop.png

Python 3.10+, stdlib only for --check; --run also needs moka (pyMilk, matplotlib) and
--callouts needs Pillow.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from xml.sax.saxutils import escape

REPO = Path(__file__).resolve().parents[2]
PAGE = REPO / "docs/quickstart/moka_tutorial.md"
IMG_REL = "img/moka_tutorial"
WORKDIR_NAME = "AOloop2"
PIPELINES = ("aoloop", "mlatsim")
TMUX = ("tmux", "-L", "milk_tut", "-f", "/dev/null")
# Seconds a TUI is given to draw before its pane is captured.
TUI_WAIT = 3.0
# The confdir edit of the "Change the configuration" stage (tut:manual id=edit-loopgain).
LOOPGAIN_EDIT = "0.3"

KINDS = {"run", "out", "py", "tui", "fig", "manual"}
ATTRS = {
    "run": {"id", "expect"},
    "out": {"id"},
    "py": {"id"},
    "tui": {"id", "cmd", "size"},
    "fig": {"id"},
    "manual": {"id"},
}
# The fenced-block language each kind must precede; None: an image link.
TARGET = {"run": "bash", "out": "text", "py": "python", "tui": None, "fig": None}
CHECKED_LANGS = {"bash", "python"}
FIGURES = {"mlat", "rm", "cm"}

# Position (fractions of the image) of the top-left corner of each numbered disc.
CALLOUTS = [
    (1, 0.02, 0.02),
    (2, 0.03, 0.26),
    (3, 0.02, 0.60),
    (4, 0.07, 0.92),
    (5, 0.48, 0.05),
    (6, 0.50, 0.42),
    (7, 0.49, 0.70),
    (8, 0.29, 0.48),
    (9, 0.68, 0.03),
    (10, 0.69, 0.51),
]

MARKER_RE = re.compile(r"^\s*<!--\s*tut:(\S+)(.*?)\s*-->\s*$")
FENCE_RE = re.compile(r"^(\s*)(`{3,}|~{3,})\s*([^\s`]*)(.*)$")
IMAGE_RE = re.compile(r"!\[[^\]]*\]\(\s*<?([^)\s>]+)")


class CaptureError(Exception):
    """A page or run error; the page is left untouched."""


# ---------------------------------------------------------------- parsing


@dataclasses.dataclass
class Fence:
    lang: str
    indent: str
    # Line indexes of the opening and closing fence lines.
    start: int
    end: int


@dataclasses.dataclass
class Marker:
    kind: str
    attrs: dict
    line: int
    fence: Fence | None = None
    image: str | None = None

    @property
    def id(self) -> str | None:
        return self.attrs.get("id")


@dataclasses.dataclass
class Page:
    lines: list
    markers: list
    fences: list
    images: list
    errors: list


def parse(text: str) -> Page:
    """Markers, fenced blocks and image links of a page; ``errors`` lists marker syntax."""
    lines = text.splitlines(keepends=True)
    fences, images, raw, errors = [], [], [], []
    i = 0
    while i < len(lines):
        line = lines[i].rstrip("\r\n")
        m = FENCE_RE.match(line)
        if m:
            indent, mark, lang = m.group(1), m.group(2), m.group(3)
            close = re.compile(rf"^\s*{mark[0]}{{{len(mark)},}}\s*$")
            j = i + 1
            while j < len(lines) and not close.match(lines[j].rstrip("\r\n")):
                j += 1
            fences.append(Fence(lang.lower(), indent, i, j))
            i = j + 1
            continue
        m = MARKER_RE.match(line)
        if m:
            raw.append((i, m.group(1), m.group(2)))
        for path in IMAGE_RE.findall(line):
            images.append((i, path))
        i += 1

    by_start = {f.start: f for f in fences}
    image_at = {ln: p for ln, p in images}
    markers = []
    for ln, kind, rest in raw:
        where = f"line {ln + 1}"
        if kind not in KINDS:
            errors.append(f"{where}: unknown marker tut:{kind}")
            continue
        try:
            words = shlex.split(rest)
        except ValueError as e:
            errors.append(f"{where}: {e}")
            continue
        attrs = {}
        for w in words:
            key, eq, value = w.partition("=")
            if not eq or key not in ATTRS[kind]:
                errors.append(f"{where}: tut:{kind} does not take {w!r}")
            else:
                attrs[key] = value
        # The block or image link follows the marker directly or after one blank line.
        t = ln + 1
        if t < len(lines) and not lines[t].strip():
            t += 1
        mk = Marker(kind, attrs, ln, by_start.get(t), image_at.get(t))
        markers.append(mk)
        if kind != "manual" and not mk.id:
            errors.append(f"{where}: tut:{kind} has no id")
        if kind == "run" and not attrs.get("expect", "0").isdigit():
            errors.append(f"{where}: expect must be an integer")
        if kind == "tui":
            if "cmd" not in attrs:
                errors.append(f"{where}: tut:tui has no cmd")
            if not re.fullmatch(r"\d+x\d+", attrs.get("size", "")):
                errors.append(f"{where}: tut:tui needs size=COLSxROWS")
    return Page(lines, markers, fences, images, errors)


# ---------------------------------------------------------------- --check


def check_page(text: str, page_dir: Path) -> list[str]:
    """Errors of a page; empty when ``--run`` can use it."""
    page = parse(text)
    errors = list(page.errors)
    marked = {}
    for mk in page.markers:
        if mk.fence is not None:
            marked.setdefault(mk.fence.start, []).append(mk)
        where = f"line {mk.line + 1}"
        if mk.kind in TARGET:
            lang = TARGET[mk.kind]
            if lang is None and mk.image is None:
                errors.append(
                    f"{where}: tut:{mk.kind} is not followed by an image link"
                )
            elif lang is not None and (mk.fence is None or mk.fence.lang != lang):
                errors.append(
                    f"{where}: tut:{mk.kind} is not followed by a {lang} block"
                )
        if mk.kind in ("tui", "fig") and mk.image is not None and mk.id:
            ext = "svg" if mk.kind == "tui" else "png"
            want = f"{IMG_REL}/{mk.id}.{ext}"
            if mk.image != want:
                errors.append(f"{where}: image link is {mk.image}, expected {want}")
        if mk.kind == "fig" and mk.id and mk.id not in FIGURES:
            errors.append(
                f"{where}: no figure named {mk.id} (one of {sorted(FIGURES)})"
            )

    for f in page.fences:
        if f.lang in CHECKED_LANGS and f.start not in marked:
            errors.append(f"line {f.start + 1}: {f.lang} block without marker")

    seen, outs = {}, {}
    for mk in page.markers:
        if mk.kind == "manual" or not mk.id:
            continue
        table = outs if mk.kind == "out" else seen
        if mk.id in table:
            errors.append(
                f"line {mk.line + 1}: duplicate id {mk.id} (line {table[mk.id] + 1})"
            )
        else:
            table[mk.id] = mk.line
    run_ids = {mk.id for mk in page.markers if mk.kind == "run"}
    for oid, ln in outs.items():
        if oid not in run_ids:
            errors.append(f"line {ln + 1}: tut:out id={oid} has no tut:run")

    for ln, path in page.images:
        if "://" in path or path.startswith("/"):
            continue
        if not (page_dir / path).is_file():
            errors.append(f"line {ln + 1}: missing image {path}")
    return errors


# ---------------------------------------------------------------- pure helpers


def normalize(text: str, home: Path) -> str:
    """$HOME to ~; nothing else."""
    return text.replace(str(home), "~")


def rewrite(text: str, outputs: dict) -> str:
    """The page with each ``tut:out`` block whose id is in ``outputs`` replaced.

    Only the lines between the fences change; the block keeps its fence indentation.
    """
    page = parse(text)
    lines = list(page.lines)
    targets = [mk for mk in page.markers if mk.kind == "out" and mk.id in outputs]
    # Bottom-up, so earlier line indexes stay valid.
    for mk in sorted(targets, key=lambda m: m.line, reverse=True):
        f = mk.fence
        if f is None:
            continue
        body = outputs[mk.id].strip("\n")
        new = [(f.indent + ln if ln else "") + "\n" for ln in body.split("\n")]
        if not body:
            new = []
        lines[f.start + 1 : f.end] = new
    return "".join(lines)


def start_refusals(home: Path, shm_dir: Path) -> list[str]:
    """Why ``--run`` must not start: an existing workdir, FPS or stream."""
    reasons = []
    if (home / WORKDIR_NAME).exists():
        reasons.append(f"{home / WORKDIR_NAME} exists")
    for pattern in ("mlatsim_*.fps.shm", "aoloop_*.fps.shm", "dm07disp*.im.shm"):
        for p in sorted(shm_dir.glob(pattern)):
            reasons.append(f"{p} exists")
    return reasons


# ---------------------------------------------------------------- ANSI to SVG

_BASE16 = [
    "#000000", "#cd0000", "#00cd00", "#cdcd00", "#0000ee", "#cd00cd", "#00cdcd", "#e5e5e5",
    "#7f7f7f", "#ff0000", "#00ff00", "#ffff00", "#5c5cff", "#ff00ff", "#00ffff", "#ffffff",
]  # fmt: skip
_FG, _BG = "#d0d0d0", "#1c1c1c"
_SGR_RE = re.compile(r"\x1b\[([0-9;:]*)([A-Za-z])|\x1b[()][A-Za-z0-9]|\x1b.")


def _xterm256(n: int) -> str:
    if n < 16:
        return _BASE16[n]
    if n < 232:
        n -= 16
        levels = [0, 95, 135, 175, 215, 255]
        r, g, b = levels[n // 36], levels[(n // 6) % 6], levels[n % 6]
        return f"#{r:02x}{g:02x}{b:02x}"
    v = 8 + 10 * (n - 232)
    return f"#{v:02x}{v:02x}{v:02x}"


def _apply_sgr(params: list, st: dict) -> None:
    i = 0
    while i < len(params):
        p = params[i]
        if p == 0:
            st.update(fg=None, bg=None, bold=False, reverse=False)
        elif p == 1:
            st["bold"] = True
        elif p == 22:
            st["bold"] = False
        elif p == 7:
            st["reverse"] = True
        elif p == 27:
            st["reverse"] = False
        elif 30 <= p <= 37 or 90 <= p <= 97:
            st["fg"] = _BASE16[p - 30 if p < 90 else p - 82]
        elif 40 <= p <= 47 or 100 <= p <= 107:
            st["bg"] = _BASE16[p - 40 if p < 100 else p - 92]
        elif p == 39:
            st["fg"] = None
        elif p == 49:
            st["bg"] = None
        elif p in (38, 48):
            key = "fg" if p == 38 else "bg"
            mode = params[i + 1] if i + 1 < len(params) else None
            if mode == 5 and i + 2 < len(params):
                st[key] = _xterm256(params[i + 2] % 256)
                i += 2
            elif mode == 2 and i + 4 < len(params):
                r, g, b = (min(v, 255) for v in params[i + 2 : i + 5])
                st[key] = f"#{r:02x}{g:02x}{b:02x}"
                i += 4
            else:
                i = len(params)
        i += 1


def ansi_to_svg(ansi: str, cols: int, rows: int) -> str:
    """SVG of a ``tmux capture-pane -e`` text: one monospace cell per character.

    Handles SGR 16/256/truecolor, bold, reverse and reset; other escapes are dropped.
    A single replaceable function: the rest of the script only calls it.
    """
    cw, lh, fs, pad = 8.4, 17.0, 14, 8
    st = {"fg": None, "bg": None, "bold": False, "reverse": False}
    bgs, texts = [], []
    for y, line in enumerate(ansi.split("\n")[:rows]):
        runs, x, pos = [], 0, 0
        for m in list(_SGR_RE.finditer(line)) + [None]:
            chunk = line[pos : m.start() if m else len(line)]
            chunk = "".join(c for c in chunk if c == "\t" or ord(c) >= 0x20)
            if chunk:
                fg, bg = st["fg"] or _FG, st["bg"]
                if st["reverse"]:
                    fg, bg = bg or _BG, fg
                runs.append((x, chunk, fg, bg, st["bold"]))
                x += len(chunk)
            if m is None:
                break
            if m.group(2) == "m":
                ps = re.split(r"[;:]", m.group(1))
                _apply_sgr([int(p) if p.isdigit() else 0 for p in ps], st)
            pos = m.end()
        ty = pad + y * lh
        for x0, chunk, fg, bg, bold in runs:
            px = pad + x0 * cw
            if bg is not None:
                bgs.append(
                    f'<rect x="{px:.1f}" y="{ty:.1f}" width="{len(chunk) * cw:.1f}" '
                    f'height="{lh:.1f}" fill="{bg}"/>'
                )
            if chunk.strip():
                weight = ' font-weight="bold"' if bold else ""
                texts.append(
                    f'<text x="{px:.1f}" y="{ty + lh * 0.78:.1f}" fill="{fg}"{weight} '
                    f'textLength="{len(chunk) * cw:.1f}">{escape(chunk)}</text>'
                )
    width, height = 2 * pad + cols * cw, 2 * pad + rows * lh
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" '
            f'height="{height:.0f}" viewBox="0 0 {width:.0f} {height:.0f}" '
            f'font-family="DejaVu Sans Mono, Menlo, Consolas, monospace" '
            f'font-size="{fs}" xml:space="preserve">',
            f'<rect width="100%" height="100%" fill="{_BG}"/>',
            *bgs,
            *texts,
            "</svg>",
            "",
        ]
    )


# ---------------------------------------------------------------- live system
"""
Everything below touches the live system (shell, tmux, moka). Each call to the outside
is a small function so it can be adjusted after the verification run on the machine.
"""


def shm_dir() -> Path:
    return Path(os.environ.get("MILK_SHM_DIR", "/milk/shm"))


class Shell:
    """One bash process for all blocks, so ``cd`` and variables carry over."""

    def __init__(self, cwd: Path):
        env = dict(os.environ, NO_COLOR="1")
        self.tmp = Path(tempfile.mkdtemp(prefix="tut_capture_"))
        self.proc = subprocess.Popen(
            ["bash", "--noprofile", "--norc"],
            cwd=cwd,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        # Records the first failing command's code without stopping the shell.
        self._send(
            "set -E; trap 'tut_r=$?; [ \"$tut_rc\" -eq 0 ] && tut_rc=$tut_r' ERR\n"
        )

    def _send(self, s: str) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(s)
        self.proc.stdin.flush()

    def run(self, code: str) -> tuple[int, str]:
        """Exit code (first failing command, else 0) and stdout+stderr of ``code``."""
        script = self.tmp / "block.sh"
        script.write_text(code)
        tag = f"__TUT_DONE_{uuid.uuid4().hex}__"
        self._send(f"tut_rc=0; source {shlex.quote(str(script))} < /dev/null\n")
        self._send(f"printf '\\n%s %d\\n' {tag} \"$tut_rc\"\n")
        assert self.proc.stdout is not None
        out = []
        for line in self.proc.stdout:
            if line.startswith(tag):
                return int(line.split()[1]), "".join(out).rstrip() + "\n"
            out.append(line)
        raise CaptureError("the shell exited")

    def python(self, code: str) -> int:
        script = self.tmp / "block.py"
        script.write_text(code)
        return self.run(f"python {shlex.quote(str(script))}")[0]

    def close(self) -> None:
        if self.proc.poll() is None:
            self._send("exit\n")
            self.proc.wait(timeout=30)
        shutil.rmtree(self.tmp, ignore_errors=True)


def tui_capture(cmd: str, cols: int, rows: int) -> str:
    """ANSI text of ``cmd`` drawn in a detached tmux pane of ``cols`` x ``rows``."""
    name = f"tut_{uuid.uuid4().hex[:8]}"
    subprocess.run(
        [*TMUX, "new-session", "-d", "-s", name, "-x", str(cols), "-y", str(rows), cmd,
         ";", "set-option", "-t", name, "status", "off"],
        check=True,
    )  # fmt: skip
    try:
        time.sleep(TUI_WAIT)
        res = subprocess.run(
            [*TMUX, "capture-pane", "-p", "-e", "-t", name],
            check=True,
            capture_output=True,
            text=True,
        )
        return res.stdout
    finally:
        subprocess.run([*TMUX, "kill-session", "-t", name], check=False)


def save_figure(fig_id: str, workdir: Path, dest: Path) -> None:
    """mokashow's figure ``fig_id`` (mlat, rm, cm) of the aoloop pipeline, headless."""
    import matplotlib

    matplotlib.use("Agg")
    rootdir = workdir / "aoloop-rootdir"
    if fig_id == "mlat":
        from moka.show.cli import mlat_figure

        fig = mlat_figure(rootdir / "work" / "fps.aoloop_mlat.datadir")
    else:
        from moka.show.modes import modes_figure

        fig = modes_figure(rootdir / "calib", fig_id)
    fig.savefig(dest, dpi=100)


def edit_loopgain(workdir: Path) -> None:
    """The confdir edit of the "Change the configuration" stage."""
    path = workdir / "aoloop-conf" / "fpsdetails.toml"
    text, n = re.subn(
        r"^loopgain = .*$", f"loopgain = {LOOPGAIN_EDIT}", path.read_text(), 1, re.M
    )
    if n != 1:
        raise CaptureError(f"no loopgain line in {path}")
    path.write_text(text)


MANUAL_ACTIONS = {"edit-loopgain": edit_loopgain}


class LiveSystem:
    """What ``run`` needs from the machine; tests pass a fake with the same methods."""

    def __init__(self, cwd: Path, workdir: Path):
        self.workdir = workdir
        self.shell = Shell(cwd)

    def bash(self, code: str) -> tuple[int, str]:
        return self.shell.run(code)

    def python(self, code: str) -> int:
        return self.shell.python(code)

    def tui(self, cmd: str, cols: int, rows: int) -> str:
        return tui_capture(cmd, cols, rows)

    def figure(self, fig_id: str, dest: Path) -> None:
        save_figure(fig_id, self.workdir, dest)

    def manual(self, action_id: str) -> None:
        MANUAL_ACTIONS[action_id](self.workdir)

    def close(self) -> None:
        """Tear down both pipelines (errors ignored), then stop the shell."""
        self.shell.close()
        if self.workdir.is_dir():
            for pipe in PIPELINES:
                subprocess.run(["moka", "teardown", "-p", pipe], cwd=self.workdir)


# ---------------------------------------------------------------- --run


def block_code(page: Page, mk: Marker) -> str:
    f = mk.fence
    assert f is not None
    body = page.lines[f.start + 1 : f.end]
    return "".join(
        ln[len(f.indent) :] if ln.startswith(f.indent) else ln for ln in body
    )


def run(
    page_path: Path,
    home: Path,
    shm: Path,
    start: str | None = None,
    system_factory=LiveSystem,
) -> None:
    """Run the page's blocks in order; write outputs and images at the end, all or none.

    Without ``start``: refuses an existing workdir, FPS or stream, and removes the
    workdir at the end. With ``start``: runs from that id on, in the existing workdir,
    and leaves it in place. Both pipelines are torn down at the end, also on failure.
    """
    text = page_path.read_text()
    errors = check_page(text, page_path.parent)
    if errors:
        raise CaptureError("page check failed:\n" + "\n".join(errors))
    page = parse(text)
    workdir = home / WORKDIR_NAME
    runnable = [mk for mk in page.markers if mk.kind in ("run", "py", "tui", "fig")]
    if start is None:
        refusals = start_refusals(home, shm)
        if refusals:
            raise CaptureError("refusing to start:\n" + "\n".join(refusals))
        cwd = home
    else:
        if start not in {mk.id for mk in runnable}:
            raise CaptureError(f"no run, py, tui or fig marker with id {start}")
        if not workdir.is_dir():
            raise CaptureError(f"--from needs {workdir}")
        cwd = workdir

    outputs, images = {}, {}
    staging = Path(tempfile.mkdtemp(prefix="tut_images_"))
    system = system_factory(cwd, workdir)
    try:
        started = start is None
        for mk in page.markers:
            if not started:
                started = mk.id == start and mk.kind in ("run", "py", "tui", "fig")
                if not started:
                    continue
            if mk.kind == "run":
                rc, out = system.bash(block_code(page, mk))
                want = int(mk.attrs.get("expect", "0"))
                if rc != want:
                    raise CaptureError(
                        f"{mk.id}: exit code {rc}, expected {want}\n{out}"
                    )
                outputs[mk.id] = normalize(out, home)
            elif mk.kind == "py":
                rc = system.python(block_code(page, mk))
                if rc != 0:
                    raise CaptureError(f"{mk.id}: exit code {rc}")
            elif mk.kind == "tui":
                cols, rows = (int(v) for v in mk.attrs["size"].split("x"))
                ansi = normalize(system.tui(mk.attrs["cmd"], cols, rows), home)
                dest = staging / f"{mk.id}.svg"
                dest.write_text(ansi_to_svg(ansi, cols, rows))
                images[mk.image] = dest
            elif mk.kind == "fig":
                dest = staging / f"{mk.id}.png"
                system.figure(mk.id, dest)
                images[mk.image] = dest
            elif mk.kind == "manual" and mk.id in MANUAL_ACTIONS:
                system.manual(mk.id)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        system.close()
        if start is None and workdir.exists():
            shutil.rmtree(workdir)

    for rel, src in images.items():
        dest = page_path.parent / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(src, dest)
    shutil.rmtree(staging, ignore_errors=True)
    page_path.write_text(rewrite(text, outputs))


# ---------------------------------------------------------------- --callouts


def callouts(src: Path, dest: Path) -> None:
    """``src`` with the numbered discs of ``CALLOUTS`` drawn on it, saved as ``dest``."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.open(src).convert("RGB")
    draw = ImageDraw.Draw(img)
    d = round(0.03 * img.width)
    font = ImageFont.load_default(size=round(0.6 * d))
    for n, fx, fy in CALLOUTS:
        x, y = round(fx * img.width), round(fy * img.height)
        draw.ellipse((x, y, x + d, y + d), fill="#d62728", outline="white", width=3)
        draw.text((x + d / 2, y + d / 2), str(n), fill="white", font=font, anchor="mm")
    dest.parent.mkdir(parents=True, exist_ok=True)
    img.save(dest)


# ---------------------------------------------------------------- CLI


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="markers, ids, images")
    mode.add_argument("--run", action="store_true", help="the full capture")
    mode.add_argument("--callouts", type=Path, metavar="IMG", help="draw the discs")
    ap.add_argument("--from", dest="start", metavar="ID", help="--run: resume at ID")
    args = ap.parse_args(argv)
    if args.start and not args.run:
        ap.error("--from needs --run")

    try:
        if args.check:
            errors = check_page(PAGE.read_text(), PAGE.parent)
            for e in errors:
                print(f"{PAGE.relative_to(REPO)}: {e}", file=sys.stderr)
            return 1 if errors else 0
        if args.run:
            run(PAGE, Path.home(), shm_dir(), args.start)
        else:
            callouts(args.callouts, PAGE.parent / IMG_REL / "desktop.png")
    except (CaptureError, OSError) as e:
        print(e, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
