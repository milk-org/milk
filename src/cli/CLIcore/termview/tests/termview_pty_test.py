#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Olivier Guyon et al
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""pty-driven tests and benchmarks for milk-termview (Python stdlib only).

Every test runs against throw-away streams in a private MILK_SHM_DIR under /dev/shm, created by
tv-test-mkstream, so production streams are never touched.

    termview_pty_test.py --termview BIN --mkstream BIN [--only NAME ...]
    termview_pty_test.py --termview BIN --mkstream BIN --bench
"""

import argparse
import fcntl
import os
import re
import resource
import select
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import termios
import time

ENTER = b"\x1b[?1049h"
LEAVE = b"\x1b[?1049l"
DSR = b"\x1b[6n"  # cursor position request: a terminal answers CSI row;col R
CUP_DSR = re.compile(rb"\x1b\[(\d+);(\d+)H\x1b\[6n")  # how the viewer asks


class Screen:
    """Minimal VT model of what milk-termview emits: CUP, CUF, ED, SGR, private modes, text.
    With @colour, SGR colours are kept per cell too (slower)."""

    def __init__(self, rows, cols, colour=False):
        self.colour = colour
        self.bg = self.fg = None
        self.rows, self.cols = rows, cols
        self.resize(rows, cols)
        self.pending = b""
        self.dsr = []  # cursor position (1-based) at each request, for the answers

    def resize(self, rows, cols):
        self.rows, self.cols = rows, cols
        self.grid = [[" "] * cols for _ in range(rows)]
        self.attr = [[(None, None)] * cols for _ in range(rows)]
        self.r = self.c = 0

    def set_size(self, rows, cols):
        """The window changes size: the cursor stays, within it."""
        r, c = self.r, self.c
        self.resize(rows, cols)
        self.r, self.c = min(r, rows - 1), min(c, cols - 1)

    def _put(self, ch):
        if 0 <= self.r < self.rows and 0 <= self.c < self.cols:
            self.grid[self.r][self.c] = ch
            if self.colour:
                self.attr[self.r][self.c] = (self.bg, self.fg)
        if self.c < self.cols - 1:
            self.c += 1

    def _sgr(self, nums):
        k = 0
        while k < len(nums):
            p = nums[k]
            if p == 0:
                self.bg = self.fg = None
            elif p in (38, 48) and k + 1 < len(nums):
                n = 5 if nums[k + 1] == 2 else 3
                val = tuple(nums[k + 2 : k + n])
                if p == 48:
                    self.bg = val
                else:
                    self.fg = val
                k += n - 1
            elif 40 <= p <= 47 or 100 <= p <= 107:
                self.bg = (p,)
            elif 30 <= p <= 37 or 90 <= p <= 97:
                self.fg = (p,)
            k += 1

    def looks(self, rows):
        """What rows @rows look like: glyph plus the colours it shows."""
        out = []
        for r in rows:
            for ch, (bg, fg) in zip(self.grid[r], self.attr[r]):
                out.append(
                    (ch, None if ch == "\u2588" else bg, None if ch == " " else fg)
                )
        return out

    def feed(self, data):
        data = self.pending + data
        self.pending = b""
        i, n = 0, len(data)
        while i < n:
            b = data[i]
            if b == 0x1B:
                if i + 1 >= n:
                    self.pending = data[i:]
                    return
                if data[i + 1] != 0x5B:
                    i += 2
                    continue
                j = i + 2
                while j < n and not 0x40 <= data[j] <= 0x7E:
                    j += 1
                if j >= n:
                    self.pending = data[i:]
                    return
                self._csi(data[i + 2 : j].decode("ascii", "replace"), chr(data[j]))
                i = j + 1
            elif b < 0x20:
                i += 1
            elif b < 0x80:
                self._put(chr(b))
                i += 1
            else:
                ln = 4 if b >= 0xF0 else 3 if b >= 0xE0 else 2 if b >= 0xC0 else 1
                if i + ln > n:
                    self.pending = data[i:]
                    return
                self._put(data[i : i + ln].decode("utf-8", "replace"))
                i += ln

    def _csi(self, params, final):
        if params.startswith("?"):
            if params == "?1049":
                self.resize(self.rows, self.cols)
            return
        nums = (
            [int(p) if p.isdigit() else 0 for p in params.split(";")] if params else []
        )
        if final == "H":  # clamped, as a terminal does
            self.r = min(self.rows, nums[0] if nums and nums[0] else 1) - 1
            self.c = min(self.cols, nums[1] if len(nums) > 1 and nums[1] else 1) - 1
        elif final == "C":
            self.c = min(self.cols - 1, self.c + (nums[0] if nums and nums[0] else 1))
        elif final == "J" and nums and nums[0] == 2:
            self.resize(self.rows, self.cols)
        elif final == "m" and self.colour:
            self._sgr(nums or [0])
        elif final == "n" and params == "6":
            self.dsr.append((self.r + 1, self.c + 1))

    def text(self):
        return "\n".join("".join(row) for row in self.grid)


class Viewer:
    """milk-termview running on its own pty, with the slave side kept for termios checks."""

    def __init__(
        self,
        exe,
        args,
        env,
        rows=40,
        cols=120,
        block_hup=False,
        colour=False,
        stderr_path=None,
        limit_as=None,
        answer=True,
        delay=0.0,
        block_tstp=False,
        ignore_tstp=False,
    ):
        self.answer, self.delay = answer, delay
        self.answers = []  # (when, bytes, is an answer) not written yet
        self.asked = self.answered = 0
        # answers still owed each time the viewer let go of the terminal
        self.leaves = []
        self.master, self.slave = os.openpty()
        fcntl.ioctl(
            self.master, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0)
        )
        self.saved = termios.tcgetattr(self.slave)
        pid = os.fork()
        if pid == 0:
            try:
                os.close(self.master)
                os.login_tty(os.dup(self.slave))
                os.close(self.slave)
                if stderr_path:
                    fd = os.open(
                        stderr_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600
                    )
                    os.dup2(fd, 2)
                    os.close(fd)
                if block_hup:
                    signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGHUP})
                if block_tstp:
                    signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTSTP})
                if ignore_tstp:
                    signal.signal(signal.SIGTSTP, signal.SIG_IGN)
                resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
                if limit_as:
                    resource.setrlimit(resource.RLIMIT_AS, (limit_as, limit_as))
                os.execve(exe, [exe] + list(args), env)
            finally:
                os._exit(127)
        self.pid = pid
        os.set_blocking(self.master, False)
        self.out = bytearray()
        self.screen = Screen(rows, cols, colour)
        self.status = None
        self.rusage = None
        self.master_open = True

    def pump(self, timeout=0.05):
        """Collect whatever the viewer writes during @timeout seconds."""
        end = time.monotonic() + timeout
        first = True
        while self.master_open:
            left = end - time.monotonic()
            if left <= 0 and not first:
                return  # bounded even when the viewer writes continuously
            first = False
            nxt = self.reply()
            if nxt is not None:
                left = min(left, nxt - time.monotonic())
            r, _, _ = select.select([self.master], [], [], max(0.0, left))
            if not r:
                if time.monotonic() < end:
                    continue  # woken for an answer
                return
            try:
                data = os.read(self.master, 1 << 16)
            except OSError:
                data = b""
            if not data:
                time.sleep(min(0.01, max(0.0, end - time.monotonic())))
                continue
            self.got(data)
        self.reply_until(end)

    def got(self, data, feed=True):
        """@data reached the terminal: kept, shown (with @feed), and its cursor requests answered.
        Without the screen model (no @feed) the answer is where the move just before put it.
        """
        start = len(self.out)
        self.out += data
        lo = max(0, start - len(DSR) + 1)  # requests that end in @data
        n = self.out.count(DSR, lo)
        i = self.out.find(LEAVE, max(0, start - len(LEAVE) + 1))
        while i >= 0:
            self.leaves.append(len(self.answers) + self.out.count(DSR, lo, i))
            i = self.out.find(LEAVE, i + 1)
        pos = []
        if feed:
            self.screen.feed(data)
            pos, self.screen.dsr = self.screen.dsr, []
        self.asked += n
        if not feed and n:
            pos = [
                (
                    min(int(m.group(1)), self.screen.rows),
                    min(int(m.group(2)), self.screen.cols),
                )
                for m in CUP_DSR.finditer(self.out, max(0, lo - 16))
                if m.end() > start
            ]
        if self.answer:
            due = time.monotonic() + self.delay
            for k in range(n):
                r, c = pos[k] if k < len(pos) else (self.screen.rows, self.screen.cols)
                self.answers.append((due, b"\x1b[%d;%dR" % (r, c), True))
        self.reply()

    def reply(self):
        """Write the answers that are due; returns when the next one is (None if none)."""
        now = time.monotonic()
        while self.answers and self.answers[0][0] <= now and self.master_open:
            try:
                os.write(self.master, self.answers[0][1])
            except BlockingIOError:
                return now + 0.001
            except OSError:
                pass
            self.answered += self.answers.pop(0)[2]
        return self.answers[0][0] if self.answers else None

    def reply_until(self, end):
        """Sleep until @end, writing answers as they fall due."""
        while True:
            nxt = self.reply()
            now = time.monotonic()
            if now >= end:
                return
            time.sleep(max(0.0, min(end, nxt or end) - now))

    def wait_for(self, pattern, timeout=3.0):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if pattern in self.out:
                return True
            if self.poll_exit() is not None:
                self.pump(0.05)
                break
            self.pump(0.02)
        return pattern in self.out

    def screen_has(self, text, timeout=1.0):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if text in self.screen.text():
                return True
            self.pump(0.05)
        return text in self.screen.text()

    def send(self, data, gap=0.0):
        if gap <= 0:
            os.write(self.master, data)
            return
        for b in data:
            os.write(self.master, bytes([b]))
            time.sleep(gap)

    def resize(self, rows, cols):
        fcntl.ioctl(
            self.master, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0)
        )
        self.screen.resize(rows, cols)

    def poll_exit(self):
        if self.status is None:
            pid, status, ru = os.wait4(self.pid, os.WNOHANG)
            if pid == self.pid:
                self.status, self.rusage = status, ru
        return self.status

    def wait_exit(self, timeout=3.0):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if self.poll_exit() is not None:
                self.pump(0.05)
                return True
            self.pump(0.02)
        return self.poll_exit() is not None

    def alive(self):
        return self.poll_exit() is None

    def close_master(self):
        if self.master_open:
            os.close(self.master)
            self.master_open = False

    def close(self):
        if self.poll_exit() is None:
            os.kill(self.pid, signal.SIGKILL)
            self.wait_exit(2.0)
        self.close_master()
        os.close(self.slave)

    def cpu_ticks(self):
        """utime + stime of the running viewer, in clock ticks."""
        with open(f"/proc/{self.pid}/stat") as f:
            fields = f.read().rsplit(")", 1)[1].split()
        return int(fields[11]) + int(fields[12])

    def wakeups(self):
        with open(f"/proc/{self.pid}/status") as f:
            for line in f:
                if line.startswith("voluntary_ctxt_switches"):
                    return int(line.split()[1])
        return 0

    def termios_restored(self):
        now = termios.tcgetattr(self.slave)
        return now[:4] == self.saved[:4] and now[6] == self.saved[6]

    def exit_code(self):
        return None if self.status is None else os.waitstatus_to_exitcode(self.status)


class JobViewer(Viewer):
    """The viewer under a job-control shell: a session leader runs it in a foreground process
    group of its own and takes the terminal back when it stops, so Ctrl-Z really stops it (a
    Viewer leads an orphaned session, where the kernel discards the stop)."""

    def __init__(self, exe, args, env, rows=40, cols=120, delay=0.0):
        self.answer, self.delay = True, delay
        self.answers = []
        self.asked = self.answered = 0
        self.leaves = []
        self.master, self.slave = os.openpty()
        fcntl.ioctl(
            self.master, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0)
        )
        self.saved = termios.tcgetattr(self.slave)
        rd, wr = os.pipe()
        shell = os.fork()
        if shell == 0:
            try:
                os.close(self.master)
                os.close(rd)
                os.setsid()
                fcntl.ioctl(self.slave, termios.TIOCSCTTY, 0)
                signal.signal(signal.SIGTTOU, signal.SIG_IGN)
                pid = os.fork()
                if pid == 0:
                    os.setpgid(0, 0)
                    for fd in (0, 1, 2):
                        os.dup2(self.slave, fd)
                    os.tcsetpgrp(0, os.getpid())
                    signal.signal(signal.SIGTTOU, signal.SIG_DFL)
                    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
                    os.execve(exe, [exe] + list(args), env)
                    os._exit(127)
                try:
                    os.setpgid(pid, pid)
                except OSError:
                    pass
                os.write(wr, b"pid %d\n" % pid)
                while True:
                    _, st = os.waitpid(pid, os.WUNTRACED | os.WCONTINUED)
                    if os.WIFSTOPPED(st):
                        os.tcsetpgrp(self.slave, os.getpgrp())  # the shell has it again
                        os.write(wr, b"stopped %d\n" % os.WSTOPSIG(st))
                    elif os.WIFCONTINUED(st):
                        os.write(wr, b"continued 0\n")
                    else:
                        os.write(wr, b"exit %d\n" % st)
                        break
            finally:
                os._exit(0)
        os.close(wr)
        self.shell, self.rep, self.events = shell, rd, []
        os.set_blocking(rd, False)
        os.set_blocking(self.master, False)
        self.out = bytearray()
        self.screen = Screen(rows, cols, False)
        self.status = self.rusage = None
        self.master_open = True
        while not self.events:
            self.collect()
            time.sleep(0.01)
        self.pid = int(self.events[0][1])

    def collect(self):
        try:
            data = os.read(self.rep, 4096)
        except BlockingIOError:
            return
        for line in data.decode().splitlines():
            ev, arg = line.split()
            self.events.append((ev, int(arg)))
            self.status = int(arg) if ev == "exit" else self.status

    def pump(self, timeout=0.05):
        super().pump(timeout)
        self.collect()

    def poll_exit(self):
        self.collect()
        return self.status

    def stops(self):
        return sum(e[0] == "stopped" for e in self.events)

    def close(self):
        if self.status is None:
            for sig in (signal.SIGKILL, signal.SIGCONT):
                try:
                    os.kill(self.pid, sig)
                except OSError:
                    pass
        os.waitpid(self.shell, 0)
        self.close_master()
        os.close(self.slave)


class ShmDir:
    """Private MILK_SHM_DIR under /dev/shm plus test streams; removed on exit."""

    def __init__(self, mkstream):
        self.mkstream = mkstream
        self.procs = []

    def __enter__(self):
        self.dir = tempfile.mkdtemp(dir="/dev/shm", prefix="tvtest-")
        self.env = dict(os.environ)
        self.env.update(
            MILK_SHM_DIR=self.dir,
            TERM="xterm-256color",
            COLORTERM="truecolor",
            ASAN_OPTIONS="handle_segv=0:handle_sigbus=0:handle_abort=0:detect_leaks=0",
            UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1:log_path="
            + self.dir
            + "/ubsan",
        )
        return self

    def stream(self, tag, *args):
        """Create stream tvtest_<pid>_<tag>; a positive -r rate keeps the generator running."""
        name = f"tvtest_{os.getpid()}_{tag}"
        args = [str(a) for a in args]
        cmd = [self.mkstream, "-n", name] + args
        live = "-r" in args and float(args[args.index("-r") + 1]) > 0
        if live and "-u" not in args:
            # the whole stream first (one frame), then the generator updates it: a viewer started
            # now never sees a half-written header, however loaded the machine is
            first = list(args)
            first[first.index("-r") + 1] = "0"
            subprocess.run(
                [self.mkstream, "-n", name] + first, env=self.env, check=True
            )
            cmd.append("-u")
        if live:
            self.procs.append(subprocess.Popen(cmd, env=self.env))
            time.sleep(0.05)
        else:
            subprocess.run(cmd, env=self.env, check=True)
        return name

    def __exit__(self, *exc):
        for p in self.procs:
            p.terminate()
        for p in self.procs:
            try:
                p.wait(3)
            except subprocess.TimeoutExpired:
                p.kill()
        reports = [
            os.path.join(self.dir, f)
            for f in os.listdir(self.dir)
            if f.startswith("ubsan")
        ]
        for r in reports:
            with open(r) as f:
                sys.stderr.write(f.read())
        shutil.rmtree(self.dir, ignore_errors=True)
        if reports and exc[0] is None:
            raise AssertionError("sanitizer reports: " + ", ".join(reports))
        return False


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def start(
    cfg,
    shm,
    stream,
    *opts,
    rows=40,
    cols=120,
    block_hup=False,
    colour=False,
    stderr=None,
    answer=True,
    delay=0.0,
    block_tstp=False,
    ignore_tstp=False,
):
    v = Viewer(
        cfg.termview,
        list(opts) + [stream],
        shm.env,
        rows,
        cols,
        block_hup,
        colour,
        stderr,
        answer=answer,
        delay=delay,
        block_tstp=block_tstp,
        ignore_tstp=ignore_tstp,
    )
    check(
        v.wait_for(ENTER, 3.0), f"viewer did not start (output {bytes(v.out[:200])!r})"
    )
    return v


def quit_clean(v, key=b"q"):
    v.send(key)
    check(v.wait_exit(3.0), "viewer did not exit after quit key")
    check(v.exit_code() == 0, f"exit status {v.exit_code()}")
    check(LEAVE in v.out, "leave sequence missing")
    check(v.termios_restored(), "terminal attributes not restored")


# ------------------------------------------------------------------ tests


def test_help_oneline(cfg):
    r = subprocess.run(
        [cfg.termview, "-h1"], capture_output=True, text=True, timeout=10
    )
    check(
        r.returncode == 0 and r.stdout.count("\n") == 1,
        f"-h1 gave {r.returncode} {r.stdout!r}",
    )


def test_missing_stream(cfg):
    with ShmDir(cfg.mkstream) as shm:
        r = subprocess.run(
            [cfg.termview, "tvtest_missing"],
            env=shm.env,
            capture_output=True,
            text=True,
            timeout=10,
        )
        check(r.returncode == 1, f"exit status {r.returncode}")
        check("Error:" in r.stderr and "not found" in r.stderr, f"stderr {r.stderr!r}")


def test_bad_options(cfg):
    for opts in (
        ["-c", "nope"],
        ["-r", "42"],
        ["-f", "0"],
        ["--min", "1"],
        ["--min", "nan", "--max", "1"],
        ["--min", "2", "--max", "1"],
        ["--colors", "88"],
    ):
        r = subprocess.run(
            [cfg.termview] + opts + ["x"], capture_output=True, text=True, timeout=10
        )
        check(
            r.returncode == 1 and "Error:" in r.stderr,
            f"{opts}: {r.returncode} {r.stderr!r}",
        )


def test_stdin_not_tty(cfg):
    with ShmDir(cfg.mkstream) as shm:
        s = shm.stream("ramp", "-s", "64x64")
        t0 = time.monotonic()
        r = subprocess.run(
            [cfg.termview, s],
            env=shm.env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=10,
        )
        check(
            r.returncode == 1 and b"Error:" in r.stderr, f"{r.returncode} {r.stderr!r}"
        )
        check(time.monotonic() - t0 < 2.0, "slow exit")


def test_quit_keys(cfg):
    with ShmDir(cfg.mkstream) as shm:
        s = shm.stream("ramp", "-s", "128x96")
        for key in (b"q", b"\x03", b"\x1c"):
            v = start(cfg, shm, s)
            try:
                quit_clean(v, key)
            finally:
                v.close()


def test_small_terminals(cfg):
    with ShmDir(cfg.mkstream) as shm:
        s = shm.stream("noise", "-s", "200x150", "-p", "noise", "-r", "100")
        for rows, cols in (
            (5, 80),
            (1, 1),
            (4, 13),
            (5, 14),
            (6, 20),
            (3, 200),
            (2, 2),
        ):
            v = start(cfg, shm, s, rows=rows, cols=cols)
            try:
                v.pump(0.4)
                check(v.alive(), f"{rows}x{cols}: viewer died, status {v.exit_code()}")
                quit_clean(v)
            finally:
                v.close()
        # too small, nothing to draw: only the data polls wake the viewer (20/s)
        still = shm.stream("still", "-s", "64x64")
        v = start(cfg, shm, still, rows=3, cols=120)
        try:
            v.pump(0.5)
            w0 = v.wakeups()
            v.pump(2.0)
            wk = (v.wakeups() - w0) / 2.0
            check(wk <= 25, f"{wk:.0f} wakeups/s while the terminal is too small")
            quit_clean(v)
        finally:
            v.close()


def test_resize_storm(cfg):
    sizes = [
        (40, 120),
        (1, 1),
        (4, 13),
        (5, 14),
        (90, 300),
        (5, 80),
        (24, 80),
        (2, 400),
    ]
    with ShmDir(cfg.mkstream) as shm:
        s = shm.stream("noise", "-s", "256x256", "-p", "noise", "-r", "200")
        v = start(cfg, shm, s)
        try:
            for k in range(40):
                v.resize(*sizes[k % len(sizes)])
                v.pump(0.015)
            v.resize(40, 120)
            v.pump(0.4)
            check(v.alive(), f"viewer died, status {v.exit_code()}")
            quit_clean(v)
        finally:
            v.close()


def test_nan_inf_frames(cfg):
    with ShmDir(cfg.mkstream) as shm:
        streams = [
            shm.stream("nanf", "-t", "f32", "-s", "96x64", "-p", "nan", "-r", "100"),
            shm.stream("inff", "-t", "f32", "-s", "96x64", "-p", "inf", "-r", "100"),
            shm.stream("nand", "-t", "f64", "-s", "96x64", "-p", "nan", "-r", "100"),
            shm.stream("constd", "-t", "f64", "-s", "8x8", "-p", "const"),
        ]
        for s in streams:
            v = start(cfg, shm, s)
            try:
                for key in b"rrrrrrrrcccsssssss+++-----0":
                    v.send(bytes([key]))
                    v.pump(0.03)
                check(v.alive(), f"{s}: viewer died, status {v.exit_code()}")
                quit_clean(v)
            finally:
                v.close()


def test_manual_range_validation(cfg):
    with ShmDir(cfg.mkstream) as shm:
        s = shm.stream("ramp", "-s", "128x96")
        v = start(cfg, shm, s)
        try:
            for text, expect in (
                (b"mnan\r", "invalid number"),
                (b"M1e999\r", "invalid number"),
                (b"m5000\r", "min must be < max"),
            ):
                v.send(text)
                check(v.screen_has(expect), f"{text!r}: no '{expect}' message")
                v.send(b"\x1b")
                v.pump(0.15)
                check(v.alive(), f"{text!r}: viewer died")
            v.send(b"m10\r")
            check(v.screen_has("LOCK"), "valid min did not lock the range")
            quit_clean(v)
        finally:
            v.close()


def test_signals_restore_terminal(cfg):
    with ShmDir(cfg.mkstream) as shm:
        s = shm.stream("ramp", "-s", "64x64")
        for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGSEGV):
            v = start(cfg, shm, s)
            try:
                v.pump(0.2)
                os.kill(v.pid, sig)
                check(v.wait_exit(3.0), f"{sig.name}: viewer did not exit")
                check(os.WIFSIGNALED(v.status), f"{sig.name}: status {v.status:#x}")
                got = os.WTERMSIG(v.status)
                check(got == sig, f"{sig.name}: killed by signal {got}")
                check(LEAVE in v.out, f"{sig.name}: leave sequence missing")
                check(
                    v.termios_restored(),
                    f"{sig.name}: terminal attributes not restored",
                )
            finally:
                v.close()


def test_stop_cont_reapplies_raw_mode(cfg):
    with ShmDir(cfg.mkstream) as shm:
        s = shm.stream("ramp", "-s", "64x64")
        v = start(cfg, shm, s)
        try:
            v.pump(0.2)
            os.kill(v.pid, signal.SIGSTOP)
            time.sleep(0.1)
            termios.tcsetattr(v.slave, termios.TCSANOW, v.saved)
            os.kill(v.pid, signal.SIGCONT)
            v.pump(0.5)
            lflag = termios.tcgetattr(v.slave)[3]
            check(not lflag & termios.ICANON, "raw mode not re-applied after SIGCONT")
            quit_clean(v)
        finally:
            v.close()


def test_hangup_exits(cfg):
    with ShmDir(cfg.mkstream) as shm:
        s = shm.stream("ramp", "-s", "64x64")
        v = start(cfg, shm, s, block_hup=True)
        try:
            v.pump(0.2)
            v.close_master()
            t0 = time.monotonic()
            check(v.wait_exit(3.0), "viewer kept running after the terminal hung up")
            check(time.monotonic() - t0 < 1.5, "slow exit after hangup")
            cpu = v.rusage.ru_utime + v.rusage.ru_stime
            check(cpu < 0.5, f"{cpu:.2f} s CPU: busy loop on hangup")
        finally:
            v.close()


def test_split_and_legacy_mouse(cfg):
    with ShmDir(cfg.mkstream) as shm:
        s = shm.stream("ramp", "-s", "128x96")
        v = start(cfg, shm, s)
        try:
            check(v.screen_has("/20 fps"), "status line not found")
            v.send(b"\x1b[<0;12;5M", gap=0.005)
            v.send(b"\x1b[<0;12;5m", gap=0.005)
            # X10 press and release at column 81: 32 + 81 is 'q'
            v.send(
                b"\x1b[M"
                + bytes([32, 32 + 81, 32 + 5])
                + b"\x1b[M"
                + bytes([35, 32 + 81, 32 + 5])
            )
            v.pump(0.4)
            check(v.alive(), "viewer quit on mouse input")
            text = v.screen.text()
            check("/20 fps" in text, "mouse bytes changed the refresh rate")
            check("Enter: lock" not in text, "mouse bytes opened the min/max prompt")
            quit_clean(v)
        finally:
            v.close()


SYNC_ON = b"\x1b[?2026h"


def test_drag_and_wheel_storms_are_capped(cfg):
    with ShmDir(cfg.mkstream) as shm:
        s = shm.stream("ramp", "-s", "256x256")
        v = start(cfg, shm, s)
        try:
            v.pump(0.5)
            before = v.screen.text()
            n0, t0 = v.out.count(SYNC_ON), time.monotonic()
            v.send(b"\x1b[<0;40;20M")
            # a moving drag report every ~2 ms, as a fast mouse sends them
            for k in range(400):
                x = 40 + (k % 40 if (k // 40) % 2 == 0 else 39 - k % 40)
                v.send(b"\x1b[<32;%d;20M" % x)
                v.pump(0)
                time.sleep(0.002)
            v.send(b"\x1b[<0;80;20m")
            for k in range(100):  # wheel notches, one per report
                v.send(b"\x1b[<65;40;20M")
                v.pump(0)
                time.sleep(0.002)
            v.pump(0.3)
            dt = time.monotonic() - t0
            frames = v.out.count(SYNC_ON) - n0
            check(
                frames <= 64 * dt + 6,
                f"{frames} frames in {dt:.2f} s: input renders not capped",
            )
            check(v.screen.text() != before, "drag and wheel did not change the view")
            check(v.alive(), "viewer died")
            quit_clean(v)
        finally:
            v.close()


def test_input_does_not_wait_for_data_frames(cfg):
    """At 1 fps, a lone Esc and two quick keys still reach the screen promptly."""
    with ShmDir(cfg.mkstream) as shm:
        s = shm.stream("ramp", "-s", "128x96")
        v = start(cfg, shm, s, "-f", "1")
        try:
            check(v.screen_has("/1 fps", 2.0), "status line not found")
            v.send(b"m")
            check(v.screen_has("Enter: lock", 1.0), "prompt did not open")
            v.pump(0.3)
            t0 = time.monotonic()
            v.send(b"\x1b")
            while "Enter: lock" in v.screen.text() and time.monotonic() - t0 < 2.0:
                v.pump(0.005)
            esc_ms = (time.monotonic() - t0) * 1000
            check(esc_ms < 250, f"lone Esc took {esc_ms:.0f} ms to close the prompt")
            v.pump(0.3)
            t0 = time.monotonic()
            v.send(b"h")
            time.sleep(0.005)
            v.send(b"v")
            while "flipV" not in v.screen.text() and time.monotonic() - t0 < 2.0:
                v.pump(0.005)
            keys_ms = (time.monotonic() - t0) * 1000
            check("flipH" in v.screen.text(), "first key lost")
            check(keys_ms < 150, f"second key took {keys_ms:.0f} ms to show")
            quit_clean(v)
        finally:
            v.close()


def test_live_and_stale_indicator(cfg):
    with ShmDir(cfg.mkstream) as shm:
        live = shm.stream("live", "-s", "64x64", "-p", "noise", "-r", "50")
        static = shm.stream("static", "-s", "64x64")
        v = start(cfg, shm, live)
        try:
            v.pump(1.5)
            check(v.screen_has("LIVE"), "a 50 Hz stream is not shown LIVE")
            quit_clean(v)
        finally:
            v.close()
        v = start(cfg, shm, static)
        try:
            check(v.screen_has("STALE", 3.0), "a stream written once never turns STALE")
            quit_clean(v)
        finally:
            v.close()


def val_range(v):
    """The '[lo : hi]' range shown in the status bar, or None."""
    import re

    m = re.search(r"\[([-+0-9.eE]+) : ([-+0-9.eE]+)\]", v.screen.text())
    return (float(m.group(1)), float(m.group(2))) if m else None


def wait_range(v, ok, timeout):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        r = val_range(v)
        if r and ok(r):
            return r
        v.pump(0.05)
    return val_range(v)


def test_range_mode_on_static_stream(cfg):
    with ShmDir(cfg.mkstream) as shm:
        s = shm.stream("ramp", "-s", "256x256")
        v = start(cfg, shm, s)
        try:
            v.pump(0.3)
            v.send(b"rrrrrrr")  # 20-80%
            r = wait_range(
                v, lambda r: abs(r[0] - 200) < 8 and abs(r[1] - 800) < 8, 1.5
            )
            check(r and abs(r[0] - 200) < 8 and abs(r[1] - 800) < 8, f"20-80% gave {r}")
            quit_clean(v)
        finally:
            v.close()


def test_range_settles_after_transient(cfg):
    with ShmDir(cfg.mkstream) as shm:
        s = shm.stream("ramp", "-s", "128x128")
        v = start(cfg, shm, s)
        try:
            v.pump(0.5)
            gen = subprocess.Popen(
                [cfg.mkstream, "-n", s, "-u", "-p", "spike", "-r", "10", "-d", "0.35"],
                env=shm.env,
            )
            peak = wait_range(v, lambda r: r[1] > 4000, 1.0)
            gen.wait(5)
            check(peak and peak[1] > 4000, f"the hot frame was not shown: {peak}")
            r = wait_range(v, lambda r: r[1] < 1100, 2.0)
            check(r and r[1] < 1100, f"range stuck at {r} after the transient frame")
            quit_clean(v)
        finally:
            v.close()


def test_stale_seeded_from_write_time(cfg):
    import re

    with ShmDir(cfg.mkstream) as shm:
        s = shm.stream("old", "-s", "64x64")
        time.sleep(2.2)
        v = start(cfg, shm, s)
        try:
            check(
                v.screen_has("STALE", 0.6),
                "a 2 s old stream is not shown STALE at startup",
            )
            m = re.search(r"STALE (\d+)s", v.screen.text())
            check(m and int(m.group(1)) >= 2, f"wrong age: {m.group(0) if m else None}")
            quit_clean(v)
        finally:
            v.close()


def test_frame_between_polls_is_shown(cfg):
    with ShmDir(cfg.mkstream) as shm:
        s = shm.stream("ramp", "-s", "64x64")
        v = start(cfg, shm, s, "-f", "1")
        try:
            v.pump(1.2)
            subprocess.run(
                [cfg.mkstream, "-n", s, "-u", "-p", "const"], env=shm.env, check=True
            )
            v.send(b"c")  # an input-driven frame right after the new data
            r = wait_range(v, lambda r: r[0] > 40 and r[1] < 44, 2.5)
            check(r and r[0] > 40 and r[1] < 44, f"new frame never shown: range {r}")
            quit_clean(v)
        finally:
            v.close()


def lowbw_diff(v):
    """Cells of the image area that differ from an exact (Ctrl-L) repaint."""
    rows = range(
        1, v.screen.rows - 1
    )  # the image and colorbar; not the menubar or status
    held = v.screen.looks(rows)
    v.send(b"\x0c")
    v.pump(0.5)
    return sum(a != b for a, b in zip(held, v.screen.looks(rows)))


def test_lowbw_converges_when_stream_stops(cfg):
    """Low-bandwidth mode holds small changes; once the stream stops (at 50, 3 or 0.5 Hz) or
    keeps posting the same frame (at 50 or 2 Hz), the image becomes exact."""
    for rate, repost, run, wait in (
        ("50", False, 1.5, 1.0),
        ("3", False, 1.5, 2.0),
        ("0.5", False, 6.0, 4.0),  # 6 s: the frame period must be known
        ("50", True, 1.5, 1.0),
        ("2", True, 1.5, 1.5),
    ):
        with ShmDir(cfg.mkstream) as shm:
            s = shm.stream("jitter", "-s", "128x128", "-p", "jitter", "-r", rate)
            gen = shm.procs[-1]
            v = start(cfg, shm, s, "--min", "0", "--max", "1010", colour=True)
            try:
                v.send(b"b")
                check(v.screen_has("lowbw"), "low-bandwidth mode did not turn on")
                v.pump(run)
                if (
                    repost
                ):  # the same frame again and again: cnt0 moves, the data does not
                    shm.stream("jitter", "-u", "-p", "repost", "-r", rate)
                    v.pump(0.2)
                gen.terminate()
                gen.wait(3)
                v.pump(wait)
                diff = lowbw_diff(v)
                what = f"{rate} Hz " + ("reposts" if repost else "stream stops")
                check(
                    diff == 0,
                    f"{diff} cells differ from an exact repaint {wait} s after {what}",
                )
                quit_clean(v)
            finally:
                v.close()


def test_lowbw_keeps_the_range(cfg):
    """Redrawing held rows does not move the smoothed auto range: not between frames, and not
    when the stream stops."""
    import re

    def ranges(v, dt):
        seen = set()
        end = time.monotonic() + dt
        while time.monotonic() < end:
            v.pump(0.05)
            m = re.search(r"\[[-+0-9.eE]+ : [-+0-9.eE]+\]", v.screen.text())
            if m:
                seen.add(m.group(0))
        return seen

    with ShmDir(cfg.mkstream) as shm:
        s = shm.stream("jitter", "-s", "8x8", "-p", "jitter", "-r", "5")
        gen = shm.procs[-1]
        v = start(cfg, shm, s)
        try:
            v.send(b"b")
            v.pump(1.5)
            seen = ranges(v, 3.0)
            check(len(seen) == 1, f"the range moved in 3 s: {sorted(seen)[:4]}")
            gen.terminate()
            gen.wait(3)
            after = ranges(v, 1.5)
            check(
                after == seen,
                f"the range moved when the stream stopped: {sorted(after)[:4]}",
            )
            quit_clean(v)
        finally:
            v.close()


def test_colors_option(cfg):
    """--colors picks the depth whatever the environment says."""
    with ShmDir(cfg.mkstream) as shm:
        s = shm.stream("ramp", "-s", "128x96")
        for depth, want, avoid in (
            ("16", b"m", (b"8;5;", b"8;2;")),
            ("256", b"8;5;", (b"8;2;",)),
            ("truecolor", b"8;2;", (b"8;5;",)),
        ):
            v = start(cfg, shm, s, "--colors", depth)
            try:
                v.pump(0.5)
                bad = [a for a in avoid if a in v.out]
                check(want in v.out and not bad, f"--colors {depth}: codes {bad} sent")
                quit_clean(v)
            finally:
                v.close()


def shm_path(shm, name):
    return os.path.join(shm.dir, name + ".im.shm")


def remake(cfg, shm, name, *args):
    """Create stream @name again (a new file, as milk does), writing one frame."""
    subprocess.run(
        [cfg.mkstream, "-n", name] + [str(a) for a in args], env=shm.env, check=True
    )


def test_stream_removed_and_recreated(cfg):
    """A removed stream shows GONE; one created again under the name, with another size and type,
    is picked up and the view reset."""
    with ShmDir(cfg.mkstream) as shm:
        s = shm.stream("life", "-s", "64x64", "-t", "u16")
        v = start(cfg, shm, s)
        try:
            check(v.screen_has("64x64 u16"), "geometry not shown")
            v.send(b"++")
            check(v.screen_has("zoom 1.44x"), "zoom did not apply")
            os.unlink(shm_path(shm, s))
            check(v.screen_has("GONE", 2.0), "a removed stream is not shown GONE")
            remake(cfg, shm, s, "-s", "200x100", "-t", "f32")
            check(
                v.screen_has("200x100 f32", 2.0),
                "the stream created again was not picked up",
            )
            v.pump(0.2)
            text = v.screen.text()
            check("GONE" not in text, "still GONE after the stream came back")
            check("zoom 1.00x" in text, "a new geometry did not reset the view")
            quit_clean(v)
        finally:
            v.close()


def test_stream_truncated(cfg):
    """A stream truncated in place: no crash, also with a key or a resize right after; GONE, an
    empty image once the view changes; a new stream under the name brings the image back.
    """
    with ShmDir(cfg.mkstream) as shm:
        for poke in ("none", "key", "resize", "key -f 1"):
            s = shm.stream("trunc", "-s", "64x64")
            opts = ["-f", "1"] if poke.endswith("-f 1") else []
            v = start(cfg, shm, s, *opts)
            try:
                v.pump(0.5)
                os.truncate(shm_path(shm, s), 0)
                if poke.startswith("key"):
                    v.send(b"+")
                elif poke == "resize":
                    v.resize(30, 100)
                v.pump(0.3)
                check(v.alive(), f"{poke}: viewer died, status {v.exit_code()}")
                check(
                    v.screen_has("GONE", 1.5),
                    f"{poke}: a truncated stream is not shown GONE",
                )
                v.send(b"-")
                check(
                    v.screen_has("stream gone", 1.0),
                    f"{poke}: no message over the empty image",
                )
                remake(cfg, shm, s, "-s", "64x64")
                end = time.monotonic() + 2.5
                while time.monotonic() < end and "GONE" in v.screen.text():
                    v.pump(0.05)
                v.pump(0.3)
                text = v.screen.text()
                check(
                    "GONE" not in text and "stream gone" not in text,
                    f"{poke}: no image",
                )
                quit_clean(v)
            finally:
                v.close()


def test_rate_after_counter_reset(cfg):
    """A stream created again restarts its frame counter: the rate shown stays sensible, whether
    the new counter is below the old one or (replaced while paused) far above it."""
    import re

    def rates(v, dt):
        got = []
        end = time.monotonic() + dt
        while time.monotonic() < end:
            v.pump(0.1)
            m = re.search(r"(-?[0-9]+\.[0-9]) Hz", v.screen.text())
            if m:
                got.append(float(m.group(1)))
        return got

    with ShmDir(cfg.mkstream) as shm:
        s = shm.stream("rate", "-s", "32x32", "-p", "noise", "-r", "50")
        gen = shm.procs[-1]
        v = start(cfg, shm, s)
        try:
            v.pump(2.5)
            gen.terminate()
            gen.wait(3)
            gen = (
                shm.stream("rate", "-s", "32x32", "-p", "noise", "-r", "20")
                and shm.procs[-1]
            )
            got = rates(v, 4.0)
            check(
                got and max(got) < 200 and min(got) >= 0, f"rates: {sorted(set(got))}"
            )
            check(
                any(15 < r < 25 for r in got), f"20 Hz never shown: {sorted(set(got))}"
            )

            v.send(b" ")
            v.pump(0.3)
            gen.terminate()
            gen.wait(3)
            shm.stream("rate", "-s", "32x32", "-p", "noise", "-r", "200")
            v.pump(4.0)
            v.send(b" ")
            got = rates(v, 4.0)
            check(
                got and max(got) < 300, f"after the paused switch: {sorted(set(got))}"
            )
            check(
                any(150 < r < 250 for r in got),
                f"200 Hz never shown: {sorted(set(got))}",
            )
            quit_clean(v)
        finally:
            v.close()


def test_reopen_leaks_nothing(cfg):
    """Each reopen (and each pause) releases the old file, mapping and memory. 12 reopens of a
    256x256 stream; the plan's 100 would take 100 s, as each waits for the 1 Hz check.
    """

    def counts(pid):
        with open(f"/proc/{pid}/maps") as f:
            maps = sum(1 for _ in f)
        with open(f"/proc/{pid}/status") as f:
            rss = next(int(line.split()[1]) for line in f if line.startswith("VmRSS"))
        return len(os.listdir(f"/proc/{pid}/fd")), maps, rss

    with ShmDir(cfg.mkstream) as shm:
        s = shm.stream("leak", "-s", "256x256", "-t", "f32")
        v = start(cfg, shm, s)
        try:
            seen = []
            for k in range(12):
                v.send(b" ")  # pause (a frame-sized copy) and resume
                v.pump(0.1)
                v.send(b" ")
                w = 255 + k % 2
                remake(
                    cfg, shm, s, "-s", f"{w}x256", "-t", ("f32", "u16", "f64")[k % 3]
                )
                check(v.screen_has(f"{w}x256", 2.5), f"reopen {k} not picked up")
                seen.append(counts(v.pid))
            fds, maps = {c[0] for c in seen[2:]}, {c[1] for c in seen[2:]}
            check(len(fds) == 1, f"fds grew over reopens: {seen}")
            if cfg.sanitized:
                return  # the sanitizer allocator maps memory as it goes
            check(len(maps) == 1, f"mappings grew over reopens: {seen}")
            # the heap keeps a freed frame or two for reuse; a leak would add one per reopen
            grew = seen[-1][2] - seen[5][2]
            check(grew < 600, f"RSS grew by {grew} kB over the last 6 reopens")
            quit_clean(v)
        finally:
            v.close()


def test_pause(cfg):
    """Space freezes the image (zoom still works on the frozen frame) and wakes the viewer about
    once a second; Space again resumes."""
    with ShmDir(cfg.mkstream) as shm:
        s = shm.stream("pause", "-s", "64x64", "-p", "noise", "-r", "100")
        v = start(cfg, shm, s, colour=True)
        try:
            rows = range(v.screen.rows - 4)
            v.pump(0.5)
            v.send(b" ")
            check(v.screen_has("PAUSED"), "no PAUSED state")
            v.pump(0.3)
            a = v.screen.looks(rows)
            w0, c0 = v.wakeups(), v.cpu_ticks()
            v.pump(3.0)
            wk = (v.wakeups() - w0) / 3.0
            cpu = (v.cpu_ticks() - c0) / os.sysconf("SC_CLK_TCK") / 3.0 * 100
            check(v.screen.looks(rows) == a, "the image changed while paused")
            check(wk <= 4 and cpu < 2, f"paused: {wk:.1f} wakeups/s, {cpu:.1f} % CPU")
            v.send(b"+")
            v.pump(0.3)
            b = v.screen.looks(rows)
            check(b != a, "zoom did nothing while paused")
            v.pump(0.5)
            check(v.screen.looks(rows) == b, "the zoomed image changed while paused")
            v.send(b"-")
            v.pump(0.3)
            check(
                v.screen.looks(rows) == a, "zooming back does not show the frozen frame"
            )
            v.send(b" ")
            v.pump(0.3)
            check("PAUSED" not in v.screen.text(), "Space did not resume")
            c = v.screen.looks(rows)
            v.pump(0.3)
            check(v.screen.looks(rows) != c, "the image does not move after resuming")
            quit_clean(v)
        finally:
            v.close()


def corrupt_copy(shm, name):
    """Replace stream @name by a copy whose header passes the viewer's checks but not
    ImageStreamIO's (the name is cleared), so the library would complain on each retry.
    """
    path = shm_path(shm, name)
    with open(path, "rb") as f:
        data = bytearray(f.read())
    data[32:112] = bytes(80)  # md->name
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.rename(tmp, path)


def test_junk_under_the_name(cfg):
    """Junk, then a header ImageStreamIO rejects, under the name: GONE, and no library message
    on the screen (stderr is /dev/null) or in a stderr file (the library printer is off).
    """
    with ShmDir(cfg.mkstream) as shm:
        s = shm.stream("junk", "-s", "64x64")
        v = start(cfg, shm, s)
        try:
            v.pump(0.3)
            if not cfg.sanitized:
                fd2 = os.readlink(f"/proc/{v.pid}/fd/2")
                check(fd2 == "/dev/null", f"stderr is {fd2} while the screen is in use")
            tmp = shm_path(shm, s) + ".tmp"
            with open(tmp, "wb") as f:
                f.write(bytes((k * 7) & 255 for k in range(100000)))
            os.rename(tmp, shm_path(shm, s))
            check(v.screen_has("GONE", 2.0), "junk under the name is not shown GONE")
            remake(cfg, shm, s, "-s", "64x64")
            check(
                v.screen_has("LIVE", 2.5) or v.screen_has("STALE", 0.1),
                "no stream again",
            )
            n0 = len(v.out)
            corrupt_copy(shm, s)
            check(v.screen_has("GONE", 2.0), "a rejected header is not shown GONE")
            v.pump(2.2)  # two more retries
            check(b"ERROR" not in v.out[n0:], "a library message reached the screen")
            check(v.alive(), "viewer died")
            quit_clean(v)
        finally:
            v.close()

        errlog = os.path.join(shm.dir, "stderr.log")
        s = shm.stream("junk2", "-s", "64x64")
        v = start(cfg, shm, s, stderr=errlog)
        try:
            v.pump(0.3)
            corrupt_copy(shm, s)
            check(v.screen_has("GONE", 2.0), "a rejected header is not shown GONE")
            v.pump(2.2)
            with open(errlog, "rb") as f:
                err = f.read()
            check(
                err == b"",
                f"the library wrote {len(err)} bytes to stderr: {err[:80]!r}",
            )
            quit_clean(v)
        finally:
            v.close()


def test_stderr_on_another_terminal(cfg):
    """stderr on another terminal than the image is left alone."""
    master, slave = os.openpty()
    try:
        with ShmDir(cfg.mkstream) as shm:
            s = shm.stream("tty", "-s", "64x64")
            other = os.ttyname(slave)
            v = start(cfg, shm, s, stderr=other)
            try:
                fd2 = os.readlink(f"/proc/{v.pid}/fd/2")
                check(fd2 == other, f"stderr on {other} became {fd2}")
                quit_clean(v)
            finally:
                v.close()
    finally:
        os.close(master)
        os.close(slave)


def test_pause_refused_without_memory(cfg):
    """No memory for the frozen copy: the pause is refused with a bell instead of a PAUSED view
    of live data."""
    if cfg.sanitized:
        print("     (skipped: an address-space limit breaks sanitizer builds)")
        return
    with ShmDir(cfg.mkstream) as shm:
        s = shm.stream("big", "-s", "4096x4096", "-t", "f32", "-p", "noise", "-r", "5")
        v = Viewer(cfg.termview, [s], shm.env, 40, 120, limit_as=110 << 20)
        try:
            check(v.wait_for(ENTER, 3.0), "viewer did not start")
            v.pump(0.5)
            n0 = len(v.out)
            v.send(b" ")
            v.pump(0.5)
            check(b"\a" in v.out[n0:], "no bell")
            check("PAUSED" not in v.screen.text(), "PAUSED without a frozen copy")
            quit_clean(v)
        finally:
            v.close()


def test_lowbw_rests_when_gone(cfg):
    """Low-bandwidth mode with held colours, then the stream is truncated away: the viewer does
    not keep redrawing; it wakes for its data polls only (120/s at -f 120)."""
    with ShmDir(cfg.mkstream) as shm:
        s = shm.stream("hg", "-s", "128x128", "-p", "jitter", "-r", "50")
        gen = shm.procs[-1]
        v = start(cfg, shm, s, "-f", "120")
        try:
            v.send(b"b")
            v.pump(1.0)
            gen.terminate()
            gen.wait(3)
            os.truncate(shm_path(shm, s), 0)
            check(v.screen_has("GONE", 1.5), "a truncated stream is not shown GONE")
            v.pump(1.0)
            w0 = v.wakeups()
            v.pump(3.0)
            wk = (v.wakeups() - w0) / 3.0
            check(wk < 135, f"{wk:.0f} wakeups/s with nothing to draw")
            quit_clean(v)
        finally:
            v.close()


def test_pause_outlives_a_replacement(cfg):
    """Replaced while paused: the frozen frame stays (zoom still works); the new stream comes
    on the resume."""
    with ShmDir(cfg.mkstream) as shm:
        s = shm.stream("pr", "-s", "64x64", "-t", "u8")
        v = start(cfg, shm, s)
        try:
            v.pump(0.3)
            v.send(b" ")
            check(v.screen_has("PAUSED"), "no PAUSED state")
            os.unlink(shm_path(shm, s))
            remake(cfg, shm, s, "-s", "400x400", "-t", "f64")
            v.pump(2.2)
            text = v.screen.text()
            check(
                "64x64" in text and "PAUSED GONE" in text,
                "the paused view did not hold",
            )
            v.send(b"++")
            v.pump(0.2)
            v.send(b"--")
            v.pump(0.2)
            check(v.alive(), f"viewer died, status {v.exit_code()}")
            v.send(b" ")
            check(
                v.screen_has("400x400", 2.5),
                "the new stream was not picked up on resume",
            )
            v.pump(0.2)
            text = v.screen.text()
            check("GONE" not in text, "still GONE after the resume")
            # written once, 2 s ago: its own write time says so
            check(
                "STALE" in text and "LIVE" not in text,
                "the new stream's age is not shown",
            )
            quit_clean(v)
        finally:
            v.close()


def test_menus(cfg):
    """A click on the menubar opens a menu; moving previews, Esc undoes, Enter keeps; F3/Tab and
    Left/Right switch; the help box shows the keys; the minibuffer locks a typed range.
    """
    with ShmDir(cfg.mkstream) as shm:
        s = shm.stream("menu", "-s", "128x96")
        v = start(cfg, shm, s)
        try:
            check(v.screen_has("Cmap greyscale"), "no menubar")
            col = v.screen.text().splitlines()[0].index("Cmap") + 2
            n0 = len(v.out)
            v.send(b"\x1b[<0;%d;1M\x1b[<0;%d;1m" % (col, col))
            check(v.screen_has("viridis", 1.0), "a click on Cmap did not open its menu")
            check(b"\x1b[?1003h" in v.out[n0:], "pointer motion reports not turned on")
            v.send(b"\x1b[B\x1b[B")  # Down, Down: previews cold
            check(v.screen_has("Cmap cold", 1.0), "moving in the menu did not preview")
            n0 = len(v.out)
            v.send(b"\x1b")
            v.pump(0.2)
            text = v.screen.text()
            check(
                "Cmap greyscale" in text and "viridis" not in text,
                "Esc did not undo the preview",
            )
            check(
                b"\x1b[?1003l\x1b[?1002h" in v.out[n0:], "motion reports not turned off"
            )
            v.send(b"\x1bOR")  # F3 (SS3 R): Scale
            check(v.screen_has("log_extreme", 1.0), "F3 did not open the Scale menu")
            v.send(b"\x1b[B\r")
            v.pump(0.2)
            check("Scale sqrt" in v.screen.text(), "Enter did not keep the choice")
            v.send(b"\t\x1b[C")  # Tab: the last menu (Scale); Right: Range
            check(
                v.screen_has("Set min / max", 1.0),
                "Tab and Right did not reach the Range menu",
            )
            v.send(b"\x1b")
            v.pump(0.2)
            v.send(b"?")
            check(v.screen_has("Keys and mouse", 1.0), "? did not show the help box")
            v.send(b"\x1b")
            v.pump(0.2)
            check(
                "Keys and mouse" not in v.screen.text(),
                "Esc did not close the help box",
            )
            v.send(b"m")
            check(v.screen_has("Enter: lock", 1.0), "m did not open the minibuffer")
            v.send(b"100\t900\r")
            check(
                v.screen_has("[100 : 900] LOCK", 1.0), "the typed range was not locked"
            )
            # xterm and tmux end all mouse reporting on ?1003l: button events must come back
            body = bytes(v.out).split(LEAVE)[0]
            k = body.find(b"\x1b[?1003l")
            while k >= 0:
                check(
                    body[k + 8 : k + 16] == b"\x1b[?1002h",
                    "?1003l without ?1002h after it",
                )
                k = body.find(b"\x1b[?1003l", k + 1)
            # a resume re-enters the screen: the open menu gets its motion reports again
            v.send(b"\x1bOQ")
            check(v.screen_has("viridis", 1.0), "F2 did not open the Cmap menu")
            os.kill(v.pid, signal.SIGSTOP)
            time.sleep(0.1)
            os.kill(v.pid, signal.SIGCONT)
            v.pump(0.5)
            after = bytes(v.out).rsplit(ENTER, 1)[-1]
            check(
                b"\x1b[?1003h" in after,
                "motion reports not turned on again after a resume",
            )
            v.send(b"\x1b")
            v.pump(0.2)
            quit_clean(v)
        finally:
            v.close()


def echoed_answers(v, dt=0.5):
    """Answers written to the terminal during @dt seconds once the viewer let go of it: the
    shell would read them as typed text (the restored terminal echoes them as ^[[...R).
    """
    n0 = len(v.out)
    v.pump(dt)
    return b"^[[" in v.out[n0:]


def wait_owed(v, timeout=2.0):
    """Pump until an answer is owed: a request reached the terminal, its answer not sent yet."""
    end = time.monotonic() + timeout
    while not any(a[2] for a in v.answers) and time.monotonic() < end:
        v.pump(0.005)
    check(any(a[2] for a in v.answers), "no request outstanding")


def leave_owing(v, act, what):
    """With an answer owed, do @act; check that none was owed when the terminal was let go,
    and that it was let go as soon as the answers were in."""
    wait_owed(v)
    n0 = len(v.leaves)
    t0 = time.monotonic()
    act()
    end = time.monotonic() + 4.0
    while len(v.leaves) == n0 and time.monotonic() < end:
        v.pump(0.01)
    took = time.monotonic() - t0
    check(len(v.leaves) > n0, f"terminal not restored on {what}")
    check(v.leaves[n0] == 0, f"{v.leaves[n0]} answer(s) still owed after {what}")
    check(
        took < v.delay + 0.3,
        f"{what}: let go {took:.2f} s later, answers {v.delay} s late",
    )


def test_cursor_answers_read_before_leaving(cfg):
    """Ctrl-Z, SIGTSTP and quit wait for the answers to outstanding cursor requests, here 150 ms
    late, so none reaches the shell. (The viewer leads its own session here, where the kernel
    ignores the stop: the terminal is restored and the viewer comes straight back.)"""
    with ShmDir(cfg.mkstream) as shm:
        s = shm.stream("noise", "-s", "64x64", "-p", "noise", "-r", "100")
        v = start(cfg, shm, s, delay=0.15)
        try:
            v.pump(1.0)
            check(v.answered > 2, f"{v.asked} requests, {v.answered} answered")
            n0 = v.out.count(ENTER)
            leave_owing(v, lambda: v.send(b"\x1a"), "Ctrl-Z")
            end = time.monotonic() + 2.0
            while v.out.count(ENTER) <= n0 and time.monotonic() < end:
                v.pump(0.02)
            check(v.out.count(ENTER) > n0, "no resume")
            v.pump(0.5)
            leave_owing(v, lambda: os.kill(v.pid, signal.SIGTSTP), "SIGTSTP")
            v.pump(0.5)
            leave_owing(v, lambda: v.send(b"q"), "q")
            check(not echoed_answers(v), "an answer reached the shell after q")
            check(v.wait_exit(1.0), "viewer did not exit")
            check(v.exit_code() == 0, f"exit status {v.exit_code()}")
            check(v.termios_restored(), "terminal attributes not restored")
        finally:
            v.close()


def test_terminate_signal_reads_answers(cfg):
    """SIGTERM from outside reads the outstanding answers too before restoring the terminal."""
    with ShmDir(cfg.mkstream) as shm:
        s = shm.stream("noise", "-s", "64x64", "-p", "noise", "-r", "100")
        v = start(cfg, shm, s, delay=0.15)
        try:
            v.pump(1.0)
            leave_owing(v, lambda: os.kill(v.pid, signal.SIGTERM), "SIGTERM")
            check(not echoed_answers(v), "an answer reached the shell after SIGTERM")
            check(v.wait_exit(1.0), "viewer did not exit")
            check(
                os.WIFSIGNALED(v.status) and os.WTERMSIG(v.status) == signal.SIGTERM,
                f"status {v.status}",
            )
            check(v.termios_restored(), "terminal attributes not restored")
        finally:
            v.close()
        # answers a second late: a second SIGTERM does not wait for them
        v = start(cfg, shm, s, delay=1.0)
        try:
            v.pump(2.5)
            wait_owed(v)
            os.kill(v.pid, signal.SIGTERM)
            v.pump(0.1)
            t0 = time.monotonic()
            os.kill(v.pid, signal.SIGTERM)
            check(v.wait_exit(3.0), "viewer did not exit")
            check(
                time.monotonic() - t0 < 0.3, f"exit {time.monotonic() - t0:.2f} s later"
            )
        finally:
            v.close()


def owed_answer(v, timeout=2.0):
    """Wait until exactly the newest request is owed, and take its answer back (to send it
    with a key in one write)."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        v.pump(0.002)
        owed = [a for a in v.answers if a[2]]
        if len(owed) == 1 and time.monotonic() < owed[0][0] - 0.05:
            v.answers.remove(owed[0])
            return owed[0][1]
    check(False, "no single answer owed")


def test_key_with_the_last_answer(cfg):
    """q or Ctrl-Z read together with the last answer owed: the terminal is let go at once."""
    with ShmDir(cfg.mkstream) as shm:
        s = shm.stream("ramp", "-s", "64x64")  # still: few requests
        v = start(cfg, shm, s, delay=0.3)
        try:
            v.pump(1.5)
            for key in (b"\x1a", b"q"):
                ans = owed_answer(v)
                n0 = len(v.leaves)
                t0 = time.monotonic()
                v.send(key + ans)
                while len(v.leaves) == n0 and time.monotonic() - t0 < 4.0:
                    v.pump(0.01)
                took = time.monotonic() - t0
                check(
                    len(v.leaves) > n0 and v.leaves[n0] == 0,
                    f"{key!r}: not let go cleanly",
                )
                check(took < 0.3, f"{key!r} with the answer: let go {took:.2f} s later")
                v.pump(0.5)
            check(v.wait_exit(2.0) and v.exit_code() == 0, "viewer did not exit")
        finally:
            v.close()


def test_ctrl_z_then_sigterm(cfg):
    """SIGTERM while Ctrl-Z waits for answers: the viewer quits (it does not stop), and no
    answer reaches the shell."""
    with ShmDir(cfg.mkstream) as shm:
        s = shm.stream("noise", "-s", "64x64", "-p", "noise", "-r", "100")
        v = start(cfg, shm, s, delay=1.0)
        try:
            v.pump(2.5)
            wait_owed(v)
            v.send(b"\x1a")
            v.pump(0.2)
            os.kill(v.pid, signal.SIGTERM)
            check(v.wait_exit(4.0), "viewer did not exit")
            check(
                os.WIFSIGNALED(v.status) and os.WTERMSIG(v.status) == signal.SIGTERM,
                f"status {v.status}",
            )
            check(
                v.leaves and v.leaves[0] == 0, f"answers owed at the leave: {v.leaves}"
            )
            check(not echoed_answers(v, 1.5), "an answer reached the shell")
            check(v.out.count(ENTER) == 1, "the terminal was entered again")
        finally:
            v.close()


def test_ctrl_c_stops_the_wait(cfg):
    """Answers late (the link got slow): after q, Ctrl-C ends the wait a second after it began,
    and at once later on. A double tap does not cut a short wait: the answers would reach the
    shell."""
    with ShmDir(cfg.mkstream) as shm:
        s = shm.stream("noise", "-s", "64x64", "-p", "noise", "-r", "100")
        for late in (0.2, 1.5):
            v = start(cfg, shm, s, delay=0.05)
            try:
                v.pump(1.5)
                v.delay = 5.0
                v.pump(0.5)
                wait_owed(v)
                t0 = time.monotonic()
                v.send(b"q")
                v.pump(late)
                v.send(b"\x03")
                check(v.wait_exit(3.0), "viewer did not exit")
                took = time.monotonic() - t0
                want = 1.0 if late < 1.0 else late
                check(
                    want - 0.1 < took < want + 0.3,
                    f"Ctrl-C {late} s after q: exit {took:.2f} s after q",
                )
            finally:
                v.close()
        v = start(cfg, shm, s, delay=0.3)
        try:
            v.pump(1.5)
            wait_owed(v)
            n0 = len(v.leaves)
            v.send(b"\x03")
            v.pump(0.1)
            v.send(b"\x03")
            check(v.wait_exit(3.0), "viewer did not exit")
            check(
                v.leaves[n0] == 0,
                f"{v.leaves[n0]} answer(s) owed after a double Ctrl-C",
            )
            check(
                not echoed_answers(v),
                "an answer reached the shell after a double Ctrl-C",
            )
        finally:
            v.close()


def test_hangup_with_answers_owed(cfg):
    """The terminal hangs up while answers are owed (SIGHUP not delivered): the viewer exits
    at once, without a busy loop."""
    with ShmDir(cfg.mkstream) as shm:
        s = shm.stream("noise", "-s", "64x64", "-p", "noise", "-r", "100")
        v = start(cfg, shm, s, block_hup=True, delay=1.0)
        try:
            v.pump(2.5)
            wait_owed(v)
            v.close_master()
            t0 = time.monotonic()
            check(v.wait_exit(3.0), "viewer kept running after the hangup")
            check(
                time.monotonic() - t0 < 0.5,
                f"exit {time.monotonic() - t0:.2f} s after it",
            )
            cpu = v.rusage.ru_utime + v.rusage.ru_stime
            check(cpu < 2.0, f"{cpu:.2f} s CPU")
        finally:
            v.close()


def test_typed_line_dropped_whole(cfg):
    """Keys typed while q waits for answers are dropped, and the rest of a line begun then
    too (it would run cut short); a note says so. The next line goes to the shell."""
    with ShmDir(cfg.mkstream) as shm:
        s = shm.stream("noise", "-s", "64x64", "-p", "noise", "-r", "100")
        err = os.path.join(shm.dir, "stderr.txt")
        v = start(cfg, shm, s, delay=0.4, stderr=err)
        try:
            v.pump(1.5)
            wait_owed(v)
            v.send(b"q")
            v.pump(0.05)
            v.send(b"ech")  # during the wait
            while any(a[2] for a in v.answers):
                v.pump(0.01)
            v.pump(0.1)
            check(LEAVE not in v.out, "the terminal was let go in the middle of a line")
            v.send(b"o cut\r")  # the rest of that line, after the answers are in
            check(v.wait_exit(3.0) and v.exit_code() == 0, "viewer did not exit")
            v.send(b"echo ok\r")
            v.pump(0.3)
            os.set_blocking(v.slave, False)
            try:
                line = os.read(v.slave, 256)
            except BlockingIOError:
                line = b""
            check(line == b"echo ok\n", f"the shell got {line!r}")
            with open(err) as f:
                note = f.read()
            check("dropped" in note, f"no note on stderr: {note!r}")
        finally:
            v.close()


def typed_after_q(cfg, shm, s, keys, err=None):
    """q while answers are owed (0.4 s late), then @keys: (delay, bytes, resize) steps from
    then on. Returns what the shell got, and how long after q the viewer exited."""
    v = start(cfg, shm, s, delay=0.4, stderr=err)
    try:
        v.pump(1.5)
        wait_owed(v)
        t0 = time.monotonic()
        v.send(b"q")
        for dt, data, resize in keys:
            v.pump(dt)
            v.resize(41, 120) if resize else v.send(data)
        check(v.wait_exit(12.0) and v.exit_code() == 0, "viewer did not exit")
        took = time.monotonic() - t0
        v.pump(0.3)
        os.set_blocking(v.slave, False)
        try:
            line = os.read(v.slave, 256)
        except BlockingIOError:
            line = b""
        return line, took
    finally:
        v.close()


def test_typed_lines_after_q(cfg):
    """A line typed while q waits for answers is dropped whole, however long it takes to type
    (at 8 keys/s here), with pauses under 1.5 s or a resize in the middle. The viewer's own
    keys (q, Ctrl-Z) begin no line: the terminal is let go as the answers are in."""
    with ShmDir(cfg.mkstream) as shm:
        s = shm.stream("noise", "-s", "64x64", "-p", "noise", "-r", "100")
        line = b"git commit -am wip && git push\r"
        steps = [(0.05, line[:1], False)] + [
            (0.12, bytes([b]), False) for b in line[1:]
        ]
        got, _ = typed_after_q(cfg, shm, s, steps)
        check(got == b"", f"a long line: the shell got {got!r}")
        steps = [(0.05, b"ls -", False), (0.7, b"la /tmp\r", False)]
        got, _ = typed_after_q(cfg, shm, s, steps)
        check(got == b"", f"a pause: the shell got {got!r}")
        steps = [(0.05, b"git st", False), (0.5, b"", True), (0.1, b"atus\r", False)]
        got, _ = typed_after_q(cfg, shm, s, steps)
        check(got == b"", f"a resize: the shell got {got!r}")
        for key in (b"q", b"\x1a"):
            got, took = typed_after_q(cfg, shm, s, [(0.05, key, False)])
            check(
                got == b"" and took < 1.0, f"{key!r} again: let go {took:.2f} s after q"
            )


def test_ctrl_z_with_sigtstp_ignored(cfg):
    """With SIGTSTP ignored by the parent (a shell without job control), Ctrl-Z and SIGTSTP do
    nothing at all: no stop, and nothing waits."""
    with ShmDir(cfg.mkstream) as shm:
        s = shm.stream("noise", "-s", "64x64", "-p", "noise", "-r", "100")
        v = start(cfg, shm, s, delay=0.3, ignore_tstp=True, cols=160)  # room for flipH
        try:
            v.pump(1.0)
            n0, asked = len(v.out), v.asked
            v.send(b"\x1a")
            os.kill(v.pid, signal.SIGTSTP)
            v.pump(0.1)
            v.send(b"h")  # acts at once: nothing waits for answers
            check(v.screen_has("flipH", 1.0), "a key after Ctrl-Z did not act")
            v.pump(1.5)
            check(LEAVE not in v.out[n0:], "the terminal was restored")
            check(v.asked > asked + 10, f"{v.asked - asked} requests after Ctrl-Z")
            quit_clean(v)
        finally:
            v.close()


def test_early_quit_without_answers(cfg):
    """A terminal that never answers: q waits until the newest request gives up (2 s after it,
    3 s after the start at most), and leaves nothing."""
    with ShmDir(cfg.mkstream) as shm:
        s = shm.stream("ramp", "-s", "64x64")
        v = start(cfg, shm, s, answer=False)
        try:
            v.pump(0.3)
            t0 = time.monotonic()
            v.send(b"q")
            check(v.wait_exit(3.0) and v.exit_code() == 0, "viewer did not exit")
            check(
                time.monotonic() - t0 < 2.0, f"quit took {time.monotonic() - t0:.2f} s"
            )
        finally:
            v.close()


def test_narrower_far_terminal(cfg):
    """The far terminal narrows while answers are on their way (it moves the cursor in, as the
    viewer learns of the size later): answers still name their requests (none is left for the
    shell), and requests go on."""
    with ShmDir(cfg.mkstream) as shm:
        s = shm.stream("noise", "-s", "64x64", "-p", "noise", "-r", "100")
        for cols in (100, 40):
            v = Viewer(cfg.termview, [s], shm.env, 40, 120)
            try:
                relay = Relay(v, 300_000, 0.6)
                check(relay.until(lambda: ENTER in v.out, 3.0), "viewer did not start")
                relay.run(4.0)
                relay.resize(40, cols)
                relay.run(4.0)
                asked = v.asked
                relay.run(3.0)
                check(
                    v.asked > asked + 5,
                    f"{v.asked - asked} requests after narrowing to {cols}",
                )
                relay.quit()
                tail = bytes(v.out[v.out.rfind(LEAVE) :])
                check(
                    b"^[[" not in tail,
                    f"{tail.count(b'^[[')} answer(s) reached the shell",
                )
            finally:
                v.close()


def test_quit_before_the_first_answer(cfg):
    """q or SIGINT before the startup request is answered, on a slow link with a long round
    trip: the viewer waits for that answer and those behind it, so the shell gets none.
    """
    with ShmDir(cfg.mkstream) as shm:
        s = shm.stream("noise", "-s", "256x256", "-p", "noise", "-r", "200")
        for how in ("q", "SIGINT"):
            v = Viewer(cfg.termview, [s], shm.env, 90, 300)
            try:
                relay = Relay(v, 20_000, 1.0)
                relay.run(0.1)
                relay.send(b"q") if how == "q" else os.kill(v.pid, signal.SIGINT)
                check(
                    relay.until(lambda: not v.alive(), 15.0),
                    f"{how}: viewer did not exit",
                )
                idle = lambda: not select.select([v.master], [], [], 0.05)[0]
                relay.until(
                    lambda: not relay.fifo
                    and not relay.wire
                    and not v.answers
                    and idle(),
                    15.0,
                )
                tail = bytes(v.out[v.out.rfind(LEAVE) :])
                check(
                    v.leaves and v.leaves[0] == 0,
                    f"{how}: answers owed at the leave: {v.leaves}",
                )
                check(
                    b"^[[" not in tail,
                    f"{how}: {tail.count(b'^[[')} answer(s) reached the shell",
                )
            finally:
                v.close()


def test_signal_during_a_blocked_write(cfg):
    """A terminal that stops reading in the middle of a frame: SIGTSTP or SIGTERM still acts at
    once (the write is cut short), not after the rest of the frame."""
    with ShmDir(cfg.mkstream) as shm:
        s = shm.stream("noise", "-s", "256x256", "-p", "noise", "-r", "200")
        for sig in (signal.SIGTSTP, signal.SIGTERM):
            v = start(cfg, shm, s, rows=90, cols=300, answer=False)
            try:
                v.pump(2.5)  # no answers: no flow control, whole frames
                time.sleep(
                    1.0
                )  # the terminal stops reading: the viewer blocks in a write
                n0 = len(v.out)
                os.kill(v.pid, sig)
                time.sleep(0.2)
                end = time.monotonic() + 3.0
                while LEAVE not in v.out[n0:] and time.monotonic() < end:
                    v.pump(0.02)
                i = v.out.find(LEAVE, n0)
                check(i >= 0, f"{sig.name}: the terminal was not restored")
                check(
                    i - n0 < 100_000, f"{sig.name}: {i - n0} bytes of the frame first"
                )
                if sig == signal.SIGTERM:
                    check(v.wait_exit(2.0), "viewer did not exit")
                else:
                    quit_clean(v)
            finally:
                v.close()


def test_stopped_then_terminated(cfg):
    """Under a job-control shell, Ctrl-Z stops the viewer, here once the answers stopped coming
    (the wait gives up). Killed while stopped (SIGTERM then SIGCONT, as bash's kill %1), it
    exits at once and leaves the line typed at the shell meanwhile alone."""
    with ShmDir(cfg.mkstream) as shm:
        s = shm.stream("noise", "-s", "64x64", "-p", "noise", "-r", "100")
        v = JobViewer(cfg.termview, [s], shm.env, delay=0.05)
        try:
            check(v.wait_for(ENTER, 3.0), "viewer did not start")
            v.pump(2.0)
            v.answer = False  # requests go overdue
            v.pump(3.5)
            v.send(b"\x1a")
            end = time.monotonic() + 8.0
            while not v.stops() and time.monotonic() < end:
                v.pump(0.02)
            check(v.stops() == 1, "Ctrl-Z did not stop the viewer")
            v.pump(0.3)
            t0 = time.monotonic()
            os.kill(v.pid, signal.SIGTERM)
            os.kill(v.pid, signal.SIGCONT)
            time.sleep(0.05)
            v.send(b"ls\r")  # at the shell
            end = time.monotonic() + 5.0
            while v.poll_exit() is None and time.monotonic() < end:
                v.pump(0.01)
            check(v.status is not None, "viewer still there after SIGTERM and SIGCONT")
            check(
                time.monotonic() - t0 < 0.5, f"exit {time.monotonic() - t0:.2f} s later"
            )
            check(v.stops() == 1, "the viewer read the terminal from the background")
            os.set_blocking(v.slave, False)
            try:
                line = os.read(v.slave, 256)
            except BlockingIOError:
                line = b""
            check(line == b"ls\n", f"the shell got {line!r}")
        finally:
            v.close()


def test_suspend_before_the_keys_after_it(cfg):
    """Ctrl-Z and q in one read: the viewer stops first, and q applies after the resume. So
    too with the last answer owed between them (read, it lets the terminal go)."""
    with ShmDir(cfg.mkstream) as shm:
        s = shm.stream("ramp", "-s", "64x64")  # still: few requests
        for answer in (False, True):
            v = start(cfg, shm, s, answer=answer, delay=0.3 if answer else 0.0)
            try:
                if answer:
                    v.pump(1.5)
                    keys = b"\x1a" + owed_answer(v) + b"q"
                else:
                    v.pump(3.5)  # past the startup request and the wait for it
                    keys = b"\x1aq"
                n0 = len(v.out)
                v.send(keys)
                check(v.wait_exit(3.0), f"viewer did not exit (answers {answer})")
                tail = bytes(v.out[n0:])
                i = tail.find(LEAVE)
                check(
                    i >= 0 and tail.find(ENTER, i) > i,
                    f"q was taken before the stop (answers {answer})",
                )
                check(v.exit_code() == 0, f"exit status {v.exit_code()}")
            finally:
                v.close()


def test_slow_link_quit_leaves_nothing(cfg):
    """Over a slow ssh link the answers take a while, behind the image queued at startup or in
    steady state: quitting waits for them, so the shell gets none."""
    with ShmDir(cfg.mkstream) as shm:
        s = shm.stream("noise", "-s", "256x256", "-p", "noise", "-r", "200")
        for rate, rtt, after in (
            (150_000, 0.04, 0.5),
            (300_000, 0.04, 0.4),
            (64_000, 0.2, 0.3),  # the first window alone takes a second
            (100_000, 0.04, 5.0),
            (60_000, 0.8, 1.0),  # satellite-like
            (20_000, 0.05, 0.3),
            (30_000, 1.0, 6.0),
        ):
            v = Viewer(cfg.termview, [s], shm.env, 90, 300)
            try:
                relay = Relay(v, rate, rtt)
                check(relay.until(lambda: ENTER in v.out, 3.0), "viewer did not start")
                relay.run(after)
                relay.quit(20.0)
                tail = bytes(v.out[v.out.rfind(LEAVE) :])
                check(
                    b"^[[" not in tail,
                    f"{tail.count(b'^[[')} answer(s) reached the shell"
                    f" at {rate // 1000} KB/s, q after {after} s",
                )
            finally:
                v.close()


def test_terminal_without_answers(cfg):
    """A terminal that never answers cursor requests gets them only until the startup request
    times out (64 KB of image meanwhile), then the viewer works as without flow control: frames
    keep coming, and quitting does not wait."""
    with ShmDir(cfg.mkstream) as shm:
        s = shm.stream("noise", "-s", "64x64", "-p", "noise", "-r", "100")
        v = start(cfg, shm, s, answer=False)
        try:
            t0 = time.monotonic()
            v.pump(1.8)
            held = len(v.out)
            v.pump(2.5 - (time.monotonic() - t0))
            n0 = v.out.count(SYNC_ON)
            v.pump(1.0)
            frames = v.out.count(SYNC_ON) - n0
            asked = v.asked
            v.pump(0.5)
            check(v.asked == asked, "requests go on to a terminal that never answers")
            check(held < 80_000, f"{held} bytes before the startup request timed out")
            check(frames >= 10, f"{frames} frames/s")
            t1 = time.monotonic()
            quit_clean(v)
            check(
                time.monotonic() - t1 < 0.2, f"quit took {time.monotonic() - t1:.2f} s"
            )
        finally:
            v.close()
        # a still image larger than the first window: the rest shows once the request timed out
        s = shm.stream("still", "-s", "256x256", "-p", "noise")
        v = Viewer(cfg.termview, [s], shm.env, 90, 300, answer=False)
        try:
            check(v.wait_for(ENTER, 3.0), "viewer did not start")
            t0 = time.monotonic()
            while len(v.out) < 250_000 and time.monotonic() - t0 < 3.0:
                if select.select([v.master], [], [], 0.01)[0]:
                    v.got(os.read(v.master, 1 << 16), feed=False)
            took = time.monotonic() - t0
            check(took < 2.5, f"250 KB of the first paint took {took:.2f} s")
            quit_clean(v)
        finally:
            v.close()


def test_resume_forgets_requests(cfg):
    """Answers lost across a stop (the shell read them) are not waited for: after the resume,
    the image flows again at once over a slow link."""
    with ShmDir(cfg.mkstream) as shm:
        s = shm.stream("noise", "-s", "64x64", "-p", "noise", "-r", "100")
        v = Viewer(cfg.termview, [s], shm.env, 40, 120)
        try:
            relay = Relay(v, 300_000, 0.2)
            relay.run(2.0)
            check(ENTER in v.out and v.answered > 5, f"{v.answered} answers")
            os.kill(v.pid, signal.SIGSTOP)
            while (
                relay.fifo or relay.wire or select.select([v.master], [], [], 0.05)[0]
            ):
                relay.run(0.05)
            v.answers.clear()
            termios.tcflush(v.slave, termios.TCIFLUSH)
            os.kill(v.pid, signal.SIGCONT)
            n0 = len(v.out)
            relay.set_rtt(1.0)  # what the viewer writes before any answer can be back
            relay.run(0.3)
            kb = (
                len(v.out) + len(relay.fifo) + sum(len(w[1]) for w in relay.wire) - n0
            ) / 1024
            check(kb > 25, f"{kb:.0f} KB written in the first 0.3 s after the resume")
            relay.set_rtt(0.2)
            relay.run(1.0)
            kb = (len(v.out) - n0) / 1024
            check(kb > 100, f"{kb:.0f} KB shown in the first 1.3 s after the resume")
            relay.quit()
        finally:
            v.close()


def test_ctrl_z_with_sigtstp_blocked(cfg):
    """With SIGTSTP blocked by the parent, Ctrl-Z does nothing, and the viewer goes on as
    before: images flow and the terminal is still asked."""
    with ShmDir(cfg.mkstream) as shm:
        s = shm.stream("noise", "-s", "64x64", "-p", "noise", "-r", "100")
        v = start(cfg, shm, s, delay=0.05, block_tstp=True)
        try:
            v.pump(1.0)
            n0, asked = len(v.out), v.asked
            v.send(b"\x1a")
            v.pump(2.0)
            check(LEAVE not in v.out[n0:], "the terminal was restored")
            check(
                v.asked > asked + 10, f"{v.asked - asked} requests in 2 s after Ctrl-Z"
            )
            check(
                len(v.out) - n0 > 50000, f"{len(v.out) - n0} bytes in 2 s after Ctrl-Z"
            )
            quit_clean(v)
        finally:
            v.close()


def test_stop_during_startup_request(cfg):
    """A stop while the startup request is out, longer than it may take: after the resume the
    terminal is asked again, before the repaint, and flow control works."""
    with ShmDir(cfg.mkstream) as shm:
        s = shm.stream("noise", "-s", "64x64", "-p", "noise", "-r", "100")
        v = start(cfg, shm, s, delay=0.5)
        try:
            os.kill(v.pid, signal.SIGSTOP)
            v.pump(3.0)
            n0 = len(v.out)
            os.kill(v.pid, signal.SIGCONT)
            v.pump(2.0)
            tail = bytes(v.out[n0:])
            i, j = tail.find(DSR), tail.find(SYNC_ON)
            check(0 <= i < j, "no request before the repaint")
            check(tail.count(DSR) > 5, f"{tail.count(DSR)} requests after the resume")
            quit_clean(v)
        finally:
            v.close()


def test_input_taken_while_stopped(cfg):
    """A key that arrived just before a stop, and that something else read meanwhile, is not
    taken for a hangup after the resume."""
    with ShmDir(cfg.mkstream) as shm:
        s = shm.stream("ramp", "-s", "64x64")
        v = start(cfg, shm, s, answer=False)
        try:
            v.pump(2.5)  # no requests: the viewer waits for input
            for _ in range(5):
                v.send(b"x")
                os.kill(v.pid, signal.SIGSTOP)
                time.sleep(0.05)
                termios.tcflush(v.slave, termios.TCIFLUSH)
                os.kill(v.pid, signal.SIGCONT)
                v.pump(0.3)
                check(v.alive(), f"the viewer exited ({v.exit_code()}) after a stop")
            quit_clean(v)
        finally:
            v.close()


TESTS = [
    test_help_oneline,
    test_missing_stream,
    test_bad_options,
    test_stdin_not_tty,
    test_quit_keys,
    test_small_terminals,
    test_resize_storm,
    test_nan_inf_frames,
    test_manual_range_validation,
    test_signals_restore_terminal,
    test_stop_cont_reapplies_raw_mode,
    test_hangup_exits,
    test_split_and_legacy_mouse,
    test_drag_and_wheel_storms_are_capped,
    test_input_does_not_wait_for_data_frames,
    test_live_and_stale_indicator,
    test_range_mode_on_static_stream,
    test_range_settles_after_transient,
    test_stale_seeded_from_write_time,
    test_frame_between_polls_is_shown,
    test_lowbw_converges_when_stream_stops,
    test_lowbw_keeps_the_range,
    test_colors_option,
    test_stream_removed_and_recreated,
    test_stream_truncated,
    test_rate_after_counter_reset,
    test_reopen_leaks_nothing,
    test_pause,
    test_junk_under_the_name,
    test_pause_outlives_a_replacement,
    test_lowbw_rests_when_gone,
    test_stderr_on_another_terminal,
    test_pause_refused_without_memory,
    test_menus,
    test_cursor_answers_read_before_leaving,
    test_terminate_signal_reads_answers,
    test_suspend_before_the_keys_after_it,
    test_slow_link_quit_leaves_nothing,
    test_terminal_without_answers,
    test_resume_forgets_requests,
    test_input_taken_while_stopped,
    test_ctrl_z_with_sigtstp_blocked,
    test_stop_during_startup_request,
    test_key_with_the_last_answer,
    test_ctrl_z_then_sigterm,
    test_ctrl_c_stops_the_wait,
    test_hangup_with_answers_owed,
    test_typed_line_dropped_whole,
    test_typed_lines_after_q,
    test_ctrl_z_with_sigtstp_ignored,
    test_early_quit_without_answers,
    test_narrower_far_terminal,
    test_quit_before_the_first_answer,
    test_signal_during_a_blocked_write,
    test_stopped_then_terminated,
]


# ------------------------------------------------------------------ perf checks


def perf_idle_static_stream(cfg):
    """A static stream costs about nothing: ~fps+1 wakeups/s, writes only for the 1 Hz status."""
    tck = os.sysconf("SC_CLK_TCK")
    with ShmDir(cfg.mkstream) as shm:
        s = shm.stream("static", "-s", "256x256")
        v = start(cfg, shm, s, rows=90, cols=300)
        try:
            v.pump(1.5)
            win = 5.0
            n0, c0, w0 = len(v.out), v.cpu_ticks(), v.wakeups()
            v.pump(win)
            n1, c1, w1 = len(v.out), v.cpu_ticks(), v.wakeups()
            cpu = (c1 - c0) / tck / win * 100
            wk = (w1 - w0) / win
            bps = (n1 - n0) / win
            print(f"     idle: cpu {cpu:.2f} %, {wk:.1f} wakeups/s, {bps:.0f} B/s")
            check(cpu <= 0.5, f"idle CPU {cpu:.2f} %")
            check(wk <= 21 * 1.3, f"{wk:.1f} wakeups/s")
            check(bps <= 2048, f"{bps:.0f} B/s written while idle")
            quit_clean(v)
        finally:
            v.close()


def perf_key_latency(cfg):
    """A key press reaches the screen within about one interactive frame."""
    with ShmDir(cfg.mkstream) as shm:
        s = shm.stream("static", "-s", "256x256")
        v = start(cfg, shm, s, rows=60, cols=200)
        try:
            v.pump(1.2)
            lat = []
            for key in b"hvhvhvhv":
                v.pump(0.25)
                n0 = len(v.out)
                t0 = time.monotonic()
                v.send(bytes([key]))
                while len(v.out) == n0 and time.monotonic() - t0 < 1.0:
                    v.pump(0.001)
                lat.append((time.monotonic() - t0) * 1000)
            lat.sort()
            print(
                f"     key latency: median {lat[len(lat) // 2]:.1f} ms, max {lat[-1]:.1f} ms"
            )
            check(
                lat[len(lat) // 2] <= 20, f"median latency {lat[len(lat) // 2]:.1f} ms"
            )
            quit_clean(v)
        finally:
            v.close()


def drain_at(v, rate, dt, feed=True):
    """Read the viewer's output at @rate bytes/s for @dt seconds, like a slow link.
    Returns (time, bytes) of each read."""
    reads = []
    end = time.monotonic() + dt
    while time.monotonic() < end:
        slot = time.monotonic() + 0.02
        quota = int(rate * 0.02)
        while quota > 0:
            left = slot - time.monotonic()
            if left <= 0:
                break
            nxt = v.reply()
            r, _, _ = select.select(
                [v.master],
                [],
                [],
                left if nxt is None else max(0.0, min(left, nxt - time.monotonic())),
            )
            if not r:
                if nxt is not None and time.monotonic() < slot:
                    continue  # woken for an answer
                break
            try:
                data = os.read(v.master, min(quota, 1 << 16))
            except OSError:
                data = b""
            if not data:
                break
            quota -= len(data)
            reads.append((time.monotonic(), len(data)))
            v.got(data, feed)
        v.reply_until(slot)
    return reads


def drain_fast(v, dt):
    """Read the viewer's output as fast as it comes for @dt seconds (no screen model).
    Returns the bytes read."""
    n = 0
    end = time.monotonic() + dt
    while time.monotonic() < end:
        r, _, _ = select.select([v.master], [], [], 0.005)
        if r:
            try:
                data = os.read(v.master, 1 << 16)
            except OSError:
                data = b""
            n += len(data)
            v.got(data, feed=False)
        else:
            v.reply()
    return n


class Relay:
    """sshd between the viewer and a terminal across a link of round trip @rtt, half each way.
    sshd reads the viewer's output as soon as it comes, into a buffer of up to @cap bytes (ssh's
    channel window), and sends it at @rate bytes/s (0: the link stalls). The terminal shows it
    @rtt/2 later and answers a cursor request once shown; answers and keys take @rtt/2 back.
    """

    def __init__(self, v, rate, rtt=0.0, cap=2 << 20):
        self.v, self.rate, self.cap = v, rate, cap
        self.fifo = bytearray()  # in sshd
        self.wire = []  # (arrival, bytes) on the way to the terminal
        self.credit = 0.0
        self.last = time.monotonic()
        self.held = []  # bytes in sshd and on the wire, sampled
        self.shown = []  # (time, bytes shown so far), sampled
        self.later = []  # (time, rows, cols): sizes on their way to the viewer
        self.set_rtt(rtt)

    def set_rtt(self, rtt):
        self.fwd = rtt / 2
        self.v.delay = rtt / 2

    def resize(self, rows, cols):
        """The far terminal changes size now; the viewer hears of it half a round trip later."""
        self.v.screen.set_size(rows, cols)
        self.later.append((time.monotonic() + self.v.delay, rows, cols))

    def send(self, key):
        """A key typed on the far terminal: it reaches the viewer half a round trip later."""
        self.v.answers.append((time.monotonic() + self.v.delay, key, False))

    def run(self, dt, feed=True):
        v = self.v
        end = time.monotonic() + dt
        while True:
            now = time.monotonic()
            self.credit += (now - self.last) * self.rate
            self.last = now
            if not self.fifo:
                self.credit = min(
                    self.credit, self.rate * 0.005
                )  # slept idle: nothing saved
            while len(self.fifo) < self.cap and v.master_open:
                r, _, _ = select.select([v.master], [], [], 0)
                if not r:
                    break
                try:
                    data = os.read(v.master, min(self.cap - len(self.fifo), 1 << 16))
                except OSError:
                    data = b""
                if not data:
                    break
                self.fifo += data
            now = time.monotonic()
            n = min(int(self.credit), len(self.fifo))
            if n:
                self.wire.append((now + self.fwd, bytes(self.fifo[:n])))
                del self.fifo[:n]
                self.credit -= n
            if not self.fifo:
                self.credit = min(
                    self.credit, self.rate * 0.005
                )  # an idle link saves nothing
            while self.wire and self.wire[0][0] <= now:
                v.got(self.wire.pop(0)[1], feed)
            while self.later and self.later[0][0] <= now:
                _, rows, cols = self.later.pop(0)
                fcntl.ioctl(
                    v.master, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0)
                )
            self.held.append(len(self.fifo) + sum(len(w[1]) for w in self.wire))
            self.shown.append((now, len(v.out)))
            nxt = v.reply()
            if now >= end:
                return
            wait = min(0.005, end - now) if self.fifo else end - now
            for t in (nxt, self.wire[0][0] if self.wire else None):
                if t is not None:
                    wait = max(0.0, min(wait, t - now))
            if len(self.fifo) < self.cap:
                select.select([v.master], [], [], wait)
            else:
                time.sleep(wait)

    def until(self, pred, timeout):
        """Run until @pred() holds or @timeout seconds pass; returns pred()."""
        end = time.monotonic() + timeout
        while not pred() and time.monotonic() < end:
            self.run(0.01)
        return pred()

    def stall(self, since, least=8192):
        """Longest time after @since during which the terminal showed fewer than @least bytes."""
        pts = [p for p in self.shown if p[0] >= since]
        worst, j = 0.0, 0
        for i in range(len(pts)):
            while pts[i][1] - pts[j][1] >= least:
                j += 1
            worst = max(worst, pts[i][0] - pts[j][0])
        return worst

    def busy(self, since):
        """Bytes shown since @since over what the link could carry."""
        pts = [p for p in self.shown if p[0] >= since]
        if len(pts) < 2:
            return 0.0
        return (pts[-1][1] - pts[0][1]) / ((pts[-1][0] - pts[0][0]) * self.rate)

    def latency(self, key, txt, gone=False, timeout=10):
        """Seconds from typing @key until @txt is on the far screen (or, with @gone, off it)."""
        t0 = time.monotonic()
        self.send(key)
        self.until(lambda: (txt in self.v.screen.text()) != gone, timeout)
        return time.monotonic() - t0

    def quit(self, timeout=5.0):
        """q, and the viewer exits, as soon as the answers on the way are in: the key's way
        there, the queue, an answer's way back. Then the link drains, answering what it still
        carries, until the terminal (with its echoes) is quiet."""
        held = len(self.fifo) + sum(len(w[1]) for w in self.wire)
        t0 = time.monotonic()
        self.send(b"q")
        self.until(lambda: not self.v.alive(), timeout)
        took = time.monotonic() - t0
        check(not self.v.alive(), "viewer did not quit")
        check(self.v.exit_code() == 0, f"exit status {self.v.exit_code()}")
        bound = 3 * self.fwd + held / self.rate + 0.35
        check(took < bound, f"quit took {took:.2f} s, the link needed {bound:.2f}")
        idle = lambda: not select.select([self.v.master], [], [], 0.05)[0]  # echoes too
        self.until(
            lambda: not self.fifo and not self.wire and not self.v.answers and idle(),
            30.0,
        )


def key_latency(v, key, txt, rate, timeout=10, gone=False):
    """Seconds from sending @key until @txt is on screen (or, with @gone, off it), reading at
    @rate."""
    t0 = time.monotonic()
    v.send(key)
    while (txt in v.screen.text()) == gone and time.monotonic() - t0 < timeout:
        drain_at(v, rate, 0.01)
    return time.monotonic() - t0


def flush_sizes(data):
    """Bytes of each synchronized update in @data."""
    return [len(c) for c in data.split(SYNC_ON)[1:]]


def runq_wait(pid):
    """Seconds @pid ("self" for this process) has waited for a CPU; 0 if unknown."""
    try:
        with open(f"/proc/{pid}/schedstat") as f:
            return int(f.read().split()[1]) / 1e9
    except (OSError, IndexError, ValueError):
        return 0.0


def quit_slow(v, rate):
    v.send(b"q")
    drain_at(v, rate, 0.5)
    check(v.wait_exit(3.0), "viewer did not quit")


def perf_slow_terminal(cfg):
    """Over a 1 MB/s link a noisy 300x90 image cannot be sent whole: keys must still show fast.
    The terminal does not answer cursor requests, so this is the blocked-write budget's work.
    """
    with ShmDir(cfg.mkstream) as shm:
        s = shm.stream("noise", "-s", "256x256", "-p", "noise", "-r", "200")
        v = start(cfg, shm, s, rows=90, cols=300, answer=False)
        try:
            drain_at(v, 1_000_000, 3.0)  # past the startup request
            lat = []
            for gone in (False, True) * 4:
                lat.append(key_latency(v, b"h", "flipH", 1_000_000, gone=gone))
                drain_at(v, 1_000_000, 0.3)
            lat.sort()
            med = lat[len(lat) // 2]
            print(
                f"     slow link: key latency median {med * 1000:.0f} ms,",
                f"max {lat[-1] * 1000:.0f} ms",
            )
            check(med < 0.15, f"median key latency {med:.2f} s over a 1 MB/s link")
            check(lat[-1] < 0.4, f"max key latency {lat[-1]:.2f} s over a 1 MB/s link")

            # the link gets fast again: whole frames go out again, one per flush (not checked
            # when the viewer or this reader waited for a CPU: then cut frames are right)
            n0, w0 = len(v.out), runq_wait(v.pid) + runq_wait("self")
            mbs = drain_fast(v, 2.0) / 2.0 / 1e6
            busy = (runq_wait(v.pid) + runq_wait("self") - w0) / 2.0
            sizes = sorted(flush_sizes(bytes(v.out[n0:]))[-20:])
            med = sizes[len(sizes) // 2] if sizes else 0
            print(
                f"     fast again: {mbs:.1f} MB/s, median flush {med / 1024:.0f} KB,"
                f" CPU wait {busy * 1000:.0f} ms/s"
            )
            if busy < 0.02:
                check(
                    med > 400 * 1024,
                    f"median flush {med / 1024:.0f} KB: the budget stayed on",
                )
            else:
                print("     (not checked: the viewer or the reader waited for a CPU)")

            # ... and slow again: the first write that blocks sets the budget
            drain_at(v, 1_000_000, 1.5, feed=False)
            n0 = len(v.out)
            drain_at(v, 1_000_000, 1.5, feed=False)
            big = max(flush_sizes(bytes(v.out[n0:])) or [0])
            print(f"     slow again: largest flush {big / 1024:.0f} KB")
            check(
                big < 200 * 1024,
                f"{big / 1024:.0f} KB flush 1.5 s after the link slowed",
            )
            quit_slow(v, 1_000_000)
        finally:
            v.close()


def perf_static_repaint_slow_link(cfg):
    """A static image over a 1 MB/s link: after a quiet spell a repaint must not stall keys
    (blocked-write budget: no answers)."""
    with ShmDir(cfg.mkstream) as shm:
        s = shm.stream("still", "-s", "256x256", "-p", "noise")
        v = start(cfg, shm, s, rows=90, cols=300, answer=False)
        try:
            drain_at(v, 1_000_000, 4.0)  # first paint, then only the 1 Hz status
            t0 = time.monotonic()
            v.send(b"h")  # flips the whole image: about 540 KB to send
            drain_at(v, 1_000_000, 0.1)
            lat = key_latency(v, b"v", "flipV", 1_000_000)
            n0 = len(v.out)
            reads = drain_at(v, 1_000_000, 3.0)
            big = [t for t, n in reads if n > 1024]
            done = (big[-1] if big else time.monotonic()) - t0
            kb = (len(v.out) - n0) / 1024
            print(
                f"     static repaint: key latency {lat * 1000:.0f} ms, {kb:.0f} KB in {done:.2f} s"
            )
            check(
                lat < 0.2,
                f"key latency {lat:.2f} s behind a repaint over a 1 MB/s link",
            )
            check(kb > 400, f"only {kb:.0f} KB of the repaint arrived")
            check(
                done < 2.6, f"two repaints of about 540 KB took {done:.2f} s at 1 MB/s"
            )
            quit_slow(v, 1_000_000)
        finally:
            v.close()


def perf_box_select_slow_link(cfg):
    """Holding a box-zoom selection over a slow link: the dimmed image stays within the budget
    (blocked-write budget: no answers)."""
    with ShmDir(cfg.mkstream) as shm:
        s = shm.stream("noise", "-s", "256x256", "-p", "noise", "-r", "200")
        v = start(cfg, shm, s, rows=90, cols=300, answer=False)
        try:
            drain_at(v, 1_000_000, 3.0)  # past the startup request
            v.send(b"\x1b[<2;10;10M\x1b[<34;250;80M")
            drain_at(v, 1_000_000, 0.5)
            n0 = len(v.out)
            drain_at(v, 1_000_000, 2.0)
            big = max(flush_sizes(bytes(v.out[n0:])) or [0])
            print(f"     box selection: largest flush {big / 1024:.0f} KB")
            check(
                big < 200 * 1024,
                f"{big / 1024:.0f} KB flush while a box selection is held",
            )
            v.send(b"\x1b")
            drain_at(v, 1_000_000, 0.2)  # past the Esc timeout: 'q' is not Alt+q
            quit_slow(v, 1_000_000)
        finally:
            v.close()


def perf_low_fps_slow_link(cfg):
    """-f 2 on a link slower than the frames: about 2 frames/s are sent, not the link's worth."""
    with ShmDir(cfg.mkstream) as shm:
        s = shm.stream("noise", "-s", "256x256", "-p", "noise", "-r", "200")
        got = {}
        for fps in ("20", "2"):
            v = start(cfg, shm, s, "-f", fps, rows=90, cols=300)
            try:
                drain_at(v, 4_000_000, 1.5, feed=False)
                n0 = len(v.out)
                drain_at(v, 4_000_000, 3.0, feed=False)
                got[fps] = (len(v.out) - n0) / 3.0 / 1e6
                print(f"     -f {fps} at 4 MB/s: {got[fps]:.2f} MB/s sent")
                quit_slow(v, 4_000_000)
            finally:
                v.close()
        # two frames of about 0.53 MB are all there is to send, however busy the machine
        check(
            got["2"] < 1.6, f"{got['2']:.2f} MB/s at -f 2: frames are not limited to -f"
        )
        if got["20"] >= 3.5:
            ratio = got["2"] / got["20"]
            check(
                ratio < 0.45,
                f"-f 2 sends {ratio:.2f} of -f 20: frames are not limited to -f",
            )


def first_paint(cfg, shm, stream, env):
    """Bytes of the first full paint of a static image at 300x90, however many flushes flow
    control cuts it into (plus a status update or two)."""
    v = Viewer(cfg.termview, [stream], env, 90, 300)
    try:
        check(v.wait_for(ENTER, 3.0), "viewer did not start")
        drain_fast(v, 1.5)
        paint = sum(flush_sizes(bytes(v.out)))
        quit_clean(v)
        return paint, bytes(v.out)
    finally:
        v.close()


def lowbw_rate(cfg, shm, stream, keys, warmup=1.5, window=4.0):
    """Bytes/s the viewer writes for @stream at 300x90 after @keys, read as fast as it comes."""
    v = Viewer(cfg.termview, [stream], shm.env, 90, 300)
    try:
        check(v.wait_for(ENTER, 3.0), "viewer did not start")
        v.send(keys)
        drain_fast(v, warmup)
        rate = drain_fast(v, window) / window
        quit_clean(v)
        return rate
    finally:
        v.close()


def perf_output_volume(cfg):
    """Output at 300x90: a noise paint in truecolor and at 256 colours; what low-bandwidth mode
    sends on a steady scene, for fast and slow streams, against a whole frame per frame.
    """
    with ShmDir(cfg.mkstream) as shm:
        still = shm.stream("still", "-s", "256x256", "-p", "noise")
        env256 = dict(shm.env)
        env256.pop("COLORTERM")
        tc, _ = first_paint(cfg, shm, still, shm.env)
        c256, out256 = first_paint(cfg, shm, still, env256)
        print(
            f"     noise paint: {tc / 1024:.0f} KB truecolor, {c256 / 1024:.0f} KB at 256 colours"
        )
        check(
            tc <= 580 * 1024, f"truecolor paint {tc / 1024:.0f} KB"
        )  # 88 x 287 image cells
        check(
            b"8;5;" in out256 and b"8;2;" not in out256,
            "256 colours: 24-bit codes sent",
        )
        check(0.4 <= c256 / tc <= 0.75, f"256 colours: {c256 / tc:.2f} of truecolor")
        paint, _ = first_paint(
            cfg, shm, shm.stream("jitter", "-s", "256x256", "-p", "jitter"), shm.env
        )
        for hz, warmup, window in ((200, 1.5, 4.0), (3, 1.5, 4.0), (0.5, 6.0, 10.0)):
            s = shm.stream(f"jitter{hz}", "-s", "256x256", "-p", "jitter", "-r", hz)
            on = lowbw_rate(cfg, shm, s, b"b", warmup, window)
            ratio = on / (
                paint * min(hz, 20)
            )  # normal mode redraws nearly every cell per frame
            print(
                f"     jitter {hz} Hz, low-bandwidth: {on / 1024:.0f} KB/s, {ratio:.2f} of a frame"
            )
            check(
                0.03 <= ratio <= 0.3,
                f"{hz} Hz: low-bandwidth sends {ratio:.2f} of a frame",
            )


def perf_menu_esc(cfg):
    """Esc closes a menu within the ESC timeout plus one interactive frame (50 + 16 ms)."""
    with ShmDir(cfg.mkstream) as shm:
        s = shm.stream(
            "esc", "-s", "256x256"
        )  # static: only the viewer's latency is timed
        v = start(cfg, shm, s, rows=60, cols=200)
        try:
            v.pump(1.0)
            lat = []
            for k in range(6):
                v.send(b"\x1bOR")  # F3: Scale
                check(v.screen_has("log_extreme", 1.0), "F3 did not open the menu")
                v.pump(0.2)
                t0 = time.monotonic()
                v.send(b"\x1b")
                while "log_extreme" in v.screen.text() and time.monotonic() - t0 < 1.0:
                    v.pump(0.002)
                lat.append((time.monotonic() - t0) * 1000)
            lat.sort()
            print(
                f"     Esc closes a menu: median {lat[3]:.0f} ms, max {lat[-1]:.0f} ms"
            )
            check(lat[3] < 66, f"median {lat[3]:.0f} ms to close a menu")
            quit_clean(v)
        finally:
            v.close()


def relay_keys(relay, n=3):
    """Median and max seconds for @n pairs of h presses to show (and clear) flipH."""
    lat = []
    for gone in (False, True) * n:
        lat.append(relay.latency(b"h", "flipH", gone=gone))
        relay.run(0.3)
    lat.sort()
    return lat[len(lat) // 2], lat[-1]


def perf_ssh_relay(cfg):
    """Over ssh, writes do not block: sshd buffers up to 2 MB, which a slow link takes seconds
    to send. The terminal's answers to cursor requests keep a round trip plus a frame period of
    output in flight: keys show within the round trip plus about a frame period, and the link
    stays busy. Without answers the whole buffer is in front of each key."""
    with ShmDir(cfg.mkstream) as shm:
        s = shm.stream("noise", "-s", "256x256", "-p", "noise", "-r", "200")
        rate = 1_000_000
        for rtt in (0.02, 0.1, 0.3):
            v = Viewer(cfg.termview, [s], shm.env, 90, 300)
            try:
                relay = Relay(v, rate, rtt)  # from the first byte: the first paint too
                check(relay.until(lambda: ENTER in v.out, 3.0), "viewer did not start")
                relay.run(1.0)
                t0 = time.monotonic()
                relay.run(2.0)
                early = relay.busy(
                    t0
                )  # the window grows a window per round trip at first
                t0 = time.monotonic()
                relay.held = []
                relay.run(6.0)  # a re-timing of the round trip falls in here
                busy, stall = relay.busy(t0), relay.stall(t0)
                held = sum(relay.held) / max(1, len(relay.held))
                med, worst = relay_keys(relay)
                print(
                    f"     ssh relay, round trip {rtt * 1000:.0f} ms: keys median",
                    f"{med * 1000:.0f} ms, max {worst * 1000:.0f} ms; link {busy * 100:.0f}%",
                    f"busy ({early * 100:.0f}% in seconds 1-3), {held / 1024:.0f} KB on the way,",
                    f"longest stall {stall * 1000:.0f} ms",
                )
                check(med < rtt + 0.15, f"median key latency {med:.2f} s")
                check(
                    early > 0.6, f"the link was {early * 100:.0f}% busy in seconds 1-3"
                )
                check(worst < rtt + 0.4, f"max key latency {worst:.2f} s")
                check(busy > 0.8, f"the link is {busy * 100:.0f}% busy")
                check(stall < 0.15 + 2 * rtt, f"the image stopped for {stall:.2f} s")
                if rtt == 0.3:
                    # the far terminal is dragged narrower, a column every 20 ms: answers to
                    # requests on the way still match theirs, nothing piles up in sshd
                    relay.held = []
                    relay.run(1.0)
                    before = max(relay.held)
                    relay.held = []
                    for k in range(30):
                        relay.resize(90, 299 - k)
                        relay.run(0.02)
                    relay.run(1.0)
                    after = max(relay.held)
                    med, worst = relay_keys(relay, 1)
                    print(
                        f"     dragged narrower: keys max {worst * 1000:.0f} ms, sshd held"
                        f" {after / 1024:.0f} KB (before {before / 1024:.0f} KB)"
                    )
                    check(
                        worst < rtt + 0.4, f"key latency {worst:.2f} s after the drag"
                    )
                    check(
                        after < 2 * before + 65536,
                        f"{after} bytes in sshd after the drag",
                    )
                if rtt == 0.02:
                    # another route: the round trip grows tenfold; the link is busy again soon
                    relay.set_rtt(0.2)
                    relay.run(7.0)
                    t1 = time.monotonic()
                    relay.run(3.0)
                    busy = relay.busy(t1)
                    print(f"     round trip now 200 ms: link {busy * 100:.0f}% busy")
                    check(
                        busy > 0.75,
                        f"the link is {busy * 100:.0f}% busy after the change",
                    )
                    # the link stalls for 8 s: then all is as before, flow control on
                    relay.rate = 0
                    relay.run(8.0)
                    relay.rate = rate
                    relay.run(3.0)
                    asked = v.asked
                    med, worst = relay_keys(relay, 2)
                    print(f"     after an 8 s stall: keys median {med * 1000:.0f} ms")
                    check(v.asked > asked + 10, "no cursor requests after the stall")
                    check(
                        worst < 0.2 + 0.4,
                        f"max key latency {worst:.2f} s after a stall",
                    )
                relay.quit()
            finally:
                v.close()

        # a fast link with a long round trip: at first the window grows a window per round
        # trip, not only with the rate it measures (bursts too short to time the link)
        v = Viewer(cfg.termview, [s], shm.env, 90, 300)
        try:
            relay = Relay(v, 2_000_000, 1.0)
            check(relay.until(lambda: ENTER in v.out, 3.0), "viewer did not start")
            relay.run(2.0, feed=False)
            t0 = time.monotonic()
            relay.run(2.0, feed=False)
            busy = relay.busy(t0)
            print(
                f"     2 MB/s, round trip 1 s: link {busy * 100:.0f}% busy in seconds 2-4"
            )
            check(busy > 0.5, f"the link was {busy * 100:.0f}% busy in seconds 2-4")
            v.send(b"q")
            relay.until(lambda: not v.alive(), 5.0)
        finally:
            v.close()

        # a slow link: the smallest window still keeps keys within a few tenths
        v = Viewer(cfg.termview, [s], shm.env, 90, 300)
        try:
            relay = Relay(v, 50_000, 0.05)
            check(relay.until(lambda: ENTER in v.out, 3.0), "viewer did not start")
            relay.run(6.0)
            asked = v.asked
            w0, t0 = v.wakeups(), time.monotonic()
            relay.run(
                3.0
            )  # the image held most of the time: the viewer waits for answers
            wk = (v.wakeups() - w0) / (time.monotonic() - t0)
            med, worst = relay_keys(relay, 2)
            print(
                f"     50 KB/s: keys median {med * 1000:.0f} ms, max {worst * 1000:.0f} ms,"
                f" {wk:.0f} wakeups/s"
            )
            check(med < 0.32, f"median key latency {med:.2f} s at 50 KB/s")
            check(worst < 0.6, f"max key latency {worst:.2f} s at 50 KB/s")
            check(v.asked > asked + 5, "flow control stopped at 50 KB/s")
            check(wk < 50, f"{wk:.0f} wakeups/s while the image is held")
            # a drag, a column every 20 ms: a new size is painted once per answer, not per step
            for k in range(30):
                relay.resize(90, 299 - k)
                relay.run(0.02)
            med, worst = relay_keys(relay, 1)
            print(f"     50 KB/s, dragged narrower: the next key {worst * 1000:.0f} ms")
            check(worst < 0.6, f"key latency {worst:.2f} s after a drag at 50 KB/s")
            relay.quit()
        finally:
            v.close()

        # a terminal that does not answer: each key waits behind the whole buffer
        v = Viewer(cfg.termview, [s], shm.env, 90, 300, answer=False)
        try:
            relay = Relay(v, rate, 0.02)
            relay.run(4.0)
            med, _ = relay_keys(relay, 1)
            print(f"     no answers: keys median {med * 1000:.0f} ms")
            check(med > 1.0, f"{med:.2f} s without answers: the relay holds nothing")
            relay.quit()
        finally:
            v.close()

        # a large terminal: its first paint (about 1 MB) does not go out in one piece either
        v = Viewer(cfg.termview, [s], shm.env, 120, 400)
        try:
            relay = Relay(v, rate, 0.02)
            check(relay.until(lambda: ENTER in v.out, 3.0), "viewer did not start")
            relay.run(0.1)
            first = relay.latency(b"h", "flipH")
            print(f"     400x120: key latency {first * 1000:.0f} ms at start")
            check(first < 0.45, f"key latency {first:.2f} s at start")
            relay.quit()
        finally:
            v.close()


def perf_first_paint_local(cfg):
    """A fast local terminal: a first paint larger than the first window still shows at once,
    each answer releasing the next part without waiting for the interactive cap."""
    with ShmDir(cfg.mkstream) as shm:
        s = shm.stream("still", "-s", "400x240", "-p", "noise")  # the whole screen
        took = []
        for _ in range(3):
            v = Viewer(cfg.termview, [s], shm.env, 120, 400)
            try:
                check(v.wait_for(ENTER, 3.0), "viewer did not start")
                t0 = time.monotonic()
                n0 = len(v.out)
                while len(v.out) - n0 < 800_000 and time.monotonic() - t0 < 2.0:
                    if select.select([v.master], [], [], 0.01)[0]:
                        v.got(os.read(v.master, 1 << 16), feed=False)  # no screen model
                took.append(time.monotonic() - t0)
                quit_clean(v)
            finally:
                v.close()
        took.sort()
        print(f"     first paint at 400x120 (1 MB): {took[1] * 1000:.0f} ms")
        check(took[1] < 0.02, f"first paint took {took[1] * 1000:.0f} ms")


def perf_outage_quit(cfg):
    """A 25 s outage on a narrow terminal (24 columns to tell requests apart): each answer still
    names its own request afterwards, so q soon after the link is back leaves none for the
    shell."""
    with ShmDir(cfg.mkstream) as shm:
        s = shm.stream("noise", "-s", "64x64", "-p", "noise", "-r", "100")
        v = Viewer(cfg.termview, [s], shm.env, 40, 41)
        try:
            relay = Relay(v, 100_000, 0.6)
            check(relay.until(lambda: ENTER in v.out, 3.0), "viewer did not start")
            relay.run(4.0)
            relay.rate = 0
            relay.run(25.0)
            relay.rate = 100_000
            relay.run(2.0)
            relay.send(b"q")
            check(relay.until(lambda: not v.alive(), 20.0), "viewer did not quit")
            idle = lambda: not select.select([v.master], [], [], 0.05)[0]
            relay.until(
                lambda: not relay.fifo and not relay.wire and not v.answers and idle(),
                20.0,
            )
            tail = bytes(v.out[v.out.rfind(LEAVE) :])
            print(f"     25 s outage, 41 columns: {tail.count(b'^[[')} answer(s) left")
            check(
                b"^[[" not in tail, f"{tail.count(b'^[[')} answer(s) reached the shell"
            )
        finally:
            v.close()


PERF = [
    perf_idle_static_stream,
    perf_key_latency,
    perf_slow_terminal,
    perf_static_repaint_slow_link,
    perf_box_select_slow_link,
    perf_low_fps_slow_link,
    perf_output_volume,
    perf_menu_esc,
    perf_ssh_relay,
    perf_outage_quit,
    perf_first_paint_local,
]


# ------------------------------------------------------------------ benchmark


def bench(cfg):
    """CPU, output bytes and wakeups of a running viewer; drives it with keys only."""
    tck = os.sysconf("SC_CLK_TCK")
    win = cfg.window
    print(
        "| terminal | content | mode | CPU % | KB/s | KB/frame | frames/s | wakeups/s |"
    )
    print("|---|---|---|---|---|---|---|---|")
    with ShmDir(cfg.mkstream) as shm:
        contents = {
            "static": shm.stream("bstatic", "-s", "256x256", "-p", "ramp"),
            "noise 200 Hz": shm.stream(
                "bnoise", "-s", "256x256", "-p", "noise", "-r", "200"
            ),
            "jitter 200 Hz": shm.stream(
                "bjitter", "-s", "256x256", "-p", "jitter", "-r", "200"
            ),
        }
        env256 = dict(shm.env)
        env256.pop("COLORTERM", None)
        modes = [("minmax", b"", shm.env), ("1-99%", b"rrr", shm.env)]
        if not cfg.bench_basic:
            modes += [("minmax lowbw", b"b", shm.env), ("minmax 256 col", b"", env256)]
        for rows, cols in ((40, 120), (60, 200), (90, 300)):
            for cname, s in contents.items():
                for mname, keys, env in modes:
                    v = Viewer(cfg.termview, [s], env, rows, cols)
                    try:
                        v.wait_for(ENTER, 3.0)
                        v.send(keys)
                        v.pump(1.0)
                        n0, t0, w0 = len(v.out), v.cpu_ticks(), v.wakeups()
                        f0 = v.out.count(b"\x1b[0m")
                        v.pump(win)
                        n1, t1, w1 = len(v.out), v.cpu_ticks(), v.wakeups()
                        frames = (v.out.count(b"\x1b[0m") - f0) / win
                        kbs = (n1 - n0) / 1024 / win
                        cpu = (t1 - t0) / tck / win * 100
                        kbf = kbs / frames if frames else 0.0
                        wk = (w1 - w0) / win
                        print(
                            f"| {cols}x{rows} | {cname} | {mname} | {cpu:.1f} | {kbs:.0f} | "
                            f"{kbf:.1f} | {frames:.1f} | {wk:.0f} |",
                            flush=True,
                        )
                        v.send(b"q")
                        v.wait_exit(2.0)
                    finally:
                        v.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--termview", required=True)
    ap.add_argument("--mkstream", required=True)
    ap.add_argument("--only", action="append", default=[])
    ap.add_argument("--bench", action="store_true")
    ap.add_argument(
        "--perf", action="store_true", help="run the CPU/latency checks instead"
    )
    ap.add_argument("--window", type=float, default=3.0)
    ap.add_argument(
        "--sanitized",
        action="store_true",
        help="sanitizer build: stderr kept, own allocator mappings, no address-space limits",
    )
    ap.add_argument(
        "--bench-basic", action="store_true", help="bench: only the two basic modes"
    )
    cfg = ap.parse_args()

    if cfg.bench:
        bench(cfg)
        return 0

    failed = 0
    for t in PERF if cfg.perf else TESTS:
        if cfg.only and not any(o in t.__name__ for o in cfg.only):
            continue
        t0 = time.monotonic()
        try:
            t(cfg)
            print(f"PASS {t.__name__} ({time.monotonic() - t0:.1f} s)", flush=True)
        except (
            Exception
        ) as e:  # a hang or crash of the viewer is a failure, not a harness error
            failed += 1
            print(f"FAIL {t.__name__}: {type(e).__name__}: {e}", flush=True)
    print(f"{failed} failure(s)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
