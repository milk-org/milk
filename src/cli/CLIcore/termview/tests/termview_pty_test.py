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


class Screen:
    """Minimal VT model of what milk-termview emits: CUP, CUF, ED, SGR, private modes, text.
    With @colour, SGR colours are kept per cell too (slower)."""

    def __init__(self, rows, cols, colour=False):
        self.colour = colour
        self.bg = self.fg = None
        self.rows, self.cols = rows, cols
        self.resize(rows, cols)
        self.pending = b""

    def resize(self, rows, cols):
        self.rows, self.cols = rows, cols
        self.grid = [[" "] * cols for _ in range(rows)]
        self.attr = [[(None, None)] * cols for _ in range(rows)]
        self.r = self.c = 0

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
        if final == "H":
            self.r = (nums[0] if nums and nums[0] else 1) - 1
            self.c = (nums[1] if len(nums) > 1 and nums[1] else 1) - 1
        elif final == "C":
            self.c = min(self.cols - 1, self.c + (nums[0] if nums and nums[0] else 1))
        elif final == "J" and nums and nums[0] == 2:
            self.resize(self.rows, self.cols)
        elif final == "m" and self.colour:
            self._sgr(nums or [0])

    def text(self):
        return "\n".join("".join(row) for row in self.grid)


class Viewer:
    """milk-termview running on its own pty, with the slave side kept for termios checks."""

    def __init__(
        self, exe, args, env, rows=40, cols=120, block_hup=False, colour=False
    ):
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
                if block_hup:
                    signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGHUP})
                resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
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
            r, _, _ = select.select([self.master], [], [], max(0.0, left))
            if not r:
                return
            try:
                data = os.read(self.master, 1 << 16)
            except OSError:
                data = b""
            if not data:
                time.sleep(min(0.01, max(0.0, end - time.monotonic())))
                continue
            self.out += data
            self.screen.feed(data)
        time.sleep(max(0.0, end - time.monotonic()))

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
        cmd = [self.mkstream, "-n", name] + [str(a) for a in args]
        live = "-r" in args and float(args[list(args).index("-r") + 1]) > 0
        if live:
            self.procs.append(subprocess.Popen(cmd, env=self.env))
            path = os.path.join(self.dir, name + ".im.shm")
            end = time.monotonic() + 3
            while not os.path.exists(path) and time.monotonic() < end:
                time.sleep(0.01)
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


def start(cfg, shm, stream, *opts, rows=40, cols=120, block_hup=False, colour=False):
    v = Viewer(
        cfg.termview, list(opts) + [stream], shm.env, rows, cols, block_hup, colour
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
            check(v.screen_has("LOCKED"), "valid min did not lock the range")
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
            check(v.screen_has("UI:  20 FPS"), "status line not found")
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
            check("UI:  20 FPS" in text, "mouse bytes changed the refresh rate")
            check("Enter Manual" not in text, "mouse bytes opened the min/max prompt")
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
            check(v.screen_has("UI:   1 FPS", 2.0), "status line not found")
            v.send(b"m")
            check(v.screen_has("Enter Manual Min", 1.0), "prompt did not open")
            v.pump(0.3)
            t0 = time.monotonic()
            v.send(b"\x1b")
            while "Enter Manual" in v.screen.text() and time.monotonic() - t0 < 2.0:
                v.pump(0.005)
            esc_ms = (time.monotonic() - t0) * 1000
            check(esc_ms < 250, f"lone Esc took {esc_ms:.0f} ms to close the prompt")
            v.pump(0.3)
            t0 = time.monotonic()
            v.send(b"h")
            time.sleep(0.005)
            v.send(b"v")
            while "v:FLIPV" not in v.screen.text() and time.monotonic() - t0 < 2.0:
                v.pump(0.005)
            keys_ms = (time.monotonic() - t0) * 1000
            check("h:FLIPH" in v.screen.text(), "first key lost")
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
    """The 'Val: [lo : hi]' range shown in the info bar, or None."""
    import re

    m = re.search(r"Val: \[([-+0-9.eE]+) : ([-+0-9.eE]+)\]", v.screen.text())
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
                v, lambda r: abs(r[0] - 200) < 3 and abs(r[1] - 800) < 3, 1.5
            )
            check(r and abs(r[0] - 200) < 3 and abs(r[1] - 800) < 3, f"20-80% gave {r}")
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
            v.send(b"m")  # an input-driven frame right after the new data
            r = wait_range(v, lambda r: r[0] > 40 and r[1] < 44, 2.5)
            check(r and r[0] > 40 and r[1] < 44, f"new frame never shown: range {r}")
            v.send(b"\x1b")
            v.pump(0.2)
            quit_clean(v)
        finally:
            v.close()


def lowbw_diff(v):
    """Cells of the image area that differ from an exact (Ctrl-L) repaint."""
    rows = range(v.screen.rows - 4)  # the image and colorbar; not the status rows
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
                check(v.screen_has("b:LOWBW"), "low-bandwidth mode did not turn on")
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
            m = re.search(r"Val: \[[^]]*\]", v.screen.text())
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
            r, _, _ = select.select([v.master], [], [], left)
            if not r:
                break
            try:
                data = os.read(v.master, min(quota, 1 << 16))
            except OSError:
                data = b""
            if not data:
                break
            quota -= len(data)
            reads.append((time.monotonic(), len(data)))
            v.out += data
            if feed:
                v.screen.feed(data)
        time.sleep(max(0.0, slot - time.monotonic()))
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
            v.out += data
    return n


def key_latency(v, key, txt, rate, timeout=10):
    """Seconds from sending @key until @txt is on screen, reading at @rate."""
    t0 = time.monotonic()
    v.send(key)
    while txt not in v.screen.text() and time.monotonic() - t0 < timeout:
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
    """Over a 1 MB/s link a noisy 300x90 image cannot be sent whole: keys must still show fast."""
    with ShmDir(cfg.mkstream) as shm:
        s = shm.stream("noise", "-s", "256x256", "-p", "noise", "-r", "200")
        v = start(cfg, shm, s, rows=90, cols=300)
        try:
            drain_at(v, 1_000_000, 2.0)
            lat = []
            for key, txt in ((b"h", "h:FLIPH"), (b"h", "h:fliph")) * 4:
                lat.append(key_latency(v, key, txt, 1_000_000))
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
    """A static image over a 1 MB/s link: after a quiet spell a repaint must not stall keys."""
    with ShmDir(cfg.mkstream) as shm:
        s = shm.stream("still", "-s", "256x256", "-p", "noise")
        v = start(cfg, shm, s, rows=90, cols=300)
        try:
            drain_at(v, 1_000_000, 4.0)  # first paint, then only the 1 Hz status
            t0 = time.monotonic()
            v.send(b"h")  # flips the whole image: about 540 KB to send
            drain_at(v, 1_000_000, 0.1)
            lat = key_latency(v, b"v", "v:FLIPV", 1_000_000)
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
    """Holding a box-zoom selection over a slow link: the dimmed image stays within the budget."""
    with ShmDir(cfg.mkstream) as shm:
        s = shm.stream("noise", "-s", "256x256", "-p", "noise", "-r", "200")
        v = start(cfg, shm, s, rows=90, cols=300)
        try:
            drain_at(v, 1_000_000, 2.0)
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
    """Bytes of the first full paint of a static image at 300x90 (no budget: one flush)."""
    v = Viewer(cfg.termview, [stream], env, 90, 300)
    try:
        check(v.wait_for(ENTER, 3.0), "viewer did not start")
        drain_fast(v, 1.5)
        paint = max(flush_sizes(bytes(v.out)) or [0])
        quit_clean(v)
        return paint, bytes(v.out)
    finally:
        v.close()


def lowbw_rate(cfg, shm, stream, keys, warmup=1.5, window=4.0):
    """Bytes/s the viewer writes for @stream at 300x90 after @keys, read as fast as it comes,
    and the size of its first (whole, unbudgeted) paint."""
    v = Viewer(cfg.termview, [stream], shm.env, 90, 300)
    try:
        check(v.wait_for(ENTER, 3.0), "viewer did not start")
        v.send(keys)
        drain_fast(v, warmup)
        paint = (flush_sizes(bytes(v.out)) or [0])[0]
        rate = drain_fast(v, window) / window
        quit_clean(v)
        return rate, paint
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
        check(tc <= 560 * 1024, f"truecolor paint {tc / 1024:.0f} KB")
        check(
            b"8;5;" in out256 and b"8;2;" not in out256,
            "256 colours: 24-bit codes sent",
        )
        check(0.4 <= c256 / tc <= 0.75, f"256 colours: {c256 / tc:.2f} of truecolor")
        for hz, warmup, window in ((200, 1.5, 4.0), (3, 1.5, 4.0), (0.5, 6.0, 10.0)):
            s = shm.stream(f"jitter{hz}", "-s", "256x256", "-p", "jitter", "-r", hz)
            on, paint = lowbw_rate(cfg, shm, s, b"b", warmup, window)
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


PERF = [
    perf_idle_static_stream,
    perf_key_latency,
    perf_slow_terminal,
    perf_static_repaint_slow_link,
    perf_box_select_slow_link,
    perf_low_fps_slow_link,
    perf_output_volume,
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
