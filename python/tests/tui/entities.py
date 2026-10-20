"""Synthetic milk entities for TUI tests: streams, FPS, processinfo loops and tmux sessions."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

# A processinfo loop at about 20 Hz that honours CTRLval (1 pause, 3 exit) and prints its PID.
# SIGUSR1 toggles a stall: loopcnt stops while loopstat stays ACTIVE.
_FAKEPROC = textwrap.dedent("""
    import os, signal, sys, time
    from pyMilk.interfacing.pinfo import ProcessInfo
    stalled = [False]
    signal.signal(signal.SIGUSR1, lambda *a: stalled.__setitem__(0, not stalled[0]))
    p = ProcessInfo()
    p.create(sys.argv[1], 0)
    p.loopstat = 1
    print(os.getpid(), flush=True)
    while True:
        if p.CTRLval == 3:
            break
        if p.CTRLval != 1 and not stalled[0]:
            p.loopcnt = p.loopcnt + 1
        time.sleep(0.05)
    p.loopstat = 3
    p.close()
    """)

_MKFPS = textwrap.dedent("""
    import sys
    from pyMilk.interfacing.fps import FPS, FPS_type, FPS_flags
    fps = FPS.create(sys.argv[1], force_recreate=True)
    for spec in sys.argv[2:]:
        key, typ, val = spec.split(':', 2)
        t = getattr(FPS_type, typ)
        fps.add_param(key, key + ' comment', t, FPS_flags.DEFAULT_INPUT)
        if val:
            fps.set_param(key, type(fps.get_param(key))(val) if t != FPS_type.STRING else val)
    """)


class Entities:
    """Create entities inside the test's isolated MILK_SHM_DIR; everything is torn down by cleanup()."""

    def __init__(self, env: dict[str, str], shm_dir: str):
        self.env = env
        self.shm = Path(shm_dir)
        self.procs: list[subprocess.Popen] = []

    def _run(self, argv: list[str]) -> subprocess.CompletedProcess:
        return subprocess.run(
            argv, env=self.env, text=True, capture_output=True, timeout=30, check=True
        )

    def make_stream(
        self, name: str, x: int = 16, y: int = 16, dtype: str = "float"
    ) -> Path:
        self._run(["milk-stream-create", "-t", dtype, name, str(x), str(y)])
        return self.shm / f"{name}.im.shm"

    def make_hostile_stream(self, name: str, hostile: str) -> Path:
        """Create a valid stream, then rename its file to a name the CLI tools would reject."""
        src = self.make_stream(name)
        dst = self.shm / f"{hostile}.im.shm"
        os.rename(src, dst)
        return dst

    def make_fps(self, name: str, *params: str) -> Path:
        """params are 'key:TYPE:value', TYPE from pyMilk FPS_type (INT64, FLOAT64, STRING, ONOFF...)."""
        self._run([sys.executable, "-c", _MKFPS, name, *params])
        return self.shm / f"{name}.fps.shm"

    def make_proc(self, name: str) -> int:
        """Start a fake processinfo loop; returns its PID once the registry entry exists."""
        p = subprocess.Popen(
            [sys.executable, "-c", _FAKEPROC, name],
            env=self.env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        self.procs.append(p)
        line = ""
        for _ in range(20):  # the library prints its own messages before our PID line
            line = p.stdout.readline().strip()
            if line.isdigit() or not line:
                break
        if not line.isdigit():
            raise RuntimeError(f"fake process {name} failed to start")
        deadline = time.monotonic() + 5
        while (
            not list(self.shm.glob(f"proc.{name}.*.shm"))
            and time.monotonic() < deadline
        ):
            time.sleep(0.05)
        return int(line)

    def kill(self, pid: int, sig: int = signal.SIGKILL) -> None:
        """Signal a process started by make_proc and reap it (no lingering zombie)."""
        os.kill(pid, sig)
        for p in self.procs:
            if p.pid == pid:
                p.wait(timeout=5)

    def tmux_session(self, server, name: str) -> None:
        """A session shaped like an FPS one: windows ctrl, conf, run running cat."""
        s = server.new_session(
            session_name=name, window_name="ctrl", window_command="cat"
        )
        s.new_window(window_name="conf", window_shell="cat")
        s.new_window(window_name="run", window_shell="cat")

    def cleanup(self) -> None:
        for p in self.procs:
            if p.poll() is None:
                p.send_signal(signal.SIGKILL)
                p.wait(timeout=5)
