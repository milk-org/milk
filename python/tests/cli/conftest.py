"""Shared test helpers and fixtures for milk CLI, milk-script, and milk.bash tests."""

from __future__ import annotations

import contextlib
import fcntl
import os
import pty
import re
import select
import shutil
import signal
import subprocess
import tempfile
import termios
import time
import typing as typ
from pathlib import Path

import pytest

from milk.cliwrap import MILK_CLI_EXEC

_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def bug(finding: str, reason: str = "", strict: bool = True):
    """Xfail marker tagged with a review finding ID (strict unless the repro is timing-dependent)."""
    return pytest.mark.xfail(strict=strict, reason=f"[{finding}] {reason}")


def _install_dir() -> Path:
    return Path(os.environ["MILK_INSTALLDIR"]).resolve()


def _session_pids(sid: int) -> list[int]:
    pids = []
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            stat = Path(f"/proc/{d}/stat").read_text()
        except OSError:
            continue
        fields = stat.rsplit(")", 1)[1].split()
        if int(fields[3]) == sid:
            pids.append(int(d))
    return pids


def _kill_session(sid: int) -> None:
    """Kill every process in session @sid (milk-cli puts bash in its own pgrp)."""
    for pid in _session_pids(sid):
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signal.SIGKILL)


def _kill_by_cmdline(token: str) -> None:
    """Kill processes whose command line contains @token (auto-spawned milk servers live in their own session)."""
    for d in os.listdir("/proc"):
        if not d.isdigit() or int(d) == os.getpid():
            continue
        try:
            cmd = (
                Path(f"/proc/{d}/cmdline")
                .read_bytes()
                .replace(b"\0", b" ")
                .decode(errors="replace")
            )
        except OSError:
            continue
        if token in cmd:
            with contextlib.suppress(ProcessLookupError):
                os.kill(int(d), signal.SIGKILL)


def _isolated_shm_env(
    extra_env: dict[str, str] | None = None,
) -> tuple[dict[str, str], str]:
    """Build a subprocess env with an isolated MILK_SHM_DIR, MILK_PROC_DIR, and HOME."""
    install_dir = _install_dir()
    base_shm = os.environ.get("MILK_SHM_DIR", "/tmp")
    os.makedirs(base_shm, exist_ok=True)
    shm_dir = tempfile.mkdtemp(prefix="milk_bash_", dir=base_shm)
    home = Path(shm_dir) / "home"
    home.mkdir(exist_ok=True)

    env = os.environ.copy()
    env.update(
        {
            "MILK_INSTALLDIR": str(install_dir),
            "PATH": f"{install_dir / 'bin'}:{env.get('PATH', '')}",
            "LD_LIBRARY_PATH": f"{install_dir / 'lib'}:{env.get('LD_LIBRARY_PATH', '')}",
            "MILK_SHM_DIR": shm_dir,
            "MILK_PROC_DIR": shm_dir,
            "HOME": str(home),
            "MILK_QUIET": "1",
        }
    )
    if extra_env:
        env.update(extra_env)
    return env, shm_dir


_env = _isolated_shm_env


def _run(
    argv: list[str],
    env: dict[str, str],
    timeout: float,
    stdin: str | None = None,
    cwd: str | None = None,
) -> subprocess.CompletedProcess[str]:
    """subprocess.run that also reaps the whole session on timeout."""
    proc = subprocess.Popen(
        argv,
        env=env,
        cwd=cwd,
        text=True,
        start_new_session=True,
        stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        out, err = proc.communicate(stdin, timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_session(proc.pid)
        try:
            out, err = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            out, err = "", ""
        return subprocess.CompletedProcess(
            argv, 124, _ANSI.sub("", out or ""), _ANSI.sub("", err or "")
        )
    _kill_session(proc.pid)
    return subprocess.CompletedProcess(
        argv, proc.returncode, _ANSI.sub("", out), _ANSI.sub("", err)
    )


def milk_bash(
    script: str,
    tmp_path: Path | None = None,
    extra_env: dict[str, str] | None = None,
    timeout: float = 30,
    errexit: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run @script in bash after `source milk.bash`, with an isolated HOME and SHM dir."""
    env, shm_dir = _env(extra_env)
    try:
        full = f'source "{_install_dir() / "bin" / "milk.bash"}"\n{script}\n'
        flags = ["-e", "-c"] if errexit else ["-c"]
        return _run(
            ["/bin/bash", *flags, full],
            env,
            timeout,
            cwd=str(tmp_path) if tmp_path else None,
        )
    finally:
        shutil.rmtree(shm_dir, ignore_errors=True)


def _run_bash(
    script: str, extra_env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return milk_bash(script, extra_env=extra_env, timeout=25, errexit=True)


def milk_cli_env(
    script: str, tmp_path: Path, env: dict[str, str], timeout: float = 20
) -> subprocess.CompletedProcess[str]:
    """`milk-cli -s` with a caller-provided environment."""
    f = tmp_path / "s.milk"
    f.write_text(script + "\n")
    return _run([MILK_CLI_EXEC, "-s", str(f)], env, timeout, cwd=str(tmp_path))


def milk_cli_script(
    script: str, tmp_path: Path, timeout: float = 20
) -> subprocess.CompletedProcess[str]:
    """Run @script as a file via `milk-cli -s`."""
    env, shm_dir = _env()
    try:
        return milk_cli_env(script, tmp_path, env, timeout)
    finally:
        shutil.rmtree(shm_dir, ignore_errors=True)


def milk_cli_stdin(
    script: str, tmp_path: Path, timeout: float = 20
) -> subprocess.CompletedProcess[str]:
    """Pipe @script into milk-cli as REPL input."""
    env, shm_dir = _env()
    try:
        return _run(
            [MILK_CLI_EXEC], env, timeout, stdin=script + "\n", cwd=str(tmp_path)
        )
    finally:
        shutil.rmtree(shm_dir, ignore_errors=True)


def _run_milk_cli(
    script: str,
    tmp_path: Path,
    mode: str = "script",
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env, shm_dir = _env(extra_env)
    try:
        if mode == "cmd":
            return _run([MILK_CLI_EXEC, "-c", script], env, 25)
        if mode == "stdin":
            return _run([MILK_CLI_EXEC], env, 25, stdin=script + "\nexit\n")
        return milk_cli_env(script, tmp_path, env, 25)
    finally:
        shutil.rmtree(shm_dir, ignore_errors=True)


def milk_script(
    script: str,
    tmp_path: Path,
    args: list[str] | None = None,
    opts: list[str] | None = None,
    timeout: float = 20,
) -> subprocess.CompletedProcess[str]:
    """Run @script with the standalone milk-script interpreter."""
    env, _ = _env()
    f = tmp_path / "ms.milk"
    f.write_text(script + "\n")
    exe = str(_install_dir() / "bin" / "milk-script")
    return _run(
        [exe, "-q", *(opts or []), str(f), *(args or [])],
        env,
        timeout,
        cwd=str(tmp_path),
    )


def translate(src: str) -> str:
    """Translate @src with `milk-call translate --legacy -` (no server needed)."""
    env, _ = _env()
    res = _run(
        [str(_install_dir() / "bin" / "milk-call"), "translate", "--legacy", "-"],
        env,
        10,
        stdin=src,
    )
    assert res.returncode == 0, res.stderr
    return res.stdout


def value(out: str, key: str) -> str | None:
    """Return the value of the last `key=value` line in @out."""
    m = re.findall(rf"^{re.escape(key)}=(.*)$", out, re.M)
    return m[-1] if m else None


def events(out: str) -> list[str]:
    return re.findall(r"^D:(.*)$", out, re.M)


def pty_session(
    steps: list[tuple[str, str, float]], tmp_path: Path, warmup: float = 2.0
) -> str:
    """Drive interactive milk-cli on a pty; steps are (kind, payload, wait) with kind in line|raw."""
    env, _ = _env()
    env["TERM"] = "dumb"
    pid, fd = pty.fork()
    if pid == 0:
        os.chdir(tmp_path)
        os.execve(MILK_CLI_EXEC, [MILK_CLI_EXEC], env)
    out = b""

    def drain(t: float) -> None:
        nonlocal out
        end = time.time() + t
        while time.time() < end:
            r, _, _ = select.select([fd], [], [], 0.05)
            if r:
                try:
                    d = os.read(fd, 65536)
                except OSError:
                    return
                if not d:
                    return
                out += d

    try:
        drain(warmup)
        for kind, payload, wait in steps:
            os.write(
                fd, (payload + "\r").encode() if kind == "line" else payload.encode()
            )
            drain(wait)
    finally:
        sid = os.getsid(pid) if Path(f"/proc/{pid}").exists() else pid
        _kill_session(sid)
        with contextlib.suppress(ChildProcessError):
            os.waitpid(pid, 0)
        os.close(fd)
    return _ANSI.sub("", out.decode(errors="replace")).replace("\r", "")


class _Milk:
    """Handle yielded by `milk_proc`."""

    def __init__(self, proc: subprocess.Popen, master: int | None) -> None:
        self.proc = proc
        self.master = master
        self.log = b""

    def write(self, data: bytes) -> None:
        assert self.master is not None
        os.write(self.master, data)

    def _read_chunk(self, wait: float) -> bytes | None:
        """None on timeout, b"" on EOF/hangup."""
        assert self.master is not None
        ready, _, _ = select.select([self.master], [], [], wait)
        if not ready:
            return None
        try:
            return os.read(self.master, 4096)
        except OSError:
            return b""

    def read_until(self, pattern: bytes | None, timeout: float = 3.0) -> bytes:
        """Read until `pattern` shows up; with None, until EOF."""
        buf = b""
        end = time.monotonic() + timeout
        while (pattern is None or pattern not in buf) and time.monotonic() < end:
            chunk = self._read_chunk(0.05)
            if chunk == b"":
                break
            buf += chunk or b""
        self.log += buf
        return buf

    def drain(self, quiet: float = 0.2, timeout: float = 3.0) -> bytes:
        """Read until the PTY has been silent for `quiet` seconds."""
        buf = b""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            chunk = self._read_chunk(quiet)
            if not chunk:
                break
            buf += chunk
        self.log += buf
        return buf


@contextlib.contextmanager
def milk_proc(
    args: list[str],
    *,
    pty_mode: bool = False,
    env: dict[str, str] | None = None,
    stdin: int = subprocess.DEVNULL,
) -> typ.Generator[_Milk]:
    """Spawn milk-cli in its own session; always reap it and its children."""
    assert MILK_CLI_EXEC is not None
    master = slave = None
    proc = None
    try:
        if pty_mode:
            master, slave = pty.openpty()
            io: dict[str, typ.Any] = dict(stdin=slave, stdout=slave, stderr=slave)
        else:
            io = dict(
                stdin=stdin,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
            )
        proc = subprocess.Popen(
            [MILK_CLI_EXEC, *args],
            env={**os.environ, "TERM": "xterm-256color", **(env or {})},
            start_new_session=True,
            preexec_fn=(
                (lambda: fcntl.ioctl(0, termios.TIOCSCTTY, 0)) if pty_mode else None
            ),
            **io,
        )
        if slave is not None:
            os.close(slave)
            slave = None
        yield _Milk(proc, master)
    finally:
        if proc is not None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            for stream in (proc.stdin, proc.stdout, proc.stderr):
                if stream is not None:
                    stream.close()
        for fd in (master, slave):
            if fd is not None:
                os.close(fd)


def assert_no_crash(returncode: int) -> None:
    assert 0 <= returncode < 128, f"milk-cli crashed (exit {returncode})"


def run_milk(
    args: list[str], *, stdin_text: str | None = None, timeout: float = 10
) -> tuple[int, str]:
    """Run milk-cli to completion; fail on crash or hang."""
    with milk_proc(
        args, stdin=subprocess.DEVNULL if stdin_text is None else subprocess.PIPE
    ) as m:
        try:
            out, _ = m.proc.communicate(stdin_text, timeout=timeout)
        except subprocess.TimeoutExpired:
            pytest.fail(f"milk-cli hung: {args!r}")
    assert_no_crash(m.proc.returncode)
    return m.proc.returncode, out
