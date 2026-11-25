from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from typing import IO


from pyMilk.interfacing.fps import FPS, FPSDoesntExistError, FpsFlag, FpsType

from ..errors import ComputeUnitError


@dataclass
class _PipeCapture:
    """A subprocess pipe, read without blocking, and everything read from it so far."""

    pipe: IO[bytes]
    data: bytes = b""

    def __post_init__(self) -> None:
        os.set_blocking(self.pipe.fileno(), False)

    def read(self) -> str:
        chunk = self.pipe.read()  # None if nothing available
        if chunk:
            self.data += chunk
        return self.data.decode(errors="replace")


# PROCESSINFO_TRIGGERMODE_SEMAPHORE (processinfo.h)
_TRIGGERMODE_SEMAPHORE = 3


class ComputeUnit:
    """Python-side handle on one milk-fpsexec-* executable and its FPS."""

    def __init__(self, exec_name: str, fpsname: str) -> None:
        self.exec_name = exec_name
        self.fpsname = fpsname

        self.fps: FPS | None = None
        self._trylink_fps()

        # Output of the last subprocess issued by runstart().
        self._run_out: _PipeCapture | None = None
        self._run_err: _PipeCapture | None = None

    @property
    def last_stdout(self) -> str:
        """Output of the last runstart() subprocess so far."""
        return self._run_out.read() if self._run_out else ""

    @property
    def last_stderr(self) -> str:
        """Error output of the last runstart() subprocess so far."""
        return self._run_err.read() if self._run_err else ""

    def _trylink_fps(self, raise_on_miss: bool = False) -> None:
        """Try to link to the named FPS, raising FPSDoesntExistError only if raise_on_miss."""
        if self.fps is not None and self.fps.is_valid():
            return
        try:
            self.fps = FPS(self.fpsname)
        except FPSDoesntExistError:
            self.fps = None
            if raise_on_miss:
                raise

    def _subprocess_start(
        self, command: str, *args: str, wrap_in_shell: bool = False
    ) -> subprocess.Popen:
        target = f"{self.fpsname}:{command}"
        argv = [self.exec_name, *args, target]
        if wrap_in_shell:
            # 2 things:
            # - Wrap in a bash call so that bash performs a wait on the underlying process that we actually desire
            #   so that this process doesn't become Zombie for long, and conf.isrunning / run.isrunning remain accurate.
            # - Add a tail call "; true" to bash -c so that we force out of bash's last-command
            #   optimization which very exactly defeats the above purpose.
            return subprocess.Popen(
                ["bash", "-c"] + [" ".join(argv) + "; true"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        else:
            return subprocess.Popen(
                argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )

    def _subprocess_complete(
        self, command: str, *args: str, wrap_in_shell: bool = False
    ) -> subprocess.Popen:
        proc = self._subprocess_start(command, *args, wrap_in_shell=wrap_in_shell)
        retcode = proc.wait()
        if retcode != 0:
            stderr = proc.stderr.read() if proc.stderr else b""
            raise ComputeUnitError(
                f"Command {proc.args!r} exited with status {retcode}",
                compu=self.fpsname,
                cmd=proc.args,
                stderr=stderr.decode().rstrip(),
            )
        return proc

    def fpsinit(self) -> None:
        """Create the FPS shared memory segment, with its procinfo."""
        self._subprocess_complete("fpsinit", "-procinfo")
        self._trylink_fps(raise_on_miss=True)
        # Guarantees FPS exists

    def trigger_stream_key(self) -> str | None:
        """Key of the first STREAMNAME parameter flagged FPFLAG_TRIGGER_STREAM, if any."""
        self._trylink_fps(raise_on_miss=True)
        assert self.fps is not None
        for key, fptype in self.fps.key_types.items():
            if (
                fptype == FpsType.STREAMNAME
                and FpsFlag.TRIGGER_STREAM in self.fps.get_flags(key)
            ):
                return key
        return None

    def autopopulate_trigger_stream(self) -> None:
        """Python counterpart of fps_autopopulate_trigger_stream (fps_lifecycle.c).

        If procinfo exists and there is a trigger stream, set triggermode=SEMAPHORE,
        loopcntMax=-1, enabled=ON. procinfo.triggersname is set to the trigger
        stream's value only if empty; a user-set triggersname is kept.
        """
        key = self.trigger_stream_key()
        assert self.fps is not None
        procinfo = self.fps.procinfo
        if procinfo is None or key is None:
            return
        if not procinfo.triggersname:
            trigger_name = self.fps[key]
            if not trigger_name:
                return
            procinfo.triggersname = str(trigger_name)
        procinfo.triggermode = _TRIGGERMODE_SEMAPHORE
        procinfo.loopcntMax = -1
        procinfo.enabled = True

    def __str__(self) -> str:
        self._trylink_fps()
        if self.fps is None:
            return f"ComputeUnit {self.exec_name} {self.fpsname} [FPS unlinked]"
        return self._subprocess_complete("fps").stdout.read().decode()  # type: ignore

    def __repr__(self) -> str:
        self._trylink_fps()
        fps_ok = "unlinked" if self.fps is None else "OK"
        return f"ComputeUnit {self.fpsname} ({self.exec_name}) [FPS {fps_ok}]"

    def confstart(self, tmux: bool = False) -> None:
        """
        This ends up calling:
        int fps_generic_conf_cb(const char *fps_name, int loop, errno_t (*confcheck_fn)(void))
        with loop = 1
        """
        self._trylink_fps(raise_on_miss=True)
        if tmux:
            self._subprocess_complete("confstart", "-tmux")
        else:
            self._subprocess_start("confstart", wrap_in_shell=True)

    def confstop(self) -> None:
        self._trylink_fps(raise_on_miss=True)
        self._subprocess_complete("confstop")

    def runstart(
        self, tmux: bool = False, loopd: float | None = None, loops: bool = False
    ) -> None:
        self._trylink_fps(raise_on_miss=True)
        assert loopd is None or not loops
        if loops:
            args = ("-loops",)
        elif loopd:
            args = ("-loopd", f"{loopd:.6f}")
        else:
            args = ()

        if tmux:
            proc = self._subprocess_complete("runstart", "-tmux", *args)
            self._run_out, self._run_err = None, None
        else:
            proc = self._subprocess_start("runstart", *args, wrap_in_shell=True)
            self._run_out = _PipeCapture(proc.stdout)  # type: ignore[arg-type]
            self._run_err = _PipeCapture(proc.stderr)  # type: ignore[arg-type]

    def runstop(self) -> None:
        self._trylink_fps(raise_on_miss=True)
        self._subprocess_complete("runstop")
