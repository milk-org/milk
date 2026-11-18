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


def fpslist(exec_name: str):
    """
    No capture of stdout, just print
    """
    proc = subprocess.Popen([exec_name, "fpslist"])
    proc.wait()


class ComputeUnit:
    """Python-side handle for a milk-fpsexec-* compute unit lifecycle.

    Wraps the CLI commands supported by standalone executables:

    Commands:
      fpsinit          Create the FPS
      fps              Print FPS content
      fpslist          List matching FPS instances
      confstart        Configuration loop
      confstep         Single config step
      confstop         Stop config loop
      runstart         Main processing loop
      runstop          Stop processing loop
    """

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

        self.procinfo = (
            self.fps.procinfo
            if (self.fps is not None and self.fps.procinfo is not None)
            else None
        )

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

    def fpsinit(self, procinfo: bool = True) -> None:
        """Create the FPS shared memory segment."""
        if procinfo:
            self._subprocess_complete("fpsinit", "-procinfo")
        else:
            self._subprocess_complete("fpsinit")
        self._trylink_fps(raise_on_miss=True)
        self.autopopulate_trigger_stream()
        # Guarantees FPS exists

    def trigger_stream_key(self) -> str | None:
        """Key of the first STREAMNAME parameter flagged FPFLAG_TRIGGER_STREAM, if any."""
        self._trylink_fps(raise_on_miss=True)
        assert self.fps is not None
        flags = self.fps.fps.flags
        for key, fptype in self.fps.key_types.items():
            if fptype == FpsType.STREAMNAME and flags[key] & FpsFlag.TRIGGER_STREAM:
                return key
        return None

    def autopopulate_trigger_stream(self) -> None:
        """Python counterpart of fps_autopopulate_trigger_stream (fps_lifecycle.c).

        If procinfo exists and procinfo.triggersname is empty, set it to the
        trigger stream's value, with triggermode=SEMAPHORE, loopcntMax=-1, enabled=ON.
        """
        key = self.trigger_stream_key()
        if self.procinfo is None or self.procinfo.triggersname or key is None:
            return
        assert self.fps is not None
        trigger_name = self.fps[key]
        if not trigger_name:
            return
        self.procinfo.triggersname = trigger_name
        self.procinfo.triggermode = _TRIGGERMODE_SEMAPHORE
        self.procinfo.loopcntMax = -1
        self.procinfo.enabled = True

    def __str__(self) -> str:
        self._trylink_fps()
        if self.fps is None:
            return f"ComputeUnit {self.exec_name} {self.fpsname} [FPS unlinked]"
        return self._subprocess_complete("fps").stdout.read().decode()  # type: ignore

    def __repr__(self) -> str:
        self._trylink_fps()
        fps_ok = "unlinked" if self.fps is None else "OK"
        return f"ComputeUnit {self.fpsname} ({self.exec_name}) [FPS {fps_ok}]"

    def fpslist(self) -> None:
        fpslist(self.exec_name)

    def confstart(self, tmux: bool = False) -> None:
        """
        This ends up calling:
        int fps_generic_conf_cb(const char *fps_name, int loop, errno_t (*confcheck_fn)(void))
        with loop = 1
        """
        self._trylink_fps(raise_on_miss=True)
        # timeout ? guaranteed completion ? return pid ?
        if tmux:
            self._subprocess_complete("confstart", "-tmux")
        else:
            self._subprocess_start("confstart", wrap_in_shell=True)

    def confstep(self) -> None:
        """
        This ends up calling:
        int fps_generic_conf_cb(const char *fps_name, int loop, errno_t (*confcheck_fn)(void))
        with loop = 0
        """
        self._trylink_fps(raise_on_miss=True)
        self._subprocess_complete("confstep")

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
