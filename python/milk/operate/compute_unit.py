from __future__ import annotations

import subprocess

from pyMilk.interfacing.fps import FPS, FPSDoesntExistError

from ..errors import ComputeUnitError


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

    def _subprocess_start(self, command: str, *args: str) -> subprocess.Popen:
        target = f"{self.fpsname}:{command}"
        argv = [self.exec_name, *args, target]
        return subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def _subprocess_complete(self, command: str, *args: str) -> subprocess.Popen:
        proc = self._subprocess_start(command, *args)
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
        # Guarantees FPS exists

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

    def confstart(self) -> None:
        """
        This ends up calling:
        int fps_generic_conf_cb(const char *fps_name, int loop, errno_t (*confcheck_fn)(void))
        with loop = 1, in the background. Returns once the conf loop runs.
        """
        self._trylink_fps(raise_on_miss=True)
        self._subprocess_complete("confstart")

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

    def runstart(self, loopd: float | None = None, loops: bool = False) -> None:
        """Start the run loop in the background; returns once it is active."""
        self._trylink_fps(raise_on_miss=True)
        assert loopd is None or not loops
        if loops:
            args = ("-loops",)
        elif loopd:
            args = ("-loopd", f"{loopd:.6f}")
        else:
            args = ()
        self._subprocess_complete("runstart", *args)

    def runstop(self) -> None:
        self._trylink_fps(raise_on_miss=True)
        self._subprocess_complete("runstop")
