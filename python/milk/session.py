from __future__ import annotations

import subprocess

from pyMilk.interfacing.fps import FPS, FPSDoesntExistError


class ComputeSessionError(RuntimeError):
    """A lifecycle command of a compute unit failed; the message holds what it printed."""


class ComputeSession:
    """Python-side handle for a milk-fpsexec-* compute unit lifecycle.

    Wraps the CLI commands supported by standalone executables:

    Commands:
      fpsinit          Create the FPS (and capture the environment its unit runs with)
      fps              Print FPS content
      fpslist          List matching FPS instances # TODO should be a classmethod, no point.
      confstart        Start the configuration loop in the background, once it runs
      confstep         Single config step
      confstop         Stop the configuration loop, once it is gone
      runstart         Start the processing loop in the background, once it is active
      runstop          Stop the processing loop, once it is gone
      set [args]       Set positional args (. to skip) # TODO probably wont do this here
      exec [args]      Auto-init + set args + run      # TODO meh

    The start and stop commands return when the state has changed: a failure raises
    ComputeSessionError, a start that is not ready in time raises TimeoutError.
    """

    def __init__(self, exec_name: str, fpsname: str) -> None:
        self.exec_name = exec_name
        self.fpsname = fpsname

        self.has_procinfo: bool

        self.fps: FPS | None = None
        self._trylink_fps()

    def _trylink_fps(self, raise_on_miss: bool = False) -> None:
        """Try to link to the named FPS, raising only if except_ is True."""

        if self.fps is not None:
            if self.fps.is_valid():
                return
            else:
                self.fps = None

        try:
            self.fps = FPS(self.fpsname)
        except FPSDoesntExistError:
            self.fps = None
            if raise_on_miss:
                raise

        self.has_procinfo = (
            self.fps is not None and "procinfo.enabled" in self.fps.key_types
        )
        # self.procinfo = # TODO add a procinfo mapping.

    def _run(self, command: str, *args: str) -> subprocess.CompletedProcess[str]:
        """Run `<exec> [args] <fpsname>:<command>` to completion, reading its output."""
        argv = [self.exec_name, *args, f"{self.fpsname}:{command}"]
        return subprocess.run(argv, capture_output=True, text=True)

    def _lifecycle(self, command: str, *args: str) -> None:
        """Run a start or stop command; raise if it did not reach its state."""
        proc = self._run(command, *args)
        if proc.returncode == 124:
            raise TimeoutError(
                f"{self.fpsname} {command}: not ready in time\n{proc.stderr}"
            )
        if proc.returncode != 0:
            raise ComputeSessionError(
                f"{self.fpsname} {command} failed ({proc.returncode})\n{proc.stderr}"
            )

    def fpsinit(self, procinfo: bool = True) -> None:
        """Create the FPS shared memory segment."""
        if procinfo:
            self._run("fpsinit")
        else:
            self._run("fpsinit", "--no-procinfo")
        self._trylink_fps(raise_on_miss=True)
        # Guarantees FPS exists

    def __str__(self) -> str:
        self._trylink_fps(True)
        return self._run("fps").stdout

    def __repr__(self) -> str:
        self._trylink_fps(True)
        return self._run("fps").stdout

    def confstart(self, timeout: float | None = None) -> None:
        self._trylink_fps(True)
        args = ("--timeout", f"{timeout}") if timeout is not None else ()
        self._lifecycle("confstart", *args)

    def confstep(self) -> None:
        # TODO I have no idea how to test this? What's the spec?
        self._trylink_fps(True)
        self._run("confstep")

    def confstop(self) -> None:
        self._trylink_fps(True)
        self._lifecycle("confstop")

    def runstart(
        self,
        loopd: float | None = None,
        loops: bool = False,
        timeout: float | None = None,
    ) -> None:
        self._trylink_fps(True)
        assert loopd is None or not loops
        if loops:
            args: tuple[str, ...] = ("-loops",)
        elif loopd:
            args = ("-loopd", f"{loopd:.6f}")
        else:
            args = ()
        if timeout is not None:
            args += ("--timeout", f"{timeout}")
        self._lifecycle("runstart", *args)

    def runstop(self) -> None:
        self._trylink_fps(True)
        self._lifecycle("runstop")

    '''
    def set(self, *args: str) -> None:
        """Set positional args in the FPS ('.' to skip a position)."""
        self._run("set", *args)

    def exec(self, *args: str) -> None:
        """Auto-init + set args + run, one-shot."""
        self._run("exec", *args)
    '''
