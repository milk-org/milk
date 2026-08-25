from __future__ import annotations

import subprocess

from pyMilk.interfacing.fps import FPS, FPSDoesntExistError


class ComputeSession:
    """Python-side handle for a milk-fpsexec-* compute unit lifecycle.

    Wraps the CLI commands supported by standalone executables:

    Commands:
      fpsinit          Create the FPS
      fps              Print FPS content
      fpslist          List matching FPS instances # TODO should be a classmethod, no point.
      confstart        Configuration loop
      confstep         Single config step
      confstop         Stop config loop
      runstart         Main processing loop
      runstop          Stop processing loop
      set [args]       Set positional args (. to skip) # TODO probably wont do this here
      exec [args]      Auto-init + set args + run      # TODO meh
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
            # TODO add an isvalid() to the FPS object itself.
            return

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

    def _runstart(self, command: str, *args: str) -> subprocess.Popen:
        target = f"{self.fpsname}:{command}"
        argv = [self.exec_name, *args, target]
        return subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def _runcomplete(self, command: str, *args: str) -> subprocess.Popen:
        proc = self._runstart(command, *args)
        proc.wait()
        return proc

    def fpsinit(self, procinfo: bool = True) -> None:
        """Create the FPS shared memory segment."""
        if procinfo:
            self._runcomplete("fpsinit", "-procinfo")
        else:
            self._runcomplete("fpsinit")
        self._trylink_fps(raise_on_miss=True)
        # Guarantees FPS exists

    def __str__(self) -> str:
        self._trylink_fps(True)
        return self._runcomplete("fps").stdout.read().decode()  # type: ignore

    def confstart(self, tmux: bool = False) -> None:
        self._trylink_fps(True)
        # timeout ? guaranteed completion ? return pid ?
        if tmux:
            self._runcomplete("confstart", "-tmux")
        else:
            self._runstart("confstart")

    def confstep(self) -> None:
        # TODO I have no idea how to test this? What's the spec?
        self._trylink_fps(True)
        self._runcomplete("confstep")

    def confstop(self) -> None:
        self._trylink_fps(True)
        self._runcomplete("confstop")

    def runstart(
        self, tmux: bool, loopd: float | None = None, loops: bool = False
    ) -> None:
        self._trylink_fps(True)
        assert loopd is None or not loops
        if loops:
            args = ("-loops",)
        elif loopd:
            args = ("-loopd", f"{loopd:.6f}")
        if tmux:
            self._runcomplete("runstart", "-tmux")
        else:
            self._runstart("runstart")

    def runstop(self) -> None:
        self._trylink_fps(True)
        self._runcomplete("runstop")

    '''
    def set(self, *args: str) -> None:
        """Set positional args in the FPS ('.' to skip a position)."""
        self._run("set", *args)

    def exec(self, *args: str) -> None:
        """Auto-init + set args + run, one-shot."""
        self._run("exec", *args)
    '''
