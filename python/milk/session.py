from __future__ import annotations

import subprocess

from pyMilk.interfacing.fps import FPS, FPSDoesntExistError


def fpslist(exec_name: str):
    """
    No capture of stdout, just print
    """
    proc = subprocess.Popen([exec_name, "fpslist"])
    proc.wait()


class ComputeSession:
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
      # set [args]       Set positional args (. to skip) # NotImplemented
      # exec [args]      Auto-init + set args + run      # NotImplemented
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
        proc.wait()
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
        self._trylink_fps(False)
        if self.fps is None:
            return f"Session {self.exec_name} {self.fpsname} [FPS unlinked]"
        return self._subprocess_complete("fps").stdout.read().decode()  # type: ignore

    def __repr__(self) -> str:
        self._trylink_fps(False)
        fps_ok = "unlinked" if self.fps is None else "OK"
        return f"Session {self.fpsname} ({self.exec_name}) [FPS {fps_ok}]"

    def fpslist(self) -> None:
        fpslist(self.exec_name)

    def confstart(self, tmux: bool = False) -> None:
        """
        This ends up calling:
        int fps_generic_conf_cb(const char *fps_name, int loop, errno_t (*confcheck_fn)(void))
        with loop = 1
        """
        self._trylink_fps(True)
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
        self._trylink_fps(True)
        self._subprocess_complete("confstep")

    def confstop(self) -> None:
        self._trylink_fps(True)
        self._subprocess_complete("confstop")

    def runstart(
        self, tmux: bool = False, loopd: float | None = None, loops: bool = False
    ) -> None:
        self._trylink_fps(True)
        assert loopd is None or not loops
        if loops:
            args = ("-loops",)
        elif loopd:
            args = ("-loopd", f"{loopd:.6f}")
        else:
            args = ()

        if tmux:
            self._subprocess_complete("runstart", "-tmux", *args)
        else:
            self._subprocess_start("runstart", *args, wrap_in_shell=True)

    def runstop(self) -> None:
        self._trylink_fps(True)
        self._subprocess_complete("runstop")

    '''
    def set(self, *args: str) -> None:
        """Set positional args in the FPS ('.' to skip a position)."""
        self._run("set", *args)

    def exec(self, *args: str) -> None:
        """Auto-init + set args + run, one-shot."""
        self._run("exec", *args)
    '''
