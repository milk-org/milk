from __future__ import annotations

import subprocess


class PipelineConfFolderNotFoundException(FileNotFoundError):
    # TODO pretty-print a message here
    ...


class PipelineTomlNotFoundException(FileNotFoundError):
    # TODO pretty-print a message here
    ...


class FPSExecException(RuntimeError):
    """Raised when a milk-fpsexec-* CLI subprocess exits with non-zero status."""

    def __init__(self, proc: subprocess.Popen) -> None:
        # proc is assumed already-completed; kill it if that's not the case
        if proc.poll() is None:
            proc.kill()
            proc.wait()

        self.proc = proc

        stderr = proc.stderr.read() if proc.stderr else b""
        if isinstance(stderr, bytes):
            stderr = stderr.decode()

        message = f"Command {proc.args!r} exited with status {proc.returncode}"
        if stderr:
            message += f"\n--- stderr ---\n{stderr.rstrip()}"
        super().__init__(message)
