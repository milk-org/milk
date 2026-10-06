from __future__ import annotations

import shutil, os
import re
import subprocess
import typing as typ
import ctypes


def _find_build_tags() -> dict[str, str]:
    MILK_INSTALLDIR = os.environ["MILK_INSTALLDIR"]
    libpath = MILK_INSTALLDIR + "/lib/libmilkcommon.so"
    dll = ctypes.CDLL(libpath)

    char_buffer = ctypes.c_char * 1024
    val = char_buffer.in_dll(dll, "_milk_build_tag_")

    taglist = val.value.decode("utf8")[12:].split(",")[:-1]
    tagdict = {s.split("=")[0]: s.split("=")[1] for s in taglist}

    return tagdict


MILK_BUILD_TAGS = _find_build_tags()
HAVE_CLI = bool(int(MILK_BUILD_TAGS["CLI"]))

MILK_CLI_EXEC = shutil.which("milk-cli") if HAVE_CLI else None

# milk-cli always ends its prompt with this suffix, with no trailing newline.
_PROMPT_SUFFIX = " >"

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_LOADED_MODULES_RE = re.compile(r"Loaded \d+ modules, \d+ commands")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


class MilkBuildException(Exception): ...


class CLICrashError(Exception):
    """Raised when milk-cli was killed by a signal (CRASH)."""


class CLI:
    open: bool = False

    def __init__(self, strip_ansi: bool = True) -> None:
        if MILK_CLI_EXEC is None:
            raise MilkBuildException(
                "MILK built without CLI support. Must build with -DUSE_CLI=ON."
            )

        self._proc = subprocess.Popen(
            [MILK_CLI_EXEC],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self.open = True
        self.strip_ansi = strip_ansi
        self.retcode: int | None = None

        self._known_prompt = None
        self._read_until_prompt()  # discard startup banner + first prompt

        self._known_prompt = self.send_line(
            ""
        )  # initializes the prompt typically "milk-cli >"
        self.timeout = 5

    def _read_until_prompt(self) -> str:
        # Prompt has no trailing newline, so this must read char-by-char
        # rather than line-by-line (which would block forever).
        assert self._proc.stdout is not None
        buf = ""
        escape = ""
        while not buf.endswith(_PROMPT_SUFFIX):
            char = self._proc.stdout.read(1)
            if char == "":
                break  # REPL process exited
            buf += char
        if self.strip_ansi:
            buf = _strip_ansi(buf)

        # It seems that the suggestion-complete echoes back... let's ditch the first line (the echo)
        buf = buf.split("\n", 1)[1]

        if self._known_prompt:
            buf = buf.removesuffix(self._known_prompt)
        return buf

    def send_line(self, line: str) -> str:
        assert self._proc.stdin is not None
        self._proc.stdin.write(line + "\n")
        self._proc.stdin.flush()
        return self._read_until_prompt().strip()

    def close(self) -> None:
        if not self.open:
            return
        if self._proc.stdin is not None:
            try:
                self._proc.stdin.write("exit\n")
                self._proc.stdin.flush()
            except (BrokenPipeError, ValueError):
                pass
        try:
            self.retcode = self._proc.wait(timeout=self.timeout)
        except subprocess.TimeoutExpired:
            self._proc.terminate()
            self.retcode = self._proc.wait(timeout=self.timeout)
        self.open = False

    def run(self, commands: typ.Sequence[str]) -> list[str]:
        """Feed a batch of commands to the REPL, returning their outputs."""
        return [self.send_line(command) for command in commands]


class CLICommands:
    """Run a batch of commands via ``milk-cli -s <temp_file>``.

    Context manager whose ``__enter__`` returns the ``CompletedProcess``
    (stdout, stderr, returncode). Raises :class:`CLICrashError` when the
    shell is killed by a signal. ``__exit__`` kills the shell if it has
    not already exited.
    """

    def __init__(
        self,
        commands: list[str],
        *,
        strip_ansi: bool = True,
        quiet: bool = True,
    ) -> None:
        if MILK_CLI_EXEC is None:
            raise MilkBuildException(
                "MILK built without CLI support. Must build with -DUSE_CLI=ON."
            )
        self.commands = commands
        self.strip_ansi = strip_ansi
        self.quiet = quiet
        self._proc: subprocess.Popen[str] | None = None
        self.result: subprocess.CompletedProcess[str] | None = None

    def __enter__(self) -> subprocess.CompletedProcess[str]:
        try:
            return self._enter_exception_unsafe()
        except:  # It's important NOT to specify the exception here !
            # If self._proc killed by external signal we may get something that is NOT a sub Exception.
            self._cleanup()
            raise
        finally:
            if os.path.exists(self.script_filename):
                os.remove(self.script_filename)

    def _enter_exception_unsafe(self) -> subprocess.CompletedProcess[str]:

        assert MILK_CLI_EXEC is not None
        env = dict(os.environ)
        # TODO MILK_QUIET is broken and disables stdout/stderr.
        # if self.quiet:
        #    env["MILK_QUIET"] = "1"
        import time

        self.script_filename = (
            f"/tmp/milk-pycli.{os.getpid()}.{int(time.time() * 1e9)}.milk"
        )
        with open(self.script_filename, "w") as f:
            f.writelines([c + "\n" for c in self.commands])
        self._proc = subprocess.Popen(
            # [MILK_CLI_EXEC, "-c", ";".join(self.commands)],
            [MILK_CLI_EXEC, "-s", self.script_filename],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        stdout, stderr = self._proc.communicate()
        if self.strip_ansi:
            stdout = _strip_ansi(stdout)
            stderr = _strip_ansi(stderr)
        if self.quiet:
            lines = stdout.splitlines(keepends=True)
            for i, line in enumerate(lines):
                if _LOADED_MODULES_RE.search(line):
                    stdout = "".join(lines[i + 2 :])
                    break
        returncode = self._proc.returncode
        self.result = subprocess.CompletedProcess(
            self._proc.args, returncode, stdout, stderr
        )
        # Negative return code == killed by signal (bash CRASH: exit >= 128)
        if returncode < 0:
            raise CLICrashError(
                f"milk-cli killed by signal {-returncode}: "
                f"{';'.join(self.commands)!r}"
            )
        return self.result

    def __exit__(self, *exc_info: object) -> None:
        self._cleanup()

    def _cleanup(self) -> None:
        """
        _cleanup can be invoked even if __enter__ excepted and did not succeed.

        This function must run and kill the child... even if something else caused the interruption

        """
        if self._proc is not None and self._proc.poll() is None:
            self._proc.kill()
            self._proc.wait()
