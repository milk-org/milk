from __future__ import annotations

import shutil, os
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


class MilkBuildException(Exception): ...


class CLI:
    open: bool = False

    def __init__(self) -> None:
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
        self._read_until_prompt()  # discard startup banner + first prompt

    def _read_until_prompt(self) -> str:
        # Prompt has no trailing newline, so this must read char-by-char
        # rather than line-by-line (which would block forever).
        assert self._proc.stdout is not None
        buf = ""
        while not buf.endswith(_PROMPT_SUFFIX):
            char = self._proc.stdout.read(1)
            if char == "":
                break  # REPL process exited
            buf += char
        # Look for last linebreak
        for k in range(1, len(buf)):
            if buf[-k] == "\n":
                return buf[:-k]

        return buf[: -len(_PROMPT_SUFFIX)]  # No newlines ??

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
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._proc.terminate()
            self._proc.wait(timeout=5)
        self.open = False

    def __enter__(self) -> typ.Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def run(self, commands: typ.Sequence[str]) -> list[str]:
        """Feed a batch of commands to the REPL, returning their outputs."""
        return [self.send_line(command) for command in commands]
