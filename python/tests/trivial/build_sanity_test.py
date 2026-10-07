from __future__ import annotations

import typing as typ

import pytest

import pathlib
import os
import subprocess
from milk.cliwrap import CLI, HAVE_CLI

TIMEOUT = 1

FPSEXEC_LINKS_CLICORE_EXCEPTIONS = [
    "milk-fpsexec-fft-dofft",
    "milk-fpsexec-fft-pup2foc",
]


def _find_fpsexecs() -> list[pathlib.Path]:
    installdir = os.environ.get("MILK_INSTALLDIR")
    if not installdir:
        return []
    bindir = pathlib.Path(installdir).resolve() / "bin"
    exes = [
        p
        for pattern in ("milk-fpsexec-*", "cacao-fpsexec-*")
        for p in bindir.glob(pattern)
        if p.is_file()
    ]
    return sorted(exes)


FPSEXECS = _find_fpsexecs()


def _params(mark_exceptions: bool):
    for exe in FPSEXECS:
        marks = []
        if mark_exceptions and exe.name in FPSEXEC_LINKS_CLICORE_EXCEPTIONS:
            marks.append(
                pytest.mark.xfail(reason="listed CLIcore dependency", strict=False)
            )
        yield pytest.param(exe, id=exe.name, marks=marks)


@pytest.mark.skipif(not HAVE_CLI, reason="MILK compiled without CLI support")
@pytest.mark.parametrize(
    "module_name",
    [
        "milkCOREMODmemory",
        "milkCOREMODarith",
        "milkCOREMODiofits",
        "milkCOREMODtools",
        "milkclustering",
        "milkcompilertest",
        "milkfft",
        "milkfpstest",
        "milkimagebasic",
        "milkimagefilter",
        "milkimageformat",
        "milkimagegen",
        "milkimgreduce",
        "milkinfo",
        "milklinalgebra",
        "milklinARfilterPred",
        "milklinoptimtools",
        "milk_module_example",
        "milkpsf",
        "milkstatistic",
        "milkZernikePolyn",
    ],
)
def test_module_import_in_cli(module_name: str):
    cli = CLI(strip_ansi=True)
    try:
        stdout = cli.send_line(f"mload {module_name}")
        last_line = stdout.split("\n")[-1]
        lib_file = f"lib{module_name}.so"

        if "COREMOD" in module_name:
            assert last_line.endswith("already loaded - no action taken")
            fullpath_to_so = pathlib.Path(last_line.split()[2]).resolve()
        else:
            assert "LOADED :" in last_line
            assert last_line.endswith(lib_file)
            fullpath_to_so = pathlib.Path(last_line.split()[-1]).resolve()

        milk_installdir = pathlib.Path(os.environ["MILK_INSTALLDIR"]).resolve()

        assert milk_installdir / "lib" / lib_file == fullpath_to_so

        stdout_lines = cli.send_line(f"m?").split("\n")
        assert len(stdout_lines) > 4
        if "COREMOD" in module_name:
            assert len(stdout_lines) == 8
        else:
            assert len(stdout_lines) == 9
    finally:
        cli.close()


def test_fpsexecs_found():
    assert os.environ.get("MILK_INSTALLDIR"), "MILK_INSTALLDIR is not set"
    assert FPSEXECS, "no fpsexec executables found in $MILK_INSTALLDIR/bin"


@pytest.mark.parametrize("fpsexec", _params(mark_exceptions=True))
def test_no_clicore_link(fpsexec: pathlib.Path):
    proc = subprocess.run(
        ["ldd", "-d", str(fpsexec)], capture_output=True, text=True, timeout=TIMEOUT
    )
    assert proc.returncode == 0, proc.stderr
    assert "CLIcore" not in proc.stdout


@pytest.mark.parametrize("fpsexec", _params(mark_exceptions=False))
def test_run_help(fpsexec: pathlib.Path):
    assert os.access(fpsexec, os.X_OK)
    proc = subprocess.run(
        [str(fpsexec), "-h"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=TIMEOUT,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip(), f"stdout is empty for fpsexec {fpsexec}"
