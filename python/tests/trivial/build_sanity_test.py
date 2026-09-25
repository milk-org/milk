from __future__ import annotations

import typing as typ

import pytest

import pathlib
import os
from milk.cliwrap import CLI, HAVE_CLI


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
    if not HAVE_CLI:
        pytest.skip("MILK compiled without CLI support")
    with CLI() as cli:
        stdout = cli.send_line(f"mload {module_name}")
        last_line = stdout.split("\n")[-1]
        lib_file = f"lib{module_name}.so"

        if "COREMOD" in module_name:
            assert last_line.endswith("already loaded - no action taken")
            fullpath_to_so = pathlib.Path(last_line.split()[2]).absolute()
        else:
            assert "LOADED :" in last_line
            assert last_line.endswith(lib_file)
            fullpath_to_so = pathlib.Path(last_line.split()[-1]).absolute()

        milk_installdir = pathlib.Path(os.environ["MILK_INSTALLDIR"]).absolute()

        assert milk_installdir / "lib" / lib_file == fullpath_to_so

        stdout_lines = cli.send_line(f"m?").split("\n")
        assert len(stdout_lines) > 6
        if "COREMOD" in module_name:
            assert len(stdout_lines) == 10
        else:
            assert len(stdout_lines) == 11
