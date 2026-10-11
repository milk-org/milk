"""
Run a function from a non-default plugin (milkimagebasic: imgbasic.cubecollapse)
through milk-cli, driven from Python via CLICommands.

    1. direct:  one command line, positional args
    2. FPS:     fpsinit -> set params -> confstart -> runstart
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from pyMilk.interfacing.fps import FPS
from pyMilk.interfacing.shm import SHM

from milk.cliwrap import CLICommands, HAVE_CLI

pytestmark = pytest.mark.skipif(
    not HAVE_CLI, reason="MILK compiled without CLI support"
)

PLUGIN = "milkimagebasic"
CMD = "imgbasic.cubecollapse"
IN_NAME = "tcinpl"
OUT_NAME = "tcoutpl"
FPS_NAME = "tccfps"

MLOAD = f"mload {PLUGIN}"


@pytest.fixture
def cube():
    arr = np.random.rand(3, 4, 5).astype(np.float32)
    shm = SHM(IN_NAME, arr)
    yield arr
    for name in (IN_NAME, OUT_NAME):
        try:
            SHM(name).destroy()
        except Exception:
            pass


@pytest.mark.timeout(1)
def test_plugin_direct_cmdline(cube):
    # imgbasic.cubecollapse is not in the default module set: mload first.
    with CLICommands(
        [MLOAD, f"mem.readshmim {IN_NAME}", f"{CMD} {IN_NAME} {OUT_NAME}"]
    ) as r:
        assert r.returncode == 0

    out = SHM(OUT_NAME).get_data()
    assert out.shape == cube.shape[:-1]
    np.testing.assert_allclose(out, cube.sum(axis=2), rtol=1e-5)


# TODO devise a test for the execution of an FPS through the CLI with fpsinit, confstart, runstart.
