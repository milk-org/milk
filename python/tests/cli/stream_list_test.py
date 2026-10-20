"""readshmimall and shmimpurge list the streams of the shm directory (libmilkdata milk_stream_list)."""

from __future__ import annotations

import shutil
import subprocess

import pytest

from milk.cliwrap import HAVE_CLI, MILK_CLI_EXEC

from tests.cli.bash_integration_test import _isolated_shm_env

pytestmark = pytest.mark.skipif(not HAVE_CLI, reason="milk-cli not built")


def test_readshmimall_loads_the_directory_streams():
    """Before libmilkdata, readshmimall resolved find_streams to a stub and loaded nothing."""
    env, shm_dir = _isolated_shm_env()
    try:
        for name in ["rsa_one", "rsa_two", "other"]:
            subprocess.run(
                ["milk-stream-create", "-t", "float", name, "4", "4"],
                env=env,
                capture_output=True,
                check=True,
                timeout=10,
            )
        r = subprocess.run(
            [MILK_CLI_EXEC, "-c", "mem.readshmimall rsa_\nmem.listim"],
            env=env,
            text=True,
            capture_output=True,
            timeout=25,
        )
        assert r.returncode == 0, r.stderr
        listing = r.stdout[r.stdout.rindex("INDEX") :]
        assert "rsa_one" in listing and "rsa_two" in listing
        assert "other" not in listing  # the filter applies
        assert "2 image(s)" in listing
    finally:
        shutil.rmtree(shm_dir, ignore_errors=True)
