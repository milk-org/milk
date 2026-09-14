import os, shutil
import libtmux

import pytest


# ConfTest.py FIXTture == ctfixt_
@pytest.fixture(scope="session", autouse=True)
def ctfixt_change_MILK_SHM_DIR():

    MILK_SHM_DIR_SPOOF = "/tmp/milk_shm_dir_pytest"
    os.makedirs(MILK_SHM_DIR_SPOOF, exist_ok=True)

    # There was something more aggressive to reload the env at runtime, wasn't there?
    os.environ["MILK_SHM_DIR"] = MILK_SHM_DIR_SPOOF
    # It is used by processinfo... really necessary?
    os.environ["MILK_PROC_DIR"] = MILK_SHM_DIR_SPOOF

    yield MILK_SHM_DIR_SPOOF

    # Fixture teardown here:
    shutil.rmtree(MILK_SHM_DIR_SPOOF)


@pytest.fixture(scope="session", autouse=True)
def ctfixt_change_tmux_server():
    TMUX_TMPDIR_SPOOF = "/tmp/milk_tmux_tmpdir_pytest"
    os.makedirs(TMUX_TMPDIR_SPOOF, exist_ok=True)

    saved_tmux = os.environ.pop("TMUX", None)  # detach from any outer tmux client
    os.environ["TMUX_TMPDIR"] = TMUX_TMPDIR_SPOOF

    yield TMUX_TMPDIR_SPOOF

    # Fixture teardown here: kill only the spoofed server, then clean up.
    libtmux.Server().kill()
    shutil.rmtree(TMUX_TMPDIR_SPOOF, ignore_errors=True)
    shutil.rmtree(TMUX_TMPDIR_SPOOF, ignore_errors=True)

    if saved_tmux is not None:
        os.environ["TMUX"] = saved_tmux


def test_check_if_executed():  # it's not!!!! Cuz of the filename.
    return 1 / 0
