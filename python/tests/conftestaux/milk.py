import os, shutil, tempfile

import pytest


# ConfTest.py FIXTture == ctfixt_
@pytest.fixture(scope="session", autouse=True)
def ctfixt_change_MILK_SHM_DIR():

    # One per pytest process, so that several can run side by side. No longer than the fixed name
    # it replaces: a unit's --file path, built under it, holds 63 characters.
    MILK_SHM_DIR_SPOOF = tempfile.mkdtemp(prefix="milk_pytest", dir="/tmp")

    # There was something more aggressive to reload the env at runtime, wasn't there?
    os.environ["MILK_SHM_DIR"] = MILK_SHM_DIR_SPOOF
    # It is used by processinfo... really necessary?
    os.environ["MILK_PROC_DIR"] = MILK_SHM_DIR_SPOOF

    yield MILK_SHM_DIR_SPOOF

    # Fixture teardown here:
    shutil.rmtree(MILK_SHM_DIR_SPOOF, ignore_errors=True)


@pytest.fixture(scope="session", autouse=True)
def ctfixt_change_tmux_server():
    """Any tmux a test starts (the TUI drivers do) gets its socket here, away from the user's."""
    TMUX_TMPDIR_SPOOF = tempfile.mkdtemp(prefix="milk_tmux_pytest", dir="/tmp")

    saved_tmux = os.environ.pop("TMUX", None)  # detach from any outer tmux client
    os.environ["TMUX_TMPDIR"] = TMUX_TMPDIR_SPOOF

    yield TMUX_TMPDIR_SPOOF

    shutil.rmtree(TMUX_TMPDIR_SPOOF, ignore_errors=True)

    if saved_tmux is not None:
        os.environ["TMUX"] = saved_tmux


def test_check_if_executed():  # it's not!!!! Cuz of the filename.
    return 1 / 0
