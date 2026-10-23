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


@pytest.fixture(scope="session", autouse=True)
def ctfixt_test_tmux_server_chage(
    ctfixt_change_tmux_server: str, ctfixt_change_MILK_SHM_DIR: str
):
    """
    Test the tmux server swap happened correctly
    Wrapped in a autouse fixture to crash the test session early.
    """
    # TODO add info to doc.

    TMUX_TMPDIR_SPOOF = ctfixt_change_tmux_server

    full_info = """
============= WARNING: for pytest/nox to work entirely, ================
    You must allow environment variables to be passed to the
    testing tmux servers.

For MILK_SHM_DIR, you must configure the .bashrc/.profile/.env
with:
    export MILK_SHM_DIR="${MILK_SHM_DIR:-/milk/shm}" # sets a default but passes a non-default
but not:
    export MILK_SHM_DIR=/milk/shm
"""

    import libtmux

    from tests.conftestaux.polling import poll_until_or_fail

    assert os.environ["TMUX_TMPDIR"] == TMUX_TMPDIR_SPOOF
    assert "TMUX" not in os.environ

    server = libtmux.Server()
    session = server.new_session(session_name="pytest_conftest_sanity")
    try:
        # The server only starts (and creates its socket) once a session exists.
        socket_path = server.cmd("display-message", "-p", "#{socket_path}").stdout[0]
        assert socket_path.startswith(TMUX_TMPDIR_SPOOF + "/")
        assert os.path.exists(socket_path)

        # Round trip through a pane's shell: the server env must carry MILK_SHM_DIR.
        pane = session.active_window.active_pane
        assert pane is not None

        pane.send_keys("echo MILK_SHM_DIR=[$MILK_SHM_DIR]")
        expected = "MILK_SHM_DIR=[/tmp/milk_shm_dir_pytest]"
        poll_until_or_fail(
            lambda: expected in pane.capture_pane(),
            t=1,
            interval=0.05,
            msg=f"{expected!r} not found in pane output. {full_info}",
        )
    finally:
        # session.kill()
        pass

    yield None
