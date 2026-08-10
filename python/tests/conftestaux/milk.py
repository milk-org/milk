import os, shutil

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


def test_check_if_executed():  # it's not!!!! Cuz of the filename.
    return 1 / 0
