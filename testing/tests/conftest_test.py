"""
This file is basic sanity to make sure the implemented fixtures
in
    - conftest.py
    - conftestaux/*.py
actually do their job.
"""

import os


def test_milk_shm_dir_fixture():
    # Fixture from milk.py -- which is scope='session', autouse=True
    MILK_SHM_DIR_SPOOF = "/tmp/milk_shm_dir_pytest"

    assert os.path.isdir(MILK_SHM_DIR_SPOOF)
    assert os.environ["MILK_SHM_DIR"] == MILK_SHM_DIR_SPOOF


def test_cwd_to_temp_fixture():
    cwd = os.getcwd()

    assert (
        cwd.startswith("/tmp") or "/tmp/" in cwd
    )  # nox will nest a ..../tmp/... under the nox session dir.
    # last dir made by the fixture...
    # if other fixtures change cwd further, they should also revert and also not be autouse.
    assert cwd.endswith("pytest_dont_use_for_anything0")
