"""
This file is basic sanity to make sure the implemented fixtures
in
    - conftest.py
    - conftestaux/*.py
actually do their job.
"""

import os


def test_milk_shm_dir_fixture(ctfixt_change_MILK_SHM_DIR):
    # Fixture from milk.py -- which is scope='session', autouse=True: one directory per pytest
    # process (a later test may have pointed MILK_SHM_DIR elsewhere, so ask the fixture)
    assert os.path.isdir(ctfixt_change_MILK_SHM_DIR)
    assert ctfixt_change_MILK_SHM_DIR.startswith("/tmp/milk_pytest")


def test_cwd_to_temp_fixture():
    cwd = os.getcwd()

    assert (
        cwd.startswith("/tmp") or "/tmp/" in cwd
    )  # nox will nest a ..../tmp/... under the nox session dir.
    # last dir made by the fixture...
    # if other fixtures change cwd further, they should also revert and also not be autouse.
    assert cwd.endswith("pytest_dont_use_for_anything0")
