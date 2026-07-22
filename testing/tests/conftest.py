import pytest

import os

# Fetch autouse MILK fixtures from pyMilk
pytest_plugins = [
    "tests.conftestaux.cacao_loop",
    "tests.conftestaux.milk",
    "tests.conftestaux.coverage",
]


# ConfTest.py FIXTture == ctfixt_
@pytest.fixture(scope="session", autouse=True)
def ctfixt_change_pwd(tmpdir_factory, request):

    # This would be better used with the --basetemp feature...

    basetemp = None
    if "PYTEST_TMPDIR" in os.environ:
        basetemp = os.environ["PYTEST_TMPDIR"]
    if basetemp:
        tmpdir_factory.config.option.basetemp = basetemp

    # from datetime import datetime
    # time_str = datetime.strftime(datetime.now(), '%Y%m%dT%H%M%S')

    # if there is no basetemp, this just straight up goes to a tree in /tmp! Perfect
    folder = tmpdir_factory.mktemp("pytest_dont_use_for_anything")
    orig_cwd = os.getcwd()

    os.chdir(folder)

    yield None

    os.chdir(orig_cwd)
