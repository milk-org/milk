import os, sys, pathlib
import nox
import shutil

nox.options.error_on_missing_interpreters = False
"""
I am investigating nox as an option to run the (some) tests with multiple "installation modes"
In particular, with the dependency on libImageStreamIO.so and the building of the python
module with pybind, and combining between editable and non-editable installs,
comprehending _what_ exactly happens is tricky.

Nox should allow making virgin python environments, installing one way or another, and
running the test suite.

The functions below test for editable/non-editable install
and whether the tests are run from pyMilk dir or externally. This matters due to PYTHONPATH resolution.
"""

# Capture system tool paths before nox sanitizes PATH
_GCOV = shutil.which("gcov")


@nox.session(default=False)
def tests_run_coverage_lazybuild(session: nox.Session): ...


@nox.session(default=False)
def tests_run_coverage(session: nox.Session):

    THIS_PATH = pathlib.Path(
        os.path.abspath(os.getcwd())
    )  # Am I expecting this to be $MILK_ROOT/testing?
    PROJECT_ROOT = THIS_PATH.parent

    tmp_dir = os.path.abspath(session.create_tmp())
    session.chdir(tmp_dir)
    os.makedirs("./build", exist_ok=True)
    session.chdir("build")

    session.install("setuptools", "coverage", "pytest")

    session.run(
        *(
            f"cmake {PROJECT_ROOT} -DCMAKE_BUILD_TYPE=Coverage -DCMAKE_INSTALL_PREFIX={tmp_dir} -DUSE_CUDA=ON"
        ).split(),
        external=True,
    )
    session.run(*(f"make -j 32").split(), external=True)
    session.run(*(f"make install").split(), external=True)

    session.chdir(tmp_dir)

    session.env["PATH"] = tmp_dir + "/milk-1.03.00/bin:" + session.env.get("PATH", "")
    session.env["LD_LIBRARY_PATH"] = (
        tmp_dir + "/milk-1.03.00/lib:" + session.env.get("LD_LIBRARY_PATH", "")
    )

    session.run("pytest", str(PROJECT_ROOT) + "/testing")

    os.makedirs("./gcov_html", exist_ok=True)
    session.run(
        "gcovr",
        "--verbose",
        "--gcov-executable",
        _GCOV,
        "-r",
        PROJECT_ROOT,
        "--html-details",
        "-o",
        str(PROJECT_ROOT) + "/testing/gcov_html/c_coverage.html",
        os.path.abspath("./build"),
    )
