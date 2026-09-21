import os, sys, pathlib
import nox
import shutil

nox.options.error_on_missing_interpreters = False
nox.options.reuse_existing_virtualenvs = True
# nox.options.default_venv_backend = "uv"
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


def milk_build_and_test(
    session: nox.Session,
    *,
    use_cuda: bool = False,
    use_lto: bool = False,
    use_cli: bool = False,
):

    session.install("setuptools", "coverage", "pytest")
    session.run(*("uv pip install -e .").split())  # install milk

    this_path = pathlib.Path(os.path.abspath(os.getcwd()))
    project_root = this_path.parent
    milk_build(
        session, project_root, use_cuda=use_cuda, use_lto=use_lto, use_cli=use_cli
    )

    session.chdir(project_root / "python")  # otherwise pytest is somewhat upset...
    session.run("pytest", str(project_root / "python"))


def milk_build(
    session: nox.Session,
    PROJECT_ROOT: pathlib.Path,
    *,
    use_cuda: bool = False,
    use_lto: bool = False,
    use_cli: bool = False,
):

    on_off = lambda b: "ON" if b else "OFF"

    tmp_dir = os.path.abspath(session.create_tmp())
    session.chdir(tmp_dir)

    if not os.path.islink("./pyMilk"):
        session.run(*("ln -sf /home/vdeo/src/pyMilk ./pyMilk").split(), external=True)
    session.run(*("uv pip install ./pyMilk").split())  # install pyMilk

    os.makedirs("./build", exist_ok=True)
    session.chdir("build")

    # Of note, we could have a mismatch of the available engine extensions (USE_CUDA in particular) between pyMilk and milk builds.
    session.run(
        *(
            f"cmake {PROJECT_ROOT} -DCMAKE_INSTALL_PREFIX={tmp_dir} "
            f"-DUSE_CUDA={on_off(use_cuda)} -DUSE_CLI={on_off(use_cli)} "
            f"-DUSE_STATIC_LTO={on_off(use_lto)}"
        ).split(),
        external=True,
    )
    session.run(*(f"make -j20 install").split(), external=True)

    session.chdir(tmp_dir)

    # Add default path's that the nox session doesn't see, but subprocessing to system tools
    # may require them (e.g. lsmod to detect nvidia driver)
    DEFAULT_PATHS = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

    session.env["PATH"] = (
        tmp_dir + "/milk-1.03.00/bin:" + DEFAULT_PATHS + session.env.get("PATH", "")
    )  # type: ignore
    session.env["LD_LIBRARY_PATH"] = (
        tmp_dir + "/milk-1.03.00/lib:" + session.env.get("LD_LIBRARY_PATH", "")
    )  # type: ignore


@nox.session(default=False)
def tests_run_coverage_lazybuild(session: nox.Session): ...


@nox.session(venv_backend="uv")
def build_and_pytest_nocli_nolto(session: nox.Session):
    milk_build_and_test(session, use_cli=False, use_lto=False)


@nox.session(venv_backend="uv")
def build_and_pytest_nocli_lto(session: nox.Session):
    milk_build_and_test(session, use_cli=False, use_lto=True)


@nox.session(venv_backend="uv")
def build_and_pytest_cli_nolto(session: nox.Session):
    milk_build_and_test(session, use_cli=True, use_lto=False)


@nox.session(venv_backend="uv")
def build_and_pytest_cli_lto(session: nox.Session):
    milk_build_and_test(session, use_cli=True, use_lto=True)


@nox.session(default=False)
def tests_run_coverage(session: nox.Session):

    THIS_PATH = pathlib.Path(
        os.path.abspath(os.getcwd())
    )  # Am I expecting this to be $MILK_ROOT/testing?
    PROJECT_ROOT = THIS_PATH.parent

    milk_build(session, PROJECT_ROOT, use_cuda=True)

    # TODO Struggling to get this to work.
    # session.env["COVERAGE_RCFILE"] = str(PROJECT_ROOT / "python" / "pyproject.toml")
    # session.run("coverage", "run", "-m", "pytest", str(PROJECT_ROOT / "python" / "tests"))
    # session.run(*(f"coverage html -d {str(PROJECT_ROOT)}/cov_py --data-file=.coverage").split())

    session.run("pytest", str(PROJECT_ROOT / "python"))

    os.makedirs(str(PROJECT_ROOT) + "/cov_c", exist_ok=True)
    session.run(
        "gcovr",
        # "--verbose",
        "--gcov-executable",
        _GCOV,
        "-r",
        PROJECT_ROOT,
        "--html-details",
        "-o",
        str(PROJECT_ROOT) + "/cov_c/index.html",
        os.path.abspath("./build"),
    )
