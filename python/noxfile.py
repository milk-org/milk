import os, sys, pathlib
import nox
import shutil

nox.options.error_on_missing_interpreters = False
nox.options.reuse_existing_virtualenvs = True
# Prefer uv; fall back to virtualenv (pip) when uv is not installed
VENV_BACKEND = "uv|virtualenv"

# Capture system tool paths before nox sanitizes PATH
_GCOV = shutil.which("gcov")


def milk_build_and_test(
    session: nox.Session,
    *,
    use_cuda: bool = False,
    lto_level: int = 1,
    use_cli: bool = False,
):

    session.install("setuptools", "coverage", "pytest", "pytest-timeout")
    session.install("-e", ".")  # install milk

    this_path = pathlib.Path(os.path.abspath(os.getcwd()))
    project_root = this_path.parent
    milk_build(
        session, project_root, use_cuda=use_cuda, lto_level=lto_level, use_cli=use_cli
    )

    session.chdir(project_root / "python")  # otherwise pytest is somewhat upset...
    session.run("pytest", str(project_root / "python"))


def milk_build(
    session: nox.Session,
    PROJECT_ROOT: pathlib.Path,
    *,
    use_cuda: bool = False,
    lto_level: int = 1,
    use_cli: bool = False,
):

    on_off = lambda b: "ON" if b else "OFF"
    lto_string = (
        "-DUSE_LTO=OFF",
        "-DUSE_LTO=ON -DUSE_STATIC_LTO=OFF",
        "-DUSE_LTO=ON -DUSE_STATIC_LTO=ON",
    )[lto_level]

    tmp_dir = os.path.abspath(session.create_tmp())
    session.chdir(tmp_dir)

    if not os.path.islink("./pyMilk"):
        HOME = os.environ["HOME"]
        PYMILK_ROOT = os.environ.get("PYMILK_ROOT", f"{HOME}/src/pyMilk")
        session.run("ln", "-sf", PYMILK_ROOT, "./pyMilk", external=True)
    session.install("./pyMilk")  # install pyMilk

    os.makedirs("./build", exist_ok=True)
    session.chdir("build")

    # Of note, we could have a mismatch of the available engine extensions (USE_CUDA in particular) between pyMilk and milk builds.
    use_ninja = False
    session.run(
        *(
            f"cmake{' -GNinja' if use_ninja else ''} {PROJECT_ROOT} -DCMAKE_INSTALL_PREFIX={tmp_dir} "
            f"-DUSE_CUDA={on_off(use_cuda)} -DUSE_CLI={on_off(use_cli)} "
            f"{lto_string}"
        ).split(),
        external=True,
        silent=False,
    )
    if use_ninja:
        session.run(*(f"ninja install").split(), external=True, silent=True)
    else:
        session.run(*(f"make -j 30 install").split(), external=True, silent=True)

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
    session.env["MILK_INSTALLDIR"] = tmp_dir + "/milk-1.03.00"


@nox.session(default=False)
def tests_run_coverage_lazybuild(session: nox.Session): ...


@nox.session(venv_backend=VENV_BACKEND)
def build_and_pytest_nocli_nolto(session: nox.Session):
    milk_build_and_test(session, use_cli=False, lto_level=0)


@nox.session(venv_backend=VENV_BACKEND)
def build_and_pytest_nocli_lto(session: nox.Session):
    milk_build_and_test(session, use_cli=False, lto_level=1)


@nox.session(venv_backend=VENV_BACKEND)
def build_and_pytest_cli_nolto(session: nox.Session):
    milk_build_and_test(session, use_cli=True, lto_level=0)


@nox.session(venv_backend=VENV_BACKEND)
def build_and_pytest_cli_lto(session: nox.Session):
    milk_build_and_test(session, use_cli=True, lto_level=1)


@nox.session(venv_backend=VENV_BACKEND, default=False)
def build_and_pytest_cli_ltostat(session: nox.Session):
    milk_build_and_test(session, use_cli=True, lto_level=2)


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
