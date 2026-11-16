import os, sys
import nox
import shutil
import subprocess
from pathlib import Path

nox.options.error_on_missing_interpreters = False


def _imagestreamio_has_cuda(session: nox.Session) -> bool:
    result = subprocess.run(
        [
            f"{session.bin}/python",
            "-c",
            "from pyMilk.interfacing.shm import IMAGESTREAMIO_HAVE_CUDA; raise SystemExit(not IMAGESTREAMIO_HAVE_CUDA)",
        ],
        capture_output=True,
    )
    return result.returncode == 0


def _cupy_package() -> str:
    """Return the cupy wheel and the minimal NVRTC headers package for the installed CUDA version.

    `cupy-cuda{N}x` is the fast pre-built wheel, but NVRTC needs compiler
    headers at runtime to JIT-compile kernels.  Instead of pulling the full
    toolkit via `cupy-cuda{N}x[ctk]`, we install only `nvidia-cuda-nvcc-cu{N}`
    which is the single pip package that provides those headers — much smaller.

    Falls back to plain 'cupy' if the CUDA version cannot be determined.
    """
    result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
    if result.returncode == 0:
        for line in result.stdout.splitlines():
            # e.g. "Cuda compilation tools, release 12.2, V12.2.140"
            if "release" in line:
                major = line.split("release")[1].strip().split(".")[0]
                return f"cupy-cuda{major}x"
    return "cupy"


def _cuda_root() -> str | None:
    result = subprocess.run(["which", "nvcc"], capture_output=True, text=True)
    if result.returncode == 0:
        # nvcc lives at $CUDA_ROOT/bin/nvcc
        return str(Path(result.stdout.strip()).parent.parent)
    return None


def _set_cuda_env(session: nox.Session) -> None:
    cuda_root = _cuda_root()
    if cuda_root and not session.env.get("CUDA_PATH"):
        session.env["CUDA_PATH"] = cuda_root


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


@nox.session
def tests_not_installed_inner(session: nox.Session):
    """
    NOT INSTALLED, ie no pip install; test from inside the project root.
    INNER testing, ie we run pytest while PWD is $PYMILK_ROOT (often $HOME/src/pyMilk/)
    ACTUALLY THAT CANNOT WORK, due to uncompiled ImageStreamIO / CacaoProcessTools which gets compiled on install.
    """
    ...


@nox.session
def tests_editable_inner_outer(session: nox.Session):
    """
    EDITABLE installation, ie `pip install -e .`; test from inside the project root.

    INNER testing, ie we run pytest while PWD is $PYMILK_ROOT (often $HOME/src/pyMilk/)
    OUTER testing, ie we run pytest while PWD is NOT $PYMILK_ROOT; here's its /tmp/nox

    We do both to ensure package name shadowing when running in the directory is not an issue.
    """

    session.install("pytest", "pyright")
    session.install("-e", ".")
    # session.install("pyright") # why pyright twice ????

    if _imagestreamio_has_cuda(session):
        session.install(_cupy_package())
        _set_cuda_env(session)

    # Run tests from inside the project root
    session.run("pytest")

    # Run tests from outside the project root
    project_dir = os.path.abspath(os.getcwd())
    os.makedirs("/tmp/pytest_workspace", exist_ok=True)

    # This gymnastics of making "tests" importable has been rendered necessary
    # When switching pytest multiprocessing from fork to spawn
    # (as otherwise necessary for CUDA)
    tmp_dir = session.create_tmp()
    session.chdir(tmp_dir)
    shutil.copytree(project_dir + "/tests", tmp_dir + "/tests")
    with session.chdir(tmp_dir):
        session.run("pytest", "-x", "--pdb", project_dir)
    shutil.rmtree(tmp_dir + "/tests")


@nox.session
def tests_non_editable_inner_and_outer(session: nox.Session):
    """
    NON-EDITABLE installation, ie `pip install .`; run from both inside and outside the project root.
    IN and OUT testing
    """
    # session.run("./clean.sh", external=True)

    session.install("pytest")
    session.install(".")

    if _imagestreamio_has_cuda(session):
        session.install(_cupy_package())
        _set_cuda_env(session)

    # Change cwd to something else...
    # Otherwise name collision with the source folder, rather than the install in the venv.
    project_dir = os.path.abspath(os.getcwd())

    tmp_dir = session.create_tmp()
    session.chdir(tmp_dir)
    shutil.copytree(project_dir + "/tests", tmp_dir + "/tests")
    with session.chdir(tmp_dir):
        session.run("pytest", project_dir)
    shutil.rmtree(tmp_dir + "/tests")


@nox.session(default=False)
def tests_run_coverage(session: nox.Session):
    print(os.path.abspath(os.getcwd()))
    session.run("./clean.sh", external=True)

    session.env["COVERAGE"] = "ON"
    session.install("nanobind", "setuptools", "coverage", "pytest")
    session.install(".")  # for dependencies only...

    session.run(*("python setup.py build_ext --inplace".split()))

    print(os.path.abspath(os.getcwd()))
    import shutil, glob, pathlib

    for file in glob.glob("./build/lib.*/pyMilk/*.so"):
        fname = pathlib.Path(file).name
        shutil.copyfile(file, f"pyMilk/{fname}")

    if _imagestreamio_has_cuda(session):
        session.install(_cupy_package())
        _set_cuda_env(session)

    session.run("coverage", "run")
    session.run("coverage", "report")  # TODO where html??
    session.run("coverage", "html")  # TODO where html??

    os.makedirs("gcov_html", exist_ok=True)
    session.run(
        "gcovr",
        "-r",
        ".",
        "--exclude",
        ".*/nanobind/.*",
        "--exclude",
        ".*/pybind.*",
        "--html-details",
        "-o",
        "gcov_html/c_coverage.html",
    )
