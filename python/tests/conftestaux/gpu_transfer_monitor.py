"""
gpu_transfer_monitor.py — CUPTI-based GPU host-transfer monitoring for pytest.

Loads the companion libcupti_spy.so (built from tests/gpu_monitor/Makefile)
to intercept **all** cudaMemcpy calls in the current process, regardless of
which library initiated them (CuPy, PyTorch, pyMilk, etc.).

Usage
-----
As a context manager in a test:

    from tests.conftestaux.gpu_transfer_monitor import assert_no_gpu_host_transfers

    def test_zero_copy_roundtrip():
        shm = SHM('my_gpu_stream')
        with assert_no_gpu_host_transfers():
            arr = cupy.asarray(shm.IMAGE.view())  # must be zero-copy
            shm.set_data(arr)

As a pytest fixture (add to conftest.py plugins list and use as parameter):

    def test_zero_copy_roundtrip(no_gpu_host_transfers):
        ...

Availability guard
------------------
If libcupti_spy.so has not been built, attempting to call either the context
manager or the fixture raises ``CuptiSpyUnavailable`` — tests that import the
fixture are automatically skipped via ``pytest.importorskip``-style logic.
Build the library first with::

    make -C tests/gpu_monitor/
"""

from __future__ import annotations

import ctypes
import os
import struct
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import pytest

# ---------------------------------------------------------------------------
# Locate the shared library relative to this file
# ---------------------------------------------------------------------------


def attempt_build_libcupti_spy():
    import subprocess

    """
    Attempts to build the CUPTI spy library from tests/conftestaux/gpu_monitor/Makefile.
    If the build fails, it will skip GPU transfer monitoring tests but not fail the session.
    """
    gpu_monitor_dir = Path(__file__).parent / "gpu_monitor"

    try:
        # Run make to build libcupti_spy.so
        subprocess.run(
            ["make"],
            cwd=gpu_monitor_dir,
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )
        print(f"Successfully built libcupti_spy.so from {gpu_monitor_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Warning: Failed to build libcupti_spy.so from {gpu_monitor_dir}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
    except FileNotFoundError:
        print(
            f"Warning: make command not found or gpu_monitor directory not found at {gpu_monitor_dir}"
        )
    except subprocess.TimeoutExpired:
        print(f"Warning: Building libcupti_spy.so timed out")
    except Exception as e:
        print(f"Warning: Unexpected error building libcupti_spy.so: {e}")


_LIB_PATH = Path(__file__).parent / "gpu_monitor" / "libcupti_spy.so"


def _load_lib() -> ctypes.CDLL:
    attempt_build_libcupti_spy()

    if not _LIB_PATH.exists():
        raise CuptiSpyUnavailable(
            f"libcupti_spy.so not found at {_LIB_PATH}. "
            "Run  make -C tests/conftestaux/gpu_monitor/  to build it."
        )

    lib = ctypes.CDLL(str(_LIB_PATH))

    lib.cupti_spy_start.restype = None
    lib.cupti_spy_start.argtypes = []

    lib.cupti_spy_stop.restype = None
    lib.cupti_spy_stop.argtypes = []

    lib.cupti_spy_get_dtoh.restype = ctypes.c_uint64
    lib.cupti_spy_get_dtoh.argtypes = []

    lib.cupti_spy_get_htod.restype = ctypes.c_uint64
    lib.cupti_spy_get_htod.argtypes = []

    return lib


class CuptiSpyUnavailable(RuntimeError):
    """Raised when libcupti_spy.so has not been built."""


# Lazy-builder-loader for CUPTY_SPY_LIB
class CUPTY_SPY_LIB_CLS:

    def __init__(self) -> None:
        self.loaded = False
        self.CUPTI_SPY_LIB: ctypes.CDLL | None = None

    def __call__(self) -> ctypes.CDLL | None:
        if not self.loaded:
            try:
                self.CUPTI_SPY_LIB = _load_lib()
            except CuptiSpyUnavailable as exc:
                print(repr(exc))
            self.loaded = True  # Which may be a failed build and load !
        return self.CUPTI_SPY_LIB


CUPTI_SPY_LIB_LAZYLOADER = CUPTY_SPY_LIB_CLS()  # Singleton, unloaded

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@contextmanager
def assert_no_gpu_host_transfers() -> Generator[None, None, None]:
    """
    Context manager that raises AssertionError if any cudaMemcpy
    DeviceToHost or HostToDevice call occurs inside the block.

    Raises CuptiSpyUnavailable if the native library has not been built.
    """
    CUPTI = CUPTI_SPY_LIB_LAZYLOADER()
    if CUPTI is None:
        raise CuptiSpyUnavailable

    CUPTI.cupti_spy_start()
    try:
        yield
    finally:
        CUPTI.cupti_spy_stop()

    dtoh: int = CUPTI.cupti_spy_get_dtoh()
    htod: int = CUPTI.cupti_spy_get_htod()

    assert dtoh == 0 and htod == 0, (
        f"Unexpected GPU host transfers detected: "
        f"{dtoh} DeviceToHost,  {htod} HostToDevice"
    )


from dataclasses import dataclass


@dataclass
class TransferCounter:
    htod: int = 0
    dtoh: int = 0


@contextmanager
def count_gpu_host_transfers() -> Generator[TransferCounter, None, None]:
    """
    Context manager that yields a mutable dict ``{'dtoh': N, 'htod': N}``
    populated with transfer counts after the block exits.

    Example::

        with count_gpu_host_transfers() as counts:
            do_work()
        assert counts['dtoh'] == 0
    """
    CUPTI = CUPTI_SPY_LIB_LAZYLOADER()
    if CUPTI is None:
        raise CuptiSpyUnavailable
    counts = TransferCounter()
    CUPTI.cupti_spy_start()
    try:
        yield counts
    finally:
        CUPTI.cupti_spy_stop()
        counts.dtoh = int(CUPTI.cupti_spy_get_dtoh())
        counts.htod = int(CUPTI.cupti_spy_get_htod())


# ---------------------------------------------------------------------------
# Pytest fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def no_gpu_host_transfers() -> Generator[None, None, None]:
    with assert_no_gpu_host_transfers():
        yield


def _subprocess_spy_wrapper(fn, args, kwargs, lib_path: str, out_path: str):
    """
    Module-level picklable wrapper.  Runs in the subprocess: loads
    libcupti_spy.so, arms CUPTI *after* Python/CUDA are fully initialized,
    calls fn, then writes counters to out_path for the parent to read.
    """
    import ctypes

    lib = ctypes.CDLL(lib_path)
    lib.cupti_spy_start.restype = None
    lib.cupti_spy_start.argtypes = []
    lib.cupti_spy_stop.restype = None
    lib.cupti_spy_stop.argtypes = []
    lib.cupti_spy_write_counts.restype = None
    lib.cupti_spy_write_counts.argtypes = [ctypes.c_char_p]

    lib.cupti_spy_start()
    try:
        fn(*args, **kwargs)
    finally:
        lib.cupti_spy_stop()
        lib.cupti_spy_write_counts(out_path.encode())


@contextmanager
def count_subprocess_transfers():
    """
    Context manager that counts GPU host-transfers produced by a subprocess.

    Yields a ``(TransferCounter, spy)`` pair.
    ``spy(fn, *args, **kwargs)`` returns a picklable callable that wraps *fn*
    with CUPTI monitoring inside the subprocess.  Use with
    ``multiprocessing.get_context('spawn')`` so the child starts fresh::

        with count_subprocess_transfers() as (counts, spy):
            ctx = multiprocessing.get_context('spawn')
            p = ctx.Process(target=spy(my_gpu_function))
            p.start()
            p.join()
        assert counts.htod == 100

    Raises ``CuptiSpyUnavailable`` if the native library has not been built.
    """
    if not _LIB_PATH.exists():
        raise CuptiSpyUnavailable(
            f"libcupti_spy.so not found at {_LIB_PATH}. "
            "Run  make -C tests/conftestaux/gpu_monitor/  to build it."
        )

    counts = TransferCounter()
    fd, out_path = tempfile.mkstemp(prefix="cupti_spy_", suffix=".bin")
    os.close(fd)
    lib_path = str(_LIB_PATH)

    def spy(fn, *args, **kwargs):
        """Return a picklable wrapper that monitors fn in the subprocess."""
        import functools

        return functools.partial(
            _subprocess_spy_wrapper, fn, args, kwargs, lib_path, out_path
        )

    try:
        yield counts, spy
    finally:
        try:
            with open(out_path, "rb") as f:
                data = f.read(16)
            if len(data) == 16:
                dtoh, htod = struct.unpack("<QQ", data)
                counts.dtoh = int(dtoh)
                counts.htod = int(htod)
        except FileNotFoundError:
            pass
        finally:
            try:
                os.unlink(out_path)
            except FileNotFoundError:
                pass


@contextmanager
def count_popen_transfers():
    if not _LIB_PATH.exists():
        raise CuptiSpyUnavailable(
            f"libcupti_spy.so not found at {_LIB_PATH}. "
            "Run  make -C tests/conftestaux/gpu_monitor/  to build it."
        )

    counts = TransferCounter()
    fd, out_path = tempfile.mkstemp(prefix="cupti_spy_", suffix=".bin")
    os.close(fd)

    extra_env = {
        "CUDA_INJECTION64_PATH": str(_LIB_PATH),
        "CUPTI_SPY_FILE": out_path,
    }
    try:
        yield counts, extra_env
    finally:
        # read file written by FinalizeInjection in the child
        try:
            with open(out_path, "rb") as f:
                data = f.read(16)
            if len(data) == 16:
                dtoh, htod = struct.unpack("<QQ", data)
                counts.dtoh = int(dtoh)
                counts.htod = int(htod)
        except FileNotFoundError:
            pass
        finally:
            try:
                os.unlink(out_path)
            except FileNotFoundError:
                pass
