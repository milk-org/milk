from __future__ import annotations

import multiprocessing

try:
    multiprocessing.set_start_method("spawn")
    # Ensure the repo root is in sys.path so that spawned processes (which
    # inherit sys.path via multiprocessing preparation data) can import
    # tests.conftestaux.async_shm_fixtures to reconstruct pickled targets.
    import sys as _sys, os as _os

    _repo_root = _os.path.abspath(_os.path.join(__file__, "../../.."))
    if _repo_root not in _sys.path:
        _sys.path.insert(0, _repo_root)
    del _sys, _os, _repo_root
except RuntimeError:
    pass

import time
import numpy as np
import pytest

from pyMilk.interfacing.shm import SHM

import typing as typ

if typ.TYPE_CHECKING:
    from multiprocessing.synchronize import Event as T_Event


def _sproc_serve_external_shm(
    name: str,
    shape: tuple[int, ...],
    nptype: np.typing.DTypeLike,
    location: int,
    event: T_Event,
):
    s = SHM(name, (shape, nptype), location=location)
    event.wait()


@pytest.fixture
def serve_external_shm(request):
    name: str
    shape: tuple[int, ...]
    nptype: np.typing.DTypeLike
    location: int

    name, shape, nptype, location = request.param

    try:
        s = SHM(name)
        s.destroy()
    except:
        pass

    event = multiprocessing.Event()
    subprocess = multiprocessing.Process(
        target=_sproc_serve_external_shm,
        args=(name, shape, nptype, location, event),
        daemon=False,
    )
    subprocess.start()

    time.sleep(1.0)

    while True:
        time.sleep(0.01)
        try:
            s = SHM(name)
            break
        except:
            pass

    yield s

    event.set()
    subprocess.join()
    # s.destroy()


def _thread_func_pingpong(
    toprocess_name: str,
    fromprocess_name: str,
    shape,
    dtype,
    location: int,
    delay: float,
    evt,
) -> None:
    shm_toprocess = SHM(toprocess_name)
    shm_fromprocess = SHM(fromprocess_name)

    shm_toprocess._checkGrabSemaphore()
    shm_toprocess.IMAGE.semflush(shm_toprocess.semID)

    shm_fromprocess._checkGrabSemaphore()
    shm_fromprocess.IMAGE.semflush(shm_fromprocess.semID)

    break_all = False

    while True:
        while not shm_toprocess.check_sem_timedwait(0.001):
            if evt.is_set():
                break_all = True
                break
        if evt.is_set() or break_all:
            break

        # Repost the semaphores.
        dat = shm_toprocess.get_data(
            False
        )  # check_sem_timedwait clicked the semaphore.
        # print('P: giggity')
        time.sleep(delay)
        shm_fromprocess.set_data(dat + 1)


def _thread_func_looper(
    toprocess_name: str,
    fromprocess_name: str,
    shape,
    dtype,
    location: int,
    delay: float,
    evt,
) -> None:
    shm_fromprocess = SHM(fromprocess_name)
    dat = shm_fromprocess.get_data()
    ii = -1
    while True:
        ii += 1
        if evt.is_set():
            break
        dat[:] = ii
        time.sleep(delay)
        shm_fromprocess.set_data(dat)


def _thread_func_looper_with_forceful_recreation(
    toprocess_name: str,
    fromprocess_name: str,
    shape,
    dtype,
    location: int,
    delay: float,
    evt,
) -> None:
    shm_fromprocess = SHM(fromprocess_name)
    dat = shm_fromprocess.get_data()
    ii = -1
    while True:
        ii += 1
        if evt.is_set():
            break

        dat[:] = ii
        time.sleep(delay)
        shm_fromprocess.close()
        shm_fromprocess = SHM(
            fromprocess_name, dat
        )  # I count that this induces a set_data and sempost


def _thread_func_create_kill_recreate(
    toprocess_name: str,
    fromprocess_name: str,
    shape,
    dtype,
    location: int,
    delay: float,
    evt,
) -> None:
    shm_toprocess = SHM(toprocess_name)
    shm_fromprocess = SHM(fromprocess_name)
    time.sleep(0.1)  # be sure we got time to instantiate in main process
    shm_toprocess.repost()
    print(f"orig -- {shm_toprocess.IMAGE.md.inode}")
    time.sleep(0.1)
    shm_toprocess = SHM(toprocess_name, (shape, dtype), location=location)
    shm_fromprocess.repost()
    print(f"recreated -- {shm_toprocess.IMAGE.md.inode}")
    time.sleep(0.1)
    shm_toprocess.repost()

    time.sleep(0.1)
    shm_toprocess = SHM(toprocess_name, ((5, 2, 3), np.int32), location=location)
    shm_fromprocess.repost()
    print(f"recreated -- {shm_toprocess.IMAGE.md.inode}")
    time.sleep(0.1)
    shm_toprocess.repost()


@pytest.fixture
def threaded_shm_data_pingpong(request):
    yield from subfixture(request, _thread_func_pingpong)


@pytest.fixture
def threaded_shm_data_pingpong_gpu(request):
    yield from subfixture(request, _thread_func_pingpong, location=0)


@pytest.fixture
def threaded_shm_poster(request):
    yield from subfixture(request, _thread_func_looper)


@pytest.fixture
def threaded_shm_poster_recreator(request):
    yield from subfixture(request, _thread_func_looper_with_forceful_recreation)


@pytest.fixture
def threaded_shm_create_kill_recreate(request):
    yield from subfixture(request, _thread_func_create_kill_recreate)


def subfixture(request, subprocess_callable, location=-1):

    prefix, shape, dtype, delay = request.param

    shm_toprocess = SHM(prefix + "_toprocess", (shape, dtype), location=location)
    shm_fromprocess = SHM(prefix + "_fromprocess", (shape, dtype), location=location)
    exit_event = multiprocessing.Event()

    # Originally attempted with a threading.Thread, but that causes issues
    # because the semID issued is identical !
    subprocess = multiprocessing.Process(
        target=subprocess_callable,
        args=(
            prefix + "_toprocess",
            prefix + "_fromprocess",
            shape,
            dtype,
            location,
            delay,
            exit_event,
        ),
        daemon=False,
    )
    subprocess.start()

    shm_toprocess._checkGrabSemaphore()
    shm_fromprocess._checkGrabSemaphore()
    shm_toprocess.IMAGE.semflush(shm_toprocess.semID)
    shm_fromprocess.IMAGE.semflush(shm_fromprocess.semID)
    yield shm_toprocess, shm_fromprocess, subprocess

    exit_event.set()
    subprocess.join(timeout=1.0)

    assert not subprocess.is_alive()

    shm_toprocess.destroy()
    shm_fromprocess.destroy()
