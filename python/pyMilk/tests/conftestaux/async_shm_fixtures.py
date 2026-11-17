from __future__ import annotations

import multiprocessing

try:
    # spawn is needed for GPU profiling in children and C/C++ coverage (although more costly)
    multiprocessing.set_start_method("spawn")
    # Ensure the repo root is in sys.path so that spawned processes (which
    # inherit sys.path via multiprocessing preparation data) can import
    # tests.conftestaux.async_shm_fixtures to reconstruct pickled targets.
    import sys as _sys, os as _os

    _repo_root = _os.path.abspath(_os.path.join(__file__, "../../.."))
    if _repo_root not in _sys.path:
        # append, don't prepend: its pyMilk/  must not shadow the install.
        _sys.path.append(_repo_root)
    del _sys, _os, _repo_root
except RuntimeError:
    pass

import abc
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


class _SubprocessTask(abc.ABC):
    """Body of the subprocess spawned by `subfixture`.

    `run` is the Process target: `setup`, then the startup handshake on
    `evt` (post once, wait for the parent to clear, so the exit checks on the
    same event don't fire), then `main`. `evt` is set again by the parent to
    request exit.
    """

    def __init__(
        self,
        toprocess_name: str,
        fromprocess_name: str,
        shape: tuple[int, ...],
        dtype: np.typing.DTypeLike,
        location: int,
        delay: float,
        evt: T_Event,
    ) -> None:
        self.toprocess_name = toprocess_name
        self.fromprocess_name = fromprocess_name
        self.shape = shape
        self.dtype = dtype
        self.location = location
        self.delay = delay
        self.evt = evt

    @abc.abstractmethod
    def setup(self) -> None: ...

    @abc.abstractmethod
    def main(self) -> None: ...

    def run(self) -> None:
        self.setup()
        self.evt.set()
        while self.evt.is_set():
            time.sleep(0.001)
        self.main()


class _Pingpong(_SubprocessTask):
    def setup(self) -> None:
        self.shm_toprocess = SHM(self.toprocess_name)
        self.shm_fromprocess = SHM(self.fromprocess_name)

        self.shm_toprocess._checkGrabSemaphore()
        self.shm_toprocess.IMAGE.semflush(self.shm_toprocess.semID)

        self.shm_fromprocess._checkGrabSemaphore()
        self.shm_fromprocess.IMAGE.semflush(self.shm_fromprocess.semID)

    def main(self) -> None:
        break_all = False

        while True:
            while not self.shm_toprocess.check_sem_timedwait(0.001):
                if self.evt.is_set():
                    break_all = True
                    break
            if self.evt.is_set() or break_all:
                break

            # Repost the semaphores.
            dat = self.shm_toprocess.get_data(
                False
            )  # check_sem_timedwait clicked the semaphore.
            time.sleep(self.delay)
            self.shm_fromprocess.set_data(dat + 1)


class _Looper(_SubprocessTask):
    def setup(self) -> None:
        self.shm_fromprocess = SHM(self.fromprocess_name)
        self.dat = self.shm_fromprocess.get_data()

    def main(self) -> None:
        ii = -1
        while True:
            ii += 1
            if self.evt.is_set():
                break
            self.dat[:] = ii
            time.sleep(self.delay)
            self.shm_fromprocess.set_data(self.dat)


class _LooperWithForcefulRecreation(_Looper):
    def main(self) -> None:
        ii = -1
        while True:
            ii += 1
            if self.evt.is_set():
                break

            self.dat[:] = ii
            time.sleep(self.delay)
            self.shm_fromprocess.close()
            self.shm_fromprocess = SHM(
                self.fromprocess_name, self.dat
            )  # I count that this induces a set_data and sempost


class _CreateKillRecreate(_SubprocessTask):
    def setup(self) -> None:
        self.shm_toprocess = SHM(self.toprocess_name)
        self.shm_fromprocess = SHM(self.fromprocess_name)

    def main(self) -> None:
        shm_toprocess, shm_fromprocess = self.shm_toprocess, self.shm_fromprocess
        time.sleep(0.1)  # be sure we got time to instantiate in main process
        shm_toprocess.repost()
        print(f"orig -- {shm_toprocess.IMAGE.md.inode}")
        time.sleep(0.1)
        shm_toprocess = SHM(
            self.toprocess_name, (self.shape, self.dtype), location=self.location
        )
        shm_fromprocess.repost()
        print(f"recreated -- {shm_toprocess.IMAGE.md.inode}")
        time.sleep(0.1)
        shm_toprocess.repost()

        time.sleep(0.1)
        shm_toprocess = SHM(
            self.toprocess_name, ((5, 2, 3), np.int32), location=self.location
        )
        shm_fromprocess.repost()
        print(f"recreated -- {shm_toprocess.IMAGE.md.inode}")
        time.sleep(0.1)
        shm_toprocess.repost()


@pytest.fixture
def threaded_shm_data_pingpong(request):
    yield from subfixture(request, _Pingpong)


@pytest.fixture
def threaded_shm_data_pingpong_gpu(request):
    yield from subfixture(request, _Pingpong, location=0)


@pytest.fixture
def threaded_shm_poster(request):
    yield from subfixture(request, _Looper)


@pytest.fixture
def threaded_shm_poster_recreator(request):
    yield from subfixture(request, _LooperWithForcefulRecreation)


@pytest.fixture
def threaded_shm_create_kill_recreate(request):
    yield from subfixture(request, _CreateKillRecreate)


def subfixture(
    request: pytest.FixtureRequest,
    task_cls: type[_SubprocessTask],
    location: int = -1,
) -> typ.Generator[tuple[SHM, SHM, multiprocessing.Process], None, None]:

    prefix, shape, dtype, delay = request.param

    shm_toprocess = SHM(prefix + "_toprocess", (shape, dtype), location=location)
    shm_fromprocess = SHM(prefix + "_fromprocess", (shape, dtype), location=location)
    # Startup handshake (child sets, parent waits and clears), then exit signal.
    sync_event = multiprocessing.Event()

    # Originally attempted with a threading.Thread, but that causes issues
    # because the semID issued is identical !
    task = task_cls(
        prefix + "_toprocess",
        prefix + "_fromprocess",
        shape,
        dtype,
        location,
        delay,
        sync_event,
    )
    subprocess = multiprocessing.Process(target=task.run, daemon=False)
    subprocess.start()

    shm_toprocess._checkGrabSemaphore()
    shm_fromprocess._checkGrabSemaphore()
    shm_toprocess.IMAGE.semflush(shm_toprocess.semID)
    shm_fromprocess.IMAGE.semflush(shm_fromprocess.semID)
    assert sync_event.wait(timeout=10.0), "subprocess never signaled ready"
    sync_event.clear()
    yield shm_toprocess, shm_fromprocess, subprocess

    sync_event.set()
    subprocess.join(timeout=1.0)

    assert not subprocess.is_alive()

    shm_toprocess.destroy()
    shm_fromprocess.destroy()
