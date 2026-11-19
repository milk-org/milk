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
import queue
import time
import warnings
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
    """Task run by the session worker (`_async_worker`) on behalf of `subfixture`.

    `run`: `setup`, then the startup handshake on `evt` (post once, wait for
    the parent to clear, so the exit checks on the same event don't fire),
    then `main`. `evt` is set again by the parent to request exit.
    Instances travel through a Queue, so `evt` is not an attribute: it is the
    worker's, passed at spawn time.
    """

    def __init__(
        self,
        toprocess_name: str,
        fromprocess_name: str,
        shape: tuple[int, ...],
        dtype: np.typing.DTypeLike,
        location: int,
        delay: float,
    ) -> None:
        self.toprocess_name = toprocess_name
        self.fromprocess_name = fromprocess_name
        self.shape = shape
        self.dtype = dtype
        self.location = location
        self.delay = delay

    @abc.abstractmethod
    def setup(self) -> None: ...

    @abc.abstractmethod
    def main(self) -> None: ...

    def run(self, evt: T_Event) -> None:
        self.evt = evt
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


def _worker_loop(
    tasks: multiprocessing.Queue, done: multiprocessing.Queue, evt: T_Event
) -> None:
    """Session worker: run tasks until None. Reports None or a traceback per task.
    Puts "ready" on `done` once started (imports done), before the first task."""
    import traceback

    done.put("ready")
    while (task := tasks.get()) is not None:
        try:
            task.run(evt)
            done.put(None)
        except BaseException:
            done.put(traceback.format_exc())


class _AsyncWorker(typ.NamedTuple):
    tasks: multiprocessing.Queue
    done: multiprocessing.Queue
    evt: T_Event


@pytest.fixture(scope="session")
def _async_worker() -> typ.Generator[_AsyncWorker, None, None]:
    """One spawned process for the whole session, so `subfixture` doesn't pay
    interpreter start-up and import cost for every test."""
    # Originally attempted with a threading.Thread, but that causes issues
    # because the semID issued is identical !
    worker = _AsyncWorker(
        multiprocessing.Queue(), multiprocessing.Queue(), multiprocessing.Event()
    )
    process = multiprocessing.Process(target=_worker_loop, args=worker, daemon=True)
    process.start()
    # Spawn + imports can take seconds on a cold CI runner: wait here, not in the
    # first test's 10 s handshake.
    try:
        ready = worker.done.get(timeout=20.0)
    except queue.Empty:
        process.kill()
        pytest.fail(f"async worker not ready after 20 s (alive={process.is_alive()})")
    assert ready == "ready", ready

    yield worker

    worker.tasks.put(None)
    process.join(timeout=1.0)
    if process.is_alive():
        # Don't raise: a session-teardown error aborts pytest before the ERRORS report.
        process.kill()
        warnings.warn("async worker still busy at session end; killed")


@pytest.fixture
def threaded_shm_data_pingpong(request, _async_worker):
    yield from subfixture(request, _async_worker, _Pingpong)


@pytest.fixture
def threaded_shm_data_pingpong_gpu(request, _async_worker):
    yield from subfixture(request, _async_worker, _Pingpong, location=0)


@pytest.fixture
def threaded_shm_poster(request, _async_worker):
    yield from subfixture(request, _async_worker, _Looper)


@pytest.fixture
def threaded_shm_poster_recreator(request, _async_worker):
    yield from subfixture(request, _async_worker, _LooperWithForcefulRecreation)


@pytest.fixture
def threaded_shm_create_kill_recreate(request, _async_worker):
    yield from subfixture(request, _async_worker, _CreateKillRecreate)


def subfixture(
    request: pytest.FixtureRequest,
    worker: _AsyncWorker,
    task_cls: type[_SubprocessTask],
    location: int = -1,
) -> typ.Generator[tuple[SHM, SHM], None, None]:

    prefix, shape, dtype, delay = request.param

    shm_toprocess = SHM(prefix + "_toprocess", (shape, dtype), location=location)
    shm_fromprocess = SHM(prefix + "_fromprocess", (shape, dtype), location=location)

    task = task_cls(
        prefix + "_toprocess",
        prefix + "_fromprocess",
        shape,
        dtype,
        location,
        delay,
    )
    worker.tasks.put(task)

    shm_toprocess._checkGrabSemaphore()
    shm_fromprocess._checkGrabSemaphore()
    shm_toprocess.IMAGE.semflush(shm_toprocess.semID)
    shm_fromprocess.IMAGE.semflush(shm_fromprocess.semID)
    # Startup handshake (worker sets, parent waits and clears), then exit signal.
    if not worker.evt.wait(timeout=10.0):
        try:
            report = worker.done.get_nowait()
        except queue.Empty:
            report = "nothing reported (worker busy or stuck)"
        pytest.fail(f"worker never signaled ready: {report}")
    worker.evt.clear()
    yield shm_toprocess, shm_fromprocess

    worker.evt.set()
    error = worker.done.get(timeout=1.0)
    worker.evt.clear()
    assert error is None, error

    shm_toprocess.destroy()
    shm_fromprocess.destroy()
