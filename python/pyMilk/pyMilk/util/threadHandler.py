from __future__ import annotations

import typing as typ

import time
from threading import Thread, Event

# THIS IS SHITE
# There is much better thread action code, see for instance the camstack code for
# a better guaranteed release on interpreter exit (thread.py)
# and base.py for the structure (which should become a mixin)
# and various occurences of threading things in swmain.


class MyThread(Thread):
    """Wrapper around `threading.Thread` that propagates exceptions."""

    def __init__(self, *args, event: Event, **kwargs):
        self.event = event
        super().__init__(*args, **kwargs)

    def run(self):
        try:
            super().run()
        except BaseException as e:
            self.exc = e
        finally:
            del self._target, self._args, self._kwargs

    def join(self, timeout=None):
        self.event.set()
        super().join(timeout)
        if self.exc:
            raise self.exc


class ThreadHandler:
    """
    Basic class template for performing looped operations going on forever,
    piloted from an interactive shell.

    self.threadPause and self.threadRun pause/restart the underlying self.thread

    self._t_runThread is the looped operation, which seeks the internal flags
    handling the start/stop/pause

    ----------
    self.threadCall is the portion executed when not paused

    That's what you should implement in subclasses using this template
    ----------
    """

    def __init__(self, sleep_ms: float = 10, daemonize: bool = True) -> None:

        self._t_pauseSleepMs = sleep_ms
        self._t_threadSuspend = True
        self._t_threadDie = False

        self.thread: MyThread | None = None

        self.re_raise_exceptions = True
        self.exception_to_raise: Exception | None = None

        self._t_initThread(daemonize)

    def _t_initThread(self, daemonize: bool) -> None:
        try:
            if self.thread is not None:
                self.thread.join()
        except:
            pass

        self.thread = MyThread(target=self._t_runThread, event=self._t_threadEvent)
        self.thread.daemon = daemonize
        self.thread.start()

    def threadPause(self) -> None:
        self._t_threadSuspend = True

    def threadRun(self) -> None:
        self._t_threadSuspend = False

    def __del__(self) -> None:
        self._t_threadSuspend = True
        self._t_threadDie = True
        if self.thread is not None:
            self.thread.join()
        print("ThreadHandler.__del__: Destruct OK")

    def _t_runThread(self) -> None:
        while True:
            if self._t_threadDie:
                return
            if self._t_threadSuspend:
                time.sleep(self._t_pauseSleepMs / 1000.0)
            else:
                # What if there's an exception here?
                # Should we pass it to the main thread? how?
                self.threadCall()

    def threadCall(self) -> None:
        raise NotImplementedError(
            "ThreadHandler.threadCall: This method must be overloaded."
        )
