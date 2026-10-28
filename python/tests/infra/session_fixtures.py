from __future__ import annotations
import pytest

import os
from milk.session import ComputeSession


def func_fpsinit(
    session_cls: type[ComputeSession], pinfo: bool = True
) -> ComputeSession:
    # As fixture
    session = session_cls()
    assert session.fps is None

    session.fpsinit(pinfo)
    assert session.fps

    if pinfo:
        assert session.has_procinfo is True
        assert "procinfo.enabled" in session.fps
        assert session.fps["procinfo.enabled"]
    else:
        assert session.has_procinfo is False
        assert not "procinfo.enabled" in session.fps

    # Configure
    # default imin, imout, delaysec = 0.1 ms, naive = OFF, timebuffsize = 1000

    return session


def tp(callable):
    try:
        callable()
    except:
        pass


def func_session_cleanup(session: ComputeSession):
    # runstop and confstop return once the processes are gone
    tp(session.runstop)
    tp(session.confstop)
    if session.fps:
        md = session.fps.fps.md()
        for pid in (md.runpid, md.confpid):
            if pid > 0:  # an exited run clears its pid
                with pytest.raises(ProcessLookupError):
                    os.kill(pid, 0)

        tp(session.fps.destroy)


def fixt_fpsinit_factory(session_cls: type[ComputeSession], pinfo: bool = True):
    @pytest.fixture
    def fixt_fpsinit():
        s = func_fpsinit(session_cls, pinfo)
        yield s
        func_session_cleanup(s)

    return fixt_fpsinit
