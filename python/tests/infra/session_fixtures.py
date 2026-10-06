from __future__ import annotations
import pytest

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


def tp(call):
    try:
        call()
    except:
        pass


def func_session_cleanup(s: ComputeSession):
    tp(s.runstop)
    tp(s.confstop)
    if s.fps:
        tp(s.fps.tmux_stop)
        tp(s.fps.destroy)


def fixt_fpsinit_factory(session_cls: type[ComputeSession], pinfo: bool = True):
    @pytest.fixture
    def fixt_fpsinit():
        s = func_fpsinit(session_cls, pinfo)
        yield s
        func_session_cleanup(s)

    return fixt_fpsinit
