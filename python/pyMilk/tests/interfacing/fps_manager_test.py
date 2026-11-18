import pytest

from pyMilk.interfacing.fps import *


class AutoFPS(SmartAttributesFPSAutoMetadata):
    gain: float
    go: bool
    a_comment: str


@pytest.fixture
def yield_4_fps():
    a = AutoFPS.create("a")
    b = AutoFPS.create("b")
    c = AutoFPS.create("c")
    d = AutoFPS.create("d")

    yield a, b, c, d

    a.destroy()
    b.destroy()
    c.destroy()
    d.destroy()


def test_fps_manager(yield_4_fps: tuple[AutoFPS, ...]):
    a, b, c, d = yield_4_fps

    mgr = FPSManager()

    assert not a.conf_start()
    assert not a.run_start()

    assert a.run_stop()
    assert a.conf_stop()

    assert not a.conf_start(timeoutsync=0.1)
    assert not a.run_start(timeoutsync=0.1)

    assert a.run_stop(timeoutsync=0.1)
    assert a.conf_stop(timeoutsync=0.1)
