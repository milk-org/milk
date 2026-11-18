import pytest

from ..conftestaux.milk import ctfixt_change_MILK_SHM_DIR
from pyMilk.interfacing.fps import FPS


def test_make_fps(ctfixt_change_MILK_SHM_DIR):
    fps = FPS.create("my_fps_name")

    input("fps is available")

    fps.destroy()

    return
