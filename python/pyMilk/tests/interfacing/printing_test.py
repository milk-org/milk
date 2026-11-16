import pytest

import numpy as np

from pyMilk.interfacing import shm
from pyMilk.interfacing import fps
from pyMilk.interfacing.shm import SHM
from pyMilk.interfacing.fps import FPS

"""
These printing tests are not very useful in their testability, but can be inspected quickly
by running this file with pytest -s
"""


@pytest.fixture
def shm_fixt():
    s = SHM("x", np.random.randn(10, 15))
    yield s
    s.destroy()


@pytest.fixture
def fps_fixt():
    f = FPS.create("some_fps")
    yield f
    f.destroy()


def test_print_shm(shm_fixt: SHM):
    shm_fixt.print_meta_data()


def test_str_shm(shm_fixt: SHM):
    print(shm_fixt)


def test_repr_shm(shm_fixt: SHM):
    print(repr(shm_fixt))


def test_print_fps(fps_fixt: FPS):
    print(fps_fixt)
    print(str(fps_fixt))
    print(repr(fps_fixt))


def test_print_shm_keywords(shm_fixt: SHM):
    shm_fixt["int_kw"] = 1
    shm_fixt["flt_kw"] = 1.234
    shm_fixt["str_kw"] = "str_kw"

    kws_fetch: dict[str, shm.Image_kw] = shm_fixt.IMAGE.get_kws()
    assert isinstance(kws_fetch["int_kw"], shm.Image_kw)

    # repr is val_<space>_comment, with comment empty here...
    assert str(kws_fetch["int_kw"]) == "1 "
    assert repr(kws_fetch["int_kw"]) == "1 "
    assert str(kws_fetch["flt_kw"]) == "1.234 "
    assert repr(kws_fetch["flt_kw"]) == "1.234 "
    assert str(kws_fetch["str_kw"]) == "str_kw "
    assert repr(kws_fetch["str_kw"]) == "str_kw "
