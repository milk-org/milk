import pytest

import pyMilk.interfacing.shm_functions as sf
from pyMilk.interfacing.shm import SHM
import numpy as np


@pytest.mark.parametrize(
    ["attempt_reuse", "delete_existing"],
    [(False, False), (False, True), (True, False), (True, True)],
)
def test_creashmim(attempt_reuse: bool, delete_existing: bool):
    s = sf.creashmim(
        "shms",
        (2, 3, 4),
        np.uint64,
        attempt_reuse=attempt_reuse,
        delete_existing=delete_existing,
    )

    ss = SHM(s.FNAME)
    assert ss.get_data().shape == (2, 3, 4)
    assert ss.get_data().dtype == np.uint64

    dat = (np.random.rand(*ss.shape) * 1e4).astype(np.uint64)

    ss.set_data(dat)

    assert np.all(s.get_data() == dat)

    ss.close()
    s.destroy()


def test_reuse_delete_0_0():
    s1 = sf.creashmim("shm", (2, 3, 4), np.uint64)
    s2 = sf.creashmim(
        "shm", (2, 3), np.float32, attempt_reuse=False, delete_existing=False
    )

    assert s2.get_data().shape == (2, 3)
    assert s2.get_data().dtype == np.float32

    s2.close()
    s1.destroy()  # no doublefree because stale pointer !!


def test_reuse_delete_0_1():
    s1 = sf.creashmim("shm", (2, 3, 4), np.uint64)
    s2 = sf.creashmim(
        "shm", (2, 3), np.float32, attempt_reuse=False, delete_existing=True
    )

    s2.close()
    s1.destroy()


def test_reuse_delete_1_0():
    s1 = sf.creashmim("shm", (2, 3, 4), np.uint64)
    s2 = sf.creashmim(
        "shm", (2, 3), np.float32, attempt_reuse=True, delete_existing=False
    )

    s2.close()
    s1.destroy()


def test_reuse_delete_1_1():
    s1 = sf.creashmim("shm", (2, 3, 4), np.uint64)
    s2 = sf.creashmim(
        "shm", (2, 3), np.float32, attempt_reuse=True, delete_existing=True
    )

    s2.close()
    s1.destroy()


def test_successful_reuse_yesno_zero():
    s1 = sf.creashmim("shm", (2, 3, 4), np.uint64)

    dat = (np.random.rand(*s1.shape) * 1e4).astype(np.uint64)
    s1.set_data(dat)
    s2 = sf.creashmim("shm", (2, 3, 4), np.uint64, attempt_reuse=True, do_zero=False)

    assert np.all(s2.get_data() == dat)

    s3 = sf.creashmim("shm", (2, 3, 4), np.uint64, attempt_reuse=True, do_zero=True)

    assert np.all(s1.get_data() == 0)
    assert np.all(s2.get_data() == 0)
    assert np.all(s3.get_data() == 0)

    s1.destroy()


def test_not_enough_keywords():
    s1 = sf.creashmim("shm", (2, 3, 4), np.uint64, attempt_reuse=True, nb_kw=50)
    s2 = sf.creashmim("shm", (2, 3, 4), np.uint64, attempt_reuse=True, nb_kw=100)

    assert s2.IMAGE.md.inode != s1.IMAGE.md.inode

    s2.destroy()
