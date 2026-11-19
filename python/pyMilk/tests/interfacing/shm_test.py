import pytest

import os

import numpy as np

from pyMilk.interfacing.shm import SHM, check_SHM_name
from pyMilk import errors


@pytest.mark.parametrize(
    "shape",
    [
        (3,),
        (2, 3),
        (2, 3, 4),
        (3, 4, 5),
        (5, 6, 7),
        (1,),
        (1, 1),
        (1, 1, 1),
        (1, 5),
        (5, 1),
        (1, 3, 5),
        (3, 1, 5),
        (3, 5, 1),
        (1, 1, 5),
        (1, 5, 1),
        (5, 1, 1),
    ],
)  # test with a lot of singleton dimensions as well, but no autosqueeze
def test_data_conservation(shape: tuple[int, ...]):

    for sym in range(8):
        for tri in range(4):
            data: np.ndarray = np.random.randn(*shape)  # type: ignore

            shm_write = SHM(
                "pyMilk_autotest",
                data,
                symcode=sym,
                triDim=tri,
                location=-1,
                shared=True,
            )
            shm_read = SHM(
                "pyMilk_autotest", symcode=sym, triDim=tri, autoSqueeze=False
            )

            dd = shm_read.get_data()
            assert (
                shm_write.shape == shm_read.shape
            ), f"{shm_write.shape}, {shm_read.shape}, {sym}, {tri}"
            assert (
                shm_write.shape_c == shm_read.shape_c
            ), f"{shm_write.shape_c}, {shm_read.shape_c}"
            assert np.all(np.abs(dd - data) < 1e-7), f"{shape}, {sym}, {tri}"

            data_backw: np.ndarray = np.random.randn(*shape)  # type: ignore

            shm_read.set_data(data_backw)
            dd2 = shm_write.get_data(check=False)
            assert np.all(np.abs(dd2 - data_backw) < 1e-7), f"{shape}, {sym}, {tri}"

            shm_read.close()
            shm_write.destroy()


@pytest.mark.parametrize(
    "shape",
    [
        (1,),
        (1, 1),
        (1, 1, 1),
        (1, 5),
        (5, 1),
        (1, 3, 5),
        (3, 1, 5),
        (3, 5, 1),
        (1, 1, 5),
        (1, 5, 1),
        (5, 1, 1),
    ],
)
def test_data_conservation_with_squeezing_dimensions(shape: tuple[int, ...]):

    # for sym in range(8):
    #    for tri in range(4): # sym / tri don't have the same meaning on create/read for squeezing SHMs...
    #                           and it's most confusing.
    sym, tri = 0, 0

    data: np.ndarray = np.random.randn(*shape)  # type: ignore

    shm_write = SHM(
        "pyMilk_autotest", data, symcode=sym, triDim=tri, location=-1, shared=True
    )
    shm_read = SHM("pyMilk_autotest", symcode=sym, triDim=tri)

    dd = shm_read.get_data()
    expected_shape = tuple((s for s in shape if s > 1))

    print(shape, expected_shape)

    assert (
        expected_shape == shm_read.shape
    ), f"{shm_write.shape}, {shm_read.shape}, {sym}, {tri}"
    assert expected_shape == dd.shape, f"{expected_shape}, {dd.shape}, {sym}, {tri}"
    assert (
        shm_write.shape_c == shm_read.shape_c
    ), f"{shm_write.shape_c}, {shm_read.shape_c}"
    assert np.all(np.abs(dd - data.squeeze()) < 1e-7), f"{shape}, {sym}, {tri}"

    data_backw: np.ndarray = np.random.randn(*expected_shape)  # type: ignore

    shm_read.set_data(data_backw)
    dd2 = shm_write.get_data(check=False)
    assert np.all(np.abs(dd2.squeeze() - data_backw) < 1e-7), f"{shape}, {sym}, {tri}"

    shm_read.close()
    shm_write.destroy()


@pytest.mark.parametrize(
    "dtype1",
    [
        np.int8,
        np.uint8,
        np.int16,
        np.uint16,
        np.int32,
        np.uint32,
        np.int64,
        np.uint64,
        np.float32,
        np.float64,
    ],
)
@pytest.mark.parametrize(
    "dtype2",
    [
        np.int8,
        np.uint8,
        np.int16,
        np.uint16,
        np.int32,
        np.uint32,
        np.int64,
        np.uint64,
        np.float32,
        np.float64,
    ],
)
def test_wrong_type(dtype1: np.typing.DTypeLike, dtype2: np.typing.DTypeLike):
    s = SHM("a", ((2, 3), dtype1))
    data = (np.random.randn(2, 3) * 100).astype(dtype2)
    if dtype1 == dtype2:
        s.set_data(data)  # nofail
    else:
        with pytest.raises(ValueError):
            s.set_data(data)  # Wrong type

    s.set_data(data, check_dt=True)


def test_wrong_shape():
    s = SHM("a", ((2, 3), np.float32))
    with pytest.raises(ValueError):
        s.set_data(np.zeros((2, 2), np.float32))  # Wrong shape

    with pytest.raises(ValueError):
        s.set_data(np.zeros((2, 3, 1), np.float32))  # Wrong number of axes


def test_as_context_manager():
    data = np.random.randn(50, 20)
    shm_write = SHM("pyMilk_autotest", data)

    with SHM("pyMilk_autotest") as shm_read:
        np.testing.assert_almost_equal(shm_read.get_data(), data)

    with pytest.raises(RuntimeError):
        shm_read.close()  # double close

    shm_write.destroy()


def test_keyword():

    data = np.random.randn(50, 20)
    shm_write = SHM("pyMilk_autotest", data, nbkw=5)
    kww = {
        "yo": ("lo", "un commentaire"),
        "toto": 17,
        "arthur": 3.1415,
        "some": ("stuff",),
    }
    shm_write.set_keywords(kww)

    shm_read = SHM("pyMilk_autotest")
    kwr = shm_read.get_keywords()
    assert kwr == {"yo": "lo", "toto": 17, "arthur": 3.1415, "some": "stuff"}

    kww = {"yo": 2.718, "roger": "trois"}
    shm_read.set_keywords(
        kww
    )  # Should erase all? the comments but not overwrite the entire keyword dict.
    assert shm_read.get_keywords() == shm_write.get_keywords()
    assert all(comm == "" for (val, comm) in shm_write.get_keywords(True).values())

    shm_write.update_keyword("roger", "12")
    shm_write.update_keyword("yo", 13, "comment")
    assert shm_read.get_keywords(True)["roger"] == ("12", "")
    assert shm_read.get_keywords(True)["yo"] == (13, "comment")

    with pytest.raises(ValueError):
        shm_write.update_keyword("does_not_exist_kw", 52)

    shm_write.reset_keywords({"a": 0, "new": 1.234, "dict": "is set"})

    assert shm_read.get_keywords() == {"a": 0, "new": 1.234, "dict": "is set"}

    # get a single kw
    assert shm_read.get_keyword("dict") == "is set"

    shm_write.destroy()


def test_keyword_getitem_setitem():
    data = np.random.randn(50, 20)
    shm_write = SHM("pyMilk_autotest", data, nbkw=6)
    kww = {
        "yo": ("lo", "un commentaire"),
        "toto": 17,
        "arthur": 3.1415,
        "some": ("stuff",),
    }
    shm_write.set_keywords(kww)

    shm_read = SHM("pyMilk_autotest")
    assert shm_read["yo"] == "lo"
    assert shm_read["toto"] == 17
    assert shm_read["arthur"] == 3.1415
    assert shm_read["some"] == "stuff"

    # Creation
    shm_read["new_kw"] = "new_val"
    assert shm_write["new_kw"] == "new_val"

    # Re-set of existing
    shm_write["arthur"] = 2.718
    assert shm_read["arthur"] == 2.718
    shm_read["arthur"] = 3.14
    assert shm_write["arthur"] == 3.14

    # Illegal read
    with pytest.raises(KeyError):
        x = shm_read["80085"]

    # Set with different type !
    shm_read["arthur"] = "string"
    assert shm_write["arthur"] == "string"


def test_fits():
    for data in [
        np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
        np.random.randn(30, 30),
        np.random.randn(23),
        np.random.randn(23, 12, 16),
    ]:

        shm_write = SHM("pyMilk_autotest", data, symcode=3)
        shm_write.save_as_fits("pyMilk_test_fits.fits")

        shm_read = SHM("pyMilk_autotest", symcode=7)
        shm_read.save_as_fits("pyMilk_test_fits2.fits")

        from pyMilk.interfacing import fits_lib

        data_f = fits_lib.multi_read("pyMilk_test_fits.fits", symcode=3)
        data_f2 = fits_lib.multi_read("pyMilk_test_fits2.fits", symcode=3)

        fits_lib.multi_write("pyMilk_test_fits3.fits", shm_read.get_data(), symcode=7)
        data_f3 = fits_lib.multi_read("pyMilk_test_fits3.fits", symcode=3)

        assert isinstance(data_f, np.ndarray)
        assert isinstance(data_f2, np.ndarray)
        assert isinstance(data_f3, np.ndarray)

        shm_write.destroy()
        os.remove("pyMilk_test_fits.fits")
        os.remove("pyMilk_test_fits2.fits")
        os.remove("pyMilk_test_fits3.fits")

        assert data.shape == data_f.shape
        assert data.shape == data_f2.shape
        assert data.shape == data_f3.shape

        assert np.all(np.abs(data - data_f) < 1e-7)
        assert np.all(np.abs(data - data_f2) < 1e-7)
        assert np.all(np.abs(data - data_f3) < 1e-7)


def test_link_no_exist():
    with pytest.raises(FileNotFoundError):
        s = SHM("test_noexist")


def test_dual_destroy():
    s = SHM("test_doublefree", np.random.randn(5, 10))

    s.destroy()  # pass
    s.destroy()  # pass as well, IMAGE.destroy() would throw a RuntimeError but it's inhibited since IMAGE.md.used == 0


def test_destroy_zombie_shm():
    s1 = SHM("test_destroy", np.random.randn(5, 10))
    s2 = SHM(
        "test_destroy", np.random.randn(5, 10).astype(np.float32)
    )  # should cause overwrite

    assert s1.FILEPATH == s2.FILEPATH
    assert os.path.isfile(s2.FILEPATH)
    s1.destroy()
    assert os.path.isfile(s2.FILEPATH)
    s2.destroy()
    assert not os.path.isfile(s2.FILEPATH)


@pytest.mark.parametrize(
    "data",
    [
        "abcd",
        ("abcd", np.int32),
        ((1, 1, 1), "yolo"),
        ((1, 2, 3, 4), np.float64),
        ((-1, 2, 3), np.float64),
        ((1.123, 2, 3), np.float64),
    ],
)
def test_create_bad_data(data):
    with pytest.raises((ValueError, TypeError)):
        s = SHM("a_name", data)


@pytest.mark.parametrize(
    "data",
    [
        ((1,), np.int8),
        ((1, 1), np.int16),
        ((1, 1, 1), np.int32),
        ((1, 2, 3), np.int64),
        ((2, 1, 3), np.uint8),
        ((2, 3, 1), np.uint16),
        ((10,), np.uint32),
        ((10, 20), np.uint64),
        ((10, 20, 30), np.float32),
    ],
)
def test_create_good_data_tuple(data):
    shape, dtype = data
    s = SHM("a_name", data, symcode=0, autoSqueeze=False)

    got_data = s.get_data()
    assert s.shape == s.shape_c
    assert s.shape == shape

    assert isinstance(got_data, np.ndarray)
    assert got_data.dtype == s.nptype
    assert got_data.dtype == dtype

    s.destroy()


def test_failing_autorelink():

    a = SHM("x_autorelink", ((2, 3), np.int32))

    b = SHM("x_autorelink", ((5, 3, 4), np.int32))
    with pytest.raises(errors.AutoRelinkSizeError):
        data = a.get_data()

    b = SHM("x_autorelink", ((2, 3), np.float32))
    with pytest.raises(errors.AutoRelinkTypeError):
        data = a.get_data()

    b = SHM("x_autorelink", ((2, 3), np.int32))
    a.get_data()


@pytest.mark.parametrize(
    "dtype",
    [
        np.int8,
        np.uint8,
        np.int16,
        np.uint16,
        np.int32,
        np.uint32,
        np.int64,
        np.uint64,
        np.float32,
        np.float64,
    ],
)
def test_copy_false(dtype):
    a = SHM("x_copyfalse", ((20, 30), dtype))

    b = SHM("x_copyfalse")
    arr = b.get_data(copy=False)

    np.testing.assert_array_equal(arr, np.zeros((20, 30), np.int32))
    arr[1, 2] = 2

    arr2 = a.get_data(copy=False)
    np.testing.assert_array_equal(arr, arr2)


def test_check_SHM_name():
    MILK_SHM_DIR = os.environ["MILK_SHM_DIR"]

    assert check_SHM_name("abcd.im.shm") == "abcd"
    assert check_SHM_name("abcd") == "abcd"
    with pytest.raises(ValueError):
        check_SHM_name("x/abcd.im.shm")
    with pytest.raises(ValueError):
        check_SHM_name("x/y/abcd.im.shm")
    with pytest.raises(ValueError):
        check_SHM_name(MILK_SHM_DIR + "/y/abcd.im.shm")
    assert check_SHM_name(MILK_SHM_DIR + "/abcd.im.shm") == "abcd"
    assert check_SHM_name(MILK_SHM_DIR + "/abcd") == "abcd"


def test_kw_expt():
    s = SHM("something", np.random.randn(2))
    assert s.get_expt() == 0.0
    s.reset_keywords({"tint": 3.14})
    assert s.get_expt() == 3.14
    s.reset_keywords({"tint": 3.14, "EXPTIME": 2.718})
    assert s.get_expt() == 2.718
    s.reset_keywords({"tint": 3.14, "EXPTIME": "rofl"})
    with pytest.raises(AssertionError):
        s.get_expt()
    s.destroy()


def test_kw_frate():
    s = SHM("something", np.random.randn(2))
    assert s.get_fps() == 0.0
    s.reset_keywords({"fps": 3.14})
    assert s.get_fps() == 3.14
    s.reset_keywords({"fps": 3.14, "FRATE": 2.718})
    assert s.get_fps() == 2.718
    s.reset_keywords({"fps": 3.14, "FRATE": "rofl"})
    with pytest.raises(AssertionError):
        s.get_fps()
    s.destroy()


def test_kw_ndr():
    s = SHM("something", np.random.randn(2))
    assert s.get_ndr() == 1
    s.reset_keywords({"NDR": 3})
    assert s.get_ndr() == 3
    s.reset_keywords({"NDR": 3, "DET-NSMP": 12})
    assert s.get_ndr() == 12
    s.reset_keywords({"NDR": 3, "DET-NSMP": "rofl"})
    with pytest.raises(AssertionError):
        s.get_ndr()
    s.destroy()


def test_get_crop():
    """
    This test probs wrong -- what should the specs for get_crop be?
    """
    s = SHM("something", np.random.randn(2, 3))
    assert s.get_crop() == (0, 2, 0, 3)
    s.reset_keywords({"x0": 5, "y0": 10, "x1": 22, "y1": 37})
    assert s.get_crop() == (5, 26, 10, 46)
    s.reset_keywords({"CROP_OR1": 5, "CROP_EN1": 10, "CROP_OR2": 22, "CROP_EN2": 37})
    assert s.get_crop() == (5, 10, 22, 37)
    s.reset_keywords({"PRD-MIN1": 5, "PRD-RNG1": 10, "PRD-MIN2": 22, "PRD-RNG2": 37})
    assert s.get_crop() == (5, 14, 22, 58)

    s.destroy()


def test_shm_repost():
    s = SHM("something", np.random.randn(2, 3))
    s2 = SHM("something")

    import time, datetime

    t1 = time.time()
    s.set_data(np.random.randn(*s.shape))
    t2 = time.time()

    assert t1 < s2.md.writetime < t2
    assert s2.md.acqtime == s2.md.writetime

    t1 = time.time()
    s.repost()
    t2 = time.time()

    assert t1 < s2.md.writetime < t2
    assert s2.md.acqtime == s2.md.writetime

    s.repost(t1)
    assert s2.md.acqtime == t1
    s.repost(datetime.datetime.fromtimestamp(t2))
    assert abs(s2.md.acqtime - t2) < 0.5e-6

    s2.repost(t1)
    assert s.md.acqtime == t1
    s2.repost(datetime.datetime.fromtimestamp(t2))
    assert abs(s.md.acqtime - t2) < 0.5e-6
