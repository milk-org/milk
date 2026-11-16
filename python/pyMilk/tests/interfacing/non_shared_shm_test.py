from __future__ import annotations

import pytest
import numpy as np

from pyMilk.interfacing.shm import SHM


@pytest.mark.parametrize("shape", [(3,), (2, 3), (2, 3, 4), (3, 4, 5), (5, 6, 7)])
@pytest.mark.parametrize("location", [-1, 0])
def test_data_conservation(shape: tuple[int, ...], location: int):

    for sym in range(8):
        for tri in range(4):
            data: np.ndarray = np.random.randn(*shape)  # type: ignore

            shm_write = SHM(
                "pyMilk_autotest",
                data,
                symcode=sym,
                triDim=tri,
                location=location,
                shared=False,
            )

            shm_write.set_data(data)
            assert np.all(shm_write.get_data() == data)

            data_backw: np.ndarray = np.random.randn(*shape)  # type: ignore
            shm_write.set_data(data_backw)
            assert np.all(shm_write.get_data() == data_backw)

            shm_write.destroy()


@pytest.mark.parametrize("location1", [-1, 0])
@pytest.mark.parametrize("location2", [-1, 0])
def test_shared_nonshared_overlap(location1: int, location2: int):
    from pyMilk.ImageStreamIOWrap import IMAGESTREAMIO_HAVE_CUDA

    if (location1 >= 0 or location2 >= 0) and not IMAGESTREAMIO_HAVE_CUDA:
        pytest.skip("No CUDA -- skipping this test")

    data_shared: np.ndarray = np.random.randn(10, 20).astype(np.float32)
    data_private: np.ndarray = np.random.randn(10, 20).astype(np.float32)
    shm_shared_1 = SHM("shared_test", data_shared, location=location1)  # Create
    shm_private = SHM(
        "shared_test", data_private, location=location2, shared=False
    )  # Create
    shm_shared_2 = SHM("shared_test")  # Open

    assert np.all(shm_shared_2.get_data() == data_shared)
    assert np.all(shm_private.get_data() == data_private)
    shm_private.destroy()

    shm_private = SHM(
        "shared_test", data_private, location=location2, shared=False
    )  # Create

    shm_shared_3 = SHM("shared_test")  # Open
    assert np.all(shm_shared_3.get_data() == data_shared)

    shm_private.destroy()

    shm_shared_3.set_data(data_shared - 1)
    assert np.all(shm_shared_1.get_data() == data_shared - 1)
    assert np.all(shm_shared_2.get_data() == data_shared - 1)

    shm_shared_1.close()
    shm_shared_2.close()
    shm_shared_3.destroy()
