import pytest

import os
import time

import numpy as np

from pyMilk.interfacing.shm import SHM, check_SHM_name
from pyMilk import errors

from ..conftestaux.async_shm_fixtures import serve_external_shm
from ..conftestaux.gpu_configure import SINGLE_GPU_TESTING, MULTI_GPU_TESTING

if SINGLE_GPU_TESTING is None:
    pytest.skip("Skipping GPU tests.", allow_module_level=True)

# TODO a pytest primitive HAVE_ONE_GPU, HAVE_TWO_GPU (for p2p)

import cupy as cp


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
    s = SHM("a_name", data, symcode=0, autoSqueeze=False, location=0)

    got_data = s.get_data()
    assert s.shape == s.shape_c
    assert s.shape == shape

    assert isinstance(got_data, np.ndarray)
    assert got_data.dtype == s.nptype
    assert got_data.dtype == dtype

    s.destroy()


def test_double_open():
    data = ((10, 20), np.uint64)
    s1 = SHM("a_name", data, symcode=0, autoSqueeze=False, location=0)
    s2 = SHM("a_name")  # pass

    s1.get_data()
    s2.get_data()

    s2.close()
    s1.destroy()


def test_double_open_with_data_sanity():
    data = np.random.randn(10, 20)
    s1 = SHM("a_name", data, autoSqueeze=False, location=0)
    s2 = SHM("a_name")  # pass

    np.testing.assert_almost_equal(s1.get_data(), data)
    np.testing.assert_almost_equal(s2.get_data(), data)
    data = np.random.randn(10, 20)
    s2.set_data(data)
    np.testing.assert_almost_equal(s1.get_data(), data)
    np.testing.assert_almost_equal(s2.get_data(), data)

    s2.close()
    s1.destroy()


# def serve_external_shm(name: str, shape: tuple[int, ...], nptype: np.typing.DTypeLike, location: int):


@pytest.mark.parametrize(
    "serve_external_shm", [("gpu_test", (100, 200), np.float64, 0)], indirect=True
)
def test_external_process_serves_shm(serve_external_shm: SHM):
    # Nothing to test, the fixture already opened the local SHM and will close it.
    assert np.all(serve_external_shm.get_data() == 0.0)


@pytest.mark.parametrize(
    "serve_external_shm", [("gpu_test", (100, 200), np.float64, 0)], indirect=True
)
@pytest.mark.parametrize("overwrite_location", [-1, 0])
def test_external_process_serves_shm_and_make_zombie(
    serve_external_shm: SHM, overwrite_location: int
):
    # Nothing to test, the fixture already opened the local SHM and will close it.
    assert np.all(serve_external_shm.get_data() == 0.0)
    z_data = serve_external_shm.get_data()

    new_handle = SHM(serve_external_shm.FNAME)

    overwrite = SHM(serve_external_shm.FNAME, z_data - 1, location=overwrite_location)

    # We purposefully _disable_ autorelink_if_need
    # Otherwise we'd just automatically relink to overwritecpu.
    # But here we're testing that the zombie is staying alive

    assert np.all(serve_external_shm.get_data(autorelink_if_need=False) == 0.0)
    assert np.all(overwrite.get_data() == -1.0)

    new_handle.set_data(z_data + 1, autorelink_if_need=False)
    assert np.all(serve_external_shm.get_data(autorelink_if_need=False) == +1.0)
    assert np.all(overwrite.get_data() == -1.0)


def test_no_gpu_memcpy():
    from ..conftestaux.gpu_transfer_monitor import assert_no_gpu_host_transfers

    with assert_no_gpu_host_transfers():
        pass


def test_yes_gpu_memcpy():
    s = SHM("yolo", ((5, 10), np.int64), location=0)
    from ..conftestaux.gpu_transfer_monitor import count_gpu_host_transfers

    with count_gpu_host_transfers() as counts:
        s.get_data()

    assert counts.dtoh > 0
    assert counts.htod == 0

    with count_gpu_host_transfers() as counts:
        s.set_data((np.random.randn(5, 10) * 100).astype(np.int64))
    assert counts.dtoh == 0
    assert counts.htod > 0


def test_cp_array_to_cpu_shm():
    x_np = np.random.randn(123, 45).astype(np.float32)
    x_cp = cp.array(-x_np)

    s_cpu = SHM("x", x_np)

    s_cpu.set_data(x_cp)

    assert np.all(s_cpu.get_data() == -x_np)

    s_cpu.destroy()


def test_gpu_shm_from_np_cp():
    x_np = np.random.randn(123, 45).astype(np.float32)
    x_cp = cp.array(-x_np)

    s_gpu = SHM("x", x_cp, location=0)
    assert type(s_gpu.get_data(copy=False)) == cp.ndarray
    assert np.all(s_gpu.get_data(copy=True) == -x_np)
    # cp.all is broken when running off cupy-cudaVERx + system CUDA
    # assert cp.all(s_gpu.get_data(copy=False) == +x_cp)
    gpu_tmp = s_gpu.get_data(copy=False) - x_cp
    assert np.all(gpu_tmp.get() == 0.0)
    s_gpu.destroy()

    s_gpu = SHM("x", x_np, location=0)
    assert np.all(s_gpu.get_data(copy=True) == x_np)
    # cp.all is broken when running off cupy-cudaVERx + system CUDA
    # assert cp.all(s_gpu.get_data(copy=False) == -x_cp)
    gpu_tmp = s_gpu.get_data(copy=False) + x_cp
    assert np.all(gpu_tmp.get() == 0.0)

    s_gpu.destroy()


def test_gpu_countcopy():
    initializer_data = np.random.randn(123, 45, 67).astype(np.float32)
    initializer_data.flat[0] = 0  # counter
    s = SHM("test", initializer_data, location=0)
    assert s.location == 0

    from ..conftestaux.gpu_transfer_monitor import count_gpu_host_transfers

    with count_gpu_host_transfers() as count:
        t1 = time.time()
        for _ in range(1_000):
            cpu_data = s.get_data(copy=True)
            cpu_data.flat[0] += 1
            s.set_data(cpu_data)
        print(f"Time in loop for test_gpu_zerocopy: {time.time() - t1:.6f} s")
    assert count.dtoh == 1000
    assert count.htod == 1000
    assert s.get_data().flat[0] == 1000

    s.destroy()


def test_gpu_zerocopy():
    # Big arrays:   PCI memcpy dominates, zerocopy is faster
    # Small arrays: Driver calls to GPU dominate, zerocopy (on-device-copy but no PCI transfer) is slower.
    initializer_data = np.random.randn(123, 45, 67).astype(np.float32)
    initializer_data.flat[0] = 0  # counter
    s = SHM("test", initializer_data, location=0)
    assert s.location == 0

    from ..conftestaux.gpu_transfer_monitor import count_gpu_host_transfers

    s.set_data(initializer_data)
    with count_gpu_host_transfers() as count:
        t1 = time.time()
        for _ in range(1_000):
            gpu_data = s.get_data(copy=False)
            gpu_data.flat[
                0
            ] += 1.0  # This kernel launch costs time, but not what we're testing here.
            s.set_data(gpu_data)
        print(f"Time in loop for test_gpu_zerocopy: {time.time() - t1:.6f} s")
    assert count.dtoh == 0
    assert count.htod == 0
    assert s.get_data().flat[0] == 1000

    s.destroy()
