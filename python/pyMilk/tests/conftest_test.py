"""
This file is basic sanity to make sure the implemented fixtures
in
    - conftest.py
    - conftestaux/*.py
actually do their job.
"""

import os
import pytest

from pyMilk.interfacing.shm import IMAGESTREAMIO_HAVE_CUDA


def test_milk_shm_dir_fixture():
    # Fixture from milk.py -- which is scope='session', autouse=True
    MILK_SHM_DIR_SPOOF = "/tmp/milk_shm_dir_pytest"

    assert os.path.isdir(MILK_SHM_DIR_SPOOF)
    assert os.environ["MILK_SHM_DIR"] == MILK_SHM_DIR_SPOOF


def test_cwd_to_temp_fixture():
    cwd = os.getcwd()

    assert (
        cwd.startswith("/tmp") or "/tmp/" in cwd
    )  # nox will nest a ..../tmp/... under the nox session dir.
    # last dir made by the fixture...
    # if other fixtures change cwd further, they should also revert and also not be autouse.
    assert cwd.endswith("pytest_dont_use_for_anything0")


def _subprocess_function():
    import numpy as np
    from pyMilk.interfacing.shm import SHM

    s = SHM("gpu", ((100, 100), np.float32), location=0)
    for _ in range(100):
        s.set_data(np.zeros((100, 100), np.float32))
    s.destroy()


def test_counts_in_subprocess():
    if not IMAGESTREAMIO_HAVE_CUDA:
        pytest.skip("No CUDA -- skipping this test")

    import multiprocessing
    from .conftestaux.gpu_transfer_monitor import count_subprocess_transfers

    # Use 'spawn' so the child gets a clean CUDA context.
    ctx = multiprocessing.get_context("spawn")

    with count_subprocess_transfers() as (counts, spy):
        p = ctx.Process(target=spy(_subprocess_function))
        p.start()
        p.join()
        assert p.exitcode == 0, f"subprocess exited with code {p.exitcode}"

    # _subprocess_function: 1 create (H→D init write) + 100 set_data H→D
    assert counts.dtoh == 0
    assert counts.htod == 101


def test_counts_in_popen():
    if not IMAGESTREAMIO_HAVE_CUDA:
        pytest.skip("No CUDA -- skipping this test")

    import subprocess, shlex
    from .conftestaux.gpu_transfer_monitor import count_popen_transfers

    with count_popen_transfers() as (counts, extra_env):
        proc = subprocess.Popen(
            shlex.split(
                "python -c \"from pyMilk.interfacing.shm import SHM; import numpy as np; s = SHM('gpu', ((10,20), np.float32),location=0); s.destroy()\""
            ),
            env={**os.environ, **extra_env},
        )
        proc.wait()

    assert proc.returncode == 0
    assert counts.dtoh == 0
    assert counts.htod == 1


def test_gpu_configure():
    print(f'\n{"--- GPU CONF SCAN ---":^40}')
    from .conftestaux import gpu_configure as g

    print(f"{'NVIDIA DRIVER':<20}{str(g.NVIDIA_DRIVER_FOUND):>20}")
    print(f"{'GPU LIST':<20}{str(g.GPULIST):>20}")
    print(f"{'GPU TESTING':<20}{str(g.SINGLE_GPU_TESTING):>20}")
    print(f"{'GPU P2P TESTING':<20}{str(g.MULTI_GPU_TESTING):>20}")
