"""
files in tests/mains/ are ignored by pytest.

They can be run with pytest -k <test_name> <test_file>
"""

from time import perf_counter


class catchtime:

    def __init__(self, message: str):
        self.message = message

    def __enter__(self):
        self.start = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.time = perf_counter() - self.start
        self.readout = f"TIME {self.message}: {self.time:.3f} seconds"
        print(self.readout)


def test_bench_do_nothing(): ...


def test_shm_ping_pong():
    import numpy as np
    from pyMilk.interfacing.shm import SHM

    a = SHM("a", np.random.randn(300, 200).astype(np.float32))

    b = SHM("a")
    with catchtime("loop 100000 data pingpong, copy.") as _:
        for _ in range(100000):
            b.set_data(a.get_data(True, checkSemAndFlush=False, copy=True))
            a.set_data(b.get_data(True, checkSemAndFlush=False, copy=True))
    with catchtime("loop 100000 data pingpong, no copy.") as _:
        for _ in range(100000):
            b.set_data(a.get_data(True, checkSemAndFlush=False, copy=False))
            a.set_data(b.get_data(True, checkSemAndFlush=False, copy=False))


def test_stress_cpp_wrapper():
    import numpy as np
    from pyMilk.interfacing.shm import SHM

    a = SHM("a", np.random.randn(300, 200).astype(np.float32))
    with catchtime("loop 1000000 keyword read.") as _:
        for _ in range(1000000):
            a.get_keywords()
