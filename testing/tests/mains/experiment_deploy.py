from __future__ import annotations

import pytest

import os
import time
import numpy as np

import subprocess, shlex

from pyMilk.interfacing.shm import SHM
from pyMilk.interfacing.fps import FPS


def exec(cmd: str):
    return subprocess.run(shlex.split(cmd))


def fps_inject_env_into_tmux(fps: FPS) -> None:
    """Inject MILK_SHM_DIR and MILK_PROC_DIR into all tmux windows of an FPS.

    Sends export commands to the ctrl, conf, and run windows of the
    tmux session named after the FPS, using the values currently set
    in os.environ.
    """
    shm_dir = os.environ.get("MILK_SHM_DIR", "")
    proc_dir = os.environ.get("MILK_PROC_DIR", "")
    session = fps.name

    for window in ("ctrl", "conf", "run"):
        target = f"{session}:{window}"
        if shm_dir:
            subprocess.run(
                shlex.split(
                    f"tmux send-keys -t {target} export MILK_SHM_DIR={shm_dir} C-m"
                )
            )
        if proc_dir:
            subprocess.run(
                shlex.split(
                    f"tmux send-keys -t {target} export MILK_PROC_DIR={proc_dir} C-m"
                )
            )


def fps_sanitize_start(fpss: list[FPS]) -> None:
    import libtmux

    TSRV = libtmux.server.Server()

    tmux_needs_start = []
    for kk, fps in enumerate(fpss):
        try:
            for win_name in ("ctrl", "conf", "run"):
                p = TSRV.windows.get(
                    session_name=fps.name, window_name=win_name
                ).active_pane
                p.send_keys("C-c", enter=False, suppress_history=False)
                p.send_keys("C-c", enter=False, suppress_history=False)
                p.send_keys("C-z", enter=False, suppress_history=False)
                p.send_keys("kill %")
            tmux_needs_start += [False]
        except:
            fps.tmux_stop()
            tmux_needs_start += [True]

    time.sleep(1)
    for fps, tstart in zip(fpss, tmux_needs_start):
        if tstart:
            fps.tmux_start()
    time.sleep(1)
    for fps in fpss:
        fps_inject_env_into_tmux(fps)


def configure_procinfo(fps: FPS) -> None:
    fps["procinfo.enabled"] = True
    fps["procinfo.NBthread"] = 1
    fps["procinfo.triggermode"] = 3
    fps["procinfo.triggersname"] = fps["in_name"]
    fps["procinfo.semindexrequested"] = 0
    fps["procinfo.loopcntMax"] = -1
    fps["procinfo.RTprio"] = 49


@pytest.fixture
def fixt_streamdelay_pingpong():

    exec("milk-fpsexec-mem-streamdelay -procinfo -tmux a:fpsinit")
    exec("milk-fpsexec-mem-streamdelay -procinfo -tmux b:fpsinit")

    fps_a = FPS("a")
    fps_b = FPS("b")
    fps_sanitize_start([fps_a, fps_b])

    fps_a["in_name"] = "ping"
    fps_a["out_name"] = "pong"
    fps_b["in_name"] = "pong"
    fps_b["out_name"] = "ping"

    configure_procinfo(fps_a)
    configure_procinfo(fps_b)

    fps_a["delaysec"] = 0.0001
    fps_b["delaysec"] = 0.0001
    fps_a["naive_mode"] = True
    fps_b["naive_mode"] = True

    fps_a["procinfo.cset"] = "t5.slice"
    fps_b["procinfo.cset"] = "t6.slice"
    fps_a["procinfo.taskset"] = "5"
    fps_b["procinfo.taskset"] = "6"

    fps_a.conf_start()
    fps_b.conf_start()
    while not fps_a.conf_isrunning() and not fps_b.conf_isrunning():
        time.sleep(0.001)

    ping = SHM("ping", np.zeros((1, 3000), np.float32))
    pong = SHM("pong", np.zeros((1, 3000), np.float32))

    time.sleep(1)

    fps_a.run_start()
    fps_b.run_start()
    while not fps_a.run_isrunning() and not fps_b.run_isrunning():
        time.sleep(0.001)

    s1 = SHM("ping")
    s2 = SHM("pong")
    for _ in range(100):
        s1.IMAGE.semflush(-1)
        s2.IMAGE.semflush(-1)

    s1.repost()

    yield fps_a, fps_b

    fps_a.run_stop()
    fps_b.run_stop()

    fps_a.conf_stop()
    fps_b.conf_stop()

    print("Exiting.")


def mk_timediff_histo(n: int, s: SHM):

    import numpy as np
    from tqdm import trange

    t = np.zeros(n, np.float64)
    trange_warmed_up = trange(n)

    print(s.md.acqtime)
    s.get_data(True, checkSemAndFlush=True)

    for i in trange_warmed_up:
        s.check_sem_timedwait(1.0)
        t[i] = s.md.acqtime
    print(s.md.acqtime)

    tt = np.diff(t) * 1e6
    counts, bins = np.histogram(tt, bins=30)
    counts = counts ** (1 / 3)

    # Normalize bar lengths
    width = 60
    max_count = counts.max()

    for count, left, right in zip(counts, bins[:-1], bins[1:]):
        bar = "█" * int(width * count / max_count)
        print(f"{left:6.2f} - {right:6.2f} | {bar}")
    print()


def test_pingpong_histogram(fixt_streamdelay_pingpong):
    fps_a, fps_b = fixt_streamdelay_pingpong

    # time.sleep(3)

    from pyMilk.interfacing.shm import SHM

    s = SHM("ping")
    N = 100000

    mk_timediff_histo(N, s)


@pytest.fixture(scope="class")
def fixt_mvm_no_run():
    exec("milk-fpsexec-linalg-MVMextract -procinfo -tmux m:fpsinit")

    fps_m = FPS("m")
    fps_sanitize_start([fps_m])

    fps_m["procinfo.enabled"] = True
    fps_m["procinfo.NBthread"] = 1
    fps_m["procinfo.triggermode"] = 3
    fps_m["procinfo.triggersname"] = "mvm_input"
    fps_m["procinfo.semindexrequested"] = 0
    fps_m["procinfo.loopcntMax"] = -1
    fps_m["procinfo.RTprio"] = 49
    fps_m["procinfo.cset"] = "t4.slice"
    fps_m["procinfo.taskset"] = "4"

    fps_m["GPUindex"] = 0
    fps_m["insname"] = "mvm_input"
    fps_m["immodes"] = "modes_matrix"
    fps_m["outcoeff"] = "mvm_output"
    fps_m["axmode"] = 0

    fps_m.conf_start()
    while not fps_m.conf_isrunning():
        time.sleep(0.001)

    time.sleep(0.5)

    yield fps_m

    # fps_m.run_stop(0.5) # Caller is in charge.
    fps_m.conf_stop(0.5)
    fps_m.destroy()


@pytest.fixture
def fixt_mvm_no_runstart_yes_runstop(fixt_mvm_no_run):
    fps_m: FPS = fixt_mvm_no_run

    yield fps_m

    assert fps_m.run_stop(5.0)
    time.sleep(0.1)


def test_fixt_mvm(fixt_mvm):
    fps_m: FPS = fixt_mvm

    input_stream = SHM("mvm_input", np.zeros((SIZE_X, SIZE_Y), np.float32))
    ping_matrix = SHM(
        "modes_matrix", np.random.randn(SIZE_X, SIZE_Y, N_MODES).astype(np.float32)
    )
    # pung = SHM('mvm_output', np.zeros((1, N_MODES), np.float32))

    assert fps_m.run_start(timeoutsync=5.0)

    input("Check yo life.")


def mvm_correctness_all_params(
    fps_m: FPS,
    axmode: int,
    normalize: bool,
    GPUindex: int,
    masking: bool,
    input_dtype: np.typing.DTypeLike,
):

    fps_m["GPUindex"] = GPUindex
    fps_m["axmode"] = axmode
    fps_m["option.MODENORM"] = normalize

    SIZE_X, SIZE_Y, N_MODES = 100, 200, 300
    OUT_SHAPE = (N_MODES,) if axmode == 0 else (SIZE_X, SIZE_Y)
    OUT_SHAPE_C = (N_MODES, 1) if axmode == 0 else (SIZE_Y, SIZE_X)

    if axmode == 0:
        input_stream = SHM(fps_m["insname"], np.zeros((SIZE_X, SIZE_Y), input_dtype))
    else:
        input_stream = SHM(fps_m["insname"], np.zeros((N_MODES, 1), input_dtype))
    modes_matrix = SHM(
        fps_m["immodes"], np.random.randn(SIZE_X, SIZE_Y, N_MODES).astype(np.float32)
    )

    if masking:
        mask = (
            np.random.rand(SIZE_X, SIZE_Y) > 0.5
        )  # masking always in spatial dimensions, not modal
        mask_stream = SHM("mask", mask.astype(np.float32))
        fps_m["inmasksname"] = "mask"

    assert fps_m.run_start(timeoutsync=5.0)
    ### OH NO !! run_isrunning is actually not a strong enough guarantee,
    # since the output is overwritten some time _after_ that.
    output_stream: SHM | None = None
    for _ in range(1000):
        try:  # May not exist
            output_stream = SHM(fps_m["outcoeff"])
            if output_stream.shape == OUT_SHAPE:
                break
            time.sleep(0.001)
        except:
            pass

    assert output_stream is not None
    assert output_stream.shape == OUT_SHAPE
    assert output_stream.shape_c == OUT_SHAPE_C

    output_stream.get_data(True, timeout=0.001)  # purge

    for _ in range(50):
        x = (np.random.rand(*input_stream.shape) * 1000).astype(input_dtype)
        input_stream.set_data(x)
        y_cacao = output_stream.get_data(True, checkSemAndFlush=False)
        modes = modes_matrix.get_data()

        if masking:
            modes_masked = modes[mask]
            if normalize:
                modes_masked /= np.sum(modes[mask] ** 2, axis=0)
            if axmode == 0:  # mask on input
                y_python = x[mask] @ modes_masked  # 2D mask_pix x n_modes
            else:  # mask on output
                y_python = np.zeros((SIZE_X, SIZE_Y), np.float32)
                y_python.flat[mask.flatten()] = (
                    modes_masked @ x
                )  # 2D mask_pix x n_modes
        else:
            if normalize:
                modes /= np.sum(modes**2, axis=(0, 1))
            if axmode == 0:
                y_python = x.flatten() @ modes.reshape(SIZE_X * SIZE_Y, N_MODES)
            else:
                y_python = (
                    modes.reshape(SIZE_X * SIZE_Y, N_MODES) @ x.flatten()
                ).reshape(SIZE_X, SIZE_Y)

        atol = 1e-5 * np.max(np.abs(y_python))  # That's pretty bad...
        rtol = 1e-2 if normalize else 1e-4
        np.testing.assert_allclose(
            y_cacao,
            y_python,
            atol=atol,
        )
        # Compare relative norm of difference
        assert (
            np.sum((y_cacao - y_python) ** 2) ** 0.5 / np.sum(y_python**2) ** 0.5 < 1e-5
        )

    # TIME REPORT
    t1 = time.time()
    for _ in range(1000):
        input_stream.repost()
        output_stream.check_sem_timedwait(1.0)
    t2 = time.time()
    act = "EXTRACT" if axmode == 0 else "EXPAND"
    print(
        f"ELAPSED [MODE {act}] [{GPUindex=:<3} {normalize=:<5} {masking=:<5} "
        f"input_dtype={str(np.dtype(input_dtype)):<8}] {(t2-t1)*1000:.2f} ms"
    )


class TestsAxmode0:
    """
    Class is used for optimizing fixture recycling between parametrize'd calls
    """

    # Debug parametrizes
    # @pytest.mark.parametrize('input_dtype', [np.uint16])
    # @pytest.mark.parametrize('normalize', [False])
    # @pytest.mark.parametrize('GPUindex', [0])
    # @pytest.mark.parametrize('masking', [False])
    # Full parametrizes
    @pytest.mark.parametrize("normalize", [False, True])
    @pytest.mark.parametrize("GPUindex", [-1, 0, 98, 99])
    @pytest.mark.parametrize("masking", [False, True])
    def test_mvm_correctness(
        self,
        fixt_mvm_no_runstart_yes_runstop,
        normalize: bool,
        GPUindex: int,
        masking: bool,
    ):
        mvm_correctness_all_params(
            fixt_mvm_no_runstart_yes_runstop,
            0,
            normalize,
            GPUindex,
            masking,
            np.float32,
        )

    @pytest.mark.parametrize(
        "input_dtype",
        [
            np.uint16,
            np.float32,
            np.int64,
            np.float64,
            np.uint64,
            np.int16,
            np.uint32,
            np.int32,
        ],
    )
    def test_mvm_correctness_dtypes(
        self, fixt_mvm_no_runstart_yes_runstop, input_dtype: np.typing.DTypeLike
    ):
        mvm_correctness_all_params(
            fixt_mvm_no_runstart_yes_runstop, 0, True, 99, True, input_dtype
        )


class TestsAxmode1:
    """
    Class is used for optimizing fixture recycling between parametrize'd calls
    """

    # Debug parametrizes
    # @pytest.mark.parametrize('input_dtype', [np.float32])
    # @pytest.mark.parametrize('normalize', [False])
    # @pytest.mark.parametrize('GPUindex', [98])
    # @pytest.mark.parametrize('masking', [False])
    # Full parametrizes
    @pytest.mark.parametrize("normalize", [False, True])
    @pytest.mark.parametrize("GPUindex", [-1, 0, 98, 99])
    @pytest.mark.parametrize("masking", [False, True])
    def test_mvm_correctness(
        self,
        fixt_mvm_no_runstart_yes_runstop,
        normalize: bool,
        GPUindex: int,
        masking: bool,
    ):
        mvm_correctness_all_params(
            fixt_mvm_no_runstart_yes_runstop,
            1,
            normalize,
            GPUindex,
            masking,
            np.float32,
        )

    @pytest.mark.parametrize(
        "input_dtype",
        [
            np.uint16,
            np.float32,
            np.int64,
            np.float64,
            np.uint64,
            np.int16,
            np.uint32,
            np.int32,
        ],
    )
    def test_mvm_correctness_dtypes(
        self, fixt_mvm_no_runstart_yes_runstop, input_dtype: np.typing.DTypeLike
    ):
        mvm_correctness_all_params(
            fixt_mvm_no_runstart_yes_runstop, 1, True, 99, True, input_dtype
        )


def test_pingpong_clock_into_linalg(fixt_streamdelay_pingpong, fixt_mvm):
    fps_a, fps_b = fixt_streamdelay_pingpong
    fps_m = fixt_mvm

    from pyMilk.interfacing.shm import SHM

    s = SHM("pung")
    N = 100000

    input("asdfadsfas")

    mk_timediff_histo(N, s)
