from __future__ import annotations

import os
import time
import numpy as np
from pyMilk.interfacing.shm import SHM

import subprocess, shlex

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
                [
                    "tmux",
                    "send-keys",
                    "-t",
                    target,
                    f"export MILK_SHM_DIR={shm_dir}",
                    "C-m",
                ]
            )
        if proc_dir:
            subprocess.run(
                [
                    "tmux",
                    "send-keys",
                    "-t",
                    target,
                    f"export MILK_PROC_DIR={proc_dir}",
                    "C-m",
                ]
            )


def fps_sanitize_start(fpss: list[FPS]) -> None:
    for fps in fpss:
        fps.tmux_stop()
    time.sleep(1)
    for fps in fpss:
        fps.tmux_start()
    time.sleep(1)
    for fps in fpss:
        fps_inject_env_into_tmux(fps)


def configure_procinfo(fps: FPS) -> None:
    fps["procinfo.enabled"] = True
    fps["procinfo.NBthread"] = 0
    fps["procinfo.triggermode"] = 3
    fps["procinfo.triggersname"] = fps["in_name"]
    fps["procinfo.semindexrequested"] = 0
    fps["procinfo.loopcntMax"] = -1


def test_main_deploy_a_loop():

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

    fps_a.conf_start()
    fps_b.conf_start()

    ping = SHM("ping", np.zeros((20, 30), np.int32))
    pong = SHM("pong", np.zeros((20, 30), np.int32))

    time.sleep(1)

    fps_a.run_start()
    fps_b.run_start()

    # breakpoint()
    input("Stalling, loop is deployed and running...")

    fps_a.run_stop()
    fps_b.run_stop()

    fps_a.conf_stop()
    fps_b.conf_stop()

    print("Exiting.")
