"""
tests.conftestaux.gpu_configure

Parse build and system information to make sure we should or shouldn't do GPU testing.
"""

from __future__ import annotations

import os
import subprocess

from pyMilk.interfacing.shm import IMAGESTREAMIO_HAVE_CUDA


def find_nvidia_in_lsmod() -> bool:
    pp = subprocess.run(["lsmod"], stdout=subprocess.PIPE)
    lines = pp.stdout.split(b"\n")
    return any([l.startswith(b"nvidia ") for l in lines])


def build_gpu_tuple() -> tuple[int, ...]:
    # It's complicated if this exists at all...
    CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if len(CUDA_VISIBLE_DEVICES) > 0:
        gpu_cuda_vis = {int(s) for s in CUDA_VISIBLE_DEVICES.split(",")}
    else:
        gpu_cuda_vis = set()

    pp = subprocess.run(["nvidia-smi", "-L"], stdout=subprocess.PIPE)
    all_gpus = len(pp.stdout.strip().split(b"\n"))
    if len(gpu_cuda_vis) == 0:
        return tuple(range(all_gpus))
    else:
        return tuple(set(range(all_gpus)) - gpu_cuda_vis)


NVIDIA_DRIVER_FOUND = find_nvidia_in_lsmod()
GPULIST = () if not NVIDIA_DRIVER_FOUND else build_gpu_tuple()

# Single GPU testing
SINGLE_GPU_TESTING = (0,) if len(GPULIST) >= 1 and IMAGESTREAMIO_HAVE_CUDA else None
MULTI_GPU_TESTING = (0, 1) if len(GPULIST) >= 2 and IMAGESTREAMIO_HAVE_CUDA else None
