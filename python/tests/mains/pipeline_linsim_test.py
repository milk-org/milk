from __future__ import annotations

import os
import pytest
from pathlib import Path

from milk.pipeline import Pipeline

from milk.infra.deploy_tasks import (
    DeployFPS,
    InitialFolderSetup,
    StartConfProcesses,
)


def test_pipeline():
    pipeline = Pipeline(Path(__file__).parent.parent / "resources", "aolinearsimulator")
    os.makedirs("AOloop/", exist_ok=True)
    pipeline = pipeline.clone_to("./AOloop")

    pipeline.task_do(InitialFolderSetup).task_do(DeployFPS).task_do(StartConfProcesses)

    from pyMilk.interfacing.shm import SHM
    import numpy as np
    from astropy.io import fits

    dmv = SHM("dmvolt", np.zeros((10, 10), np.float32))

    simu_modes = SHM(
        "aol1_simu_modes",
        fits.getdata(pipeline.root_folder / "data" / "sim_matrix.fits")
        .reshape(20, 10, 100)
        .astype(np.float32),
        symcode=0,
    )

    # TODO
    # Need a cleanup <--- actually
    # rm rootdir, kill tmuxes

    # Need a syntax for reversing tasks.
    # What about pipe.do().do()...
    # and .undo().undo() with a dirty_ok... flag?
