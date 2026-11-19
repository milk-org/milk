from __future__ import annotations

import os
import glob

from .isio_shmlib import SHM, check_SHM_name
from .shm import MILK_SHM_DIR
from pyMilk.util import img_shapes

import numpy as np

import typing as typ

if typ.TYPE_CHECKING:
    import numpy.typing as npt


def creashmim(
    name: str,
    shape: tuple[int, ...],
    data_type: npt.DTypeLike = np.float32,
    nb_kw: int = 50,
    symcode: int = 4,  # Should it be 4 ??
    tri_dim: int = img_shapes.Which3DState.LAST2LAST,
    delete_existing: bool = False,
    attempt_reuse: bool = True,
    do_zero: bool = True,
) -> SHM:
    """
    See isio_shmlib.py

    Note attempt_reuse intervenes before delete_existing:
    reuse | delete
    False | False   -> systematically overwrite
    False | True    -> delete .im.shm and semaphores if any, recreate
    True  | False   -> Attempt re-use (shape, type, keyword quantity), overwrite upon fail.
    True  | True   -> Attempt re-use (shape, type, keyword quantity). Delete existing files and re-create upon fail.
    """

    shm_handle = None

    if attempt_reuse:
        try:
            # Errors upon non-existence
            shm_handle = SHM(name, symcode=symcode, triDim=tri_dim)

            if not (shm_handle.nptype == data_type and shm_handle.shape == shape):
                # FIXME autosqueeze will make this perform wrong with singleton dims
                # FIXME numpy type aliases will probably make this wrong as well
                raise ValueError("Existing SHM wrong dtype or shape")

            # Errors upon lack of keyword space
            if shm_handle.IMAGE.md.NBkw < nb_kw:
                raise ValueError("Existing SHM overwrite due to not enough kw space.")
            # Success in reopening !
            if do_zero:
                data = shm_handle.get_data().copy()
                data[:] = 0  # Should work with all dtypes
                shm_handle.set_data(data)
            return shm_handle
        except:
            print(f"creashmim {name}: attempt_reuse failed.")

    name_no_ext = check_SHM_name(name)

    if delete_existing:
        try:
            os.remove(f"{MILK_SHM_DIR()}/{name_no_ext}.im.shm")
        except FileNotFoundError:
            pass

    shm_handle = SHM(
        name, (shape, data_type), nbkw=nb_kw, symcode=symcode, triDim=tri_dim
    )

    if do_zero:
        shm_handle.set_data(np.zeros(shape, dtype=data_type))

    return shm_handle
