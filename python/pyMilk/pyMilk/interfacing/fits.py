"""
Fits manipulation routines

with symcode and 3D axis management
As in shmlib, convention is to perform decode on read and encode on write

TODO: write tests
"""

from __future__ import annotations

from astropy.io import fits
import typing as typ

from pyMilk.interfacing.isio_shmlib import SHM
from pyMilk.util import img_shapes

import numpy as np


def _fix_byte_order_on_copy(hdu: fits.PrimaryHDU | fits.ImageHDU) -> np.ndarray:
    return hdu.data.astype(hdu.data.dtype.newbyteorder("="), copy=True)  #


def multi_read(
    file_path: str, symcode: int = 4, tri_dim: int = img_shapes.Which3DState.LAST2LAST
) -> np.ndarray | tuple[np.ndarray, ...]:
    """
    Handle reading of tuples in fits files
    """
    hdu_list = fits.open(file_path, mmap=False)
    if len(hdu_list) == 1:
        data = _fix_byte_order_on_copy(hdu_list[0])  # type: ignore
        n_dim = len(data.shape)
        if n_dim == 2:
            data = img_shapes.image_decode(data, symcode)
        elif n_dim == 3:
            data = img_shapes.full_cube_decode(data, symcode, tri_dim)
    else:
        data = [_fix_byte_order_on_copy(hh) for hh in hdu_list]  # type: ignore
        data = tuple(data)

    # Cleanup
    hdu_list.close()
    for hh in hdu_list:
        del hh.data  # type: ignore

    return data


def multi_write(
    file_path: str,
    data: typ.Iterable[np.ndarray] | np.ndarray,
    symcode: int = 4,
    tri_dim: int = img_shapes.Which3DState.LAST2LAST,
) -> None:
    """
    Handle saving of tuples in fits files
    TODO: symcode and 3Dord handlers
    """
    if isinstance(data, np.ndarray):
        n_dim = len(data.shape)
        if n_dim == 2:
            data = img_shapes.image_encode(data, symcode)
        elif n_dim == 3:
            data = img_shapes.full_cube_encode(data, symcode, tri_dim)
        data = (data,)

    headers = [fits.ImageHDU(m) for m in data]
    headers[0] = fits.PrimaryHDU(data[0])  # type: ignore
    hdu_list = fits.HDUList(headers)
    hdu_list.writeto(file_path, overwrite=True)


def read_fits_load_shm(
    file_path: str,
    shm_name: str,
    symcode: int = 0,
    tri_dim: int = img_shapes.Which3DState.LAST2LAST,
    create_shm: bool = False,
) -> SHM:
    """
    Grab a fits file, load it into the given SHM
    Only a single ndarray per fit - no tuples

    Will fail on size mismatches
    """
    data = multi_read(file_path, symcode, tri_dim)
    assert isinstance(
        data, np.ndarray
    ), "No can work if file_path contains multiple HDUs"

    if create_shm:
        shm = SHM(
            shm_name, data, symcode=symcode, triDim=tri_dim, location=-1, shared=1
        )
    else:
        shm = SHM(shm_name, symcode=symcode, triDim=tri_dim)

    shm.set_data(data)

    return shm


def shm_to_fits(shm_obj_or_name: SHM | str, file_path: str) -> None:
    """
    Get a shm name or object, get the data, write into a fits file
    TODO: symcode and 3Dord handlers
    """
    if isinstance(shm_obj_or_name, str):
        shm = SHM(shm_obj_or_name)
    else:
        shm = shm_obj_or_name

    shm.save_as_fits(file_path)
