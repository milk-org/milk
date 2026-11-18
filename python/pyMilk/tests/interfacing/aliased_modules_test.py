"""
Test that the deprecated aliased modules
    pyMilk.interfacing.fits_lib and
    pyMilk.interfacing.isio_shmlib
correctly map 1:1 to pyMilk.interfacing.fits and
    pyMilk.interfacing.shm, respectively.
"""

import pytest

from pyMilk.interfacing import fits_lib, fits
from pyMilk.interfacing import isio_shmlib, shm


@pytest.mark.parametrize("attr", [a for a in dir(fits) if not a.startswith("_")])
def test_fits_alias(attr):
    assert getattr(fits_lib, attr) is getattr(fits, attr)


@pytest.mark.parametrize("attr", [a for a in dir(fits_lib) if not a.startswith("_")])
def test_fits_lib_alias(attr):
    assert getattr(fits_lib, attr) is getattr(fits, attr)


@pytest.mark.parametrize("attr", [a for a in dir(shm) if not a.startswith("_")])
def test_shm_alias(attr):
    assert getattr(isio_shmlib, attr) is getattr(shm, attr)


@pytest.mark.parametrize("attr", [a for a in dir(isio_shmlib) if not a.startswith("_")])
def test_isio_shmlib_alias(attr):
    assert getattr(isio_shmlib, attr) is getattr(shm, attr)
