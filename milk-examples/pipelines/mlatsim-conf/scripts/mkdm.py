"""Conf hook: write a 16x16 float32 zero DM to the FITS file named by argv[1]."""

import sys

import numpy as np
from astropy.io import fits

fits.writeto(sys.argv[1], np.zeros((16, 16), dtype=np.float32))
