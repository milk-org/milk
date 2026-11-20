"""Conf hook: write 16x16 zeros to the FITS file named by argv[1], of dtype argv[2] (float32)."""

import sys

import numpy as np
from astropy.io import fits

dtype = sys.argv[2] if len(sys.argv) > 2 else "float32"
fits.writeto(sys.argv[1], np.zeros((16, 16), dtype=dtype))
