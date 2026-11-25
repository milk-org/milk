"""Conf hook: write zeros to the FITS file named by argv[1], of dtype argv[2] (float32) and size
argv[3] x argv[3] (16)."""

import sys

import numpy as np
from astropy.io import fits

dtype = sys.argv[2] if len(sys.argv) > 2 else "float32"
size = int(sys.argv[3]) if len(sys.argv) > 3 else 16
fits.writeto(sys.argv[1], np.zeros((size, size), dtype=dtype))
