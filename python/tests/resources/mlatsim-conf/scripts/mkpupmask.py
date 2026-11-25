"""Conf hook: write a float32 disk (1 inside, 0 outside) to the FITS file named by argv[1], on a
grid of argv[2] x argv[2], centred on (argv[3], argv[4]) with radius argv[5]."""

import sys

import numpy as np
from astropy.io import fits
from pycacao.calib.mkmodes import make_disk

size = int(sys.argv[2])
disk = tuple(float(v) for v in sys.argv[3:6])
fits.writeto(sys.argv[1], make_disk(disk, (size, size)).astype(np.float32))
