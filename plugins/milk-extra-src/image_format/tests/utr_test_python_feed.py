#!/usr/bin/env python

import time
import numpy as np
from pyMilk.interfacing import fits_lib as fli
from pyMilk.interfacing.isio_shmlib import SHM

#DATf = fli.multi_read('/home/vdeo/data/chuck_imro/20210826/buffer_ircam0.fits').astype(np.int16) # CRED2 nocoro

#DATf = fli.multi_read('/home/vdeo/data/chuck_imro/20210827/buffer_ircam0.fits').astype(np.int16) # CRED2 coro

# synth stuff = bias + RON
DATf = np.clip(np.random.randn(8192, 128, 128) * 60. + 2000, 10,
               None).astype(np.int16)
DATf[:, 0] = 0
DATf[:, 0, 2] = (np.arange(8192) % 256)[::-1]

print(DATf.shape)
print(DATf.dtype)
a = SHM('a', DATf[0])

#DATf = np.clip(DATf, None, 3000.)

while True:
    for i in range(len(DATf)):
        time.sleep(0.002)
        a.set_data(DATf[i])
