#!/bin/bash
# Synoptic: mlatsim (simulated DM -> WFS) + aoloop, up to the linear response.
# Steps: docs/guide/pipelines_tour.md. Linkage: aoloop-conf [systemlinkage.delaysim].
set -e

mkdir -p ${HOME}/AOloop2
cd ${HOME}/AOloop2

mokainit clone --milk mlatsim .
mokainit resolve mlatsim-conf
mokainit init mlatsim-conf          # hook writes data/dm.fits
moka deploy -p mlatsim
moka runstart -p mlatsim

mokainit clone --milk aoloop .
mokainit resolve aoloop-conf
mokainit init aoloop-conf

# 3. Deploy (FPS init, dataloads, conf processes)
moka deploy -p aoloop

# 4. Link aoloop onto the simulator: aol0_wfsim -> mlatsim_wfs, aol0_dmC and aol0_dmRM -> mlatsim_dm
moka link delaysim -p aoloop

# 5. Run: the simulator, then aoloop's acquWFS and mlat (waits for mlat to finish)
moka mlat -p aoloop

# 6. Show the latency measurement
mokashow mlat -p aoloop

# 7. Poke cube, then the linear response (reads mlat's latency; writes calib/RMmodesWFS.fits).
#    zonal: the simulated WFS is a delayed copy of the DM, so RMmodesWFS should be ~ RMmodesDM.
moka mkpokes -p aoloop zonal
moka measlinresp -p aoloop
python3 -c "from astropy.io import fits; import numpy as np; \
d = fits.getdata('aoloop-rootdir/calib/RMmodesDM.fits'); \
w = fits.getdata('aoloop-rootdir/calib/RMmodesWFS.fits'); \
print('max |RMmodesWFS - RMmodesDM|:', np.abs(w - d).max())"

# Back down:
# moka teardown -p aoloop ; moka teardown -p mlatsim
