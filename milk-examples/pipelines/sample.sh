#!/bin/bash
# Synoptic: mlatsim (simulated DM -> WFS) + aoloop, up to the loop running open.
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

# 4. Link aoloop onto the simulator: aol0_wfsim -> mlatsim_wfs, aol0_dmC -> dm07disp03, aol0_dmRM -> dm07disp02
moka link delaysim -p aoloop

# 5. Run: the simulator, then aoloop's acquWFS and mlat (waits for mlat to finish)
moka mlat -p aoloop

# 6. Show the latency measurement
mokashow mlat -p aoloop

# 7. Poke cube, then the linear response (reads mlat's latency; writes calib/RMmodesWFS.fits).
#    mkpokes [TYPE]: hadamard (default) or zonal.
#    measlinresp [-a AMPL]: poke amplitude (default: the measlinresp FPS conf).
moka mkpokes -p aoloop
moka measlinresp -p aoloop

# 8. Control matrices from the response (writes calib/CMmodesWFS.fits, calib/CMmodesDM.fits).
#    mkcm [FLAVOR] [-n NMODES] [-r REG]: FLAVOR tsvd (default), tikhonov (needs -r) or pinv
#    Options: docs/guide/pipelines_plans.md.
moka mkcm -p aoloop

# 9. Load the control matrices, start wfs2modes, mfilt, modes2dm with loopON off.
#    Then close the loop by hand: milk-fpsCTRL, mfilt FPS: loopON, loopgain, loopmult, looplimit.
moka spool -p aoloop

# Back down:
# moka teardown -p aoloop ; moka teardown -p mlatsim
