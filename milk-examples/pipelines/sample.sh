#!/bin/bash
# Synoptic: mlatsim (simulated DM -> WFS) + aoloop, up to the mlat display.
# Steps: docs/guide/pipelines_tour.md. Linkage: aoloop-conf [systemlinkage.delaysim].
set -e

mkdir -p ${HOME}/AOloop2
cd ${HOME}/AOloop2

# 1. Clone (conf files + scripts/ only)
mokainit clone --milk mlatsim .
mokainit clone --milk aoloop .

# 2. Provision: check, then run hooks and build <name>-rootdir/
mokainit resolve mlatsim-conf
mokainit resolve aoloop-conf
mokainit init mlatsim-conf          # hook writes data/dm.fits
mokainit init aoloop-conf

# 3. Deploy (FPS init, dataloads, conf processes)
moka deploy -p mlatsim
moka deploy -p aoloop

# 4. Link aoloop onto the simulator: aol0_wfsim -> mlatsim_wfs, aol0_dmC -> mlatsim_dm
moka link delaysim -p aoloop

# 5. Run: simulator, WFS acquisition, then mlat (Python only, no CLI yet)
python3 - <<'EOF'
from moka.operate.pipeline import Pipeline
from moka.operate.tasks import RunStart, WaitRunEnd

sim = Pipeline("mlatsim")
sim.task_do(RunStart)                            # dmdelay + wfscam

ao = Pipeline("aoloop")
ao.task_do(RunStart(ao, ["acquWFS"]))            # aol0_wfsim -> aol0_imWFS2
ao.task_do(RunStart(ao, ["mlat"]))
ao.task_do(WaitRunEnd(ao, ["mlat"]))             # single pass, up to 120 s
EOF

# 6. Show the latency measurement
mokashow mlat -p aoloop

# Back down:
# moka teardown -p aoloop ; moka teardown -p mlatsim
