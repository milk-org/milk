#!/bin/bash
# Synoptic: mlatsim (simulated DM -> WFS) + aoloop, up to the mlat display.
# Steps: docs/guide/pipelines_tour.md. Linkage: aoloop-conf [systemlinkage.delaysim].
set -e

EX=${MILK_ROOT}/milk-examples/pipelines
mkdir -p ${HOME}/AOloop2
cd ${HOME}/AOloop2

# 1. Clone (conf files + scripts/ only)
milk-pipeinit clone ${EX}/mlatsim-conf .
milk-pipeinit clone ${EX}/aoloop-conf .

# 2. Provision: check, then run hooks and build <name>-rootdir/
milk-pipeinit resolve mlatsim-conf
milk-pipeinit resolve aoloop-conf
milk-pipeinit init mlatsim-conf          # hook writes data/dm.fits
milk-pipeinit init aoloop-conf

# 3. Deploy (FPS init, dataloads, conf processes)
milk-pipes deploy -p mlatsim             # creates mlatsim_dm, mlatsim_wfs
milk-pipes deploy -p aoloop

# 4. Link aoloop onto the simulator: aol0_wfsim -> mlatsim_wfs, aol0_dmC -> mlatsim_dm
milk-pipes link delaysim -p aoloop

# 5. Run: simulator, WFS acquisition, then mlat (Python only, no CLI yet)
python3 - <<'EOF'
from milk.operate.pipeline import Pipeline
from milk.operate.tasks import RunStart, WaitRunEnd

sim = Pipeline("mlatsim")
sim.task_do(RunStart)                            # dmdelay + wfscam

ao = Pipeline("aoloop")
ao.task_do(RunStart(ao, ["acquWFS"]))            # aol0_wfsim -> aol0_imWFS2
ao.task_do(RunStart(ao, ["mlat"]))
ao.task_do(WaitRunEnd(ao, ["mlat"]))             # single pass, up to 120 s
EOF

# 6. Show the latency measurement
milk-show mlat -p aoloop

# Back down:
# milk-pipes teardown -p aoloop ; milk-pipes teardown -p mlatsim
