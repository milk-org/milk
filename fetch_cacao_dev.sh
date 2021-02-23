#!/usr/bin/env bash

./fetch_milk_dev.sh

mkdir -p plugins/cacao-src
cd plugins/cacao-src

for mod in AOloopControl AOloopControl_DM AOloopControl_IOtools AOloopControl_PredictiveControl AOloopControl_acquireCalib AOloopControl_compTools AOloopControl_computeCalib AOloopControl_perfTest FPAOloopControl
do
echo "Module ${mod}"
if [ -d "${mod}" ]; then
  echo "	Already installed -> pulling"
  (cd ${mod}; git pull)
else
  git clone -b dev https://github.com/cacao-org/${mod}
  echo ""
fi
done
