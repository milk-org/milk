#!/usr/bin/env bash

CWD=$PWD
cd plugins/milk-extra-src
for mod in WFpropagate OpticsMaterials AtmosphereModel AtmosphericTurbulence
do
    echo "Module ${mod}"
    if [ -d "${mod}" ]; then
        echo "	Already installed -> pulling"
        (cd ${mod}; git pull)
    else
        git clone -b dev https://github.com/milk-org/${mod}
        echo ""
    fi
done
cd $CWD
