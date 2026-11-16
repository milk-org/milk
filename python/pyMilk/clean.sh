#!/bin/bash

(
    cd
    pip uninstall -y pyMilk
)

(
    cd "$(dirname "$(readlink -f "$0")")"
    rm -rf build
    rm -rf pyMilk.egg-info
    #rm ./*.so
    rm pyMilk/*.so
)

exit 0
