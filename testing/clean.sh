#!/bin/bash

(
    cd $HOME/src/pyMilk
    rm -rf build
    rm -rf pyMilk.egg-info
    #rm ./*.so
    rm pyMilk/*.so
)

exit 0
