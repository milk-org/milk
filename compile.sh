#!/bin/sh

# Compile script for milk_package
# Customize / add you own options
# Do not commit

mkdir -p _build
cd _build

cmake .. -Dpython_build=ON

NCPUS=`fgrep processor /proc/cpuinfo | wc -l`

make -j$NCPUS

# sudo make install

# # MANUAL stuff - just as reminder
# # Grab version and link to folder - if you switch versions regularly, don't forget to hack your way around:
# sudo ln -s /usr/local/milk-<version> /usr/local/milk

# # If it's your first compilation EVER
# # Check you bashrc for
# export MILK_ROOT=${HOME}/src/milk  # point to source code directory. Edit as needed.
# export MILK_INSTALLDIR=/usr/local/milk
# export PATH=${PATH}:${MILK_INSTALLDIR}/bin
# export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:${MILK_INSTALLDIR}/lib/pkgconfig
