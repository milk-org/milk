#!/bin/sh

# Compile script for milk_package
# Customize / add you own options
# Do not commit

mkdir -p _build
cd _build

if [ ! -z $1 ]
then
    CREAM_INSTALL_ROOT=$1
fi

if [ -z $CREAM_INSTALL_ROOT ]
then
    CREAM_INSTALL_ROOT=/usr/local
fi

cmake .. -Dbuild_python_module=ON -DPYTHON_EXECUTABLE=$(which python) -DCMAKE_INSTALL_PREFIX=$CREAM_INSTALL_ROOT
# cmake .. -DCMAKE_INSTALL_PREFIX=$CREAM_INSTALL_ROOT

NCPUS=`fgrep processor /proc/cpuinfo | wc -l`

cmake --build . --target install -- -j $NCPUS

# # MANUAL stuff - just as reminder
# # Grab version and link to folder - if you switch versions regularly, don't forget to hack your way around:
# sudo ln -s /usr/local/milk-<version> /usr/local/milk

# # If it's your first compilation EVER
# # Check you bashrc for
# export MILK_ROOT=${HOME}/src/milk  # point to source code directory. Edit as needed.
# export MILK_INSTALLDIR=/usr/local/milk
# export PATH=${PATH}:${MILK_INSTALLDIR}/bin
# export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:${MILK_INSTALLDIR}/lib/pkgconfig
