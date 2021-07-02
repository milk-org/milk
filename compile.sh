#!/usr/bin/env bash

# Compile script for milk_package
# Customize / add you own options
# Use as-is, or make a local custom copy (private, do not commit)

mkdir -p _build
cd _build

if [ ! -z $1 ]
then
    MILK_INSTALL_ROOT=$1
fi

if [ -z $MILK_INSTALL_ROOT ]
then
    MILK_INSTALL_ROOT=/usr/local
fi

# find python executable
pythonexec=$(which python)
if command -v python3 &> /dev/null
then
    pythonexec=$(which python3)
fi

echo "using python at ${pythonexec}"

# CMAKE_OPT could be "-DUSE_CUDA=ON" : CMAKE_OPT="-DUSE_CUDA=ON" ./compile.sh
cmake .. $CMAKE_OPT -Dbuild_python_module=ON -DPYTHON_EXECUTABLE=${pythonexec} -DCMAKE_INSTALL_PREFIX=$MILK_INSTALL_ROOT -DCMAKE_BUILD_TYPE=Debug
# cmake .. -DCMAKE_INSTALL_PREFIX=$MILK_INSTALL_ROOT

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
