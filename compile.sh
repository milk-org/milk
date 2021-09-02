#!/usr/bin/env bash


usage()
{
	echo -n "############################################################
# Compile script for milk_package
# Customize / add you own options
# Use as-is, or make a local custom copy (private, do not commit)
############################################################

Examples:

Install to deafult directory (/usr/local)
$ $(basename $0)
$ cd _build; sudo make install

Do not include Python wrapper, build in local dir
$ MILK_PYTHON=\"OFF\" $(basename $0) \$PWD/local
$ cd _build; make install

Build type Debug (default: Release)
$ MILK_BUILD_TYPE=\"Debug\" $(basename $0)
$ cd _build; sudo make install

Use CUDA
$ MILK_CMAKE_OPT=\"-DUSE_CUDA=ON\" $(basename $0)
$ cd _build; sudo make install

### NOTES ###

Grab version and link to folder - if you switch versions regularly, don't forget to hack your way around:
$ sudo ln -s /usr/local/milk-<version> /usr/local/milk

If it's your first compilation EVER
Check you bashrc for
  export MILK_ROOT=${HOME}/src/milk  # point to source code directory. Edit as needed.
  export MILK_INSTALLDIR=/usr/local/milk
  export PATH=${PATH}:${MILK_INSTALLDIR}/bin
  export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:${MILK_INSTALLDIR}/lib/pkgconfig


"
}

if [ "$1" == "-h" ]; then
usage
exit 1
fi



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

BUILDTYPE=${MILK_BUILD_TYPE:-"Release"}


if [ "${MILK_PYTHON}" == "OFF" ]; then

echo "Compiling without Python wrapper"
cmake .. $MILK_CMAKE_OPT -DCMAKE_INSTALL_PREFIX=$MILK_INSTALL_ROOT -DCMAKE_BUILD_TYPE=${BUILDTYPE}

else
# find python executable
pythonexec=$(which python)
if command -v python3 &> /dev/null
then
    pythonexec=$(which python3)
fi
echo "using python at ${pythonexec}"
cmake .. $MILK_CMAKE_OPT -Dbuild_python_module=ON -DPYTHON_EXECUTABLE=${pythonexec} -DCMAKE_INSTALL_PREFIX=$MILK_INSTALL_ROOT -DCMAKE_BUILD_TYPE=${BUILDTYPE}
fi




NCPUS=`fgrep processor /proc/cpuinfo | wc -l`
cmake --build . -- -j $NCPUS



