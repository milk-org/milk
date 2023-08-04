#!/usr/bin/env bash

echo "Module CACAO"
if [ -d "plugins/cacao-src" ]; then
    echo "	Already installed -> pulling"
    (cd plugins/cacao-src; git pull)
else
    git clone -b dev https://github.com/cacao-org/cacao plugins/cacao-src
    echo ""
fi

topdir="$(dirname $0)"
mkl_sdl=$(grep '^[\t ]*pkg_check_modules( *MKL  *mkl-sdl *) *$' "$topdir/CMakeLists.txt")
[ "$mkl_sdl" ] \
&& sed -i -e '/^[	 ]*lapacke$/s/lapacke/#&/' plugins/cacao-src/computeCalib/CMakeLists.txt \
&& echo "Updated plugins/cacao-src/computeCalib/CMakeLists.txt to remove lapacke library" \
|| true
