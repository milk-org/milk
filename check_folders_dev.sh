#!/usr/bin/env bash

# Keep original PWD
INIT_PWD=$(pwd)

# CACAO

echo "Module CACAO"
if [ -d "plugins/cacao-src" ]; then
    echo "	Already installed -> status:"
    git -C ${INIT_PWD}/plugins/cacao-src status # git -C doesn't cd back to original dir
else
    echo "	Not present."
fi

mkdir -p plugins/milk-extra-src
cd plugins/milk-extra-src




# MILK PLUGINS

for mod in clustering fft ZernikePolyn cudacomp image_basic image_filter image_format image_gen img_reduce info kdtree linARfilterPred linopt_imtools psf statistic
do
    echo "Module ${mod}"
    if [ -d "${mod}" ]; then
        echo "	Already installed -> status:"
	git -C ${INIT_PWD}/plugins/milk-extra-src/${mod} status
    else
        echo "	Not present."
    fi
done
