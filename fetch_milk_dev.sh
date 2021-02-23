#!/usr/bin/env bash

mkdir -p plugins/milk-extra-src
cd plugins/milk-extra-src

for mod in fft ZernikePolyn cudacomp image_basic image_filter image_format image_gen img_reduce info kdtree linARfilterPred linopt_imtools psf statistic
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
