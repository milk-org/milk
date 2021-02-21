#!/usr/bin/env bash

mkdir -p extra
cd extra

for mod in fft ZernikePolyn cudacomp image_basic image_filter image_format image_gen img_reduce info kdtree linARfilterPred linopt_imtools psf statistic
do
echo "Module ${mod}"
if [ -d "${mod}" ]; then
  echo "	Already installed -> skipping"
else
  git clone -b dev https://github.com/milk-org/${mod}
  echo ""
fi
done
