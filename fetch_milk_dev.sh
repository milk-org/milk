cd extra
for mod in fft ZernikePolyn cudacomp image_basic image_filter image_format image_gen img_reduce info kdtree linARfilterPred linopt_imtools psf statistic 
do
git clone -b dev https://github.com/milk-org/${mod}
done
