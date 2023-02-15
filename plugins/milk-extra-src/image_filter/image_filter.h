#if !defined(FILTER_H)
#define FILTER_H

void __attribute__((constructor)) libinit_image_filter();

#include "image_filter/cubepercentile.h"
#include "image_filter/fconvolve.h"
#include "image_filter/fit1D.h"
#include "image_filter/fit2DcosKernel.h"
#include "image_filter/fit2Dcossin.h"
#include "image_filter/gaussfilter.h"
#include "image_filter/medianfilter.h"
#include "image_filter/percentile_interpolation.h"

int f_filter(const char *ID_name, const char *ID_out, float f1, float f2);

#endif
