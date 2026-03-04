/**
 * @file    gaussfilter.h
 * @brief   Gaussian 2D image filtering.
 */

#ifndef IMAGE_FILTER_GAUSSFILTER_H
#define IMAGE_FILTER_GAUSSFILTER_H

errno_t gaussfilter_addCLIcmd();

imageID gauss_filter(
    const char *ID_name,
    const char *out_name,
    float       sigma,
    int         filter_size
);

imageID gauss_3Dfilter(
    const char *ID_name,
    const char *out_name,
    float       sigma,
    int         filter_size
);

#endif
