// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    imfunctions.h
 * @brief   Block-vectorized function-pointer dispatch and 2D median filter for image arithmetic
 */

#ifndef COREMOD_ARITH_IMFUNCTIONS_H
#define COREMOD_ARITH_IMFUNCTIONS_H

#include <libfps/IMGID.h>

errno_t arith_image_function_1_1(const char *ID_name,
                                 const char *ID_out,
                                 double (*pt2function)(double));

errno_t arith_image_function_1_1_IMGID(IMGID *imgin, IMGID *imgout, double (*pt2function)(double));

errno_t arith_image_function_1_1_inplace_IMGID(IMGID *imgin, double (*pt2function)(double));

errno_t arith_image_function_1_1_inplace(const char *ID_name, double (*pt2function)(double));

errno_t arith_image_function_2_1(const char *ID_name1,
                                 const char *ID_name2,
                                 const char *ID_out,
                                 double (*pt2function)(double, double));

errno_t arith_image_function_2_1_IMGID(IMGID *imgin1,
                                       IMGID *imgin2,
                                       IMGID *imgout,
                                       double (*pt2function)(double, double));

errno_t arith_img_function_2_1(IMGID  inimg1,
                               IMGID  inimg2,
                               IMGID *outimg,
                               double (*pt2function)(double, double));

errno_t arith_image_function_2_1_inplace(const char *ID_name1,
                                         const char *ID_name2,
                                         double (*pt2function)(double, double));

int arith_image_function_1f_1(const char *ID_name,
                              double      f1,
                              const char *ID_out,
                              double (*pt2function)(double, double));

int arith_image_function_1f_1_IMGID(IMGID *imgin,
                                    double f1,
                                    IMGID *imgout,
                                    double (*pt2function)(double, double));

int arith_image_function_1f_1_inplace_IMGID(IMGID *imgin,
                                            double f1,
                                            double (*pt2function)(double, double));

int arith_image_function_1f_1_inplace(const char *ID_name,
                                      double      f1,
                                      double (*pt2function)(double, double));

int arith_image_function_1ff_1(const char *ID_name,
                               double      f1,
                               double      f2,
                               const char *ID_out,
                               double (*pt2function)(double, double, double));

int arith_image_function_1ff_1_IMGID(IMGID *imgin,
                                     double f1,
                                     double f2,
                                     IMGID *imgout,
                                     double (*pt2function)(double, double, double));

int arith_image_function_1ff_1_inplace(const char *ID_name,
                                       double      f1,
                                       double      f2,
                                       double (*pt2function)(double, double, double));

int arith_image_function_1ff_1_inplace_IMGID(IMGID *imgin,
                                             double f1,
                                             double f2,
                                             double (*pt2function)(double, double, double));

/**
 * arith_image_medfilt2d_im - Run 2D Young-tableau SIMD median filter of half-width @filter_size
 * @src:         Source image (2D or 3D)
 * @dst:         Destination image with matching geometry
 * @filter_size: Window half-width `R` in `{0, 1, 2, 3}` (window `(2*R + 1) x (2*R + 1)`)
 *
 * Return: `RETURN_SUCCESS` on success, `RETURN_FAILURE` if @filter_size is outside `0..3` or NULL.
 */
errno_t arith_image_medfilt2d_im(const IMAGE *src, IMAGE *dst, int filter_size);

#endif // COREMOD_ARITH_IMFUNCTIONS_H
