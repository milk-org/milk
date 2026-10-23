// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    imfunctions.h
 * @brief   Per-pixel function-pointer dispatch for image arithmetic
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

#endif // COREMOD_ARITH_IMFUNCTIONS_H
