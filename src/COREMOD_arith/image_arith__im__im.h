// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    image_arith__im__im.h
 *
 */

double Ppositive(double a);

/* ------------------------------------------------------------------------- */
/* image  -> image                                                           */
/* ------------------------------------------------------------------------- */

int arith_image_acos_byID(long ID, long IDout);
int arith_image_asin_byID(long ID, long IDout);
int arith_image_atan_byID(long ID, long IDout);
int arith_image_ceil_byID(long ID_name, long IDout);
int arith_image_cos_byID(long ID, long IDout);
int arith_image_cosh_byID(long ID, long IDout);
int arith_image_exp_byID(long ID, long IDout);
int arith_image_fabs_byID(long ID, long IDout);
int arith_image_floor_byID(long ID, long IDout);
int arith_image_ln_byID(long ID, long IDout);
int arith_image_log_byID(long ID, long IDout);
int arith_image_sqrt_byID(long ID, long IDout);
int arith_image_sin_byID(long ID, long IDout);
int arith_image_sinh_byID(long ID, long IDout);
int arith_image_tan_byID(long ID, long IDout);
int arith_image_tanh_byID(long ID, long IDout);

int arith_image_acos(const char *ID_name, const char *ID_out);
int arith_image_acos_IMGID(IMGID *imgin, IMGID *imgout);

int arith_image_asin(const char *ID_name, const char *ID_out);
int arith_image_asin_IMGID(IMGID *imgin, IMGID *imgout);

int arith_image_atan(const char *ID_name, const char *ID_out);
int arith_image_atan_IMGID(IMGID *imgin, IMGID *imgout);

int arith_image_ceil(const char *ID_name, const char *ID_out);
int arith_image_ceil_IMGID(IMGID *imgin, IMGID *imgout);

int arith_image_cos(const char *ID_name, const char *ID_out);
int arith_image_cos_IMGID(IMGID *imgin, IMGID *imgout);

int arith_image_cosh(const char *ID_name, const char *ID_out);
int arith_image_cosh_IMGID(IMGID *imgin, IMGID *imgout);

int arith_image_exp(const char *ID_name, const char *ID_out);
int arith_image_exp_IMGID(IMGID *imgin, IMGID *imgout);

int arith_image_fabs(const char *ID_name, const char *ID_out);
int arith_image_fabs_IMGID(IMGID *imgin, IMGID *imgout);

int arith_image_floor(const char *ID_name, const char *ID_out);
int arith_image_floor_IMGID(IMGID *imgin, IMGID *imgout);

int arith_image_ln(const char *ID_name, const char *ID_out);
int arith_image_ln_IMGID(IMGID *imgin, IMGID *imgout);

int arith_image_log(const char *ID_name, const char *ID_out);
int arith_image_log_IMGID(IMGID *imgin, IMGID *imgout);

int arith_image_sqrt(const char *ID_name, const char *ID_out);
int arith_image_sqrt_IMGID(IMGID *imgin, IMGID *imgout);

int arith_image_sin(const char *ID_name, const char *ID_out);
int arith_image_sin_IMGID(IMGID *imgin, IMGID *imgout);

int arith_image_sinh(const char *ID_name, const char *ID_out);
int arith_image_sinh_IMGID(IMGID *imgin, IMGID *imgout);

int arith_image_tan(const char *ID_name, const char *ID_out);
int arith_image_tan_IMGID(IMGID *imgin, IMGID *imgout);

int arith_image_tanh(const char *ID_name, const char *ID_out);
int arith_image_tanh_IMGID(IMGID *imgin, IMGID *imgout);

int arith_image_positive(const char *ID_name, const char *ID_out);
int arith_image_positive_IMGID(IMGID *imgin, IMGID *imgout);

int arith_image_acos_inplace_byID(long ID);
int arith_image_asin_inplace_byID(long ID);
int arith_image_atan_inplace_byID(long ID);
int arith_image_ceil_inplace_byID(long ID);
int arith_image_cos_inplace_byID(long ID);
int arith_image_cosh_inplace_byID(long ID);
int arith_image_exp_inplace_byID(long ID);
int arith_image_fabs_inplace_byID(long ID);
int arith_image_floor_inplace_byID(long ID);
int arith_image_ln_inplace_byID(long ID);
int arith_image_log_inplace_byID(long ID);
int arith_image_sqrt_inplace_byID(long ID);
int arith_image_sin_inplace_byID(long ID);
int arith_image_sinh_inplace_byID(long ID);
int arith_image_tan_inplace_byID(long ID);
int arith_image_tanh_inplace_byID(long ID);

int arith_image_acos_inplace(const char *ID_name);
int arith_image_asin_inplace(const char *ID_name);
int arith_image_atan_inplace(const char *ID_name);
int arith_image_ceil_inplace(const char *ID_name);
int arith_image_cos_inplace(const char *ID_name);
int arith_image_cosh_inplace(const char *ID_name);
int arith_image_exp_inplace(const char *ID_name);
int arith_image_fabs_inplace(const char *ID_name);
int arith_image_floor_inplace(const char *ID_name);
int arith_image_ln_inplace(const char *ID_name);
int arith_image_log_inplace(const char *ID_name);
int arith_image_sqrt_inplace(const char *ID_name);
int arith_image_sin_inplace(const char *ID_name);
int arith_image_sinh_inplace(const char *ID_name);
int arith_image_tan_inplace(const char *ID_name);
int arith_image_tanh_inplace(const char *ID_name);
