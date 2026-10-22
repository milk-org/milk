// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file image_stats.h
 * @brief Image stats module
 */

/**
 * @file    image_stats.h
 *
 *
 */

#include <libfps/IMGID.h>

double arith_image_mean(const char *ID_name);
double arith_image_mean_IMGID(IMGID *imgin);

double arith_image_min(const char *ID_name);
double arith_image_min_IMGID(IMGID *imgin);

double arith_image_max(const char *ID_name);
double arith_image_max_IMGID(IMGID *imgin);

double arith_image_percentile(const char *ID_name, double fraction);
double arith_image_percentile_IMGID(IMGID *imgin, double fraction);

double arith_image_median(const char *ID_name);
double arith_image_median_IMGID(IMGID *imgin);

double arith_image_dot(const char *ID1_name, const char *ID2_name);
double arith_image_dot_IMGID(IMGID *imgin1, IMGID *imgin2);

double arith_image_norm(const char *ID_name);
double arith_image_norm_IMGID(IMGID *imgin);

double arith_image_rms(const char *ID_name);
double arith_image_rms_IMGID(IMGID *imgin);

double arith_image_var(const char *ID_name);
double arith_image_var_IMGID(IMGID *imgin);

double arith_image_count(const char *ID_name);
double arith_image_count_IMGID(IMGID *imgin);

/* argmax/argmin/iamax: linear pixel index (x + y*xsize + ...), -1 if every pixel is NaN */
double arith_image_argmax(const char *ID_name);
double arith_image_argmax_IMGID(IMGID *imgin);

double arith_image_argmin(const char *ID_name);
double arith_image_argmin_IMGID(IMGID *imgin);

double arith_image_iamax(const char *ID_name);
double arith_image_iamax_IMGID(IMGID *imgin);

double arith_image_asum(const char *ID_name);
double arith_image_asum_IMGID(IMGID *imgin);

double arith_image_trace(const char *ID_name);
double arith_image_trace_IMGID(IMGID *imgin);
