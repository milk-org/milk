// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    image_stats.h
 *
 *
 */

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
