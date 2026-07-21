// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file image_total.h
 * @brief Image total module
 */

/**
 * @file    image_total.h
 *
 */

#include <libfps/IMGID.h>

double arith_image_total(const char *ID_name);
double arith_image_total_IMGID(IMGID *imgin);

double arith_image_sumsquare(const char *ID_name);
double arith_image_sumsquare_IMGID(IMGID *imgin);
