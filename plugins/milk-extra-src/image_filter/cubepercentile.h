// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file cubepercentile.h
 * @brief Cubepercentile module
 */

/** @file cubepercentile.h
 */

imageID filter_CubePercentile(const char *__restrict IDcin_name,
                              float perc,
                              const char *__restrict IDout_name);

imageID filter_CubePercentileLimit(const char *__restrict IDcin_name,
                                   float perc,
                                   float limit,
                                   const char *__restrict IDout_name);
