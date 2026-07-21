// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file fconvolve.h
 * @brief Fconvolve module
 */

/** @file fconvolve.h
 *
 */

errno_t CLIADDCMD_image_filter__fconvolve();

imageID fconvolve(const char *__restrict name_in,
                  const char *__restrict name_ke,
                  const char *__restrict name_out);

imageID fconvolve_padd(const char *__restrict name_in,
                       const char *__restrict name_ke,
                       long paddsize,
                       const char *__restrict name_out);

imageID fconvolve_1(const char *__restrict name_in,
                    const char *__restrict kefft,
                    const char *__restrict name_out);

imageID fconvolveblock(const char *__restrict name_in,
                       const char *__restrict name_ke,
                       const char *__restrict name_out,
                       long blocksize);
