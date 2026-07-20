// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    improfile.h
 */

errno_t improfile_addCLIcmd();

errno_t profile(const char *ID_name,
                const char *outfile,
                double      xcenter,
                double      ycenter,
                double      step,
                long        nb_step);

errno_t profile2im(const char   *profile_name,
                   long          nbpoints,
                   unsigned long size,
                   double        xcenter,
                   double        ycenter,
                   double        radius,
                   const char   *out);
