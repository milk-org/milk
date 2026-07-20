// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/** @file loadfitsimgcube.h
 */

errno_t __attribute__((cold)) loadfitsimgcube_addCLIcmd();

long load_fitsimages_cube(const char *__restrict strfilter,
                          const char *__restrict ID_out_name);
