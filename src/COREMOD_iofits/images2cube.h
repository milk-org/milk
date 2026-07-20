// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    images2cube.h
 */

errno_t images2cube_addCLIcmd();

errno_t images_to_cube(const char *restrict img_name,
                       long nbframes,
                       const char *restrict cube_name);
