// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file FITS_to_ushortintbin_lock.h
 * @brief Write ushort binary with file locking
 */

errno_t CLIADDCMD_image_format__ushortintbin_lock();

imageID IMAGE_FORMAT_FITS_to_ushortintbin_lock(const char *__restrict IDname,
                                               const char *__restrict fname);
