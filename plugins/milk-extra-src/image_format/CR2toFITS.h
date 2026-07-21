// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file CR2toFITS.h
 * @brief Convert CR2 file to FITS
 */

errno_t CLIADDCMD_image_format__CR2toFITS();

imageID CR2toFITS(const char *__restrict fnameCR2, const char *__restrict fnameFITS);
