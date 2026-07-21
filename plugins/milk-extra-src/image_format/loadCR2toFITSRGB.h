// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file loadCR2toFITSRGB.h
 * @brief Load CR2 file into R G B images
 */

errno_t CLIADDCMD_image_format__loadCR2toFITSRGB();

errno_t loadCR2toFITSRGB(const char *__restrict fnameCR2,
                         const char *__restrict fnameFITSr,
                         const char *__restrict fnameFITSg,
                         const char *__restrict fnameFITSb);
