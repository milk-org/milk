// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef IMAGE_FORMAT_MKBMPIMAGE_H
#define IMAGE_FORMAT_MKBMPIMAGE_H

errno_t CLIADDCMD_image_format__mkBMPimage();

errno_t image_writeBMP(const char *__restrict IDnameR,
                       const char *__restrict IDnameG,
                       const char *__restrict IDnameB,
                       char *__restrict outname);

#endif
