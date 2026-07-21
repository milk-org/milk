// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    imresize.h
 * @brief   Resize 2D image.
 */

#ifndef IMAGE_BASIC_IMRESIZE_H
#define IMAGE_BASIC_IMRESIZE_H

errno_t CLIADDCMD_image_basic__imresize();

long basic_resizeim(const char *imname_in, const char *imname_out, long xsizeout, long ysizeout);

#endif
