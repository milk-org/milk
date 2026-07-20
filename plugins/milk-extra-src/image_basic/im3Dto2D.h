// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/** @file im3Dto2D.h
 */

errno_t __attribute__((cold)) im3Dto2D_addCLIcmd();

imageID image_basic_3Dto2D_byID(imageID ID);

imageID image_basic_3Dto2D(const char *__restrict IDname);
