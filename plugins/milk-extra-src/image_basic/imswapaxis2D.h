// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/** @file imswapaxis2D.h
 */

errno_t __attribute__((cold)) imswapaxis2D_addCLIcmd();

imageID image_basic_SwapAxis2D_byID(imageID IDin, const char *__restrict IDout_name);

imageID image_basic_SwapAxis2D(const char *__restrict IDin_name, const char *__restrict IDout_name);
