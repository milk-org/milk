// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/** @file streamfeed.h
 */

errno_t __attribute__((cold)) streamfeed_addCLIcmd();

long IMAGE_BASIC_streamfeed(const char *__restrict IDname,
                            const char *__restrict streamname,
                            float frequ);
