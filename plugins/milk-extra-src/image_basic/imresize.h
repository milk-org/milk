// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/** @file imresize.h
 */

errno_t imresize_addCLIcmd();

long basic_resizeim(const char *imname_in, const char *imname_out, long xsizeout, long ysizeout);
