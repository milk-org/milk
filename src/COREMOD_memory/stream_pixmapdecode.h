// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/** @file stream_pixmapdecode.h
 */

errno_t stream_pixmapdecode_addCLIcmd();

imageID COREMOD_MEMORY_PixMapDecode_U(const char *inputstream_name,
                                      uint32_t    xsizeim,
                                      uint32_t    ysizeim,
                                      const char *NBpix_fname,
                                      const char *IDmap_name,
                                      const char *IDout_name,
                                      const char *IDout_pixslice_fname,
                                      uint32_t    reverse);
