// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/** @file streamtiming_stats.h
 */


errno_t info_image_streamtiming_stats(imageID ID,
                                      int     sem,
                                      long    NBsamplesmax,
                                      float   samplestimeout,
                                      int     buffinit);
