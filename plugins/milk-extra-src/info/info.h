// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file info.h
 * @brief Info module
 */

#if !defined(INFO_H)
#    define INFO_H

void __attribute__((constructor)) libinit_info();

#    include "info/cubeMatchMatrix.h"
#    include "info/cubestats.h"
#    include "info/image_stats.h"

#    include "info/improfile.h"
#    include "info/kbdhit.h"
#    include "info/percentile.h"
#    include "info/print_header.h"

imageID info_cubeMatchMatrix(const char *IDin_name, const char *IDout_name);

#endif
