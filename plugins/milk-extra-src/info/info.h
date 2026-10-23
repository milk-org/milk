// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file info.h
 * @brief Info module
 */

#if !defined(INFO_H)
#    define INFO_H

#    include "milk_runnable.h"

void __attribute__((constructor)) libinit_info();

#    include "info/cubeMatchMatrix.h"
#    include "info/cubestats.h"
#    include "info/image_stats.h"

#    include "info/improfile.h"
#    include "info/kbdhit.h"
#    include "info/percentile.h"
#    include "info/print_header.h"

imageID info_cubeMatchMatrix(const char *IDin_name, const char *IDout_name);

extern const MILK_RUNNABLE runnable_info_cubeMatchMatrix;
extern const MILK_RUNNABLE runnable_info_cubestats;
extern const MILK_RUNNABLE runnable_info_stream_monproc;
extern const MILK_RUNNABLE runnable_info_image_stats_1;
extern const MILK_RUNNABLE runnable_info_image_stats_2;
extern const MILK_RUNNABLE runnable_info_improfile;

#endif
